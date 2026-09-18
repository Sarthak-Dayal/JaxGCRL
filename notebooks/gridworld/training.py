"""The two objectives -- InfoNCE and TD -- and the sweep over all eight critics."""

import jax
import jax.numpy as jnp
import optax

from .critics import monolithic, sa_g_bilinear, sg_a_bilinear
from .data import collect, make_batches
from .world import all_chunks


# Every trainer takes `batches` as one dict of stacked arrays, (steps, batch, ...) per key, and
# runs the whole loop as a single `lax.scan`. Feeding a step at a time from the host costs a
# device copy and a blocking `float(loss)` per step, which on a GPU is most of the wall-clock
# for networks this small. Losses come back as a list so notebook code can test them for truth.
#
# The compiled loop is cached per (critic builder, objective, hyperparameters): a sweep trains
# the same architecture on dataset after dataset, and without the cache every one of those is a
# fresh XLA compile that costs more than the training it does.

_compiled = {}


def _cached(key, build, make):
    """The jitted training loop for `build` under `key`, compiled once. The builder object is
    kept alongside so a recycled id() cannot hand back another architecture's loop."""
    hit = _compiled.get(key)
    if hit is None or hit[0] is not build:
        hit = _compiled[key] = (build, *make())
    return hit[1:]


def train_td(build, batches, feats, chunks, kind, key, alpha=0.05, tau=0.01, lr=3e-4):
    """Soft (SAC-style) or clipped-double (TD3-style) TD, with the next-chunk value taken exactly
    over every chunk rather than from an actor."""
    def make():
        # Both use clipped double-Q: SAC takes the min over twin critics in its target just as
        # TD3 does. With a single head the sampled max overestimates and the critic diverges --
        # badly enough in a continuous action space to point the policy away from the goal.
        init, q, _ = build(2)
        opt = optax.adam(lr)

        def next_value(tp, ns, gf, F, C):
            sf = jnp.repeat(F[ns][:, None, :], C.shape[0], 1)
            gg = jnp.repeat(gf[:, None, :], C.shape[0], 1)
            qs = q(tp, sf, gg, jnp.broadcast_to(C, (ns.shape[0],) + C.shape))
            qs = jnp.min(qs, -1)
            # soft value for SAC, hard max for TD3 -- that is now the only difference between them
            return alpha * jax.nn.logsumexp(qs / alpha, -1) if kind == "sac" else jnp.max(qs, -1)

        @jax.jit
        def run(key, batches, F, C):
            params = target = init(key)
            opt_state = opt.init(params)

            def step(carry, b):
                params, target, opt_state = carry
                gf = F[b["goal"]]
                tq = jax.lax.stop_gradient(
                    b["reward"] + b["discount"] * next_value(target, b["next_state"], gf, F, C))

                def loss(p):
                    return jnp.mean((q(p, F[b["state"]], gf, b["seq"]) - tq[:, None]) ** 2)

                l, g = jax.value_and_grad(loss)(params)
                upd, opt_state = opt.update(g, opt_state)
                params = optax.apply_updates(params, upd)
                target = jax.tree_util.tree_map(lambda t, p: (1 - tau) * t + tau * p, target, params)
                return (params, target, opt_state), l

            (params, _, _), losses = jax.lax.scan(step, (params, target, opt_state), batches)
            return params, losses

        return run, q

    run, q = _cached(("td", id(build), kind, alpha, tau, lr), build, make)
    params, losses = run(key, jax.device_put(batches), jnp.asarray(feats), jnp.asarray(chunks))
    return params, q, losses.tolist(), []   # no classification accuracy for a regression loss


def train_contrastive(build, batches, feats, key, lr=3e-4, logsumexp_coeff=0.1,
                      loss_fn="infonce"):
    """Contrastive fitting over the batch.

    `infonce` is what JaxGCRL's CRL does: forward InfoNCE on the (state-action, goal) logits with
    the logsumexp penalty that keeps them from drifting.

    `binary_nce` scores every pair independently -- softplus(-logit) on the diagonal and
    softplus(logit) off it -- rather than normalizing across a row. That suits the phi([s;g]),
    psi(a_tau) factorization better: an InfoNCE row there normalizes over the batch's *action
    sequences*, so the loss depends on which actions happen to be in the batch rather than on
    anything about the goal, and identical action sequences appear as false negatives.
    """
    def make():
        init, q, logits = build(1)
        opt = optax.adam(lr)

        @jax.jit
        def run(key, batches, F):
            params = init(key)
            opt_state = opt.init(params)

            def step(carry, b):
                params, opt_state = carry

                def loss(p):
                    # For CRL every (s, a) is scored against every goal in the batch; for CARL
                    # every (s, g) against every action sequence. The diagonal is the pair that
                    # occurred.
                    lg = logits(p, F[b["state"]], F[b["goal"]], b["seq"])
                    eye = jnp.eye(lg.shape[0])
                    if loss_fn == "binary_nce":
                        # Balanced accuracy: a batch is 255/256 negatives, so plain accuracy is
                        # ~1.0 for a critic that calls everything a negative. Average the classes.
                        pos = jnp.sum((lg > 0) * eye) / jnp.sum(eye)
                        neg = jnp.sum((lg <= 0) * (1 - eye)) / jnp.sum(1 - eye)
                        acc = 0.5 * (pos + neg)
                        return jnp.mean(eye * jax.nn.softplus(-lg)
                                        + (1 - eye) * jax.nn.softplus(lg)), acc
                    # accuracy: does the true pair win its row? (carl_crl: categorical_accuracy)
                    acc = jnp.mean(jnp.argmax(lg, 1) == jnp.arange(lg.shape[0]))
                    lse = jax.nn.logsumexp(lg, 1)
                    fwd = -jnp.mean(jnp.diag(lg) - lse)                      # rows: over goals
                    bwd = -jnp.mean(jnp.diag(lg) - jax.nn.logsumexp(lg, 0))  # columns: over (s, a)
                    nce = {"infonce": fwd, "bwd_infonce": bwd, "sym_infonce": fwd + bwd}[loss_fn]
                    return nce + logsumexp_coeff * jnp.mean(lse**2), acc

                (l, acc), g = jax.value_and_grad(loss, has_aux=True)(params)
                upd, opt_state = opt.update(g, opt_state)
                return (optax.apply_updates(params, upd), opt_state), (l, acc)

            (params, _), (losses, accs) = jax.lax.scan(step, (params, opt_state), batches)
            return params, losses, accs

        return run, q

    run, q = _cached(("contrastive", id(build), loss_fn, logsumexp_coeff, lr), build, make)
    params, losses, accs = run(key, jax.device_put(batches), jnp.asarray(feats))
    return params, q, losses.tolist(), accs.tolist()


def train_regression(build, batches, feats, key, lr=3e-4):
    """Regress the architecture straight onto the optimal Q. No bootstrapping, no negatives --
    the ceiling a given factorization can reach on a given dataset."""
    def make():
        init, q, _ = build(1)
        opt = optax.adam(lr)

        @jax.jit
        def run(key, batches, F):
            params = init(key)
            opt_state = opt.init(params)

            def step(carry, b):
                params, opt_state = carry

                def loss(p):
                    pred = q(p, F[b["state"]], F[b["goal"]], b["seq"])[..., 0]
                    return jnp.mean((pred - b["q_star"]) ** 2)

                l, g = jax.value_and_grad(loss)(params)
                upd, opt_state = opt.update(g, opt_state)
                return (optax.apply_updates(params, upd), opt_state), l

            (params, _), losses = jax.lax.scan(step, (params, opt_state), batches)
            return params, losses

        return run, q

    run, q = _cached(("regression", id(build), lr), build, make)
    params, losses = run(key, jax.device_put(batches), jnp.asarray(feats))
    return params, q, losses.tolist(), []


def train_all(W, n, steps, batch=256, seed=0, shaping=0.0, n_traj=400, horizon=None):
    """Three factorizations crossed with the objectives that suit them, on identical data.

    The baselines and the sa_g variants see n = 1, so the only thing that changes between SAC and
    sa_g_sac is the factorization; CARL additionally chunks.
    """
    S, A = collect(W, n_traj=n_traj, T=horizon)
    B1 = make_batches(W, S, A, 1, steps, batch, shaping=shaping)
    Bn = make_batches(W, S, A, n, steps, batch, shaping=shaping)
    one, many = all_chunks(1), all_chunks(n)
    fd = W["feats"].shape[1]

    specs = [
        # name        factorization                     batches    objective       chunks
        ("CRL",       sa_g_bilinear(1, fd, energy="norm"), B1, "contrastive", one),
        ("CARL-CRL",  sg_a_bilinear(n, fd, energy="norm"), Bn, "contrastive", many),
        ("SAC",       monolithic(1, fd),                   B1, "sac", one),
        ("sa_g_sac",  sa_g_bilinear(1, fd, energy="dot"),  B1, "sac", one),
        ("CARL-SAC",  sg_a_bilinear(n, fd, energy="dot"),  Bn, "sac", many),
        ("TD3",       monolithic(1, fd),                   B1, "td3", one),
        ("sa_g_td3",  sa_g_bilinear(1, fd, energy="dot"),  B1, "td3", one),
        ("CARL-TD3",  sg_a_bilinear(n, fd, energy="dot"),  Bn, "td3", many),
    ]
    out = {}
    for i, (name, build, batches, kind, chunks) in enumerate(specs):
        key = jax.random.PRNGKey(seed + i)
        if kind == "contrastive":
            p, q, losses, accs = train_contrastive(build, batches, W["feats"], key)
        else:
            p, q, losses, accs = train_td(build, batches, W["feats"], chunks, kind, key)
        out[name] = dict(params=p, q=q, chunks=chunks, losses=losses, accs=accs, data=(S, A),
                         objective=kind)
        yield name, out
