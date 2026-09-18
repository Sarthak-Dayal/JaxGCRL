"""The objectives. The only structural change from the discrete world is the max over actions.

With a continuous action space the chunk space is R^(2h), so the exact max and logsumexp of the
discrete notebook become a *sampled* max over `n_cand` candidate chunks drawn uniformly from
[-1, 1]^(2h) -- the same approximation QT-Opt makes. It keeps the setup actor-free; the cost is
that the target is a lower bound on the true max, biased down when `n_cand` is small.

Every trainer takes `batches` as one dict of stacked arrays, (steps, batch, ...) per key, and runs
the whole loop as a single `lax.scan`. Losses come back as a list so notebook code can test them
for truth.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .critics import monolithic, sa_g_bilinear, sg_a_bilinear
from .data import collect, make_batches


def candidates(h, n_cand, seed=0):
    """A fixed pool of action chunks, reused everywhere so the max is over the same set."""
    return np.random.default_rng(seed).uniform(-1, 1, (n_cand, 2 * h)).astype(np.float32)


def train_td(build, batches, cand, kind, key, alpha=0.05, tau=0.01, lr=3e-4):
    # Both use clipped double-Q: SAC takes the min over twin critics in its target just as TD3
    # does. With a single head the sampled max overestimates and the critic diverges -- badly
    # enough in a continuous action space to point the policy away from the goal.
    heads = 2
    init, q, _ = build(heads)
    params = target = init(key)
    opt = optax.adam(lr)
    opt_state = opt.init(params)
    C = jnp.asarray(cand)

    def next_value(tp, ns, g):
        s = jnp.repeat(ns[:, None, :], C.shape[0], 1)
        gg = jnp.repeat(g[:, None, :], C.shape[0], 1)
        qs = q(tp, s, gg, jnp.broadcast_to(C, (ns.shape[0],) + C.shape))
        qs = jnp.min(qs, -1)
        # Soft value for SAC, hard max for TD3 -- now the only difference between them.
        #
        # The soft value over a continuous action space is alpha * log integral exp(Q/alpha) da.
        # Estimating that from n_cand samples needs the - log(n_cand) term: without it the bonus
        # grows with the pool size, and worse, it grows with how flat Q is across candidates --
        # which is largest far from the goal, so it tilts the policy away from it.
        if kind == "sac":
            return alpha * (jax.nn.logsumexp(qs / alpha, -1) - jnp.log(qs.shape[-1]))
        return jnp.max(qs, -1)

    def step(carry, b):
        params, target, opt_state = carry
        tq = jax.lax.stop_gradient(
            b["reward"] + b["discount"] * next_value(target, b["next_state"], b["goal"]))

        def loss(p):
            return jnp.mean((q(p, b["state"], b["goal"], b["seq"]) - tq[:, None]) ** 2)

        l, g = jax.value_and_grad(loss)(params)
        upd, opt_state = opt.update(g, opt_state)
        params = optax.apply_updates(params, upd)
        target = jax.tree_util.tree_map(lambda t, p: (1 - tau) * t + tau * p, target, params)
        return (params, target, opt_state), l

    (params, _, _), losses = jax.lax.scan(step, (params, target, opt_state),
                                          jax.device_put(batches))
    return params, q, losses.tolist(), []


def train_contrastive(build, batches, key, lr=3e-4, logsumexp_coeff=0.1, loss_fn="infonce"):
    """Contrastive fitting over the batch: forward InfoNCE (rows normalized over the right
    tower's entries), backward (columns, over the left tower's), symmetric, or per-pair binary
    NCE. The logsumexp penalty is the repo's."""
    init, q, logits = build(1)
    params = init(key)
    opt = optax.adam(lr)
    opt_state = opt.init(params)

    def step(carry, b):
        params, opt_state = carry

        def loss(p):
            lg = logits(p, b["state"], b["goal"], b["seq"])
            eye = jnp.eye(lg.shape[0])
            if loss_fn == "binary_nce":
                pos = jnp.sum((lg > 0) * eye) / jnp.sum(eye)
                neg = jnp.sum((lg <= 0) * (1 - eye)) / jnp.sum(1 - eye)
                return jnp.mean(eye * jax.nn.softplus(-lg) + (1 - eye) * jax.nn.softplus(lg)), \
                    0.5 * (pos + neg)
            acc = jnp.mean(jnp.argmax(lg, 1) == jnp.arange(lg.shape[0]))
            lse = jax.nn.logsumexp(lg, 1)
            fwd = -jnp.mean(jnp.diag(lg) - lse)
            bwd = -jnp.mean(jnp.diag(lg) - jax.nn.logsumexp(lg, 0))
            nce = {"infonce": fwd, "bwd_infonce": bwd, "sym_infonce": fwd + bwd}[loss_fn]
            return nce + logsumexp_coeff * jnp.mean(lse**2), acc

        (l, acc), g = jax.value_and_grad(loss, has_aux=True)(params)
        upd, opt_state = opt.update(g, opt_state)
        return (optax.apply_updates(params, upd), opt_state), (l, acc)

    (params, _), (losses, accs) = jax.lax.scan(step, (params, opt_state),
                                               jax.device_put(batches))
    return params, q, losses.tolist(), accs.tolist()


def train_regression(build, batches, key, lr=3e-4):
    """Regress the architecture straight onto the optimal Q: the ceiling a given factorization
    can reach on a given dataset."""
    init, q, _ = build(1)
    params = init(key)
    opt = optax.adam(lr)
    opt_state = opt.init(params)

    def step(carry, b):
        params, opt_state = carry

        def loss(p):
            pred = q(p, b["state"], b["goal"], b["seq"])[..., 0]
            return jnp.mean((pred - b["q_star"]) ** 2)

        l, g = jax.value_and_grad(loss)(params)
        upd, opt_state = opt.update(g, opt_state)
        return (optax.apply_updates(params, upd), opt_state), l

    (params, _), losses = jax.lax.scan(step, (params, opt_state), jax.device_put(batches))
    return params, q, losses.tolist(), []


def train_all(W, h, steps, batch=256, seed=0, shaping=0.0, n_cand=64, n_traj=300, horizon=None):
    """Eight critics on identical data; baselines and sa_g variants at h = 1, CARL at h."""
    P, A = collect(W, n_traj=n_traj, T=horizon)
    B1 = make_batches(W, P, A, 1, steps, batch, shaping=shaping)
    Bh = make_batches(W, P, A, h, steps, batch, shaping=shaping)
    c1, ch = candidates(1, n_cand), candidates(h, n_cand)
    scale = 1.0 / W["n"]

    specs = [
        ("CRL",       sa_g_bilinear(1, scale, energy="norm"), B1, "contrastive", c1),
        ("CARL-CRL",  sg_a_bilinear(h, scale, energy="norm"), Bh, "contrastive", ch),
        ("SAC",       monolithic(1, scale),                   B1, "sac", c1),
        ("sa_g_sac",  sa_g_bilinear(1, scale, energy="dot"),  B1, "sac", c1),
        ("CARL-SAC",  sg_a_bilinear(h, scale, energy="dot"),  Bh, "sac", ch),
        ("TD3",       monolithic(1, scale),                   B1, "td3", c1),
        ("sa_g_td3",  sa_g_bilinear(1, scale, energy="dot"),  B1, "td3", c1),
        ("CARL-TD3",  sg_a_bilinear(h, scale, energy="dot"),  Bh, "td3", ch),
    ]
    out = {}
    for i, (name, build, batches, kind, cand) in enumerate(specs):
        key = jax.random.PRNGKey(seed + i)
        if kind == "contrastive":
            p, q, losses, _ = train_contrastive(build, batches, key)
        else:
            p, q, losses, _ = train_td(build, batches, cand, kind, key)
        out[name] = dict(params=p, q=q, cand=cand, losses=losses, data=(P, A))
        yield name, out
