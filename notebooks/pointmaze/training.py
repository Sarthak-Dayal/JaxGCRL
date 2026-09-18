"""The objectives. The only structural change from the discrete world is the max over actions.

With a continuous action space the chunk space is R^(2h), so the exact max and logsumexp of the
discrete notebook become a *sampled* max over `n_cand` candidate chunks drawn uniformly from
[-1, 1]^(2h) -- the same approximation QT-Opt makes. It keeps the setup actor-free; the cost is
that the target is a lower bound on the true max, biased down when `n_cand` is small.
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


def train_td(build, batches, cand, kind, steps, key, alpha=0.05, tau=0.01, lr=3e-4):
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

    @jax.jit
    def step(params, target, opt_state, b):
        tq = jax.lax.stop_gradient(
            b["reward"] + b["discount"] * next_value(target, b["next_state"], b["goal"]))

        def loss(p):
            return jnp.mean((q(p, b["state"], b["goal"], b["seq"]) - tq[:, None]) ** 2)

        l, g = jax.value_and_grad(loss)(params)
        upd, opt_state = opt.update(g, opt_state)
        params = optax.apply_updates(params, upd)
        target = jax.tree_util.tree_map(lambda t, p: (1 - tau) * t + tau * p, target, params)
        return params, target, opt_state, l

    losses = []
    for i in range(steps):
        params, target, opt_state, l = step(params, target, opt_state, batches(i))
        losses.append(float(l))
    return params, q, losses


def train_contrastive(build, batches, steps, key, lr=3e-4, logsumexp_coeff=0.1):
    init, q, logits = build(1)
    params = init(key)
    opt = optax.adam(lr)
    opt_state = opt.init(params)

    @jax.jit
    def step(params, opt_state, b):
        def loss(p):
            lg = logits(p, b["state"], b["goal"], b["seq"])
            lse = jax.nn.logsumexp(lg, 1)
            return -jnp.mean(jnp.diag(lg) - lse) + logsumexp_coeff * jnp.mean(lse**2)

        l, g = jax.value_and_grad(loss)(params)
        upd, opt_state = opt.update(g, opt_state)
        return optax.apply_updates(params, upd), opt_state, l

    losses = []
    for i in range(steps):
        params, opt_state, l = step(params, opt_state, batches(i))
        losses.append(float(l))
    return params, q, losses


def train_all(W, h, steps, batch=256, seed=0, shaping=0.0, n_cand=64, n_traj=300, horizon=None):
    """Eight critics on identical data; baselines and sa_g variants at h = 1, CARL at h."""
    P, A = collect(W, n_traj=n_traj, T=horizon)
    B1 = make_batches(W, P, A, 1, steps, batch, shaping=shaping)
    Bh = make_batches(W, P, A, h, steps, batch, shaping=shaping)
    take = lambda B: (lambda i: {k: jnp.asarray(v[i]) for k, v in B.items()})
    c1, ch = candidates(1, n_cand), candidates(h, n_cand)

    specs = [
        ("CRL",       sa_g_bilinear(1, energy="norm"), take(B1), "contrastive", c1),
        ("CARL-CRL",  sg_a_bilinear(h, energy="norm"), take(Bh), "contrastive", ch),
        ("SAC",       monolithic(1),                   take(B1), "sac", c1),
        ("sa_g_sac",  sa_g_bilinear(1, energy="dot"),  take(B1), "sac", c1),
        ("CARL-SAC",  sg_a_bilinear(h, energy="dot"),  take(Bh), "sac", ch),
        ("TD3",       monolithic(1),                   take(B1), "td3", c1),
        ("sa_g_td3",  sa_g_bilinear(1, energy="dot"),  take(B1), "td3", c1),
        ("CARL-TD3",  sg_a_bilinear(h, energy="dot"),  take(Bh), "td3", ch),
    ]
    out = {}
    for i, (name, build, batches, kind, cand) in enumerate(specs):
        key = jax.random.PRNGKey(seed + i)
        if kind == "contrastive":
            p, q, losses = train_contrastive(build, batches, steps, key)
        else:
            p, q, losses = train_td(build, batches, cand, kind, steps, key)
        out[name] = dict(params=p, q=q, cand=cand, losses=losses, data=(P, A))
        yield name, out
