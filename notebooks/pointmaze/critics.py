"""The same three factorizations, over continuous positions and action chunks."""

import flax.linen as nn
import jax
import jax.numpy as jnp


class MLP(nn.Module):
    out: int
    width: int = 256
    depth: int = 2

    @nn.compact
    def __call__(self, x):
        for _ in range(self.depth):
            x = nn.swish(nn.Dense(self.width)(x))
        return nn.Dense(self.out)(x)


def energy_fn(name, x, y):
    if name == "norm":
        return -jnp.sqrt(((x - y) ** 2).sum(-1) + 1e-6)
    if name == "l2":
        return -((x - y) ** 2).sum(-1)
    if name == "dot":
        return (x * y).sum(-1)
    raise ValueError(name)


def _bilinear(left_of, right_of, left_dim, right_dim, repr_dim, energy, width, depth):
    """Two towers scored against each other. `norm` for the contrastive critics and `dot` for the
    TD ones -- a distance energy is bounded above by 0 and cannot represent a positive return."""

    def build(heads):
        L, R = MLP(repr_dim, width, depth), MLP(repr_dim * heads, width, depth)

        def init(key):
            k1, k2 = jax.random.split(key)
            return {"l": L.init(k1, jnp.zeros((1, left_dim))),
                    "r": R.init(k2, jnp.zeros((1, right_dim)))}

        def towers(p, s, g, seq):
            left = L.apply(p["l"], left_of(s, g, seq))
            right = R.apply(p["r"], right_of(s, g, seq))
            return left, right.reshape(right.shape[:-1] + (heads, repr_dim))

        def q(p, s, g, seq):
            left, right = towers(p, s, g, seq)
            return energy_fn(energy, left[..., None, :], right)

        def logits(p, s, g, seq):
            left, right = towers(p, s, g, seq)
            return energy_fn(energy, left[:, None, :], right[None, :, 0, :])

        return init, q, logits

    return build


def encoder(scale, n_freq):
    """How a position enters a network: scaled to [0, 1], then (with n_freq > 0) alongside
    sin/cos features at 1, 2, ..., n_freq cycles across the world.

    Raw coordinates are the continuous analogue of the gridworld's `coords` features, and fail
    the same way: a value function jumps across a wall, and an MLP of (x, y) can only bend so
    sharply, so the regression control tops out around 0.5 here. The Fourier features are the
    analogue of one-hot: they let the network resolve position at the scale of a cell, and the
    control climbs to where the objectives can be compared against it.
    """
    freqs = 2 * jnp.pi * jnp.arange(1, n_freq + 1, dtype=jnp.float32)

    def encode(pos):
        x = pos * scale
        if n_freq == 0:
            return x
        ang = x[..., :, None] * freqs                                   # (..., 2, n_freq)
        return jnp.concatenate([x, jnp.sin(ang).reshape(*x.shape[:-1], -1),
                                jnp.cos(ang).reshape(*x.shape[:-1], -1)], -1)

    return encode, 2 + 4 * n_freq


def sa_g_bilinear(h, scale, repr_dim=64, energy="norm", width=256, depth=2, n_freq=16):
    """CRL's factorization: energy(f([s; a_tau]), g(goal))."""
    enc, dim = encoder(scale, n_freq)
    return _bilinear(lambda s, g, seq: jnp.concatenate([enc(s), seq], -1),
                     lambda s, g, seq: enc(g),
                     left_dim=dim + 2 * h, right_dim=dim, repr_dim=repr_dim, energy=energy,
                     width=width, depth=depth)


def sg_a_bilinear(h, scale, repr_dim=64, energy="dot", width=256, depth=2, n_freq=16):
    """CARL's factorization: energy(phi([s; g]), psi(a_tau))."""
    enc, dim = encoder(scale, n_freq)
    return _bilinear(lambda s, g, seq: jnp.concatenate([enc(s), enc(g)], -1),
                     lambda s, g, seq: seq,
                     left_dim=2 * dim, right_dim=2 * h, repr_dim=repr_dim, energy=energy,
                     width=width, depth=depth)


def monolithic(h, scale, width=256, depth=2, n_freq=16):
    """One MLP over [s; g; a_tau]."""
    enc, dim = encoder(scale, n_freq)

    def build(heads):
        net = MLP(heads, width, depth)

        def init(key):
            return {"q": net.init(key, jnp.zeros((1, 2 * dim + 2 * h)))}

        def q(p, s, g, seq):
            return net.apply(p["q"], jnp.concatenate([enc(s), enc(g), seq], -1))

        return init, q, None

    return build
