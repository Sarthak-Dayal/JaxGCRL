"""The same three factorizations, over continuous action chunks."""

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


def _bilinear(left_of, right_of, left_dim, right_dim, repr_dim, energy):
    """Two towers scored against each other. `norm` for the contrastive critics and `dot` for the
    TD ones -- a distance energy is bounded above by 0 and cannot represent a positive return."""

    def build(heads):
        L, R = MLP(repr_dim), MLP(repr_dim * heads)

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


def sa_g_bilinear(h, repr_dim=64, energy="norm"):
    """CRL's factorization: energy(f([s; a_tau]), g(goal))."""
    return _bilinear(lambda s, g, seq: jnp.concatenate([s, seq], -1),
                     lambda s, g, seq: g,
                     left_dim=2 + 2 * h, right_dim=2, repr_dim=repr_dim, energy=energy)


def sg_a_bilinear(h, repr_dim=64, energy="dot"):
    """CARL's factorization: energy(phi([s; g]), psi(a_tau))."""
    return _bilinear(lambda s, g, seq: jnp.concatenate([s, g], -1),
                     lambda s, g, seq: seq,
                     left_dim=4, right_dim=2 * h, repr_dim=repr_dim, energy=energy)


def monolithic(h):
    """One MLP over [s; g; a_tau]."""

    def build(heads):
        net = MLP(heads)

        def init(key):
            return {"q": net.init(key, jnp.zeros((1, 4 + 2 * h)))}

        def q(p, s, g, seq):
            return net.apply(p["q"], jnp.concatenate([s, g, seq], -1))

        return init, q, None

    return build
