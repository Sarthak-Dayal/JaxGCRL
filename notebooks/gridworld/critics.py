"""The three critic factorizations the experiment compares."""

import flax.linen as nn
import jax
import jax.numpy as jnp

from .world import N_ACT


class MLP(nn.Module):
    out: int
    width: int = 128
    depth: int = 2

    @nn.compact
    def __call__(self, x):
        for _ in range(self.depth):
            x = nn.swish(nn.Dense(self.width)(x))
        return nn.Dense(self.out)(x)


def _onehot(seq, n):
    """A chunk as flat one-hots -- the discrete stand-in for `psi_input`."""
    return jnp.reshape(jax.nn.one_hot(seq, N_ACT), seq.shape[:-1] + (n * N_ACT,))


def energy_fn(name, x, y):
    """The repo's energies, on the same axis convention."""
    if name == "norm":
        return -jnp.sqrt(((x - y) ** 2).sum(-1) + 1e-6)
    if name == "l2":
        return -((x - y) ** 2).sum(-1)
    if name == "dot":
        return (x * y).sum(-1)
    raise ValueError(name)


def _bilinear(n, left_of, right_of, left_dim, right_dim, repr_dim, energy, width):
    """A two-tower critic: energy(left(...), right(...)).

    `energy` matters more than it looks. `norm` and `l2` are bounded above by 0, which is fine for
    a contrastive loss -- only the ordering of the logits enters InfoNCE -- but a TD target here is
    a positive discounted return, and a critic that cannot represent a positive number collapses
    to Q = 0 everywhere. Hence `norm` for the contrastive critics and `dot` for the TD ones, which
    is exactly the split between carl_crl and carl_sac/carl_td3 in the repo.
    """

    def build(heads):
        L, R = MLP(repr_dim, width), MLP(repr_dim * heads, width)

        def init(key):
            k1, k2 = jax.random.split(key)
            return {"l": L.init(k1, jnp.zeros((1, left_dim))),
                    "r": R.init(k2, jnp.zeros((1, right_dim)))}

        def towers(p, sf, gf, seq):
            left = L.apply(p["l"], left_of(sf, gf, seq, n))
            right = R.apply(p["r"], right_of(sf, gf, seq, n))
            return left, right.reshape(right.shape[:-1] + (heads, repr_dim))

        def q(p, sf, gf, seq):
            left, right = towers(p, sf, gf, seq)
            return energy_fn(energy, left[..., None, :], right)

        def logits(p, sf, gf, seq):
            """(B, B): every left tower against every right tower in the batch."""
            left, right = towers(p, sf, gf, seq)
            return energy_fn(energy, left[:, None, :], right[None, :, 0, :])

        return init, q, logits

    return build


def sa_g_bilinear(n, f_dim=2, repr_dim=32, energy="norm", width=128):
    """CRL's factorization: energy(f([s; a_tau]), g(goal)). Its negatives are other goals."""
    return _bilinear(
        n,
        left_of=lambda sf, gf, seq, n: jnp.concatenate([sf, _onehot(seq, n)], -1),
        right_of=lambda sf, gf, seq, n: gf,
        left_dim=f_dim + n * N_ACT, right_dim=f_dim, repr_dim=repr_dim, energy=energy,
        width=width,
    )


def sg_a_bilinear(n, f_dim=2, repr_dim=32, energy="dot", width=128):
    """CARL's factorization: energy(phi([s; g]), psi(a_tau)). Its negatives are other chunks."""
    return _bilinear(
        n,
        left_of=lambda sf, gf, seq, n: jnp.concatenate([sf, gf], -1),
        right_of=lambda sf, gf, seq, n: _onehot(seq, n),
        left_dim=2 * f_dim, right_dim=n * N_ACT, repr_dim=repr_dim, energy=energy,
        width=width,
    )


def monolithic(n, f_dim=2, width=128):
    """The baselines' critic: one MLP over [s; g; a_tau], with no factorization at all."""

    def build(heads):
        net = MLP(heads, width)

        def init(key):
            return {"q": net.init(key, jnp.zeros((1, 2 * f_dim + n * N_ACT)))}

        def q(p, sf, gf, seq):
            return net.apply(p["q"], jnp.concatenate([sf, gf, _onehot(seq, n)], -1))

        return init, q, None      # no two-tower structure, so no contrastive form

    return build
