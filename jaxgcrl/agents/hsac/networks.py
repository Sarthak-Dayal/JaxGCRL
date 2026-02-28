"""HSAC networks.

GoalRep:   phi(s, g) -> R^rep_dim  (length-normalized, reused from HIQL)
QFunction: Q(s, a, phi) -> R       (double-Q with n_critics independent heads)
SACActor:  pi(a | s, phi) -> TanhNormal  (SAC-style, state-dependent std)
HighActor: pi_h(z | s, g) -> Normal      (proposes subgoals in phi-space)
"""

from typing import Optional, Sequence

import distrax
import jax
import jax.numpy as jnp
import optax
from flax import linen


class MLP(linen.Module):
    layer_sizes: Sequence[int]
    activation: callable = linen.relu
    kernel_init: callable = jax.nn.initializers.lecun_uniform()
    activate_final: bool = False
    bias: bool = True
    layer_norm: bool = False

    @linen.compact
    def __call__(self, data: jnp.ndarray):
        hidden = data
        for i, hidden_size in enumerate(self.layer_sizes):
            hidden = linen.Dense(
                hidden_size,
                name=f"hidden_{i}",
                kernel_init=self.kernel_init,
                use_bias=self.bias,
            )(hidden)
            if i != len(self.layer_sizes) - 1 or self.activate_final:
                if self.layer_norm:
                    hidden = linen.LayerNorm()(hidden)
                hidden = self.activation(hidden)
        return hidden


class GoalRep(linen.Module):
    """Subgoal representation phi([s; g]).

    Maps concat(observation, goal) through an MLP to rep_dim dimensions,
    then length-normalizes to lie on a sphere of radius sqrt(rep_dim).
    """

    layer_sizes: Sequence[int] = (512, 512, 512)
    rep_dim: int = 10
    layer_norm: bool = True

    @linen.compact
    def __call__(self, observations: jnp.ndarray, goals: jnp.ndarray):
        x = jnp.concatenate([observations, goals], axis=-1)
        x = MLP(
            layer_sizes=list(self.layer_sizes) + [self.rep_dim],
            activate_final=False,
            layer_norm=self.layer_norm,
        )(x)
        x = x / optax.safe_norm(x, min_norm=1e-8, ord=2, axis=-1, keepdims=True) * jnp.sqrt(x.shape[-1])
        return x


class QFunction(linen.Module):
    """Double-Q critic: Q(s, a, phi) -> (n_critics,) per batch element."""

    layer_sizes: Sequence[int] = (512, 512, 512)
    n_critics: int = 2
    layer_norm: bool = True

    @linen.compact
    def __call__(self, observation: jnp.ndarray, action: jnp.ndarray, goal_rep: jnp.ndarray):
        x = jnp.concatenate([observation, action, goal_rep], axis=-1)
        res = []
        for i in range(self.n_critics):
            q = MLP(
                layer_sizes=list(self.layer_sizes) + [1],
                activate_final=False,
                layer_norm=self.layer_norm,
                name=f"critic_{i}",
            )(x)
            res.append(q)
        return jnp.concatenate(res, axis=-1)  # (batch, n_critics)


class SACActor(linen.Module):
    """SAC-style actor with tanh-squashed Gaussian and state-dependent std.

    Outputs (mean, log_std) for a TanhNormal distribution.
    Used for the low-level actor: pi_l(a | s, phi).
    """

    action_dim: int
    layer_sizes: Sequence[int] = (512, 512, 512)
    log_std_min: float = -5.0
    log_std_max: float = 2.0

    @linen.compact
    def __call__(self, observation: jnp.ndarray, goal_rep: jnp.ndarray):
        x = jnp.concatenate([observation, goal_rep], axis=-1)
        x = MLP(
            layer_sizes=list(self.layer_sizes),
            activate_final=True,
            layer_norm=False,
        )(x)
        mean = linen.Dense(self.action_dim, name="mean")(x)
        log_std = linen.Dense(self.action_dim, name="log_std")(x)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std


class HighActor(linen.Module):
    """High-level actor: pi_h(z | s, g) -> Gaussian in phi-space.

    Proposes a subgoal z in R^rep_dim given (state, goal).
    Not tanh-squashed since phi-space is already bounded (sphere-normalized).
    """

    rep_dim: int
    layer_sizes: Sequence[int] = (512, 512, 512)
    log_std_min: float = -5.0
    log_std_max: float = 2.0

    @linen.compact
    def __call__(self, observation: jnp.ndarray, goal: jnp.ndarray):
        x = jnp.concatenate([observation, goal], axis=-1)
        x = MLP(
            layer_sizes=list(self.layer_sizes),
            activate_final=True,
            layer_norm=False,
        )(x)
        mean = linen.Dense(self.rep_dim, name="mean")(x)
        log_std = linen.Dense(self.rep_dim, name="log_std")(x)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std
