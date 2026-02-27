from typing import Optional, Sequence

import distrax
import jax
import jax.numpy as jnp
import optax
from flax import linen


class MLP(linen.Module):
    """MLP module."""

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

class Value(linen.Module):
    """Value function V(s)."""

    layer_sizes: Sequence[int] = (512, 512, 512)
    rep_dim: int = 10
    layer_norm: bool = True

    @linen.compact
    def __call__(self, observation: jnp.ndarray, goal_rep: jnp.ndarray):
        x = jnp.concatenate([observation, goal_rep], axis=-1)
        x = MLP(
            layer_sizes=list(self.layer_sizes) + [1],
            activate_final=False,
            layer_norm=self.layer_norm,
        )(x)
        return x

class GCActor(linen.Module):
    """Low-level actor pi_l(a | s, phi).

    Takes observation and a pre-encoded goal representation (phi),
    concatenates them, and outputs a Gaussian action distribution.
    """

    action_dim: int
    layer_sizes: Sequence[int] = (512, 512, 512)
    log_std_min: Optional[float] = -5.0
    log_std_max: Optional[float] = 2.0
    const_std: bool = True
    final_fc_init_scale: float = 1e-2

    def setup(self):
        self.actor_net = MLP(self.layer_sizes, activate_final=True)
        final_init = jax.nn.initializers.variance_scaling(
            self.final_fc_init_scale, "fan_avg", "uniform"
        )
        self.mean_net = linen.Dense(self.action_dim, kernel_init=final_init)
        if not self.const_std:
            self.log_stds = self.param(
                "log_stds", linen.initializers.zeros, (self.action_dim,)
            )

    def __call__(
        self,
        observation: jnp.ndarray,
        goal_rep: jnp.ndarray,
        temperature: float = 1.0,
    ):
        x = jnp.concatenate([observation, goal_rep], axis=-1)
        x = self.actor_net(x)
        means = self.mean_net(x)

        if self.const_std:
            log_stds = jnp.zeros_like(means) # std = 1
        else:
            log_stds = self.log_stds

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        return distrax.MultivariateNormalDiag(
            loc=means, scale_diag=jnp.exp(log_stds) * temperature
        )

# Low Actor = GCActor(
#     hidden_dims=config['actor_hidden_dims'],
#     action_dim=action_dim,
#     state_dependent_std=False,
#     const_std=config['const_std'],
#     gc_encoder=low_actor_encoder_def,
# )

# High Actor = GCActor( hidden_dims=config['actor_hidden_dims'],
#     action_dim=config['rep_dim'],
#     state_dependent_std=False,
#     const_std=config['const_std'],
#     gc_encoder=high_actor_encoder_def,
# )

