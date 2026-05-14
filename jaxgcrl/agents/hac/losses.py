"""Twin Delayed Deep Deterministic Policy Gradient (TD3) losses."""
from typing import Tuple, Callable, List

import jax
import jax.numpy as jnp
from flax import nnx

from . import networks
from .hac import GCTransition


def _as_scalar_vector(x: jax.Array) -> jax.Array:
    if x.ndim > 0 and x.shape[-1] == 1:
        return jnp.squeeze(x, axis=-1)
    return x


def mean_squared_error(predictions: jax.Array, targets: jax.Array) -> jax.Array:
    return jnp.mean(jnp.square(predictions - targets))


def make_td3_losses(
    reward_scaling: float,
    discounting: float,
    smoothing: float,
    noise_clip: float,
    max_action: float = 1.0,
    bc: bool = False,
    alpha: float = 2.5,
) -> Tuple[Callable[[networks.GCTD3Networks, GCTransition, nnx.Rngs], jax.Array], Callable[[networks.GCTD3Networks, GCTransition], jax.Array]]:
    """Creates TD3 actor and critic losses for goal-conditioned HAC levels."""

    def critic_loss(
        td3_network: networks.GCTD3Networks,
        transitions: GCTransition,
        rngs: nnx.Rngs,
    ) -> jax.Array:
        """Calculates the TD3 critic loss."""
        current_q_values = td3_network.q_values(
            transitions.observation,
            transitions.action,
            transitions.goal,
        )

        next_actions = td3_network.target_action(
            transitions.next_observation,
            transitions.goal,
        )
        smoothing_noise = (rngs.loss.normal(next_actions.shape) * smoothing).clip(
            -noise_clip, noise_clip
        )
        next_actions = (next_actions + smoothing_noise).clip(-max_action, max_action)

        next_q_values = td3_network.target_q_values(
            transitions.next_observation,
            next_actions,
            transitions.goal,
        )
        target_q = jnp.min(next_q_values, axis=-1)

        reward = _as_scalar_vector(transitions.reward)
        discount = _as_scalar_vector(transitions.discount)
        target_q = jax.lax.stop_gradient(reward * reward_scaling + discount * discounting * target_q)

        q_error = current_q_values - jnp.expand_dims(target_q, axis=-1)
        return 0.5 * jnp.mean(jnp.square(q_error))

    def actor_loss(
        td3_network: networks.GCTD3Networks,
        transitions: GCTransition,
    ) -> jax.Array:
        """Calculates the TD3 actor loss."""
        new_actions = td3_network.policy_network(
            td3_network.observation_preprocessor(transitions.observation),
            transitions.goal,
        )
        q_new_actions = td3_network.q_values(
            transitions.observation,
            new_actions,
            transitions.goal,
        )
        q1_new_actions = q_new_actions[..., 0]

        bc_scale = float(bc)
        denom = jnp.maximum(jnp.mean(jnp.abs(q1_new_actions)), 1e-6)
        lmbda = jax.lax.stop_gradient(bc_scale * alpha / denom + (1.0 - bc_scale))
        return -lmbda * jnp.mean(q1_new_actions) + bc_scale * mean_squared_error(
            new_actions, transitions.action
        )

    return critic_loss, actor_loss

def hac_loss_fn(
    agent: networks.HACAgent,
    transitions: List[GCTransition],
    rngs: nnx.Rngs,
    td3_policy_loss_fn: Callable[[networks.GCTD3Networks, GCTransition], jax.Array],
    td3_critic_loss_fn: Callable[[networks.GCTD3Networks, GCTransition, nnx.Rngs], jax.Array]
):
    policy_losses = [
        td3_policy_loss_fn(network, transition)
        for network, transition in zip(agent.networks, transitions, strict=True)
    ]
    critic_losses = [
        td3_critic_loss_fn(network, transition, rngs)
        for network, transition in zip(agent.networks, transitions, strict=True)
    ]

    return sum(policy_losses) + sum(critic_losses)
