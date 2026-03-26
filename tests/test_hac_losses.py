import os
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
from flax import nnx

from jaxgcrl.agents.hac.hac import GCTransition
from jaxgcrl.agents.hac.losses import hac_loss_fn, make_td3_losses


class _ShiftPreprocessor:
    def __call__(self, observations):
        return observations + 5.0


class _AnalyticTD3Network:
    def __init__(self):
        self.observation_preprocessor = _ShiftPreprocessor()
        self.policy_network = self._policy_network

    def _policy_network(self, processed_observations, goals):
        return jnp.stack(
            (
                processed_observations[:, 0] - 0.5 * goals[:, 0],
                processed_observations[:, 1] + 0.25 * goals[:, 1],
            ),
            axis=-1,
        )

    def q_values(self, observations, actions, goals):
        q1 = 0.5 * observations[:, 0] + actions[:, 0] - goals[:, 0]
        q2 = -observations[:, 1] + 2.0 * actions[:, 1] + goals[:, 1]
        return jnp.stack((q1, q2), axis=-1)

    def target_action(self, next_observations, goals):
        return jnp.stack(
            (
                0.4 * next_observations[:, 0] + goals[:, 0],
                -0.3 * next_observations[:, 1] - goals[:, 1],
            ),
            axis=-1,
        )

    def target_q_values(self, next_observations, actions, goals):
        q1 = next_observations[:, 0] + actions[:, 0] + 0.25 * goals[:, 0]
        q2 = next_observations[:, 1] - 1.5 * actions[:, 1] - 0.5 * goals[:, 1]
        return jnp.stack((q1, q2), axis=-1)


def _make_transition():
    return GCTransition(
        observation=jnp.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
            dtype=jnp.float32,
        ),
        goal=jnp.array(
            [
                [2.0, -1.0],
                [0.5, 1.5],
            ],
            dtype=jnp.float32,
        ),
        action=jnp.array(
            [
                [0.5, -1.0],
                [1.5, 0.25],
            ],
            dtype=jnp.float32,
        ),
        reward=jnp.array([[1.0], [2.0]], dtype=jnp.float32),
        discount=jnp.array([[0.25], [0.5]], dtype=jnp.float32),
        next_observation=jnp.array(
            [
                [10.0, 20.0],
                [30.0, 40.0],
            ],
            dtype=jnp.float32,
        ),
        extras={},
    )


def test_td3_critic_loss_matches_manual_formula():
    td3_network = _AnalyticTD3Network()
    transitions = _make_transition()
    reward_scaling = 1.5
    discounting = 0.75
    smoothing = 0.5
    noise_clip = 0.25
    max_action = 0.5
    critic_loss, _ = make_td3_losses(
        reward_scaling=reward_scaling,
        discounting=discounting,
        smoothing=smoothing,
        noise_clip=noise_clip,
        max_action=max_action,
    )

    actual = critic_loss(td3_network, transitions, nnx.Rngs(loss=7))

    manual_rngs = nnx.Rngs(loss=7)
    current_q_values = td3_network.q_values(
        transitions.observation,
        transitions.action,
        transitions.goal,
    )
    next_actions = td3_network.target_action(
        transitions.next_observation,
        transitions.goal,
    )
    smoothing_noise = (
        manual_rngs.loss.normal(next_actions.shape) * smoothing
    ).clip(-noise_clip, noise_clip)
    next_actions = (next_actions + smoothing_noise).clip(-max_action, max_action)
    next_q_values = td3_network.target_q_values(
        transitions.next_observation,
        next_actions,
        transitions.goal,
    )
    target_q = (
        jnp.squeeze(transitions.reward, axis=-1) * reward_scaling
        + jnp.squeeze(transitions.discount, axis=-1)
        * discounting
        * jnp.min(next_q_values, axis=-1)
    )
    expected = 0.5 * jnp.mean(
        jnp.square(current_q_values - jnp.expand_dims(target_q, axis=-1))
    )

    assert jnp.allclose(actual, expected)


def test_td3_actor_loss_matches_manual_formula_without_bc():
    td3_network = _AnalyticTD3Network()
    transitions = _make_transition()
    _, actor_loss = make_td3_losses(
        reward_scaling=1.0,
        discounting=0.99,
        smoothing=0.2,
        noise_clip=0.5,
        bc=False,
    )

    actual = actor_loss(td3_network, transitions)

    processed_observations = td3_network.observation_preprocessor(
        transitions.observation
    )
    new_actions = td3_network.policy_network(
        processed_observations,
        transitions.goal,
    )
    q1_new_actions = td3_network.q_values(
        transitions.observation,
        new_actions,
        transitions.goal,
    )[..., 0]
    expected = -jnp.mean(q1_new_actions)

    assert jnp.allclose(actual, expected)


def test_td3_actor_loss_matches_manual_formula_with_bc():
    td3_network = _AnalyticTD3Network()
    transitions = _make_transition()
    alpha = 2.5
    _, actor_loss = make_td3_losses(
        reward_scaling=1.0,
        discounting=0.99,
        smoothing=0.2,
        noise_clip=0.5,
        bc=True,
        alpha=alpha,
    )

    actual = actor_loss(td3_network, transitions)

    processed_observations = td3_network.observation_preprocessor(
        transitions.observation
    )
    new_actions = td3_network.policy_network(
        processed_observations,
        transitions.goal,
    )
    q1_new_actions = td3_network.q_values(
        transitions.observation,
        new_actions,
        transitions.goal,
    )[..., 0]
    denom = jnp.maximum(jnp.mean(jnp.abs(q1_new_actions)), 1e-6)
    expected = (
        -(alpha / denom) * jnp.mean(q1_new_actions)
        + jnp.mean(jnp.square(new_actions - transitions.action))
    )

    assert jnp.allclose(actual, expected)


def test_hac_loss_fn_sums_per_level_losses():
    transitions = [
        GCTransition(
            observation=jnp.zeros((1, 2), dtype=jnp.float32),
            goal=jnp.zeros((1, 2), dtype=jnp.float32),
            action=jnp.zeros((1, 2), dtype=jnp.float32),
            reward=jnp.array([1.0], dtype=jnp.float32),
            discount=jnp.array([0.5], dtype=jnp.float32),
            next_observation=jnp.zeros((1, 2), dtype=jnp.float32),
            extras={},
        ),
        GCTransition(
            observation=jnp.zeros((1, 2), dtype=jnp.float32),
            goal=jnp.zeros((1, 2), dtype=jnp.float32),
            action=jnp.zeros((1, 2), dtype=jnp.float32),
            reward=jnp.array([2.0], dtype=jnp.float32),
            discount=jnp.array([0.25], dtype=jnp.float32),
            next_observation=jnp.zeros((1, 2), dtype=jnp.float32),
            extras={},
        ),
        GCTransition(
            observation=jnp.zeros((1, 2), dtype=jnp.float32),
            goal=jnp.zeros((1, 2), dtype=jnp.float32),
            action=jnp.zeros((1, 2), dtype=jnp.float32),
            reward=jnp.array([3.0], dtype=jnp.float32),
            discount=jnp.array([0.125], dtype=jnp.float32),
            next_observation=jnp.zeros((1, 2), dtype=jnp.float32),
            extras={},
        ),
    ]
    agent = SimpleNamespace(
        networks=[
            SimpleNamespace(scale=jnp.array(1.0, dtype=jnp.float32)),
            SimpleNamespace(scale=jnp.array(2.0, dtype=jnp.float32)),
            SimpleNamespace(scale=jnp.array(-0.5, dtype=jnp.float32)),
        ]
    )

    def policy_loss_fn(network, transition):
        return network.scale * jnp.sum(transition.reward)

    def critic_loss_fn(network, transition, rngs):
        del transition
        return network.scale + rngs.loss.uniform(())

    actual = hac_loss_fn(
        agent,
        transitions,
        nnx.Rngs(loss=17),
        policy_loss_fn,
        critic_loss_fn,
    )

    manual_rngs = nnx.Rngs(loss=17)
    manual_keys = jax.random.split(manual_rngs.loss(), len(agent.networks))
    expected = sum(
        policy_loss_fn(network, transition)
        + critic_loss_fn(network, transition, nnx.Rngs(loss=key))
        for network, transition, key in zip(
            agent.networks,
            transitions,
            manual_keys,
            strict=True,
        )
    )

    assert jnp.allclose(actual, expected)
