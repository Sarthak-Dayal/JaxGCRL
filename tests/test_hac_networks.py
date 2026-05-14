import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
from brax.training.acme import running_statistics, specs
from flax import nnx

from jaxgcrl.agents.hac.hac import GCTransition, HAC
from jaxgcrl.agents.hac.losses import make_td3_losses
from jaxgcrl.agents.hac.networks import (
    GCTD3Networks,
    HACAgent,
    ObservationPreprocessor,
    make_networks_and_buffers,
)
from jaxgcrl.utils.config import RunConfig


def _make_transition(batch_size=4, obs_size=6, goal_size=3, action_size=2):
    return GCTransition(
        observation=jnp.ones((batch_size, obs_size), dtype=jnp.float32),
        goal=jnp.ones((batch_size, goal_size), dtype=jnp.float32),
        action=jnp.ones((batch_size, action_size), dtype=jnp.float32),
        reward=jnp.ones((batch_size,), dtype=jnp.float32),
        discount=0.99 * jnp.ones((batch_size,), dtype=jnp.float32),
        next_observation=2.0 * jnp.ones((batch_size, obs_size), dtype=jnp.float32),
        extras={},
    )


def _collect_counter_history(agent, obs, goal, num_steps):
    @nnx.jit
    def act(agent, obs, goal):
        return agent(obs, goal, deterministic=True)

    history = []
    for _ in range(num_steps):
        _, extras = act(agent, obs, goal)
        history.append(tuple(int(x) for x in extras["counters"]))
    return history


def _make_hac_parts(
    *,
    num_levels=2,
    k_step=25,
    enable_temporal_abstraction=True,
):
    hac_config = HAC(
        num_levels=num_levels,
        k_step=k_step,
        enable_temporal_abstraction=enable_temporal_abstraction,
    )
    run_config = RunConfig(
        env="ant",
        total_env_steps=8,
        episode_length=3,
        num_envs=2,
        num_eval_envs=2,
        log_wandb=False,
        cuda=False,
    )
    networks, replay_buffers = make_networks_and_buffers(
        hac_config=hac_config,
        run_config=run_config,
        observation_size=6,
        action_size=2,
        subgoal_size=3,
        goal_size=4,
        rngs=nnx.Rngs(params=0, policy=1, replay_buffer=2),
    )
    agent = HACAgent(
        subgoal_dim=3,
        networks=networks,
        k_step=k_step,
        num_levels=num_levels,
        enable_temporal_abstraction=enable_temporal_abstraction,
    )
    return networks, replay_buffers, agent


def test_gctd3_network_shapes():
    networks = GCTD3Networks(
        observation_size=6,
        subgoal_size=3,
        action_size=2,
        rngs=nnx.Rngs(params=0, policy=1),
    )

    obs = jnp.ones((4, 6), dtype=jnp.float32)
    goal = jnp.ones((4, 3), dtype=jnp.float32)
    action = jnp.ones((4, 2), dtype=jnp.float32)

    policy_action, _ = networks(obs, goal, exploration_noise=0.1, noise_clip=0.2)
    q_values = networks.q_values(obs, action, goal)
    target_action = networks.target_action(obs, goal)
    target_q_values = networks.target_q_values(obs, action, goal)

    assert policy_action.shape == (4, 2)
    assert q_values.shape == (4, 2)
    assert target_action.shape == (4, 2)
    assert target_q_values.shape == (4, 2)


def test_td3_losses_return_scalars():
    td3_network = GCTD3Networks(
        observation_size=6,
        subgoal_size=3,
        action_size=2,
        rngs=nnx.Rngs(params=0, policy=1),
    )
    transitions = _make_transition()
    critic_loss, actor_loss = make_td3_losses(
        reward_scaling=1.0,
        discounting=0.99,
        smoothing=0.2,
        noise_clip=0.5,
    )

    critic_value = critic_loss(td3_network, transitions, nnx.Rngs(loss=2))
    actor_value = actor_loss(td3_network, transitions)

    assert critic_value.shape == ()
    assert actor_value.shape == ()
    assert jnp.isfinite(critic_value)
    assert jnp.isfinite(actor_value)


def test_make_networks_and_buffers_builds_per_level_dims():
    networks, replay_buffers, _ = _make_hac_parts(num_levels=2)

    assert len(networks) == 2
    assert len(replay_buffers) == 2

    obs = jnp.ones((5, 6), dtype=jnp.float32)
    low_goal = jnp.ones((5, 3), dtype=jnp.float32)
    high_goal = jnp.ones((5, 4), dtype=jnp.float32)

    low_action, _ = networks[0](obs, low_goal)
    high_action, _ = networks[1](obs, high_goal)

    assert low_action.shape == (5, 2)
    assert high_action.shape == (5, 3)


def test_gctd3_network_uses_stateful_observation_preprocessor():
    preprocessor = ObservationPreprocessor(
        observation_size=6,
        preprocess_observations_fn=running_statistics.normalize,
    )
    network = GCTD3Networks(
        observation_size=6,
        subgoal_size=3,
        action_size=2,
        rngs=nnx.Rngs(params=0, policy=1),
        observation_preprocessor=preprocessor,
    )
    obs = jnp.ones((4, 6), dtype=jnp.float32)
    goal = jnp.ones((4, 3), dtype=jnp.float32)

    action_one, _ = network(obs, goal, deterministic=True)
    network.observation_preprocessor(obs, update_stats=True)
    action_two, _ = network(obs, goal, deterministic=True)

    assert action_one.shape == action_two.shape == (4, 2)
    assert not jnp.allclose(action_one, action_two)


def test_observation_preprocessor_running_statistics_updates_and_resets():
    preprocessor = ObservationPreprocessor(
        observation_size=3,
        preprocess_observations_fn=running_statistics.normalize,
    )
    batch = jnp.array(
        [
            [1.0, 2.0, 3.0],
            [3.0, 4.0, 5.0],
        ],
        dtype=jnp.float32,
    )

    initial_state = preprocessor.normalizer_params.get_value()
    initial_output = preprocessor(batch)
    updated_output = preprocessor(batch, update_stats=True)
    updated_state = preprocessor.normalizer_params.get_value()
    updated_output = preprocessor(batch)

    assert int(updated_state.count.lo) > int(initial_state.count.lo)
    assert initial_output.shape == updated_output.shape == batch.shape
    assert not jnp.allclose(initial_output, updated_output)

    preprocessor.normalizer_params.set_value(
        running_statistics.init_state(specs.Array((3,), jnp.dtype("float32")))
    )
    reset_state = preprocessor.normalizer_params.get_value()
    reset_output = preprocessor(batch)

    assert int(reset_state.count.lo) == int(initial_state.count.lo)
    assert jnp.allclose(reset_output, initial_output)


def test_gctd3_network_can_disable_preprocessor_updates_during_call():
    network = GCTD3Networks(
        observation_size=6,
        subgoal_size=3,
        action_size=2,
        rngs=nnx.Rngs(params=0, policy=1),
        observation_preprocessor=ObservationPreprocessor(
            observation_size=6,
            preprocess_observations_fn=running_statistics.normalize,
        ),
    )
    obs = jnp.ones((4, 6), dtype=jnp.float32)
    goal = jnp.ones((4, 3), dtype=jnp.float32)

    state_before = network.observation_preprocessor.normalizer_params.get_value()
    network(obs, goal, deterministic=True, update_preprocessor=False)
    state_after_eval = network.observation_preprocessor.normalizer_params.get_value()
    network(obs, goal, deterministic=True, update_preprocessor=True)
    state_after_update = network.observation_preprocessor.normalizer_params.get_value()

    assert int(state_after_eval.count.lo) == int(state_before.count.lo)
    assert int(state_after_update.count.lo) > int(state_after_eval.count.lo)


def test_hac_agent_runs_under_nnx_jit():
    _, _, agent = _make_hac_parts(num_levels=2)
    obs = jnp.ones((6,), dtype=jnp.float32)
    goal = jnp.ones((4,), dtype=jnp.float32)

    @nnx.jit
    def act(agent, obs, goal):
        return agent(obs, goal, exploration_noise=0.1, noise_clip=0.2)

    action, extras = act(agent, obs, goal)

    assert action.shape == (2,)
    assert extras["counters"].shape == (1,)
    assert extras["subgoals"].shape == (1, 3)


def test_hac_agent_supports_single_level():
    _, _, agent = _make_hac_parts(num_levels=1)
    action, extras = agent(
        jnp.ones((6,), dtype=jnp.float32),
        jnp.ones((4,), dtype=jnp.float32),
        deterministic=True,
    )

    assert action.shape == (2,)
    assert extras["counters"].shape == (0,)
    assert extras["subgoals"].shape == (0, 3)


def test_hac_agent_uses_hierarchical_counter_schedule():
    _, _, agent = _make_hac_parts(num_levels=4, k_step=2)
    obs = jnp.ones((6,), dtype=jnp.float32)
    goal = jnp.ones((4,), dtype=jnp.float32)

    counter_history = _collect_counter_history(agent, obs, goal, num_steps=10)

    assert counter_history == [
        (1, 1, 1),
        (2, 1, 1),
        (1, 2, 1),
        (2, 2, 1),
        (1, 1, 2),
        (2, 1, 2),
        (1, 2, 2),
        (2, 2, 2),
        (1, 1, 1),
        (2, 1, 1),
    ]


def test_hac_agent_schedule_supports_larger_k_step():
    _, _, agent = _make_hac_parts(num_levels=3, k_step=3)
    obs = jnp.ones((6,), dtype=jnp.float32)
    goal = jnp.ones((4,), dtype=jnp.float32)

    counter_history = _collect_counter_history(agent, obs, goal, num_steps=10)

    assert counter_history == [
        (1, 1),
        (2, 1),
        (3, 1),
        (1, 2),
        (2, 2),
        (3, 2),
        (1, 3),
        (2, 3),
        (3, 3),
        (1, 1),
    ]


def test_hac_agent_reset_restarts_schedule():
    _, _, agent = _make_hac_parts(num_levels=4, k_step=2)
    obs = jnp.ones((6,), dtype=jnp.float32)
    goal = jnp.ones((4,), dtype=jnp.float32)

    first_prefix = _collect_counter_history(agent, obs, goal, num_steps=4)
    agent.reset()
    second_prefix = _collect_counter_history(agent, obs, goal, num_steps=4)

    assert first_prefix == [
        (1, 1, 1),
        (2, 1, 1),
        (1, 2, 1),
        (2, 2, 1),
    ]
    assert second_prefix == first_prefix


def test_hac_agent_without_temporal_abstraction_refreshes_every_step():
    _, _, agent = _make_hac_parts(
        num_levels=4,
        k_step=2,
        enable_temporal_abstraction=False,
    )
    obs = jnp.ones((6,), dtype=jnp.float32)
    goal = jnp.ones((4,), dtype=jnp.float32)

    counter_history = _collect_counter_history(agent, obs, goal, num_steps=4)

    assert counter_history == [
        (1, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
    ]
