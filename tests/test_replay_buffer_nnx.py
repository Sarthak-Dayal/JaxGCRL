import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from jaxgcrl.utils.replay_buffer_nnx import TrajectoryUniformSamplingQueueNNX


def _make_buffer(*, max_replay_size=5, num_envs=1, episode_length=3, seed=0):
    sample = {"x": jnp.zeros((1,), dtype=jnp.float32)}
    return TrajectoryUniformSamplingQueueNNX.create(
        max_replay_size=max_replay_size,
        dummy_data_sample=sample,
        sample_batch_size=1,
        num_envs=num_envs,
        episode_length=episode_length,
        rngs=nnx.Rngs(replay_buffer=seed),
    )


def _make_samples(values, *, num_envs=1):
    values = jnp.asarray(values, dtype=jnp.float32).reshape(len(values), 1, 1)
    values = jnp.repeat(values, repeats=num_envs, axis=1)
    return {"x": values}


def test_insert_rejects_wrong_env_count():
    buffer = _make_buffer(num_envs=2)

    with pytest.raises(ValueError, match="expected 2 envs per insert, got 1"):
        buffer.insert(_make_samples([0, 1, 2], num_envs=1))


def test_sample_requires_full_trajectory():
    buffer = _make_buffer(episode_length=3)
    buffer.insert(_make_samples([0, 1], num_envs=1))

    assert not bool(buffer.can_sample())

    with pytest.raises(ValueError, match="Need at least 3 steps, found 2"):
        buffer.check_can_sample()


def test_wraparound_keeps_latest_steps():
    buffer = _make_buffer(max_replay_size=5, episode_length=3)
    buffer.insert(_make_samples([0, 1, 2], num_envs=1))
    buffer.insert(_make_samples([3, 4, 5], num_envs=1))

    assert int(buffer.size()) == 5
    assert int(buffer.insert_position) == 5
    assert jnp.array_equal(buffer.data[:, 0, 0], jnp.array([1, 2, 3, 4, 5], dtype=jnp.float32))


def test_sample_can_reach_latest_valid_start_after_wraparound():
    buffer = _make_buffer(max_replay_size=5, episode_length=3, seed=7)
    buffer.insert(_make_samples([0, 1, 2], num_envs=1))
    buffer.insert(_make_samples([3, 4, 5], num_envs=1))

    first_steps = set()
    for _ in range(64):
        trajectory = buffer.sample()["x"][0, :, 0]
        assert jnp.array_equal(jnp.diff(trajectory), jnp.ones((2,), dtype=jnp.float32))
        first_steps.add(float(trajectory[0]))

    assert first_steps.issubset({1.0, 2.0, 3.0})
    assert 3.0 in first_steps


def test_buffer_works_inside_jax_jit_when_returned():
    buffer = _make_buffer(max_replay_size=5, episode_length=3, seed=0)
    first = _make_samples([0, 1, 2], num_envs=1)
    second = _make_samples([3, 4, 5], num_envs=1)

    @jax.jit
    def step(buffer, first, second):
        buffer.insert(first)
        buffer.insert(second)
        trajectory = buffer.sample()
        return buffer, trajectory

    buffer, trajectory = step(buffer, first, second)

    assert int(buffer.size()) == 5
    assert trajectory["x"].shape == (1, 3, 1)


def test_buffer_works_inside_nnx_jit_with_in_place_updates():
    buffer = _make_buffer(max_replay_size=5, episode_length=3, seed=0)
    first = _make_samples([0, 1, 2], num_envs=1)
    second = _make_samples([3, 4, 5], num_envs=1)

    @nnx.jit
    def step(buffer, first, second):
        buffer.insert(first)
        buffer.insert(second)
        return buffer.sample()

    trajectory = step(buffer, first, second)

    assert int(buffer.size()) == 5
    assert trajectory["x"].shape == (1, 3, 1)
