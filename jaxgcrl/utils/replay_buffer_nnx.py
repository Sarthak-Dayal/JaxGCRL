from typing import Any, Callable, Tuple

from flax import nnx
import jax
import jax.numpy as jnp
from jax import flatten_util


@nnx.dataclass
class TrajectoryUniformSamplingQueueNNX(nnx.Pytree):
    """Fixed-size FIFO queue that samples contiguous per-env trajectories.

    Inserts accept any leading time dimension `1 <= T <= max_replay_size`.
    `episode_length` only controls the trajectory window returned by
    `sample()`. The one hard invariant is `max_replay_size >= episode_length`,
    otherwise there is no legal window to sample.
    """

    data: jax.Array = nnx.data()
    insert_position: jax.Array = nnx.data()
    sample_position: jax.Array = nnx.data()
    rngs: nnx.Rngs = nnx.data()

    _data_shape: Tuple[int, int, int] = nnx.static()
    _data_dtype: Any = nnx.static()
    _sample_batch_size: int = nnx.static()
    _num_envs: int = nnx.static()
    _episode_length: int = nnx.static()

    _flatten_fn: Callable[[Any], jax.Array] = nnx.static()
    _unflatten_fn: Callable[[jax.Array], Any] = nnx.static()

    @classmethod
    def create(
        cls,
        max_replay_size: int,
        dummy_data_sample,
        sample_batch_size: int,
        num_envs: int,
        episode_length: int,
        rngs: nnx.Rngs,
    ):
        if max_replay_size <= 0:
            raise ValueError(f"max_replay_size must be positive, got {max_replay_size}")
        if sample_batch_size <= 0:
            raise ValueError(f"sample_batch_size must be positive, got {sample_batch_size}")
        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}")
        if episode_length <= 0:
            raise ValueError(f"episode_length must be positive, got {episode_length}")
        if max_replay_size < episode_length:
            raise ValueError(
                "max_replay_size must be at least episode_length so a full trajectory can be sampled. "
                f"Got max_replay_size={max_replay_size}, episode_length={episode_length}"
            )

        flatten_fn = jax.vmap(jax.vmap(lambda x: flatten_util.ravel_pytree(x)[0]))
        dummy_flatten, unflatten_fn = flatten_util.ravel_pytree(dummy_data_sample)
        unflatten_fn = jax.vmap(jax.vmap(unflatten_fn))
        data_size = len(dummy_flatten)

        data_shape = (max_replay_size, num_envs, data_size)
        data_dtype = dummy_flatten.dtype

        return cls(
            data=jnp.zeros(data_shape, data_dtype),
            insert_position=jnp.array(0, dtype=jnp.int32),
            sample_position=jnp.array(0, dtype=jnp.int32),
            rngs=rngs,
            _data_shape=data_shape,
            _data_dtype=data_dtype,
            _sample_batch_size=sample_batch_size,
            _num_envs=num_envs,
            _episode_length=episode_length,
            _flatten_fn=flatten_fn,
            _unflatten_fn=unflatten_fn,
        )

    def insert(self, samples):
        """Insert a `(time, env, ...)` batch into the replay queue."""
        update = self._prepare_update(samples)
        self.insert_internal(update)

    def _prepare_update(self, samples) -> jax.Array:
        leaves = jax.tree_util.tree_leaves(samples)
        if not leaves:
            raise ValueError("Replay buffer insert expects a non-empty pytree of samples")

        first_leaf = leaves[0]
        if first_leaf.ndim < 2:
            raise ValueError(
                "Replay buffer samples must have leading `(time, env)` dimensions; "
                f"got leaf shape {first_leaf.shape}"
            )

        insert_size, sample_num_envs = first_leaf.shape[:2]
        if sample_num_envs != self._num_envs:
            raise ValueError(
                f"Replay buffer expected {self._num_envs} envs per insert, got {sample_num_envs}"
            )
        if insert_size > self._data_shape[0]:
            raise ValueError(
                "Trying to insert a batch of samples larger than the maximum replay size. "
                f"num_samples: {insert_size}, max replay size: {self._data_shape[0]}"
            )

        for leaf in leaves[1:]:
            if leaf.ndim < 2 or leaf.shape[:2] != (insert_size, sample_num_envs):
                raise ValueError(
                    "All replay buffer leaves must share the same leading `(time, env)` dimensions. "
                    f"Expected {(insert_size, sample_num_envs)}, got {leaf.shape[:2]}"
                )

        update = self._flatten_fn(samples)
        expected_shape = (insert_size, self._num_envs, self._data_shape[-1])
        if update.shape != expected_shape:
            raise ValueError(
                f"Flattened replay data has shape {update.shape}, expected {expected_shape}. "
                "This usually means the inserted pytree structure does not match the dummy sample "
                "used to create the buffer."
            )
        return update

    def check_can_insert(self, samples):
        """Host-side validation helper for insert."""
        self._prepare_update(samples)

    def can_sample(self) -> jax.Array:
        """Jittable predicate indicating whether a full trajectory can be sampled."""
        return self.size() >= self._episode_length

    def check_can_sample(self):
        """Host-side validation helper for sample."""
        current_size = int(self.size())
        if current_size < self._episode_length:
            raise ValueError(
                "Cannot sample a full trajectory yet. "
                f"Need at least {self._episode_length} steps, found {current_size}."
            )

    def insert_internal(self, update: jax.Array):
        """Insert flattened replay data into the queue."""
        if self.data.shape != self._data_shape:
            raise ValueError(
                f"buffer_state.data.shape ({self.data.shape}) "
                f"doesn't match the expected value ({self._data_shape})"
            )

        data = self.data

        # Roll left when the new batch would overflow so the valid region stays contiguous.
        position = self.insert_position
        roll = jnp.minimum(0, len(data) - position - len(update))
        data = jax.lax.cond(roll != 0, lambda: jnp.roll(data, roll, axis=0), lambda: data)
        position = position + roll

        self.data = jax.lax.dynamic_update_slice_in_dim(data, update, position, axis=0)
        self.insert_position = (position + len(update)) % (len(data) + 1)
        self.sample_position = jnp.maximum(0, self.sample_position + roll)

    def sample(self):
        """Sample one contiguous trajectory window per environment.

        Note that this returns `num_envs` trajectories. `_sample_batch_size` is
        kept for API parity with the older replay-buffer code, but this class
        does not currently subsample environments.

        This method is JIT-friendly and assumes `can_sample()` is true. In
        eager mode, call `check_can_sample()` first if you want a Python
        exception instead of undefined behavior from invalid sampling bounds.
        """
        return self.sample_internal()

    def sample_internal(self):
        if self.data.shape != self._data_shape:
            raise ValueError(
                f"Data shape expected by the replay buffer ({self._data_shape}) does "
                f"not match the shape of the buffer state ({self.data.shape})"
            )

        sample_key = self.rngs.replay_buffer()
        env_key, start_key = jax.random.split(sample_key)

        envs_idxs = jax.random.permutation(env_key, self._num_envs)
        max_start = self.insert_position - self._episode_length + 1
        start_positions = jax.random.randint(
            start_key,
            shape=(self._num_envs,),
            minval=self.sample_position,
            maxval=max_start,
        )
        time_offsets = jnp.arange(self._episode_length, dtype=start_positions.dtype)
        time_indices = start_positions[:, None] + time_offsets[None, :]

        data_by_env = jnp.swapaxes(self.data[:, envs_idxs, :], 0, 1)
        batch = jax.vmap(lambda env_data, idx: env_data[idx], in_axes=(0, 0))(data_by_env, time_indices)
        return self._unflatten_fn(batch)

    def size(self) -> jax.Array:
        """Returns the number of valid time steps currently stored."""
        return self.insert_position - self.sample_position
