"""Goal relabeling for hierarchical goal-conditioned agents (HIQL / Q-CARL).

This is the online analogue of CARL's ``HGCDataset`` / ``QCARLDataset``: instead of indexing
into a static offline dataset, it relabels a trajectory segment sampled from the replay buffer.
The sampling scheme itself is kept identical to CARL:

- value goals come from a mixture of the current state, a future state in the same trajectory
  (geometrically or uniformly sampled), and a random state,
- the low-level actor's goal is the state ``subgoal_steps`` ahead (clipped to the trajectory end),
- the high-level actor's goal is a future state and its prediction target is the state
  ``subgoal_steps`` ahead of the current one (clipped to the goal),
- rewards/masks follow the sparse goal-conditioned convention (``0 if s == g else -1``).

Q-CARL additionally needs the action sequences the bilinear critic is conditioned on and the
n-step windows it is fit over; those are produced when ``include_critic`` is set.

:class:`HierarchicalBatch` is what :func:`flatten_batch` turns a sampled segment into, and what
every loss reads.

As in ``jaxgcrl.agents.crl``, trajectory boundaries inside a segment are recovered from the
``traj_id`` state extra rather than from a terminals array, and this function is meant to be
``vmap``-ed over the environment axis of a sampled batch.
"""

import functools
from typing import NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp

Array = jnp.ndarray


class CriticBatch(NamedTuple):
    """The Q-CARL-only half of a batch: action sequences and the n-step windows to fit them on."""

    # a_{t..t+K-1}, flattened and zero-padded past the trajectory end. Shorter horizons are this
    # same vector with more slots masked, which the losses do.
    action_seq: Array
    # horizon the high-level actor queries the critic at, min(subgoal_steps, steps left)
    high_actor_horizon: Array
    # the critic's own (goal, horizon) draw and the n-step window it implies
    goal: Array
    horizon: Array
    next_state: Array
    num_steps: Array
    reached_goal: Array


class HierarchicalBatch(NamedTuple):
    """A relabeled batch: what every HIQL / Q-CARL loss reads.

    All fields carry the same leading axes. Goals are goal-space vectors (``state[goal_indices]``),
    not full observations. ``critic`` is populated only for Q-CARL.
    """

    state: Array
    next_state: Array
    action: Array

    # V(s, g) is fit on this goal, and it defines the reward/mask below
    value_goal: Array
    gc_reward: Array
    mask: Array

    # pi^l is trained to reach this subgoal
    low_actor_goal: Array

    # pi^h is trained to predict `high_actor_target_goal` given `high_actor_goal`;
    # `high_actor_target_state` is the same subgoal as a full state, for the advantage
    high_actor_goal: Array
    high_actor_target_goal: Array
    high_actor_target_state: Array

    critic: Optional[CriticBatch] = None


class BatchConfig(NamedTuple):
    """Static configuration for :func:`flatten_batch` (must be hashable: it is a jit static arg)."""

    discounting: float
    state_size: int
    action_size: int
    goal_indices: Tuple[int, ...]
    subgoal_steps: int

    # goal sampling distribution (CARL's GCDataset knobs)
    value_p_curgoal: float
    value_p_trajgoal: float
    value_p_randomgoal: float
    value_geom_sample: bool
    actor_p_randomgoal: float
    actor_geom_sample: bool
    gc_negative: bool

    # Reward on arrival (`r = 0 iff s' == g`) instead of on occupancy (`r = 0 iff s == g`).
    # Occupancy suits a state value V(s, g); arrival suits an action value Q(s, g, a), where the
    # reward belongs to the transition.
    reward_on_arrival: bool = False

    # Q-CARL critic extras
    include_critic: bool = False
    action_stride: int = 1
    critic_p_uniform: float = 0.5
    critic_p_low: float = 0.25
    critic_p_high: float = 0.25


def _categorical_from_mask(key, mask):
    """Sample one index per row from a (possibly unnormalized) non-negative mask."""
    return jax.random.categorical(key, jnp.log(mask + 1e-12))


@functools.partial(jax.jit, static_argnames=("cfg",))
def flatten_batch(cfg: BatchConfig, transition, sample_key: Array) -> HierarchicalBatch:
    """Relabel a single trajectory segment with hierarchical goals.

    ``transition`` is the agent's replay-buffer record; this reads its ``observation``, ``action``
    and ``extras["state_extras"]["traj_id"]``. It is vmapped, so ``transition.observation`` has
    shape ``(seq_len, obs_size)``.
    Every returned field has a leading ``seq_len - 1`` axis: the last timestep of the segment is
    dropped, since it has neither a next state nor a future goal (matching the CRL agent).
    """
    seq_len = transition.observation.shape[0]
    idx = jnp.arange(seq_len)
    goal_indices = jnp.asarray(cfg.goal_indices)
    subgoal_steps = cfg.subgoal_steps

    traj_id = transition.extras["state_extras"]["traj_id"]
    same_traj = jnp.equal(traj_id[:, None], traj_id[None, :])
    is_future = jnp.array(idx[:, None] < idx[None, :], dtype=jnp.float32)
    valid_future = is_future * same_traj

    # Last index of the trajectory the timestep belongs to (the offline dataset's `final_state_idxs`).
    final_idx = jnp.max(jnp.where(same_traj, idx[None, :], -1), axis=1)

    keys = jax.random.split(sample_key, 8)

    # Future goals: one categorical per timestep over a (seq_len, seq_len) matrix whose row i
    # weights picking j as i's goal. Six steps spanning an episode boundary, gamma = 0.9,
    # traj_id = [0 0 0 1 1 1], with e = 1e-5:
    #
    #   gamma^(j-i)                 valid_future = (j>i) * same_traj   geom_probs = valid*g + eye*e
    #    1   .9  .81 .73 .66 .59     0  1  1  0  0  0                   e  .9  .81  .   .   .
    #   1.1  1   .9  .81 .73 .66     0  0  1  0  0  0                   .   e  .9   .   .   .
    #   1.2 1.1  1   .9  .81 .73     0  0  0  0  0  0  <- ep 0 ends     .   .   e   .   .   .
    #   1.4 1.2 1.1  1   .9  .81     0  0  0  0  1  1                   .   .   .   e  .9  .81
    #   1.5 1.4 1.2 1.1  1   .9      0  0  0  0  0  1                   .   .   .   .   e  .9
    #   1.7 1.5 1.4 1.2 1.1  1       0  0  0  0  0  0  <- ep 1 ends     .   .   .   .   .   e
    #
    # The mask zeroes the lower triangle, so gamma^(j-i) can be built over the whole grid.
    # `categorical` normalizes each row of geom_probs, giving:
    #
    #    .  .53 .47  .   .   .
    #    .   .  1.   .   .   .
    #    .   .  1.   .   .   .
    #    .   .   .   .  .53 .47
    #    .   .   .   .   .  1.
    #    .   .   .   .   .  1.
    #
    # Row 0 draws t=1 or t=2, weighted toward the nearer one. Rows 2 and 5 end their episodes and
    # have no future: without the eye they would be all zero, and with it they draw themselves.
    # A self-goal is the already-reached case, so those rows get `gc_reward = 0`, `mask = 0`.
    # unif_probs is the same matrix without the gamma factor, making its row 0 [. .5 .5 . . .].
    eye = jnp.eye(seq_len) * 1e-5
    discount = cfg.discounting ** jnp.array(idx[None] - idx[:, None], dtype=jnp.float32)
    geom_probs = valid_future * discount + eye
    unif_probs = valid_future + eye
    geom_goal = _categorical_from_mask(keys[0], geom_probs)
    unif_goal = _categorical_from_mask(keys[1], unif_probs)
    random_goal = jax.random.randint(keys[2], (seq_len,), 0, seq_len)

    def sample_value_goals(key):
        """Mixture of current / future / random state (20/50/30), as in CARL's `sample_goals`.

        The current-state component is the only real source of `gc_reward = 0`, `mask = 0` rows,
        which is what terminates the TD backup. The `where`s are nested, not disjoint, hence the
        `p_trajgoal / (1 - p_curgoal)` renormalization.
        """
        traj_goal = geom_goal if cfg.value_geom_sample else unif_goal
        if cfg.value_p_curgoal == 1.0:
            return idx
        k1, k2 = jax.random.split(key)
        goal = jnp.where(
            jax.random.uniform(k1, (seq_len,)) < cfg.value_p_trajgoal / (1.0 - cfg.value_p_curgoal),
            traj_goal,
            random_goal,
        )
        return jnp.where(jax.random.uniform(k2, (seq_len,)) < cfg.value_p_curgoal, idx, goal)

    value_goal_idx = sample_value_goals(keys[3])

    # Low-level actor goal: not drawn, just K = `subgoal_steps` steps out.
    low_goal_idx = jnp.minimum(idx + subgoal_steps, final_idx)

    # High-level actor: a (goal, target) pair, the target being the subgoal it regresses onto.
    # The branches clip it differently because their goals index different things:
    #
    #     trajectory goal   min(t+K, g)          g is in this episode -- never overshoot it
    #     random goal       min(t+K, final_idx)  g is elsewhere -- its index says nothing about t
    #
    high_traj_goal = geom_goal if cfg.actor_geom_sample else unif_goal
    high_traj_target = jnp.minimum(idx + subgoal_steps, high_traj_goal)
    high_random_goal = jax.random.randint(keys[4], (seq_len,), 0, seq_len)
    high_random_target = jnp.minimum(idx + subgoal_steps, final_idx)
    pick_random = jax.random.uniform(keys[5], (seq_len,)) < cfg.actor_p_randomgoal
    high_goal_idx = jnp.where(pick_random, high_random_goal, high_traj_goal)
    high_target_idx = jnp.where(pick_random, high_random_target, high_traj_target)

    next_idx = jnp.minimum(idx + 1, final_idx)

    state = transition.observation[:, : cfg.state_size]
    goal_of = lambda i: state[i][:, goal_indices]

    reached_at = next_idx if cfg.reward_on_arrival else idx
    successes = jnp.array(reached_at == value_goal_idx, dtype=jnp.float32)

    batch = HierarchicalBatch(
        state=state,
        next_state=state[next_idx],
        action=transition.action,
        value_goal=goal_of(value_goal_idx),
        gc_reward=successes - (1.0 if cfg.gc_negative else 0.0),
        mask=1.0 - successes,
        low_actor_goal=goal_of(low_goal_idx),
        high_actor_goal=goal_of(high_goal_idx),
        high_actor_target_goal=goal_of(high_target_idx),
        high_actor_target_state=state[high_target_idx],
        critic=(
            _critic_batch(cfg, transition, idx, final_idx, same_traj, low_goal_idx, geom_probs, keys[6])
            if cfg.include_critic
            else None
        ),
    )

    # Drop the final timestep of the segment (no next state / no future).
    return jax.tree_util.tree_map(lambda x: x[:-1], batch)


def _critic_batch(
    cfg, transition, idx, final_idx, same_traj, low_goal_idx, traj_goal_probs, key
) -> CriticBatch:
    """Action sequences and n-step windows for the Q-CARL bilinear critic."""
    seq_len = transition.observation.shape[0]
    goal_indices = jnp.asarray(cfg.goal_indices)
    state = transition.observation[:, : cfg.state_size]
    subgoal_steps = cfg.subgoal_steps

    keys = jax.random.split(key, 4)

    # One gather of the action sequence at the subgoal horizon covers every query: a shorter
    # sequence is the same sequence with more slots zeroed, which the losses do with a mask.
    offsets = jnp.arange(0, subgoal_steps, cfg.action_stride)
    action_idx = idx[:, None] + offsets[None, :]
    within_traj = action_idx < final_idx[:, None]
    actions = transition.action[jnp.minimum(action_idx, seq_len - 1)]
    action_seq = (actions * within_traj[..., None]).reshape(seq_len, -1)

    max_horizon = jnp.maximum(final_idx - idx, 1)
    high_horizon = jnp.minimum(subgoal_steps, max_horizon)

    # The critic's own (goal, horizon) draw, independent of the value goals. Goals reuse the
    # mixture above; horizons are a mixture over what the two actors actually query:
    # n = 1 (low), n = min(K, steps left) (high), n ~ U[1, K] otherwise.
    traj_goal = _categorical_from_mask(keys[0], traj_goal_probs)
    random_goal = jax.random.randint(keys[1], (seq_len,), 0, seq_len)
    k1, k2 = jax.random.split(keys[2])
    critic_goal_idx = jnp.where(
        jax.random.uniform(k1, (seq_len,)) < cfg.value_p_trajgoal / max(1.0 - cfg.value_p_curgoal, 1e-6),
        traj_goal,
        random_goal,
    )
    critic_goal_idx = jnp.where(
        jax.random.uniform(k2, (seq_len,)) < cfg.value_p_curgoal, idx, critic_goal_idx
    )

    ck1, ck2 = jax.random.split(keys[3])
    total_p = cfg.critic_p_uniform + cfg.critic_p_low + cfg.critic_p_high
    draws = jax.random.uniform(ck1, (seq_len,)) * total_p
    is_low = (draws >= cfg.critic_p_uniform) & (draws < cfg.critic_p_uniform + cfg.critic_p_low)
    is_high = draws >= cfg.critic_p_uniform + cfg.critic_p_low

    critic_horizon = jax.random.randint(ck2, (seq_len,), 1, subgoal_steps + 1)
    critic_goal_idx = jnp.where(is_low, low_goal_idx, critic_goal_idx)
    critic_horizon = jnp.where(is_low, 1, jnp.where(is_high, high_horizon, critic_horizon))
    critic_horizon = jnp.minimum(critic_horizon, max_horizon)

    # Where the n-step window ends: at the first state reaching the goal, or `n` steps in.
    goal_offsets = critic_goal_idx - idx
    reached_goal = (
        jnp.take_along_axis(same_traj, critic_goal_idx[:, None], axis=1)[:, 0]
        & (goal_offsets >= 0)
        & (goal_offsets < critic_horizon)
    )
    num_steps = jnp.where(reached_goal, goal_offsets, critic_horizon)

    return CriticBatch(
        action_seq=action_seq,
        high_actor_horizon=high_horizon,
        goal=state[critic_goal_idx][:, goal_indices],
        horizon=critic_horizon,
        next_state=state[jnp.minimum(idx + num_steps, final_idx)],
        num_steps=num_steps.astype(jnp.float32),
        reached_goal=reached_goal.astype(jnp.float32),
    )
