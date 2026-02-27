# JaxGCRL Architecture

## Overview

JaxGCRL is a JAX-based goal-conditioned reinforcement learning framework. It achieves extreme training speed by exploiting JAX's compilation model: the **entire training loop** — environment stepping, replay buffer operations, goal relabeling, and gradient updates — is compiled into a single fused XLA program that runs on GPU without ever returning to Python.

---

## Why It's Fast

### Everything is a `jax.lax.scan`

The core insight is nesting three levels of `jax.lax.scan`:

```
training_epoch                         ← scan over training_steps
  └─ training_step                     ← one iteration
       ├─ get_experience               ← scan over unroll_length env steps
       │    └─ actor_step → env.step   ← pure JAX (Brax)
       ├─ replay_buffer.sample         ← pure JAX array ops
       ├─ flatten_batch (vmap)         ← goal relabeling, pure JAX
       └─ update_networks             ← scan over minibatches
            └─ jax.grad → apply_gradients
```

Because every function in this tree is JIT-compatible, `training_epoch` compiles into **one XLA program**. The Python `for` loop only runs at the eval boundary (typically 200 times total), and each iteration dispatches a single GPU kernel that does thousands of environment steps + gradient updates before returning.

### Vectorized environments

Brax environments are written in JAX. `env.step` is a pure function — no Python loops, no C++ bindings, no inter-process communication. With `num_envs=256`, each `env.step` call advances 256 environments in parallel on a single GPU.

### Vectorized goal relabeling

`flatten_batch` (CRL) / `flatten_batch_hiql` (HIQL) is `vmap`-ed over the sampled trajectory batch. Goal sampling, reward computation, and observation slicing all happen as batched array operations on GPU.

### No host-device sync during training

The only `block_until_ready()` calls happen at eval boundaries. Between evals, the GPU runs autonomously — there are zero host-device round trips.

---

## Key Abstractions

### 1. Agent Dataclass

Each agent (CRL, HIQL, SAC, TD3) is a frozen Flax `@dataclass` that holds hyperparameters and defines `train_fn`. The dataclass is the CLI interface — `tyro` auto-generates command-line flags from the fields.

```python
@dataclass
class CRL:
    policy_lr: float = 3e-4
    batch_size: int = 256
    discounting: float = 0.99
    ...
    def train_fn(self, config, train_env, eval_env, ...):
        # Everything below lives inside this method as closures
```

`train_fn` is a monolithic factory that:
1. Creates networks, optimizers, and `TrainingState`
2. Defines all inner functions as closures (capturing env, buffer, networks)
3. Runs the training loop
4. Returns the final policy

### 2. `TrainingState`

A frozen Flax dataclass holding everything needed for a training step. Because it's a pytree, it flows cleanly through `jax.lax.scan`.

```python
# CRL
@dataclass
class TrainingState:
    env_steps: jnp.ndarray
    gradient_steps: jnp.ndarray
    actor_state: TrainState      # Flax TrainState (params + optimizer state)
    critic_state: TrainState
    alpha_state: TrainState

# HIQL
@dataclass
class TrainingState:
    env_steps: jnp.ndarray
    gradient_steps: jnp.ndarray
    hiql_state: TrainState       # Single TrainState for all networks
    target_params: Any           # Polyak-averaged params (goal_rep, value1, value2)
```

CRL uses separate `TrainState` per network; HIQL uses a single `TrainState` with a dict of params for all networks (goal_rep, value1, value2, low_actor, high_actor) because the `total_loss` is a single backward pass.

### 3. `Transition`

A `NamedTuple` representing one environment transition. Stacks of these flow through the replay buffer.

```python
class Transition(NamedTuple):
    observation: jnp.ndarray   # concat([state, goal]) from env
    action: jnp.ndarray
    reward: jnp.ndarray
    discount: jnp.ndarray
    extras: dict               # traj_id, truncation flags
```

### 4. `TrajectoryUniformSamplingQueue`

A JAX-native replay buffer that stores and samples **contiguous trajectory segments**.

- **Storage**: a 3D array `(max_replay_size, num_envs, data_size)` where each row is one timestep across all envs. Data is flattened via `jax.flatten_util.ravel_pytree` for uniform storage.
- **Insert**: FIFO with `jax.lax.dynamic_update_slice_in_dim`. When full, rolls the buffer.
- **Sample**: For each of `num_envs` environments, picks a random start index and reads `episode_length` contiguous timesteps (wrapping). Returns shape `(num_envs, episode_length, data_size)`.

The buffer **does not** enforce episode boundaries — sampled segments can span episode resets. That's why `traj_id` exists and `flatten_batch` uses it to mask cross-episode goal sampling.

### 5. `flatten_batch` / `flatten_batch_hiql`

Transforms raw trajectory segments into training batches by **relabeling goals**.

**CRL's `flatten_batch`**: For each timestep, samples a geometrically-discounted future state from the same trajectory as the goal. Produces `(state, goal, action, future_state, future_action)` tuples for contrastive learning.

**HIQL's `flatten_batch_hiql`**: More complex — produces separate goals for three losses:
- **Value goals**: Mix of current-state (20%), trajectory-future (50%), and random (30%) goals
- **Low-actor goals**: Fixed `subgoal_steps` into the future within the same trajectory
- **High-actor goals + targets**: Geometric future goals with subgoal-step targets

Both are `vmap`-ed over the sampled batch dimension, so goal relabeling for all trajectories happens in parallel.

### 6. `TrajectoryIdWrapper`

A Brax env wrapper that tags each timestep with a `traj_id`. The ID increments when an episode resets (detected via `state.info["steps"] == 0`). This lets `flatten_batch` identify which timesteps belong to the same episode without storing episode boundaries explicitly.

### 7. `ActorEvaluator`

Wraps an eval environment in Brax's `EvalWrapper` and runs deterministic rollouts via `jax.lax.scan`. The entire eval (256 envs × 1001 steps) is a single JIT-compiled function. Metrics (reward, success, distance) are aggregated by the `EvalWrapper` automatically.

### 8. `MetricsRecorder`

The bridge between training and logging. Each eval boundary:
1. Agent calls `progress_fn(step, metrics, make_policy, params, env)`
2. `MetricsRecorder.progress` records metrics, logs to wandb, optionally renders a trajectory

---

## How a Training Step Works (CRL)

```
training_step(training_state, env_state, buffer_state, key)
│
├─ 1. COLLECT EXPERIENCE
│   get_experience() — scan over unroll_length (62) steps:
│     actor_step: obs → actor network → sample action → env.step
│     Produces (unroll_length, num_envs) transitions
│     Inserts into replay buffer
│
├─ 2. SAMPLE FROM BUFFER
│   replay_buffer.sample() → (num_envs, episode_length) raw transitions
│
├─ 3. RELABEL GOALS
│   vmap(flatten_batch)(transitions, keys)
│     For each trajectory segment:
│       - Sample future goals using geometric discount + traj_id masking
│       - Extract state, goal, future_state from observations
│     Produces (num_envs, episode_length-1) relabeled transitions
│
├─ 4. RESHAPE INTO MINIBATCHES
│   Flatten (num_envs × (episode_length-1)) → shuffle → reshape to (num_batches, batch_size)
│
└─ 5. UPDATE NETWORKS
    scan over minibatches:
      update_critic: contrastive loss on (state, action, future_state, goal)
      update_actor_and_alpha: SAC-style entropy-regularized policy gradient
```

## How a Training Step Works (HIQL)

Same structure as CRL, but step 3 and 5 differ:

```
├─ 3. RELABEL GOALS (HIQL)
│   vmap(flatten_batch_hiql)(transitions, keys)
│     Produces separate goal sets for value, low-actor, high-actor losses
│     value_goals: geometric/random future states (goal_indices only)
│     low_actor_goals: subgoal_steps-ahead states (goal_indices only)  
│     high_actor_goals: geometric future states (goal_indices only)
│     high_actor_targets: subgoal_steps-ahead states (full state, used as obs)
│
└─ 5. UPDATE NETWORKS (HIQL)
    Single jax.grad over total_loss = value_loss + low_actor_loss + high_actor_loss
    Then Polyak-average target_params (goal_rep, value1, value2)
```

---

## Data Flow Dimensions (Ant example)

| Quantity | Value |
|---|---|
| `state_size` | 29 |
| `goal_size` (len of `goal_indices`) | 2 |
| `obs_size` (state + goal from env) | 31 |
| `action_size` | 8 |
| `num_envs` | 256 |
| `episode_length` | 1001 |
| `unroll_length` | 62 |
| `batch_size` | 1024 (HIQL) / 256 (CRL) |

Per training step:
- **Collected**: `256 × 62 = 15,872` transitions
- **Sampled**: `256` trajectory segments of length `1001`
- **After flatten**: `256 × 1000 = 256,000` training samples
- **Minibatches**: `256,000 / 1024 = 250` gradient updates (HIQL)

---

## File Map

```
run.py                          Entry point (tyro CLI → main → agent.train_fn)
jaxgcrl/
  utils/
    config.py                   Config/RunConfig dataclasses, AgentConfig union type
    env.py                      create_env(), MetricsRecorder, render()
    replay_buffer.py            TrajectoryUniformSamplingQueue
    evaluator.py                ActorEvaluator, Evaluator
  envs/
    wrappers.py                 TrajectoryIdWrapper
    ant.py, humanoid.py, ...    Brax PipelineEnv subclasses (state_dim, goal_indices)
  agents/
    __init__.py                 Exports CRL, HIQL, SAC, TD3, PPO
    crl/
      crl.py                    CRL dataclass + train_fn
      networks.py               Actor, Encoder
      losses.py                 update_critic, update_actor_and_alpha
    hiql/
      hiql.py                   HIQL dataclass + train_fn
      networks.py               GoalRep, Value, GCActor, MLP
      losses.py                 value_loss, low/high_actor_loss, update_hiql
```
