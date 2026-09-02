"""Hierarchical Implicit Q-Learning (HIQL), without the CARL auxiliary loss.

The value function V(s, phi([s; g])) is learned with a plain TD loss and a target network (no
expectile loss, matching the way the other agents in this repo learn critics), and both levels of
the hierarchy are extracted from it with AWR. The high-level actor predicts the subgoal
representation phi([s; w]) and the low-level actor is conditioned on it.

Goal relabeling follows CARL's `HGCDataset` and lives in
`jaxgcrl.utils.hierarchical_batch.flatten_batch`.
"""

import functools
import logging
import pickle
import random
import time
from typing import Any, Callable, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import optax
from brax import base, envs
from brax.training import types
from etils import epath
from flax.struct import dataclass
from flax.training.train_state import TrainState

from jaxgcrl.envs.wrappers import TrajectoryIdWrapper
from jaxgcrl.utils.evaluator import ActorEvaluator
from jaxgcrl.utils.hierarchical_batch import BatchConfig, HierarchicalBatch, flatten_batch
from jaxgcrl.utils.replay_buffer import TrajectoryUniformSamplingQueue
from jaxgcrl.utils.run_config import RunConfig

from .losses import LossContext, update_high_actor, update_low_actor, update_value
from .networks import Actor, GoalRep, ValueEnsemble

Metrics = types.Metrics
Env = Union[envs.Env, envs.Wrapper]
State = envs.State


@dataclass
class TrainingState:
    """Contains training state for the learner"""

    env_steps: jnp.ndarray
    gradient_steps: jnp.ndarray
    value_state: TrainState
    target_value_params: Any
    low_actor_state: TrainState
    high_actor_state: TrainState


class Transition(NamedTuple):
    """Container for a transition"""

    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    discount: jnp.ndarray
    extras: Any = ()


def load_params(path: str):
    with epath.Path(path).open("rb") as fin:
        buf = fin.read()
    return pickle.loads(buf)


def save_params(path: str, params: Any):
    """Saves parameters in flax format."""
    with epath.Path(path).open("wb") as fout:
        fout.write(pickle.dumps(params))


@dataclass
class HIQL:
    """Hierarchical Implicit Q-Learning (HIQL) agent."""

    policy_lr: float = 3e-4
    value_lr: float = 3e-4
    batch_size: int = 256

    # gamma
    discounting: float = 0.99

    # target network update rate
    tau: float = 0.005

    # AWR temperatures
    low_alpha: float = 3.0
    high_alpha: float = 3.0

    # number of steps to the low-level goal
    subgoal_steps: int = 25

    # phi([s; g]) representation dimension
    rep_dim: int = 10

    # whether low-level actor gradients flow into the subgoal representation
    low_actor_rep_grad: bool = False

    # exploration noise added on top of the deterministic policies during rollouts
    exploration_noise: float = 0.1
    high_exploration_noise: float = 0.1

    train_step_multiplier: int = 1

    max_replay_size: int = 10000
    min_replay_size: int = 1000
    unroll_length: int = 62
    h_dim: int = 256
    n_hidden: int = 2
    use_relu: bool = False
    use_ln: bool = True

    # goal sampling (CARL's GCDataset knobs)
    value_p_curgoal: float = 0.2
    value_p_trajgoal: float = 0.5
    value_p_randomgoal: float = 0.3
    value_geom_sample: bool = True
    actor_p_randomgoal: float = 0.0
    actor_geom_sample: bool = False
    gc_negative: bool = True

    # Training metrics the logger surfaces (see `run.py`). Deliberately unannotated: it is a
    # class attribute, not a config field, so it stays off the CLI.
    metrics_to_log = (
        "sps",
        "value_loss",
        "v_mean",
        "low_actor_loss",
        "low_adv",
        "high_actor_loss",
        "high_adv",
    )

    def check_config(self, config):
        """
        episode_length: the maximum length of an episode
            NOTE: `num_envs * (episode_length - 1)` must be divisible by
            `batch_size` due to the way data is stored in replay buffer.
        """
        assert config.num_envs * (config.episode_length - 1) % self.batch_size == 0, (
            "num_envs * (episode_length - 1) must be divisible by batch_size"
        )
        assert np.isclose(self.value_p_curgoal + self.value_p_trajgoal + self.value_p_randomgoal, 1.0), (
            "value goal probabilities must sum to 1"
        )

        # Budget checks live here, not at the end of `train_fn`, so a schedule that cannot reach
        # `total_env_steps` fails before the run rather than after it.
        schedule = self.schedule(config)
        assert schedule["num_training_steps_per_epoch"] > 0, (
            "total_env_steps too small for given num_envs, unroll_length and num_evals"
        )
        assert schedule["planned_env_steps"] >= config.total_env_steps, (
            f"schedule reaches {schedule['planned_env_steps']} env steps, short of the requested "
            f"{config.total_env_steps}"
        )

    def schedule(self, config):
        """Resolve the env-step budget into loop trip counts.

        `total_env_steps` is treated as a floor: the per-epoch training step count is rounded up,
        so a budget that does not divide evenly into `prefill + num_evals * actor steps` overshoots
        rather than silently falling short of what was asked for.
        """
        env_steps_per_actor_step = config.num_envs * self.unroll_length
        num_prefill_env_steps = self.min_replay_size * config.num_envs
        num_prefill_actor_steps = int(np.ceil(self.min_replay_size / self.unroll_length))
        num_training_steps_per_epoch = int(
            np.ceil(
                (config.total_env_steps - num_prefill_env_steps)
                / (config.num_evals * env_steps_per_actor_step)
            )
        )
        planned_env_steps = (
            num_prefill_actor_steps + config.num_evals * num_training_steps_per_epoch
        ) * env_steps_per_actor_step

        return {
            "env_steps_per_actor_step": env_steps_per_actor_step,
            "num_prefill_env_steps": num_prefill_env_steps,
            "num_prefill_actor_steps": num_prefill_actor_steps,
            "num_training_steps_per_epoch": num_training_steps_per_epoch,
            "planned_env_steps": planned_env_steps,
        }

    def loss_context(self) -> LossContext:
        """The hyperparameters the loss functions read."""
        return LossContext(
            discounting=self.discounting,
            tau=self.tau,
            low_alpha=self.low_alpha,
            high_alpha=self.high_alpha,
            low_actor_rep_grad=self.low_actor_rep_grad,
        )

    def batch_config(self, state_size, action_size, goal_indices):
        return BatchConfig(
            discounting=self.discounting,
            state_size=state_size,
            action_size=action_size,
            goal_indices=tuple(np.asarray(goal_indices).tolist()),
            subgoal_steps=self.subgoal_steps,
            value_p_curgoal=self.value_p_curgoal,
            value_p_trajgoal=self.value_p_trajgoal,
            value_p_randomgoal=self.value_p_randomgoal,
            value_geom_sample=self.value_geom_sample,
            actor_p_randomgoal=self.actor_p_randomgoal,
            actor_geom_sample=self.actor_geom_sample,
            gc_negative=self.gc_negative,
        )

    def train_fn(
        self,
        config: RunConfig,
        train_env: Env,
        eval_env: Optional[Env] = None,
        randomization_fn: Optional[
            Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
        ] = None,
        progress_fn: Callable[..., None] = lambda *args, **kwargs: None,
    ):
        self.check_config(config)

        unwrapped_env = train_env
        train_env = TrajectoryIdWrapper(train_env)
        train_env = envs.training.wrap(
            train_env,
            episode_length=config.episode_length,
            action_repeat=config.action_repeat,
        )

        eval_env = TrajectoryIdWrapper(eval_env)
        eval_env = envs.training.wrap(
            eval_env,
            episode_length=config.episode_length,
            action_repeat=config.action_repeat,
        )

        schedule = self.schedule(config)
        env_steps_per_actor_step = schedule["env_steps_per_actor_step"]
        num_prefill_env_steps = schedule["num_prefill_env_steps"]
        num_prefill_actor_steps = schedule["num_prefill_actor_steps"]
        num_training_steps_per_epoch = schedule["num_training_steps_per_epoch"]

        logging.info("num_prefill_env_steps: %d", num_prefill_env_steps)
        logging.info("num_prefill_actor_steps: %d", num_prefill_actor_steps)
        logging.info("num_training_steps_per_epoch: %d", num_training_steps_per_epoch)
        logging.info("planned_env_steps: %d", schedule["planned_env_steps"])

        random.seed(config.seed)
        np.random.seed(config.seed)
        key = jax.random.PRNGKey(config.seed)
        key, buffer_key, eval_env_key, env_key, value_key, low_key, high_key = jax.random.split(key, 7)

        env_keys = jax.random.split(env_key, config.num_envs)
        env_state = jax.jit(train_env.reset)(env_keys)
        train_env.step = jax.jit(train_env.step)

        # Dimensions definitions and sanity checks
        action_size = train_env.action_size
        state_size = train_env.state_dim
        goal_size = len(train_env.goal_indices)
        obs_size = state_size + goal_size
        assert obs_size == train_env.observation_size, (
            f"obs_size: {obs_size}, observation_size: {train_env.observation_size}"
        )

        network_kwargs = dict(
            network_width=self.h_dim,
            network_depth=self.n_hidden,
            use_relu=self.use_relu,
            use_ln=self.use_ln,
        )

        # Subgoal representation phi([s; g]) and value function V(s, phi([s; g])). Both are trained
        # by the value loss, so they share one optimizer (and one target copy).
        goal_rep = GoalRep(rep_dim=self.rep_dim, **network_kwargs)
        goal_rep_params = goal_rep.init(value_key, np.ones([1, obs_size]))
        value = ValueEnsemble(**network_kwargs)
        value_params = value.init(value_key, np.ones([1, state_size + self.rep_dim]))
        value_state = TrainState.create(
            apply_fn=None,
            params={"goal_rep": goal_rep_params, "value": value_params},
            tx=optax.adam(learning_rate=self.value_lr),
        )

        # pi^l(. | s, phi([s; w])) and pi^h(phi([s; w]) | s, g)
        low_actor = Actor(output_size=action_size, **network_kwargs)
        low_actor_state = TrainState.create(
            apply_fn=low_actor.apply,
            params=low_actor.init(low_key, np.ones([1, state_size + self.rep_dim])),
            tx=optax.adam(learning_rate=self.policy_lr),
        )
        high_actor = Actor(output_size=self.rep_dim, **network_kwargs)
        high_actor_state = TrainState.create(
            apply_fn=high_actor.apply,
            params=high_actor.init(high_key, np.ones([1, obs_size])),
            tx=optax.adam(learning_rate=self.policy_lr),
        )

        training_state = TrainingState(
            env_steps=jnp.zeros(()),
            gradient_steps=jnp.zeros(()),
            value_state=value_state,
            target_value_params=value_state.params,
            low_actor_state=low_actor_state,
            high_actor_state=high_actor_state,
        )

        networks = dict(
            goal_rep=goal_rep,
            value=value,
            low_actor=low_actor,
            high_actor=high_actor,
        )

        # Replay Buffer
        dummy_transition = Transition(
            observation=jnp.zeros((obs_size,)),
            action=jnp.zeros((action_size,)),
            reward=0.0,
            discount=0.0,
            extras={
                "state_extras": {
                    "truncation": 0.0,
                    "traj_id": 0.0,
                }
            },
        )

        def jit_wrap(buffer):
            buffer.insert_internal = jax.jit(buffer.insert_internal)
            buffer.sample_internal = jax.jit(buffer.sample_internal)
            return buffer

        replay_buffer = jit_wrap(
            TrajectoryUniformSamplingQueue(
                max_replay_size=self.max_replay_size,
                dummy_data_sample=dummy_transition,
                sample_batch_size=self.batch_size,
                num_envs=config.num_envs,
                episode_length=config.episode_length,
            )
        )
        buffer_state = jax.jit(replay_buffer.init)(buffer_key)

        def select_action(low_params, high_params, obs, key=None):
            """Query the high-level actor for a subgoal representation, then the low-level actor.

            Without a `key` this is the deterministic policy used for evaluation and rendering;
            with one, exploration noise is added at both levels.
            """
            state = obs[:, :state_size]

            rep = high_actor.apply(high_params, obs)
            if key is not None:
                high_key, key = jax.random.split(key)
                rep = rep + self.high_exploration_noise * jax.random.normal(high_key, shape=rep.shape)
            rep = rep / (jnp.linalg.norm(rep, axis=-1, keepdims=True) + 1e-6) * jnp.sqrt(self.rep_dim)

            actions = low_actor.apply(low_params, jnp.concatenate([state, rep], axis=-1))
            if key is not None:
                actions = actions + self.exploration_noise * jax.random.normal(key, shape=actions.shape)
            return jnp.clip(actions, -1.0, 1.0)

        def actor_step(training_state, env, env_state, extra_fields=(), key=None):
            """Step the env with the hierarchical policy.

            Omitting `key` steps deterministically, which is the form `ActorEvaluator` calls.
            """
            actions = select_action(
                training_state.low_actor_state.params,
                training_state.high_actor_state.params,
                env_state.obs,
                key,
            )

            nstate = env.step(env_state, actions)
            state_extras = {x: nstate.info[x] for x in extra_fields}

            return nstate, Transition(
                observation=env_state.obs,
                action=actions,
                reward=nstate.reward,
                discount=1 - nstate.done,
                extras={"state_extras": state_extras},
            )

        @functools.partial(jax.jit, static_argnames=("num_actor_steps",))
        def collect_experience(training_state, env_state, buffer_state, key, num_actor_steps):
            """Roll out `num_actor_steps` chunks of `unroll_length` steps, inserting each chunk.

            The networks are frozen throughout, so only the env state, buffer and key are carried
            and the training state is left untouched — callers own the `env_steps` bookkeeping.
            """

            def unroll(carry, unused_t):
                env_state, current_key = carry
                current_key, next_key = jax.random.split(current_key)
                env_state, transition = actor_step(
                    training_state,
                    train_env,
                    env_state,
                    extra_fields=("truncation", "traj_id"),
                    key=current_key,
                )
                return (env_state, next_key), transition

            def chunk(carry, unused_t):
                env_state, buffer_state, key = carry
                key, next_key = jax.random.split(key)
                (env_state, _), data = jax.lax.scan(
                    unroll, (env_state, key), (), length=self.unroll_length
                )
                buffer_state = replay_buffer.insert(buffer_state, data)
                return (env_state, buffer_state, next_key), ()

            (env_state, buffer_state, _), _ = jax.lax.scan(
                chunk, (env_state, buffer_state, key), (), length=num_actor_steps
            )
            return env_state, buffer_state

        loss_context = self.loss_context()

        @jax.jit
        def update_networks(carry, batch: HierarchicalBatch):
            training_state, key = carry
            key, value_key, low_key, high_key = jax.random.split(key, 4)

            training_state, value_metrics = update_value(
                loss_context, networks, batch, training_state, value_key
            )
            training_state, low_metrics = update_low_actor(
                loss_context, networks, batch, training_state, low_key
            )
            training_state, high_metrics = update_high_actor(
                loss_context, networks, batch, training_state, high_key
            )
            training_state = training_state.replace(gradient_steps=training_state.gradient_steps + 1)

            metrics = {}
            metrics.update(value_metrics)
            metrics.update(low_metrics)
            metrics.update(high_metrics)

            return (training_state, key), metrics

        batch_config = self.batch_config(state_size, action_size, train_env.goal_indices)

        @jax.jit
        def training_step(training_state, env_state, buffer_state, key):
            experience_key1, experience_key2, sampling_key, training_key = jax.random.split(key, 4)

            # update buffer with one actor step of fresh experience
            env_state, buffer_state = collect_experience(
                training_state, env_state, buffer_state, experience_key1, num_actor_steps=1
            )

            training_state = training_state.replace(
                env_steps=training_state.env_steps + env_steps_per_actor_step,
            )

            # sample actor-step worth of transitions
            buffer_state, transitions = replay_buffer.sample(buffer_state)

            # relabel transitions with hierarchical goals
            batch_keys = jax.random.split(sampling_key, transitions.observation.shape[0])
            batch: HierarchicalBatch = jax.vmap(flatten_batch, in_axes=(None, 0, 0))(
                batch_config, transitions, batch_keys
            )
            batch = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"), batch
            )

            # shuffle and split into minibatches
            permutation = jax.random.permutation(experience_key2, len(batch.state))
            batch = jax.tree_util.tree_map(lambda x: x[permutation], batch)
            batch = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1, self.batch_size) + x.shape[1:]),
                batch,
            )

            # take actor-step worth of training-step
            (training_state, _), metrics = jax.lax.scan(
                update_networks, (training_state, training_key), batch
            )

            return (training_state, env_state, buffer_state), metrics

        @jax.jit
        def training_epoch(training_state, env_state, buffer_state, key):
            @jax.jit
            def f(carry, unused_t):
                ts, es, bs, k = carry
                k, train_key = jax.random.split(k, 2)
                (ts, es, bs), metrics = training_step(ts, es, bs, train_key)
                return (ts, es, bs, k), metrics

            (training_state, env_state, buffer_state, key), metrics = jax.lax.scan(
                f,
                (training_state, env_state, buffer_state, key),
                (),
                length=num_training_steps_per_epoch,
            )

            metrics["buffer_current_size"] = replay_buffer.size(buffer_state)
            return training_state, env_state, buffer_state, metrics

        key, prefill_key = jax.random.split(key, 2)

        # Prefill the buffer to `min_replay_size` rows before the first gradient step.
        env_state, buffer_state = collect_experience(
            training_state,
            env_state,
            buffer_state,
            prefill_key,
            num_actor_steps=num_prefill_actor_steps,
        )
        training_state = training_state.replace(
            env_steps=training_state.env_steps + num_prefill_actor_steps * env_steps_per_actor_step,
        )

        """Setting up evaluator"""
        evaluator = ActorEvaluator(
            actor_step,
            eval_env,
            num_eval_envs=config.num_eval_envs,
            episode_length=config.episode_length,
            key=eval_env_key,
        )

        def make_policy(params):
            low_params, high_params = params[0], params[1]

            def policy(obs, key):
                # Rendering passes a key, but the policy it renders is the deterministic one.
                return select_action(low_params, high_params, obs), {}

            return policy

        training_walltime = 0
        logging.info("starting training....")
        for ne in range(config.num_evals):
            t = time.time()

            key, epoch_key = jax.random.split(key)

            training_state, env_state, buffer_state, metrics = training_epoch(
                training_state, env_state, buffer_state, epoch_key
            )

            metrics = jax.tree_util.tree_map(jnp.mean, metrics)
            metrics = jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

            epoch_training_time = time.time() - t
            training_walltime += epoch_training_time

            sps = (env_steps_per_actor_step * num_training_steps_per_epoch) / epoch_training_time
            metrics = {
                "training/sps": sps,
                "training/walltime": training_walltime,
                "training/envsteps": training_state.env_steps.item(),
                **{f"training/{name}": value for name, value in metrics.items()},
            }
            current_step = int(training_state.env_steps.item())

            metrics = evaluator.run_evaluation(training_state, metrics)
            logging.info("step: %d", current_step)

            params = (
                training_state.low_actor_state.params,
                training_state.high_actor_state.params,
                training_state.value_state.params,
            )

            do_render = ne % config.visualization_interval == 0
            progress_fn(
                current_step,
                metrics,
                make_policy,
                params,
                unwrapped_env,
                do_render=do_render,
            )

            if config.checkpoint_logdir:
                path = f"{config.checkpoint_logdir}/step_{int(training_state.env_steps)}.pkl"
                save_params(path, params)

        logging.info("total steps: %s", current_step)

        return make_policy, params, metrics
