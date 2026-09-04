"""CARL-CRL: CRL with the critic factored as Q(s, g, a_tau) = phi([s; g]) . psi(a_tau).

CRL scores `f(s, a) . g(goal)`; this moves the goal to the state side and gives the action
its own encoder. At `action_seq_len = 1` the data pipeline, the actor and the environment
loop are identical to CRL -- only the critic factorization and hence the loss differ. Beyond
1, psi consumes the action sequence a_{t:t+n-1} actually executed from s, clipped to the end
of s's own trajectory, and the actor proposes a chunk of that length.
"""

import functools
import logging
import pickle
import random
import time
from typing import Any, Callable, Literal, NamedTuple, Optional, Tuple, Union

import flax.linen as nn
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
from jaxgcrl.utils.replay_buffer import TrajectoryUniformSamplingQueue
from jaxgcrl.utils.run_config import RunConfig

from .losses import update_actor_and_alpha, update_critic
from .networks import Actor, Encoder, sample_actions

Metrics = types.Metrics
Env = Union[envs.Env, envs.Wrapper]
State = envs.State


@dataclass
class TrainingState:
    """Contains training state for the learner"""

    env_steps: jnp.ndarray
    gradient_steps: jnp.ndarray
    actor_state: TrainState
    critic_state: TrainState
    alpha_state: TrainState


class Transition(NamedTuple):
    """Container for a transition"""

    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    discount: jnp.ndarray
    extras: jnp.ndarray = ()


@functools.partial(jax.jit, static_argnames=("buffer_config"))
def flatten_batch(buffer_config, transition, sample_key):
    gamma, state_size, goal_indices, action_seq_len = buffer_config

    # Because it's vmaped transition.obs.shape is of shape (episode_len, obs_dim)
    seq_len = transition.observation.shape[0]
    arrangement = jnp.arange(seq_len)
    is_future_mask = jnp.array(
        arrangement[:, None] < arrangement[None], dtype=jnp.float32
    )  # upper triangular matrix of shape seq_len, seq_len where all non-zero entries are 1
    discount = gamma ** jnp.array(arrangement[None] - arrangement[:, None], dtype=jnp.float32)
    probs = is_future_mask * discount

    # probs is an upper triangular matrix of shape seq_len, seq_len of the form:
    #    [[0.        , 0.99      , 0.98010004, 0.970299  , 0.960596 ],
    #    [0.        , 0.        , 0.99      , 0.98010004, 0.970299  ],
    #    [0.        , 0.        , 0.        , 0.99      , 0.98010004],
    #    [0.        , 0.        , 0.        , 0.        , 0.99      ],
    #    [0.        , 0.        , 0.        , 0.        , 0.        ]]
    # assuming seq_len = 5
    # the same result can be obtained using probs = is_future_mask * (gamma ** jnp.cumsum(is_future_mask, axis=-1))

    single_trajectories = jnp.concatenate(
        [transition.extras["state_extras"]["traj_id"][:, jnp.newaxis].T] * seq_len,
        axis=0,
    )
    # array of seq_len x seq_len where a row is an array of traj_ids that correspond to the episode index from which that time-step was collected
    # timesteps collected from the same episode will have the same traj_id. All rows of the single_trajectories are same.

    same_traj = jnp.equal(single_trajectories, single_trajectories.T)
    probs = probs * same_traj + jnp.eye(seq_len) * 1e-5
    # ith row of probs will be non zero only for time indices that
    # 1) are greater than i
    # 2) have the same traj_id as the ith time index

    goal_index = jax.random.categorical(sample_key, jnp.log(probs))
    future_state = jnp.take(
        transition.observation, goal_index[:-1], axis=0
    )  # the last goal_index cannot be considered as there is no future.
    future_action = jnp.take(transition.action, goal_index[:-1], axis=0)
    goal = future_state[:, goal_indices]
    future_state = future_state[:, :state_size]
    state = transition.observation[:-1, :state_size]  # all states are considered
    new_obs = jnp.concatenate([state, goal], axis=1)

    # The action sequence psi scores: the n actions actually executed from t, clipped to the end
    # of t's own trajectory. Rows are timesteps t, columns the n offsets. Diagrams use
    # seq_len = 6, traj_id = [7 7 7 8 8 8], n = 3.
    #
    #   final_idx[t] = last index sharing t's traj_id = [2 2 2 5 5 5]
    #
    #     idx = t + offsets   within = idx <= final_idx   action_seq  (. = zeroed)
    #        0 1 2                  1 1 1                   a0 a1 a2
    #        1 2 3                  1 1 0                   a1 a2  .
    #        2 3 4                  1 0 0                   a2  .  .
    #        3 4 5                  1 1 1                   a3 a4 a5
    #        4 5 6                  1 1 0                   a4 a5  .
    #        5 6 7                  1 0 0                   a5  .  .
    #
    # `within` is what zeroes the tail, so a sequence never borrows the next episode's actions;
    # the jnp.minimum only keeps the gather in bounds. The trailing row is dropped with the rest
    # of the batch, since it has no future to relabel against.
    #
    # At n = 1 the columns collapse to `idx = [[0] [1] ...]` and `within` is all ones, leaving
    # exactly `transition.action[:-1]` -- bit for bit the array CRL trains on.
    final_idx = jnp.max(jnp.where(same_traj, arrangement[None, :], -1), axis=1)
    offsets = jnp.arange(action_seq_len)
    idx = arrangement[:, None] + offsets[None, :]
    within = idx <= final_idx[:, None]
    action_seq = transition.action[jnp.minimum(idx, seq_len - 1)] * within[..., None]
    action_seq = action_seq.reshape(seq_len, -1)[:-1]
    num_actions = within.sum(axis=1)[:-1]

    extras = {
        "policy_extras": {},
        "state_extras": {
            "truncation": jnp.squeeze(transition.extras["state_extras"]["truncation"][:-1]),
            "traj_id": jnp.squeeze(transition.extras["state_extras"]["traj_id"][:-1]),
        },
        "state": state,
        "future_state": future_state,
        "future_action": future_action,
        "action_seq": action_seq,
        "num_actions": num_actions,
    }

    return transition._replace(
        observation=jnp.squeeze(new_obs),  # this has shape (num_envs, episode_length-1, obs_size)
        action=jnp.squeeze(transition.action[:-1]),
        reward=jnp.squeeze(transition.reward[:-1]),
        discount=jnp.squeeze(transition.discount[:-1]),
        extras=extras,
    )


def load_params(path: str):
    with epath.Path(path).open("rb") as fin:
        buf = fin.read()
    return pickle.loads(buf)


def save_params(path: str, params: Any):
    """Saves parameters in flax format."""
    with epath.Path(path).open("wb") as fout:
        fout.write(pickle.dumps(params))


@dataclass
class CarlCRL:
    """CRL with a Q(s, g, a_tau) = phi([s; g]) . psi(a_tau) critic."""

    policy_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    batch_size: int = 256

    # gamma
    discounting: float = 0.99

    # forward CRL logsumexp penalty
    logsumexp_penalty_coeff: float = 0.1

    train_step_multiplier: int = 1

    disable_entropy_actor: bool = False

    max_replay_size: int = 10000
    min_replay_size: int = 1000
    unroll_length: int = 62
    h_dim: int = 256
    n_hidden: int = 2
    skip_connections: int = 4
    use_relu: bool = False

    # phi([s; g]) and psi(a_tau) repr dimension
    repr_dim: int = 64

    # How many executed actions psi consumes
    action_seq_len: int = 1

    # How many actions of the proposed chunk the environment actually executes each actor query
    action_exec_len: int = 1

    # What the actor reads. "goal" is CRL's pi([s; g]) and leaves phi to the critic; "rep" is
    # pi(phi([s; g])); "state_rep" is pi([s; phi([s; g])]).
    actor_conditioning: Literal["goal", "rep", "state_rep"] = "goal"

    # layer norm
    use_ln: bool = False

    contrastive_loss_fn: Literal["fwd_infonce", "sym_infonce", "bwd_infonce", "binary_nce"] = "fwd_infonce"
    energy_fn: Literal["norm", "l2", "dot", "cosine"] = "norm"

    def check_config(self, config):
        """
        episode_length: the maximum length of an episode
            NOTE: `num_envs * (episode_length - 1)` must be divisible by
            `batch_size` due to the way data is stored in replay buffer.
        """
        assert config.num_envs * (config.episode_length - 1) % self.batch_size == 0, (
            "num_envs * (episode_length - 1) must be divisible by batch_size"
        )
        assert 1 <= self.action_exec_len <= self.action_seq_len, (
            f"action_exec_len ({self.action_exec_len}) must be in [1, action_seq_len ({self.action_seq_len})]"
        )

    def train_fn(
        self,
        config: RunConfig,
        train_env: Env,
        eval_env: Optional[Env] = None,
        randomization_fn: Optional[
            Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
        ] = None,
        progress_fn: Callable[..., None] = lambda *args: None,
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

        env_steps_per_actor_step = config.num_envs * self.unroll_length
        num_prefill_env_steps = self.min_replay_size * config.num_envs
        num_prefill_actor_steps = np.ceil(self.min_replay_size / self.unroll_length)
        num_training_steps_per_epoch = (config.total_env_steps - num_prefill_env_steps) // (
            config.num_evals * env_steps_per_actor_step
        )

        assert num_training_steps_per_epoch > 0, (
            "total_env_steps too small for given num_envs and episode_length"
        )

        logging.info(
            "num_prefill_env_steps: %d",
            num_prefill_env_steps,
        )
        logging.info(
            "num_prefill_actor_steps: %d",
            num_prefill_actor_steps,
        )
        logging.info(
            "num_training_steps_per_epoch: %d",
            num_training_steps_per_epoch,
        )

        random.seed(config.seed)
        np.random.seed(config.seed)
        key = jax.random.PRNGKey(config.seed)
        key, buffer_key, eval_env_key, env_key, actor_key, sa_key, g_key = jax.random.split(key, 7)

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

        # Network setup
        # Actor
        actor_obs_size = {
            "goal": obs_size,
            "rep": self.repr_dim,
            "state_rep": state_size + self.repr_dim,
        }[self.actor_conditioning]

        actor = Actor(
            action_size=action_size,
            action_seq_len=self.action_seq_len,
            network_width=self.h_dim,
            network_depth=self.n_hidden,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
        )
        actor_state = TrainState.create(
            apply_fn=actor.apply,
            params=actor.init(actor_key, np.ones([1, actor_obs_size])),
            tx=optax.adam(learning_rate=self.policy_lr),
        )

        # Critic: Q(s, g, a_tau) = phi([s; g]) . psi(a_tau)
        sg_encoder = Encoder(
            repr_dim=self.repr_dim,
            network_width=self.h_dim,
            network_depth=self.n_hidden,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
            use_ln=self.use_ln,
        )
        sg_encoder_params = sg_encoder.init(sa_key, np.ones([1, state_size + goal_size]))
        a_encoder = Encoder(
            repr_dim=self.repr_dim,
            network_width=self.h_dim,
            network_depth=self.n_hidden,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
            use_ln=self.use_ln,
        )

        psi_length_dim = self.action_seq_len if self.action_seq_len > 1 else 0
        a_encoder_params = a_encoder.init(
            g_key, np.ones([1, action_size * self.action_seq_len + psi_length_dim])
        )
        critic_state = TrainState.create(
            apply_fn=None,
            params={"sg_encoder": sg_encoder_params, "a_encoder": a_encoder_params},
            tx=optax.adam(learning_rate=self.critic_lr),
        )

        # Entropy coefficient
        target_entropy = -0.5 * action_size * self.action_seq_len
        log_alpha = jnp.asarray(0.0, dtype=jnp.float32)
        alpha_state = TrainState.create(
            apply_fn=None,
            params={"log_alpha": log_alpha},
            tx=optax.adam(learning_rate=self.alpha_lr),
        )

        # Trainstate
        training_state = TrainingState(
            env_steps=jnp.zeros(()),
            gradient_steps=jnp.zeros(()),
            actor_state=actor_state,
            critic_state=critic_state,
            alpha_state=alpha_state,
        )

        # Replay Buffer
        dummy_obs = jnp.zeros((obs_size,))
        dummy_action = jnp.zeros((action_size,))

        dummy_transition = Transition(
            observation=dummy_obs,
            action=dummy_action,
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

        def deterministic_actor_step(training_state, env, env_state, extra_fields):
            obs = env_state.obs
            chunk, _ = sample_actions(
                {"actor": actor, "sg_encoder": sg_encoder},
                {
                    "actor": training_state.actor_state.params,
                    "sg_encoder": training_state.critic_state.params["sg_encoder"],
                },
                obs[:, :state_size],
                obs[:, state_size:],
                self.actor_conditioning,
            )
            actions = chunk[..., :action_size]

            nstate = env.step(env_state, actions)
            state_extras = {x: nstate.info[x] for x in extra_fields}

            return nstate, Transition(
                observation=env_state.obs,
                action=actions,
                reward=nstate.reward,
                discount=1 - nstate.done,
                extras={"state_extras": state_extras},
            )

        def actor_step(actor_state, sg_params, env, env_state, chunk, offset, key, extra_fields):
            obs = env_state.obs
            proposal, _ = sample_actions(
                {"actor": actor, "sg_encoder": sg_encoder},
                {"actor": actor_state.params, "sg_encoder": sg_params},
                obs[:, :state_size],
                obs[:, state_size:],
                self.actor_conditioning,
                key=key,
            )
            chunk = jnp.where((offset == 0)[:, None], proposal, chunk)

            per_step = chunk.reshape(chunk.shape[0], -1, action_size)
            actions = jnp.take_along_axis(per_step, offset[:, None, None], axis=1)[:, 0, :]

            nstate = env.step(env_state, actions)
            state_extras = {x: nstate.info[x] for x in extra_fields}
            next_offset = jnp.where(nstate.done > 0, 0, (offset + 1) % self.action_exec_len)

            return (
                nstate,
                chunk,
                next_offset,
                Transition(
                    observation=env_state.obs,
                    action=actions,
                    reward=nstate.reward,
                    discount=1 - nstate.done,
                    extras={"state_extras": state_extras},
                ),
            )

        @jax.jit
        def get_experience(actor_state, sg_params, env_state, buffer_state, key):
            @jax.jit
            def f(carry, unused_t):
                env_state, chunk, offset, current_key = carry
                current_key, next_key = jax.random.split(current_key)
                env_state, chunk, offset, transition = actor_step(
                    actor_state,
                    sg_params,
                    train_env,
                    env_state,
                    chunk,
                    offset,
                    current_key,
                    extra_fields=("truncation", "traj_id"),
                )
                return (env_state, chunk, offset, next_key), transition

            num_envs = env_state.obs.shape[0]
            init_chunk = jnp.zeros((num_envs, action_size * self.action_seq_len))
            init_offset = jnp.zeros((num_envs,), dtype=jnp.int32)
            (env_state, _, _, _), data = jax.lax.scan(
                f, (env_state, init_chunk, init_offset, key), (), length=self.unroll_length
            )

            buffer_state = replay_buffer.insert(buffer_state, data)
            return env_state, buffer_state

        def prefill_replay_buffer(training_state, env_state, buffer_state, key):
            @jax.jit
            def f(carry, unused):
                del unused
                training_state, env_state, buffer_state, key = carry
                key, new_key = jax.random.split(key)
                env_state, buffer_state = get_experience(
                    training_state.actor_state,
                    training_state.critic_state.params["sg_encoder"],
                    env_state,
                    buffer_state,
                    key,
                )
                training_state = training_state.replace(
                    env_steps=training_state.env_steps + env_steps_per_actor_step,
                )
                return (training_state, env_state, buffer_state, new_key), ()

            return jax.lax.scan(
                f,
                (training_state, env_state, buffer_state, key),
                (),
                length=num_prefill_actor_steps,
            )[0]

        @jax.jit
        def update_networks(carry, transitions):
            training_state, key = carry
            key, critic_key, actor_key = jax.random.split(key, 3)

            context = dict(
                **vars(self),
                **vars(config),
                state_size=state_size,
                action_size=action_size,
                goal_size=goal_size,
                obs_size=obs_size,
                goal_indices=train_env.goal_indices,
                target_entropy=target_entropy,
            )

            networks = dict(
                actor=actor,
                sg_encoder=sg_encoder,
                a_encoder=a_encoder,
            )

            training_state, actor_metrics = update_actor_and_alpha(
                context, networks, transitions, training_state, actor_key
            )
            training_state, critic_metrics = update_critic(
                context, networks, transitions, training_state, critic_key
            )
            training_state = training_state.replace(gradient_steps=training_state.gradient_steps + 1)

            metrics = {}
            metrics.update(actor_metrics)
            metrics.update(critic_metrics)

            return (
                training_state,
                key,
            ), metrics

        @jax.jit
        def training_step(training_state, env_state, buffer_state, key):
            experience_key1, experience_key2, sampling_key, training_key = jax.random.split(key, 4)

            # update buffer
            env_state, buffer_state = get_experience(
                training_state.actor_state,
                training_state.critic_state.params["sg_encoder"],
                env_state,
                buffer_state,
                experience_key1,
            )

            training_state = training_state.replace(
                env_steps=training_state.env_steps + env_steps_per_actor_step,
            )

            # sample actor-step worth of transitions
            buffer_state, transitions = replay_buffer.sample(buffer_state)

            # process transitions for training
            batch_keys = jax.random.split(sampling_key, transitions.observation.shape[0])
            transitions = jax.vmap(flatten_batch, in_axes=(None, 0, 0))(
                (
                    self.discounting,
                    state_size,
                    tuple(np.asarray(train_env.goal_indices)),
                    self.action_seq_len,
                ),
                transitions,
                batch_keys,
            )
            transitions = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"), transitions
            )

            # permute transitions
            permutation = jax.random.permutation(experience_key2, len(transitions.observation))
            transitions = jax.tree_util.tree_map(lambda x: x[permutation], transitions)
            transitions = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1, self.batch_size) + x.shape[1:]),
                transitions,
            )

            # take actor-step worth of training-step
            (
                (
                    training_state,
                    _,
                ),
                metrics,
            ) = jax.lax.scan(update_networks, (training_state, training_key), transitions)

            return (
                training_state,
                env_state,
                buffer_state,
            ), metrics

        @jax.jit
        def training_epoch(
            training_state,
            env_state,
            buffer_state,
            key,
        ):
            @jax.jit
            def f(carry, unused_t):
                ts, es, bs, k = carry
                k, train_key = jax.random.split(k, 2)
                (
                    (
                        ts,
                        es,
                        bs,
                    ),
                    metrics,
                ) = training_step(ts, es, bs, train_key)
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

        training_state, env_state, buffer_state, _ = prefill_replay_buffer(
            training_state, env_state, buffer_state, prefill_key
        )

        """Setting up evaluator"""
        evaluator = ActorEvaluator(
            deterministic_actor_step,
            eval_env,
            num_eval_envs=config.num_eval_envs,
            episode_length=config.episode_length,
            key=eval_env_key,
        )

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

            do_render = ne % config.visualization_interval == 0

            def make_policy(params):
                """Deterministic policy for the evaluator's renderer."""
                actor_params, sg_params = params

                def policy(obs, rng):
                    chunk, _ = sample_actions(
                        {"actor": actor, "sg_encoder": sg_encoder},
                        {"actor": actor_params, "sg_encoder": sg_params},
                        obs[:, :state_size],
                        obs[:, state_size:],
                        self.actor_conditioning,
                    )
                    return chunk[..., :action_size], {}

                return policy

            progress_fn(
                current_step,
                metrics,
                make_policy,
                (
                    training_state.actor_state.params,
                    training_state.critic_state.params["sg_encoder"],
                ),
                unwrapped_env,
                do_render=do_render,
            )

            params = (
                training_state.alpha_state.params,
                training_state.actor_state.params,
                training_state.critic_state.params,
            )

            if config.checkpoint_logdir:
                # Save current policy and critic params.
                path = f"{config.checkpoint_logdir}/step_{int(training_state.env_steps)}.pkl"
                save_params(path, params)

        total_steps = current_step
        assert total_steps >= config.total_env_steps

        logging.info("total steps: %s", total_steps)

        return make_policy, params, metrics
