"""HIQL losses: a TD value function plus AWR for the low- and high-level actors.

Unlike the original HIQL, the value function is not fit with an expectile loss; it is a plain
TD regression with a target network, matching the way the other agents in this repo learn
critics. The two actors are then extracted from it with advantage-weighted regression.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from jaxgcrl.utils.hierarchical_batch import HierarchicalBatch


class LossContext(NamedTuple):
    """The hyperparameters the losses read, pulled out of the agent config once."""

    discounting: float
    tau: float
    low_alpha: float
    high_alpha: float
    low_actor_rep_grad: bool


def soft_update(target_params, params, tau):
    return jax.tree_util.tree_map(lambda t, p: (1 - tau) * t + tau * p, target_params, params)


def goal_rep(networks, params, states, goals):
    """phi([s; g]), the length-normalized subgoal representation."""
    return networks["goal_rep"].apply(params["goal_rep"], jnp.concatenate([states, goals], axis=-1))


def values(networks, params, states, goals):
    """V(s, phi([s; g])) for both ensemble members."""
    rep = goal_rep(networks, params, states, goals)
    return networks["value"].apply(params["value"], jnp.concatenate([states, rep], axis=-1))


def gaussian_log_prob(mean, target):
    """Log density of a unit-variance diagonal Gaussian, up to an additive constant."""
    return -0.5 * jnp.sum((target - mean) ** 2, axis=-1)


def awr_weights(adv, alpha):
    return jnp.minimum(jnp.exp(adv * alpha), 100.0)


def update_value(ctx: LossContext, networks, batch: HierarchicalBatch, training_state, key):
    """TD regression of V(s, phi([s; g])) onto r + gamma * mask * min(V_target(s', g))."""

    def value_loss(params, target_params, batch):
        state, next_state, goal = batch.state, batch.next_state, batch.value_goal

        next_v1_t, next_v2_t = values(networks, target_params, next_state, goal)
        next_v_t = jnp.minimum(next_v1_t, next_v2_t)
        target_v = jax.lax.stop_gradient(
            batch.gc_reward + ctx.discounting * batch.mask * next_v_t
        )

        v1, v2 = values(networks, params, state, goal)
        loss = jnp.mean((v1 - target_v) ** 2) + jnp.mean((v2 - target_v) ** 2)

        v = (v1 + v2) / 2
        return loss, (v, target_v)

    (loss, (v, target_v)), grad = jax.value_and_grad(value_loss, has_aux=True)(
        training_state.value_state.params,
        training_state.target_value_params,
        batch,
    )
    new_value_state = training_state.value_state.apply_gradients(grads=grad)
    new_target_params = soft_update(
        training_state.target_value_params, new_value_state.params, ctx.tau
    )
    training_state = training_state.replace(
        value_state=new_value_state, target_value_params=new_target_params
    )

    metrics = {
        "value_loss": loss,
        "v_mean": v.mean(),
        "v_max": v.max(),
        "v_min": v.min(),
        "v_target_mean": target_v.mean(),
    }
    return training_state, metrics


def update_low_actor(ctx: LossContext, networks, batch: HierarchicalBatch, training_state, key):
    """AWR on the one-step advantage V(s', w) - V(s, w) towards the subgoal w."""

    def low_actor_loss(actor_params, value_params, batch):
        state, next_state, goal = batch.state, batch.next_state, batch.low_actor_goal

        v1, v2 = values(networks, value_params, state, goal)
        nv1, nv2 = values(networks, value_params, next_state, goal)
        adv = (nv1 + nv2) / 2 - (v1 + v2) / 2
        exp_a = awr_weights(adv, ctx.low_alpha)

        # The low-level policy is conditioned on the same representation the value function uses.
        rep = goal_rep(networks, value_params, state, goal)
        if not ctx.low_actor_rep_grad:
            rep = jax.lax.stop_gradient(rep)
        mean = networks["low_actor"].apply(actor_params, jnp.concatenate([state, rep], axis=-1))
        log_prob = gaussian_log_prob(mean, batch.action)

        loss = -jnp.mean(exp_a * log_prob)
        return loss, (adv, exp_a, log_prob, mean)

    value_params = jax.lax.stop_gradient(training_state.value_state.params)
    (loss, (adv, exp_a, log_prob, mean)), grad = jax.value_and_grad(low_actor_loss, has_aux=True)(
        training_state.low_actor_state.params, value_params, batch
    )
    new_actor_state = training_state.low_actor_state.apply_gradients(grads=grad)
    training_state = training_state.replace(low_actor_state=new_actor_state)

    metrics = {
        "low_actor_loss": loss,
        "low_adv": adv.mean(),
        "low_exp_a": exp_a.mean(),
        "low_bc_log_prob": log_prob.mean(),
        "low_mse": jnp.mean((mean - batch.action) ** 2),
    }
    return training_state, metrics


def update_high_actor(ctx: LossContext, networks, batch: HierarchicalBatch, training_state, key):
    """AWR on the subgoal-horizon advantage, regressing onto phi([s; w])."""

    def high_actor_loss(actor_params, value_params, batch):
        state, goal = batch.state, batch.high_actor_goal
        target_state = batch.high_actor_target_state

        v1, v2 = values(networks, value_params, state, goal)
        nv1, nv2 = values(networks, value_params, target_state, goal)
        adv = (nv1 + nv2) / 2 - (v1 + v2) / 2
        exp_a = awr_weights(adv, ctx.high_alpha)

        target_rep = goal_rep(networks, value_params, state, batch.high_actor_target_goal)
        mean = networks["high_actor"].apply(actor_params, jnp.concatenate([state, goal], axis=-1))
        log_prob = gaussian_log_prob(mean, target_rep)

        loss = -jnp.mean(exp_a * log_prob)
        return loss, (adv, exp_a, log_prob, mean, target_rep)

    value_params = jax.lax.stop_gradient(training_state.value_state.params)
    (loss, (adv, exp_a, log_prob, mean, target_rep)), grad = jax.value_and_grad(
        high_actor_loss, has_aux=True
    )(training_state.high_actor_state.params, value_params, batch)
    new_actor_state = training_state.high_actor_state.apply_gradients(grads=grad)
    training_state = training_state.replace(high_actor_state=new_actor_state)

    metrics = {
        "high_actor_loss": loss,
        "high_adv": adv.mean(),
        "high_exp_a": exp_a.mean(),
        "high_bc_log_prob": log_prob.mean(),
        "high_mse": jnp.mean((mean - target_rep) ** 2),
    }
    return training_state, metrics
