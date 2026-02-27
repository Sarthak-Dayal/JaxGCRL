"""HIQL loss and update functions.

All networks are passed as a dict of linen modules. All params are passed
explicitly so that jax.grad traces through the right ones.

Batch is a dict with keys produced by flatten_batch_hiql:
    observations, next_observations, actions,
    value_goals, rewards, masks,
    low_actor_goals, low_rewards, low_masks,
    high_actor_goals, high_actor_targets.
"""

import jax
import jax.numpy as jnp



# Helpers
def expectile_loss(adv, diff, expectile):
    weight = jnp.where(adv >= 0, expectile, (1 - expectile))
    return weight * (diff ** 2)


def _apply_value(value_module, params, obs, goals, goal_rep_module, goal_rep_params):
    """Run goal_rep then value.  Returns scalar per batch element."""
    phi = goal_rep_module.apply(goal_rep_params, obs, goals)
    v = value_module.apply(params, obs, phi)        # (batch, 1)
    return v.squeeze(-1)                            # (batch,)



# Individual losses
def value_loss(batch, networks, params, target_params, config):
    """IQL expectile value loss (high-level).

    Uses double-value trick: advantage weights come from *target* V,
    but the regression target for gradient uses current V.
    """
    goal_rep = networks["goal_rep"]

    # --- target value at s' ---
    next_v1_t = _apply_value(
        networks["value"], target_params["value1"], batch["next_observations"],
        batch["value_goals"], goal_rep, target_params["goal_rep"],
    )
    next_v2_t = _apply_value(
        networks["value"], target_params["value2"], batch["next_observations"],
        batch["value_goals"], goal_rep, target_params["goal_rep"],
    )
    next_v_t = jnp.minimum(next_v1_t, next_v2_t)
    q = batch["rewards"] + config["discount"] * batch["masks"] * next_v_t

    # --- target value at s (for advantage weighting) ---
    v1_t = _apply_value(
        networks["value"], target_params["value1"], batch["observations"],
        batch["value_goals"], goal_rep, target_params["goal_rep"],
    )
    v2_t = _apply_value(
        networks["value"], target_params["value2"], batch["observations"],
        batch["value_goals"], goal_rep, target_params["goal_rep"],
    )
    v_t = (v1_t + v2_t) / 2
    adv = q - v_t

    # --- current value at s (gradient flows here) ---
    q1 = batch["rewards"] + config["discount"] * batch["masks"] * next_v1_t
    q2 = batch["rewards"] + config["discount"] * batch["masks"] * next_v2_t

    v1 = _apply_value(
        networks["value"], params["value1"], batch["observations"],
        batch["value_goals"], goal_rep, params["goal_rep"],
    )
    v2 = _apply_value(
        networks["value"], params["value2"], batch["observations"],
        batch["value_goals"], goal_rep, params["goal_rep"],
    )
    v = (v1 + v2) / 2

    loss1 = expectile_loss(adv, q1 - v1, config["expectile"]).mean()
    loss2 = expectile_loss(adv, q2 - v2, config["expectile"]).mean()

    return loss1 + loss2, {
        "value/loss": loss1 + loss2,
        "value/v_mean": v.mean(),
        "value/v_max": v.max(),
        "value/v_min": v.min(),
    }


def low_actor_loss(batch, networks, params, target_params, config):
    """AWR-style low-level actor loss.

    Advantage comes from V(s', w) - V(s, w) where w = low_actor_goals.
    Action distribution is conditioned on phi([s; w]).
    Uses current params (stop-gradient) for advantage, matching ogbench.
    """
    goal_rep = networks["goal_rep"]
    sg = jax.lax.stop_gradient

    # advantage from current value (stop-gradient, matching ogbench)
    v1 = _apply_value(
        networks["value"], sg(params["value1"]), batch["observations"],
        batch["low_actor_goals"], goal_rep, sg(params["goal_rep"]),
    )
    v2 = _apply_value(
        networks["value"], sg(params["value2"]), batch["observations"],
        batch["low_actor_goals"], goal_rep, sg(params["goal_rep"]),
    )
    nv1 = _apply_value(
        networks["value"], sg(params["value1"]), batch["next_observations"],
        batch["low_actor_goals"], goal_rep, sg(params["goal_rep"]),
    )
    nv2 = _apply_value(
        networks["value"], sg(params["value2"]), batch["next_observations"],
        batch["low_actor_goals"], goal_rep, sg(params["goal_rep"]),
    )
    v = (v1 + v2) / 2
    nv = (nv1 + nv2) / 2
    adv = nv - v

    exp_a = jnp.minimum(jnp.exp(adv * config["low_alpha"]), 100.0)

    # subgoal representation -- gradient flows through goal_rep and low_actor
    phi = goal_rep.apply(params["goal_rep"], batch["observations"], batch["low_actor_goals"])
    if not config["low_actor_rep_grad"]:
        phi = jax.lax.stop_gradient(phi)

    dist = networks["low_actor"].apply(params["low_actor"], batch["observations"], phi)
    log_prob = dist.log_prob(batch["actions"])

    loss = -(exp_a * log_prob).mean()

    return loss, {
        "low_actor/loss": loss,
        "low_actor/adv": adv.mean(),
        "low_actor/bc_log_prob": log_prob.mean(),
        "low_actor/mse": jnp.mean((dist.mode() - batch["actions"]) ** 2),
        "low_actor/std": jnp.mean(dist.scale_diag),
    }


def high_actor_loss(batch, networks, params, target_params, config):
    """AWR-style high-level actor loss.

    Advantage: V(target, g) - V(s, g) where g = high_actor_goals.
    Prediction target: phi([s; high_actor_targets]) (stop-grad).
    Uses current params (stop-gradient) for advantage, matching ogbench.
    """
    goal_rep = networks["goal_rep"]
    sg = jax.lax.stop_gradient

    # advantage from current value (stop-gradient, matching ogbench)
    v1 = _apply_value(
        networks["value"], sg(params["value1"]), batch["observations"],
        batch["high_actor_goals"], goal_rep, sg(params["goal_rep"]),
    )
    v2 = _apply_value(
        networks["value"], sg(params["value2"]), batch["observations"],
        batch["high_actor_goals"], goal_rep, sg(params["goal_rep"]),
    )
    nv1 = _apply_value(
        networks["value"], sg(params["value1"]), batch["high_actor_targets"],
        batch["high_actor_goals"], goal_rep, sg(params["goal_rep"]),
    )
    nv2 = _apply_value(
        networks["value"], sg(params["value2"]), batch["high_actor_targets"],
        batch["high_actor_goals"], goal_rep, sg(params["goal_rep"]),
    )
    v = (v1 + v2) / 2
    nv = (nv1 + nv2) / 2
    adv = nv - v

    exp_a = jnp.minimum(jnp.exp(adv * config["high_alpha"]), 100.0)

    # high actor predicts subgoal rep; target is phi([s; target_obs]) with stop-grad
    dist = networks["high_actor"].apply(
        params["high_actor"], batch["observations"], batch["high_actor_goals"],
    )
    goal_indices_arr = jnp.array(config["goal_indices"])
    target_rep = goal_rep.apply(
        sg(params["goal_rep"]), batch["observations"],
        batch["high_actor_targets"][:, goal_indices_arr],
    )
    log_prob = dist.log_prob(target_rep)

    loss = -(exp_a * log_prob).mean()

    return loss, {
        "high_actor/loss": loss,
        "high_actor/adv": adv.mean(),
        "high_actor/bc_log_prob": log_prob.mean(),
        "high_actor/mse": jnp.mean((dist.mode() - target_rep) ** 2),
        "high_actor/std": jnp.mean(dist.scale_diag),
    }


# Combined loss (single backward pass over all params)
def total_loss(batch, networks, params, target_params, config):
    """Sum of value + low_actor + (optionally) high_actor losses."""
    info = {}

    v_loss, v_info = value_loss(batch, networks, params, target_params, config)
    info.update(v_info)

    la_loss, la_info = low_actor_loss(batch, networks, params, target_params, config)
    info.update(la_info)

    loss = v_loss + la_loss

    if not config.get("flat_policy", False):
        ha_loss, ha_info = high_actor_loss(batch, networks, params, target_params, config)
        info.update(ha_info)
        loss = loss + ha_loss

    return loss, info



# Update step
def update_hiql(config, networks, batch, training_state, key):
    """One gradient step on all HIQL parameters.

    Args:
        config: dict of hyperparams.
        networks: dict of linen modules (goal_rep, value, low_actor, high_actor).
        batch: dict produced by flatten_batch_hiql.
        training_state: TrainingState dataclass.
        key: PRNGKey (unused currently, reserved for future stochastic losses).

    Returns:
        (training_state, metrics) with updated params and target params.
    """
    target_params = {
        "goal_rep": training_state.target_params["goal_rep"],
        "value1": training_state.target_params["value1"],
        "value2": training_state.target_params["value2"],
    }

    def loss_fn(params):
        return total_loss(batch, networks, params, target_params, config)

    (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        training_state.hiql_state.params,
    )
    new_hiql_state = training_state.hiql_state.apply_gradients(grads=grads)

    # Polyak-average only the target param keys (goal_rep, value1, value2)
    tau = config["tau"]
    new_target_params = {
        k: jax.tree_util.tree_map(
            lambda p, tp: p * tau + tp * (1 - tau),
            new_hiql_state.params[k],
            training_state.target_params[k],
        )
        for k in training_state.target_params
    }

    training_state = training_state.replace(
        hiql_state=new_hiql_state,
        target_params=new_target_params,
        gradient_steps=training_state.gradient_steps + 1,
    )

    info["total_loss"] = loss
    return training_state, info
