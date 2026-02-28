"""HSAC loss and update functions.

SAC-style Q-learning and policy optimization with HIQL's GoalRep.

Batch is a dict with keys produced by flatten_batch_hsac:
    observations, next_observations, actions,
    value_goals, rewards, masks,
    low_actor_goals,
    high_actor_goals, high_actor_targets.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn


# ---------------------------------------------------------------------------
# Tanh-normal helpers (matching CRL / standard SAC)
# ---------------------------------------------------------------------------

def _sample_tanh_normal(mean, log_std, key):
    """Reparameterized sample from TanhNormal. Returns (action, log_prob)."""
    std = jnp.exp(log_std)
    noise = jax.random.normal(key, shape=mean.shape, dtype=mean.dtype)
    x_t = mean + std * noise
    action = nn.tanh(x_t)
    log_prob = jax.scipy.stats.norm.logpdf(x_t, loc=mean, scale=std)
    log_prob -= jnp.log(1 - jnp.square(action) + 1e-6)
    log_prob = log_prob.sum(-1)
    return action, log_prob


def _sample_normal(mean, log_std, key):
    """Reparameterized sample from Normal. Returns (sample, log_prob)."""
    std = jnp.exp(log_std)
    noise = jax.random.normal(key, shape=mean.shape, dtype=mean.dtype)
    sample = mean + std * noise
    log_prob = jax.scipy.stats.norm.logpdf(sample, loc=mean, scale=std)
    log_prob = log_prob.sum(-1)
    return sample, log_prob


# ---------------------------------------------------------------------------
# Individual losses
# ---------------------------------------------------------------------------

def critic_loss(batch, networks, params, target_params, config, key):
    """SAC Bellman critic loss with double-Q and GoalRep.

    Q(s, a, phi(s, g)) is trained toward:
        y = r + gamma * mask * (min_i Q_targ(s', a', phi(s', g)) - alpha * log pi(a'))
    where a' ~ pi_l(s', phi(s', g)).
    """
    sg = jax.lax.stop_gradient
    goal_rep = networks["goal_rep"]

    phi = goal_rep.apply(params["goal_rep"], batch["observations"], batch["value_goals"])
    phi_next = goal_rep.apply(
        sg(target_params["goal_rep"]), batch["next_observations"], batch["value_goals"]
    )

    # Current Q values
    q_both = networks["q"].apply(
        params["q"], batch["observations"], batch["actions"], phi,
    )  # (batch, 2)

    # Next actions from current low actor (for soft value target)
    next_mean, next_log_std = networks["low_actor"].apply(
        sg(params["low_actor"]), batch["next_observations"], sg(phi_next),
    )
    next_action, next_log_prob = _sample_tanh_normal(next_mean, next_log_std, key)

    # Target Q at (s', a')
    next_q_both = networks["q"].apply(
        target_params["q"], batch["next_observations"], next_action, sg(phi_next),
    )
    next_q_min = jnp.min(next_q_both, axis=-1)

    alpha = jnp.exp(sg(params["log_alpha_low"]))
    next_v = next_q_min - alpha * next_log_prob
    target_q = batch["rewards"] + config["discount"] * batch["masks"] * next_v
    target_q = sg(target_q)

    loss = 0.0
    for i in range(q_both.shape[-1]):
        loss = loss + 0.5 * jnp.mean((q_both[:, i] - target_q) ** 2)

    return loss, {
        "critic/loss": loss,
        "critic/q_mean": jnp.mean(q_both),
        "critic/q_min": jnp.min(q_both),
        "critic/q_max": jnp.max(q_both),
        "critic/target_q_mean": jnp.mean(target_q),
    }


def low_actor_loss(batch, networks, params, target_params, config, key):
    """SAC actor loss for the low-level policy.

    Maximizes Q(s, a, phi(s, g)) - alpha * log pi_l(a | s, phi(s, g))
    using the reparameterization trick. Gradients flow into GoalRep.
    """
    sg = jax.lax.stop_gradient
    goal_rep = networks["goal_rep"]

    phi = goal_rep.apply(params["goal_rep"], batch["observations"], batch["low_actor_goals"])

    mean, log_std = networks["low_actor"].apply(
        params["low_actor"], batch["observations"], phi,
    )
    action, log_prob = _sample_tanh_normal(mean, log_std, key)

    q_both = networks["q"].apply(
        sg(params["q"]), batch["observations"], action, sg(phi),
    )
    q_min = jnp.min(q_both, axis=-1)

    alpha = jnp.exp(sg(params["log_alpha_low"]))
    loss = jnp.mean(alpha * log_prob - q_min)

    return loss, {
        "low_actor/loss": loss,
        "low_actor/entropy": -log_prob.mean(),
        "low_actor/q_pi": q_min.mean(),
        "low_actor/std": jnp.mean(jnp.exp(log_std)),
    }


def high_actor_loss(batch, networks, params, target_params, config, key):
    """SAC actor loss for the high-level policy.

    The high actor proposes a subgoal z in phi-space.
    Then the low actor (frozen params) converts z to action a.
    We evaluate a against Q(s, a, phi(s, g)) for the final goal.

    Gradients flow: dQ/da * da/dz * dz/d(params_h).
    """
    sg = jax.lax.stop_gradient
    goal_rep = networks["goal_rep"]

    key_high, key_low = jax.random.split(key)

    # High actor proposes subgoal z
    z_mean, z_log_std = networks["high_actor"].apply(
        params["high_actor"], batch["observations"], batch["high_actor_goals"],
    )
    z, z_log_prob = _sample_normal(z_mean, z_log_std, key_high)

    # Low actor converts subgoal to action (frozen low actor params, grad flows through z)
    low_mean, low_log_std = networks["low_actor"].apply(
        sg(params["low_actor"]), batch["observations"], z,
    )
    action, _ = _sample_tanh_normal(low_mean, low_log_std, key_low)

    # Evaluate against Q with the final goal's representation
    phi_goal = goal_rep.apply(
        sg(params["goal_rep"]), batch["observations"], batch["high_actor_goals"],
    )
    q_both = networks["q"].apply(
        sg(params["q"]), batch["observations"], action, phi_goal,
    )
    q_min = jnp.min(q_both, axis=-1)

    alpha = jnp.exp(sg(params["log_alpha_high"]))
    loss = jnp.mean(alpha * z_log_prob - q_min)

    return loss, {
        "high_actor/loss": loss,
        "high_actor/entropy": -z_log_prob.mean(),
        "high_actor/q_pi": q_min.mean(),
        "high_actor/std": jnp.mean(jnp.exp(z_log_std)),
    }


def alpha_loss_fn(log_alpha, log_prob, target_entropy):
    """Auto-tuned entropy temperature."""
    alpha = jnp.exp(log_alpha)
    return jnp.mean(alpha * jax.lax.stop_gradient(-log_prob - target_entropy))


# ---------------------------------------------------------------------------
# Combined update
# ---------------------------------------------------------------------------

def update_hsac(config, networks, batch, training_state, key):
    """One gradient step on all HSAC parameters.

    Unlike HIQL (single combined loss), we update critic, actors, and alphas
    in sequence since their gradients don't need to be coupled.
    """
    sg = jax.lax.stop_gradient
    key_c, key_la, key_ha, key_la_alpha, key_ha_alpha = jax.random.split(key, 5)
    all_metrics = {}

    params = training_state.hsac_state.params
    target_params = training_state.target_params

    # --- Critic update ---
    def _critic_loss(critic_and_rep_params):
        p = {**params, "q": critic_and_rep_params["q"], "goal_rep": critic_and_rep_params["goal_rep"]}
        return critic_loss(batch, networks, p, target_params, config, key_c)

    critic_grad_params = {"q": params["q"], "goal_rep": params["goal_rep"]}
    (c_loss, c_info), c_grads = jax.value_and_grad(_critic_loss, has_aux=True)(critic_grad_params)
    all_metrics.update(c_info)

    # --- Low actor update (also trains GoalRep) ---
    def _low_actor_loss(la_and_rep_params):
        p = {**params, "low_actor": la_and_rep_params["low_actor"], "goal_rep": la_and_rep_params["goal_rep"]}
        return low_actor_loss(batch, networks, p, target_params, config, key_la)

    la_grad_params = {"low_actor": params["low_actor"], "goal_rep": params["goal_rep"]}
    (la_loss, la_info), la_grads = jax.value_and_grad(_low_actor_loss, has_aux=True)(la_grad_params)
    all_metrics.update(la_info)

    # --- High actor update (hierarchical only) ---
    ha_grads = {}
    if not config.get("flat_policy", False):
        def _high_actor_loss(ha_params):
            p = {**params, "high_actor": ha_params}
            return high_actor_loss(batch, networks, p, target_params, config, key_ha)

        (ha_loss, ha_info), ha_grad = jax.value_and_grad(_high_actor_loss, has_aux=True)(params["high_actor"])
        all_metrics.update(ha_info)
        ha_grads["high_actor"] = ha_grad

    # --- Alpha updates ---
    # Low alpha
    low_mean, low_log_std = networks["low_actor"].apply(
        sg(params["low_actor"]), batch["observations"],
        networks["goal_rep"].apply(sg(params["goal_rep"]), batch["observations"], batch["low_actor_goals"]),
    )
    _, low_log_prob = _sample_tanh_normal(low_mean, low_log_std, key_la_alpha)

    def _alpha_low_loss(log_alpha):
        return alpha_loss_fn(log_alpha, low_log_prob, config["target_entropy_low"])

    alpha_low_loss_val, alpha_low_grad = jax.value_and_grad(_alpha_low_loss)(params["log_alpha_low"])
    all_metrics["alpha/low_loss"] = alpha_low_loss_val
    all_metrics["alpha/low"] = jnp.exp(params["log_alpha_low"])

    alpha_high_grad = jnp.zeros_like(params["log_alpha_high"])
    if not config.get("flat_policy", False):
        z_mean, z_log_std = networks["high_actor"].apply(
            sg(params["high_actor"]), batch["observations"], batch["high_actor_goals"],
        )
        _, z_log_prob = _sample_normal(z_mean, z_log_std, key_ha_alpha)

        def _alpha_high_loss(log_alpha):
            return alpha_loss_fn(log_alpha, z_log_prob, config["target_entropy_high"])

        alpha_high_loss_val, alpha_high_grad = jax.value_and_grad(_alpha_high_loss)(params["log_alpha_high"])
        all_metrics["alpha/high_loss"] = alpha_high_loss_val
        all_metrics["alpha/high"] = jnp.exp(params["log_alpha_high"])

    # --- Merge gradients and apply ---
    # Build full gradient dict matching params structure
    full_grads = jax.tree_util.tree_map(jnp.zeros_like, params)

    # Critic grads: q and goal_rep
    full_grads = {**full_grads, "q": c_grads["q"]}
    goal_rep_grad = jax.tree_util.tree_map(
        lambda a, b: a + b, c_grads["goal_rep"], la_grads["goal_rep"]
    )
    full_grads = {**full_grads, "goal_rep": goal_rep_grad}

    # Low actor grads
    full_grads = {**full_grads, "low_actor": la_grads["low_actor"]}

    # High actor grads
    if "high_actor" in ha_grads:
        full_grads = {**full_grads, "high_actor": ha_grads["high_actor"]}

    # Alpha grads
    full_grads = {**full_grads, "log_alpha_low": alpha_low_grad, "log_alpha_high": alpha_high_grad}

    new_hsac_state = training_state.hsac_state.apply_gradients(grads=full_grads)

    # --- Target network Polyak update (goal_rep and q only) ---
    tau = config["tau"]
    new_target_params = {
        k: jax.tree_util.tree_map(
            lambda p, tp: p * tau + tp * (1 - tau),
            new_hsac_state.params[k],
            training_state.target_params[k],
        )
        for k in training_state.target_params
    }

    training_state = training_state.replace(
        hsac_state=new_hsac_state,
        target_params=new_target_params,
        gradient_steps=training_state.gradient_steps + 1,
    )

    all_metrics["total_loss"] = c_loss + la_loss
    return training_state, all_metrics
