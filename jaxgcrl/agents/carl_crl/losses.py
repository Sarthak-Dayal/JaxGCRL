import jax
import jax.numpy as jnp

from .networks import sample_actions


def energy_fn(name, x, y):
    if name == "norm":
        return -jnp.sqrt(jnp.sum((x - y) ** 2, axis=-1) + 1e-6)
    elif name == "dot":
        return jnp.sum(x * y, axis=-1)
    elif name == "cosine":
        return jnp.sum(x * y, axis=-1) / (jnp.linalg.norm(x) * jnp.linalg.norm(y) + 1e-6)
    elif name == "l2":
        return -jnp.sum((x - y) ** 2, axis=-1)
    else:
        raise ValueError(f"Unknown energy function: {name}")


def contrastive_loss_fn(name, logits):
    if name == "fwd_infonce":
        critic_loss = -jnp.mean(jnp.diag(logits) - jax.nn.logsumexp(logits, axis=1))
    elif name == "bwd_infonce":
        critic_loss = -jnp.mean(jnp.diag(logits) - jax.nn.logsumexp(logits, axis=0))
    elif name == "sym_infonce":
        critic_loss = -jnp.mean(
            2 * jnp.diag(logits) - jax.nn.logsumexp(logits, axis=1) - jax.nn.logsumexp(logits, axis=0)
        )
    elif name == "binary_nce":
        eye = jnp.eye(logits.shape[0])
        critic_loss = jnp.mean(eye * jax.nn.softplus(-logits) + (1 - eye) * jax.nn.softplus(logits))
    else:
        raise ValueError(f"Unknown contrastive loss function: {name}")
    return critic_loss


def psi_input(config, action_seq, num_actions):
    n = config["action_seq_len"]
    if n == 1:
        return action_seq
    return jnp.concatenate([action_seq, jax.nn.one_hot(num_actions - 1, n)], axis=-1)


def update_actor_and_alpha(config, networks, transitions, training_state, key):
    def actor_loss(actor_params, critic_params, log_alpha, transitions, key):
        obs = transitions.observation  # expected_shape = self.batch_size, obs_size + goal_size
        state = obs[:, : config["state_size"]]
        goal = obs[:, config["state_size"] :]

        action, log_prob = sample_actions(
            networks,
            {"actor": actor_params, "sg_encoder": critic_params["sg_encoder"]},
            state,
            goal,
            config["actor_conditioning"],
            key=key,
        )

        sg_encoder_params, a_encoder_params = (
            critic_params["sg_encoder"],
            critic_params["a_encoder"],
        )
        sg_repr = networks["sg_encoder"].apply(sg_encoder_params, jnp.concatenate([state, goal], axis=-1))
        a_repr = networks["a_encoder"].apply(
            a_encoder_params,
            psi_input(config, action, jnp.full((action.shape[0],), config["action_seq_len"])),
        )

        qf_pi = energy_fn(config["energy_fn"], sg_repr, a_repr)

        actor_loss = jnp.mean(jnp.exp(log_alpha) * log_prob - qf_pi)

        return actor_loss, log_prob

    def alpha_loss(alpha_params, log_prob):
        alpha = jnp.exp(alpha_params["log_alpha"])
        alpha_loss = alpha * jnp.mean(jax.lax.stop_gradient(-log_prob - config["target_entropy"]))
        return jnp.mean(alpha_loss)

    (actor_loss, log_prob), actor_grad = jax.value_and_grad(actor_loss, has_aux=True)(
        training_state.actor_state.params,
        training_state.critic_state.params,
        training_state.alpha_state.params["log_alpha"],
        transitions,
        key,
    )
    new_actor_state = training_state.actor_state.apply_gradients(grads=actor_grad)

    alpha_loss, alpha_grad = jax.value_and_grad(alpha_loss)(training_state.alpha_state.params, log_prob)
    new_alpha_state = training_state.alpha_state.apply_gradients(grads=alpha_grad)

    training_state = training_state.replace(actor_state=new_actor_state, alpha_state=new_alpha_state)

    metrics = {
        "entropy": -log_prob,
        "actor_loss": actor_loss,
        "alpha_loss": alpha_loss,
        "log_alpha": training_state.alpha_state.params["log_alpha"],
    }

    return training_state, metrics


def update_critic(config, networks, transitions, training_state, key):
    def critic_loss(critic_params, transitions, key):
        sg_encoder_params, a_encoder_params = (
            critic_params["sg_encoder"],
            critic_params["a_encoder"],
        )

        state = transitions.observation[:, : config["state_size"]]
        action_seq = transitions.extras["action_seq"]

        sg_repr = networks["sg_encoder"].apply(sg_encoder_params, transitions.observation)
        a_repr = networks["a_encoder"].apply(
            a_encoder_params, psi_input(config, action_seq, transitions.extras["num_actions"])
        )

        # InfoNCE
        logits = energy_fn(config["energy_fn"], sg_repr[:, None, :], a_repr[None, :, :])
        critic_loss = contrastive_loss_fn(config["contrastive_loss_fn"], logits)

        # logsumexp regularisation
        logsumexp = jax.nn.logsumexp(logits + 1e-6, axis=1)
        critic_loss += config["logsumexp_penalty_coeff"] * jnp.mean(logsumexp**2)

        I = jnp.eye(logits.shape[0])
        correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
        logits_pos = jnp.sum(logits * I) / jnp.sum(I)
        logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)

        return critic_loss, (logsumexp, I, correct, logits_pos, logits_neg)

    (loss, (logsumexp, I, correct, logits_pos, logits_neg)), grad = jax.value_and_grad(
        critic_loss, has_aux=True
    )(training_state.critic_state.params, transitions, key)
    new_critic_state = training_state.critic_state.apply_gradients(grads=grad)
    training_state = training_state.replace(critic_state=new_critic_state)

    metrics = {
        "categorical_accuracy": jnp.mean(correct),
        "logits_pos": logits_pos,
        "logits_neg": logits_neg,
        "logsumexp": logsumexp.mean(),
        "critic_loss": loss,
    }

    return training_state, metrics
