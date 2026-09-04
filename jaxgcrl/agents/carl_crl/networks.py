import logging
from typing import Literal, Optional, Tuple, TypedDict

import flax.linen as nn
import jax
import jax.numpy as jnp
from brax.training.types import Params, PRNGKey
from flax.linen.initializers import variance_scaling


class Encoder(nn.Module):
    repr_dim: int = 64
    network_width: int = 256
    network_depth: int = 4
    skip_connections: int = (
        0  # 0 for no skip connections, >= 0 means the frequency of skip connections (every X layers)
    )
    use_relu: bool = False
    use_ln: bool = False

    @nn.compact
    def __call__(self, data: jnp.ndarray):
        logging.info("encoder input shape: %s", data.shape)
        lecun_unfirom = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        if self.use_ln:
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x

        if self.use_relu:
            activation = nn.relu
        else:
            activation = nn.swish

        x = data
        for i in range(self.network_depth):
            x = nn.Dense(self.network_width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
            x = normalize(x)
            x = activation(x)

            if self.skip_connections:
                if i == 0:
                    skip = x
                if i > 0 and i % self.skip_connections == 0:
                    x = x + skip
                    skip = x

        x = nn.Dense(self.repr_dim, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        return x


class Actor(nn.Module):
    action_size: int
    action_seq_len: int = 1
    network_width: int = 256
    network_depth: int = 4
    skip_connections: int = (
        0  # 0 for no skip connections, >= 0 means the frequency of skip connections (every X layers)
    )
    use_relu: bool = False
    use_ln: bool = False
    LOG_STD_MAX = 2
    LOG_STD_MIN = -5

    @nn.compact
    def __call__(self, x):
        if self.use_ln:
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x

        if self.use_relu:
            activation = nn.relu
        else:
            activation = nn.swish

        lecun_unfirom = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        out_size = self.action_size * self.action_seq_len

        logging.info("actor input shape: %s", x.shape)
        for i in range(self.network_depth):
            x = nn.Dense(self.network_width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
            x = normalize(x)
            x = activation(x)

            if self.skip_connections:
                if i == 0:
                    skip = x
                if i > 0 and i % self.skip_connections == 0:
                    x = x + skip
                    skip = x

        mean = nn.Dense(out_size, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        log_std = nn.Dense(out_size, kernel_init=lecun_unfirom, bias_init=bias_init)(x)

        log_std = nn.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (
            log_std + 1
        )  # From SpinUp / Denis Yarats

        return mean, log_std


class ActorNetworks(TypedDict):
    """The modules the actor path touches.

    The agent's full registry also carries `a_encoder` for the critic; a dict with that extra key
    still satisfies this, so the losses can pass their registry straight through.
    """

    actor: "Actor"
    sg_encoder: "Encoder"


class ActorParams(TypedDict):
    actor: Params
    sg_encoder: Params


def actor_inputs(
    sg_encoder: "Encoder",
    sg_params: Params,
    state: jnp.ndarray,
    goal: jnp.ndarray,
    conditioning: Literal["goal", "rep", "state_rep"],
) -> jnp.ndarray:
    if conditioning == "goal":
        return jnp.concatenate([state, goal], axis=-1)

    rep = jax.lax.stop_gradient(sg_encoder.apply(sg_params, jnp.concatenate([state, goal], axis=-1)))
    if conditioning == "rep":
        return rep
    if conditioning == "state_rep":
        return jnp.concatenate([state, rep], axis=-1)


def sample_actions(
    networks: ActorNetworks,
    params: ActorParams,
    state: jnp.ndarray,
    goal: jnp.ndarray,
    conditioning: Literal["goal", "rep", "state_rep"],
    *,
    key: Optional[PRNGKey] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    inputs = actor_inputs(networks["sg_encoder"], params["sg_encoder"], state, goal, conditioning)
    means, log_stds = networks["actor"].apply(params["actor"], inputs)
    stds = jnp.exp(log_stds)

    x = means if key is None else means + stds * jax.random.normal(key, shape=means.shape, dtype=means.dtype)
    actions = nn.tanh(x)

    # tanh-Gaussian density: log N(x; mu, sigma) - sum log(1 - tanh(x)^2)
    log_prob = jax.scipy.stats.norm.logpdf(x, loc=means, scale=stds)
    log_prob -= 2 * (jnp.log(2.0) - x - nn.softplus(-2.0 * x))
    return actions, log_prob.sum(-1)
