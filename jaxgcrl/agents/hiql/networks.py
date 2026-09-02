"""HIQL networks: a subgoal representation, an ensembled value function and the two actors."""

import logging

import flax.linen as nn
import jax.numpy as jnp
from flax.linen.initializers import variance_scaling


class MLP(nn.Module):
    """Plain MLP trunk shared by every HIQL network."""

    output_size: int
    network_width: int = 256
    network_depth: int = 2
    use_relu: bool = False
    use_ln: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        lecun_uniform = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros
        activation = nn.relu if self.use_relu else nn.swish

        for _ in range(self.network_depth):
            x = nn.Dense(self.network_width, kernel_init=lecun_uniform, bias_init=bias_init)(x)
            if self.use_ln:
                x = nn.LayerNorm()(x)
            x = activation(x)

        return nn.Dense(self.output_size, kernel_init=lecun_uniform, bias_init=bias_init)(x)


class GoalRep(nn.Module):
    """Subgoal representation phi([s; g]), normalized to length sqrt(rep_dim)."""

    rep_dim: int = 10
    network_width: int = 256
    network_depth: int = 2
    use_relu: bool = False
    use_ln: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        logging.info("goal rep input shape: %s", x.shape)
        rep = MLP(
            output_size=self.rep_dim,
            network_width=self.network_width,
            network_depth=self.network_depth,
            use_relu=self.use_relu,
            use_ln=self.use_ln,
        )(x)
        return rep / (jnp.linalg.norm(rep, axis=-1, keepdims=True) + 1e-6) * jnp.sqrt(self.rep_dim)


class ValueEnsemble(nn.Module):
    """Two independent value heads, used as a `min` for the TD target."""

    network_width: int = 256
    network_depth: int = 2
    use_relu: bool = False
    use_ln: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        logging.info("value input shape: %s", x.shape)
        kwargs = dict(
            output_size=1,
            network_width=self.network_width,
            network_depth=self.network_depth,
            use_relu=self.use_relu,
            use_ln=self.use_ln,
        )
        v1 = MLP(name="v1", **kwargs)(x)
        v2 = MLP(name="v2", **kwargs)(x)
        return jnp.squeeze(v1, axis=-1), jnp.squeeze(v2, axis=-1)


class Actor(nn.Module):
    """Deterministic mean network. AWR uses a unit-variance Gaussian around it (CARL's `const_std`)."""

    output_size: int
    network_width: int = 256
    network_depth: int = 2
    use_relu: bool = False
    use_ln: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        logging.info("actor input shape: %s", x.shape)
        return MLP(
            output_size=self.output_size,
            network_width=self.network_width,
            network_depth=self.network_depth,
            use_relu=self.use_relu,
            use_ln=self.use_ln,
        )(x)
