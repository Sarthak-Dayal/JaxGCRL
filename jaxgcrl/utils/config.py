from typing import Union

from flax.struct import dataclass

from jaxgcrl.agents import CRL, HIQL, PPO, QCARL, SAC, TD3

from .run_config import RunConfig

__all__ = ["AgentConfig", "Config", "RunConfig"]

# agent configurations
AgentConfig = Union[CRL, HIQL, PPO, QCARL, SAC, TD3]


@dataclass
class Config:
    # agent type
    agent: AgentConfig
    # run config
    run: RunConfig
