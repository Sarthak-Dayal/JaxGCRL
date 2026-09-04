from typing import Union

from flax.struct import dataclass

from jaxgcrl.agents import CRL, HAC, HIQL, PPO, QCARL, SAC, TD3, CarlCRL, CarlSAC, CarlTD3

from .run_config import RunConfig

__all__ = ["AgentConfig", "Config", "RunConfig"]

# agent configurations
AgentConfig = Union[CRL, HAC, HIQL, PPO, QCARL, SAC, TD3, CarlCRL, CarlSAC, CarlTD3]


@dataclass
class Config:
    # agent type
    agent: AgentConfig
    # run config
    run: RunConfig
