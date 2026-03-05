"""Native-first public entry points for SMAC-family environments."""

from .adapters import NormalizedEnvAdapter, VectorEnvPool
from .config import (
    ActionMode,
    CapabilityMode,
    OpponentMode,
    RewardPositiveMode,
    TeamInitMode,
    VariantName,
    VariantSwitches,
    default_switches,
    merge_switches,
)
from .env_factory import (
    EnvFactoryConfig,
    UnifiedFactory,
    make_env,
    make_env_pool,
)
from .core import (
    SMACEnv,
    SC2EnvRawSession,
    SC2SessionConfig,
)
from .types import StepBatch

__all__ = [
    'ActionMode',
    'CapabilityMode',
    'EnvFactoryConfig',
    'NormalizedEnvAdapter',
    'OpponentMode',
    'RewardPositiveMode',
    'SC2EnvRawSession',
    'SC2SessionConfig',
    'SMACEnv',
    'StepBatch',
    'TeamInitMode',
    'UnifiedFactory',
    'VariantName',
    'VariantSwitches',
    'default_switches',
    'make_env',
    'make_env_pool',
    'merge_switches',
    'VectorEnvPool',
]
