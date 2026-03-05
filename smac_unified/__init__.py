"""Native-first public entry points for SMAC-family environments."""

from .core import (
    EnvFactoryConfig,
    NormalizedEnvAdapter,
    SC2EnvRawSession,
    SC2SessionConfig,
    SMACEnvCore,
    UnifiedFactory,
    VectorEnvPool,
    make_env,
    make_env_pool,
)
from .core.switches import (
    ActionMode,
    CapabilityMode,
    OpponentMode,
    RewardPositiveMode,
    TeamInitMode,
    VariantName,
    VariantSwitches,
)
from .core.variants import default_switches, merge_switches
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
    'SMACEnvCore',
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
