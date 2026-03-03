"""Standalone unified entry points for SMAC-family environments."""

from .adapters import NormalizedEnvAdapter
from .builders import (
    BuildContext,
    DefaultObservationBuilder,
    DefaultRewardBuilder,
    DefaultStateBuilder,
    ObservationBuilder,
    RewardBuilder,
    RewardContext,
    StateBuilder,
)
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
    BackendConfig,
    EnvFactoryConfig,
    UnifiedFactory,
    build_default_backend_registry,
    make_env,
)
from .native import NativeStarCraft2Env, SC2EnvRawSession, SC2SessionConfig
from .opponents import (
    EngineBotOpponentRuntime,
    HardScriptCompatibilityWrapper,
    OpponentEpisodeContext,
    OpponentPolicy,
    OpponentRuntime,
    OpponentStepContext,
    ScriptedOpponentConfig,
    ScriptedOpponentRuntime,
)
from .types import StepBatch

__all__ = [
    "BackendConfig",
    "BuildContext",
    "DefaultObservationBuilder",
    "DefaultRewardBuilder",
    "DefaultStateBuilder",
    "EngineBotOpponentRuntime",
    "EnvFactoryConfig",
    "ActionMode",
    "CapabilityMode",
    "HardScriptCompatibilityWrapper",
    "NativeStarCraft2Env",
    "OpponentMode",
    "NormalizedEnvAdapter",
    "ObservationBuilder",
    "OpponentEpisodeContext",
    "OpponentPolicy",
    "OpponentRuntime",
    "OpponentStepContext",
    "RewardBuilder",
    "RewardPositiveMode",
    "RewardContext",
    "SC2EnvRawSession",
    "SC2SessionConfig",
    "ScriptedOpponentConfig",
    "ScriptedOpponentRuntime",
    "StateBuilder",
    "TeamInitMode",
    "StepBatch",
    "UnifiedFactory",
    "VariantName",
    "VariantSwitches",
    "build_default_backend_registry",
    "default_switches",
    "make_env",
    "merge_switches",
]
