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
from .env_factory import (
    BackendConfig,
    EnvFactoryConfig,
    UnifiedFactory,
    build_default_backend_registry,
    make_env,
)
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
    "HardScriptCompatibilityWrapper",
    "NormalizedEnvAdapter",
    "ObservationBuilder",
    "OpponentEpisodeContext",
    "OpponentPolicy",
    "OpponentRuntime",
    "OpponentStepContext",
    "RewardBuilder",
    "RewardContext",
    "ScriptedOpponentConfig",
    "ScriptedOpponentRuntime",
    "StateBuilder",
    "StepBatch",
    "UnifiedFactory",
    "build_default_backend_registry",
    "make_env",
]
