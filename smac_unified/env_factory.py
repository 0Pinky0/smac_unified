from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

from .adapters import NormalizedEnvAdapter
from .backends import (
    BackendConfig,
    BackendRegistry,
    SmacBridgeBackend,
    SmacHardBridgeBackend,
    SmacV2BridgeBackend,
)
from .builders import ObservationBuilder, RewardBuilder, StateBuilder
from .opponents import (
    EngineBotOpponentRuntime,
    OpponentRuntime,
    ScriptedOpponentRuntime,
    build_scripted_runtime_from_config,
)


def build_default_backend_registry() -> BackendRegistry:
    registry = BackendRegistry()
    registry.register(SmacBridgeBackend())
    registry.register(SmacV2BridgeBackend())
    registry.register(SmacHardBridgeBackend())
    return registry


@dataclass
class EnvFactoryConfig:
    family: str
    map_name: str = "8m"
    normalized_api: bool = True
    capability_config: Optional[Dict[str, Any]] = None
    # Kept for backward compatibility with earlier smac_unified config style.
    kwargs: Dict[str, Any] = field(default_factory=dict)
    # Preferred explicit env kwargs.
    env_kwargs: Dict[str, Any] = field(default_factory=dict)
    source_root: str | None = None
    backend_registry: BackendRegistry | None = None
    observation_builder: ObservationBuilder | None = None
    state_builder: StateBuilder | None = None
    reward_builder: RewardBuilder | None = None
    opponent_runtime: OpponentRuntime | None = None
    opponent_config: Dict[str, Any] = field(default_factory=dict)

    def resolved_env_kwargs(self) -> Dict[str, Any]:
        merged = dict(self.kwargs)
        merged.update(self.env_kwargs)
        return merged


def _default_opponent_runtime(
    family: str,
    opponent_config: Mapping[str, Any] | None,
) -> OpponentRuntime:
    if family == "smac-hard":
        return build_scripted_runtime_from_config(opponent_config or {})
    return EngineBotOpponentRuntime()


def make_env(
    *,
    family: str,
    map_name: str = "8m",
    normalized_api: bool = True,
    capability_config: Optional[Dict[str, Any]] = None,
    source_root: str | None = None,
    backend_registry: BackendRegistry | None = None,
    observation_builder: ObservationBuilder | None = None,
    state_builder: StateBuilder | None = None,
    reward_builder: RewardBuilder | None = None,
    opponent_runtime: OpponentRuntime | None = None,
    opponent_config: Mapping[str, Any] | None = None,
    **kwargs,
):
    registry = backend_registry or build_default_backend_registry()
    backend = registry.get(family)
    backend_config = BackendConfig(
        family=family,
        map_name=map_name,
        capability_config=capability_config,
        env_kwargs=dict(kwargs),
        source_root=source_root,
    )
    env = backend.make_env(backend_config)
    if not normalized_api:
        return env

    runtime = opponent_runtime or _default_opponent_runtime(
        family,
        opponent_config,
    )
    return NormalizedEnvAdapter(
        env,
        family=family,
        observation_builder=observation_builder,
        state_builder=state_builder,
        reward_builder=reward_builder,
        opponent_runtime=runtime,
        opponent_config=opponent_config,
    )


class UnifiedFactory:
    """Unified entry point for smac/smacv2/smac-hard environments."""

    @staticmethod
    def make_env(config: EnvFactoryConfig):
        return make_env(
            family=config.family,
            map_name=config.map_name,
            normalized_api=config.normalized_api,
            capability_config=config.capability_config,
            source_root=config.source_root,
            backend_registry=config.backend_registry,
            observation_builder=config.observation_builder,
            state_builder=config.state_builder,
            reward_builder=config.reward_builder,
            opponent_runtime=config.opponent_runtime,
            opponent_config=config.opponent_config,
            **config.resolved_env_kwargs(),
        )
