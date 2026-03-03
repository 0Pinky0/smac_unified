from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

from .adapters import NormalizedEnvAdapter
from .backends import (
    BackendConfig,
    BackendRegistry,
    NativeUnifiedBackend,
    SmacBridgeBackend,
    SmacHardBridgeBackend,
    SmacV2BridgeBackend,
)
from .config import VariantSwitches, default_switches
from .players import (
    EngineBotOpponentRuntime,
    OpponentRuntime,
    build_scripted_runtime_from_config,
)


def build_default_backend_registry() -> BackendRegistry:
    registry = BackendRegistry()
    registry.register(NativeUnifiedBackend("smac"))
    registry.register(NativeUnifiedBackend("smacv2"))
    registry.register(NativeUnifiedBackend("smac-hard"))
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
    backend_mode: str = "native"
    backend_registry: BackendRegistry | None = None
    logic_switches: Dict[str, str] = field(default_factory=dict)
    native_options: Dict[str, Any] = field(default_factory=dict)
    bridge_options: Dict[str, Any] = field(default_factory=dict)
    observation_builder: Any | None = None
    state_builder: Any | None = None
    reward_builder: Any | None = None
    action_builder: Any | None = None
    opponent_runtime: OpponentRuntime | None = None
    opponent_config: Dict[str, Any] = field(default_factory=dict)

    def resolved_env_kwargs(self) -> Dict[str, Any]:
        merged = dict(self.kwargs)
        merged.update(self.env_kwargs)
        return merged


def _default_opponent_runtime(
    family: str,
    switches: VariantSwitches,
    opponent_config: Mapping[str, Any] | None,
) -> OpponentRuntime:
    if switches.opponent_mode == "scripted_pool" or family == "smac-hard":
        return build_scripted_runtime_from_config(opponent_config or {})
    return EngineBotOpponentRuntime()


def make_env(
    *,
    family: str,
    map_name: str = "8m",
    normalized_api: bool = True,
    capability_config: Optional[Dict[str, Any]] = None,
    source_root: str | None = None,
    backend_mode: str = "native",
    backend_registry: BackendRegistry | None = None,
    logic_switches: Mapping[str, str] | None = None,
    native_options: Mapping[str, Any] | None = None,
    bridge_options: Mapping[str, Any] | None = None,
    observation_builder: Any | None = None,
    state_builder: Any | None = None,
    reward_builder: Any | None = None,
    action_builder: Any | None = None,
    opponent_runtime: OpponentRuntime | None = None,
    opponent_config: Mapping[str, Any] | None = None,
    **kwargs,
):
    registry = backend_registry or build_default_backend_registry()
    switches = default_switches(family)
    if logic_switches:
        switches = VariantSwitches(
            variant=switches.variant,
            action_mode=logic_switches.get("action_mode", switches.action_mode),
            opponent_mode=logic_switches.get(
                "opponent_mode", switches.opponent_mode
            ),
            capability_mode=logic_switches.get(
                "capability_mode", switches.capability_mode
            ),
            reward_positive_mode=logic_switches.get(
                "reward_positive_mode", switches.reward_positive_mode
            ),
            team_init_mode=logic_switches.get(
                "team_init_mode", switches.team_init_mode
            ),
        )
    builder_overrides: Dict[str, Any] = {}
    if action_builder is not None:
        builder_overrides["action_builder"] = action_builder
    if _has_callable_attr(observation_builder, "build_agent_obs"):
        builder_overrides["observation_builder"] = observation_builder
    if _has_callable_attr(state_builder, "build_state"):
        builder_overrides["state_builder"] = state_builder
    if _has_callable_attr(reward_builder, "build_step_reward"):
        builder_overrides["reward_builder"] = reward_builder

    backend_config = BackendConfig(
        family=family,
        map_name=map_name,
        capability_config=capability_config,
        env_kwargs=dict(kwargs),
        source_root=source_root,
        backend_mode=backend_mode,
        native_options=dict(native_options or {}),
        bridge_options=dict(bridge_options or {}),
        logic_switches=switches,
        builder_overrides=builder_overrides,
    )
    backend = registry.get(
        family,
        mode=backend_mode,
        config=backend_config,
    )
    env = backend.make_env(backend_config)
    if not normalized_api:
        return env

    runtime = opponent_runtime or _default_opponent_runtime(
        family,
        switches,
        opponent_config,
    )
    adapter_observation_builder = (
        observation_builder
        if _has_callable_attr(observation_builder, "build")
        else None
    )
    adapter_state_builder = (
        state_builder if _has_callable_attr(state_builder, "build") else None
    )
    adapter_reward_builder = (
        reward_builder if _has_callable_attr(reward_builder, "build") else None
    )
    return NormalizedEnvAdapter(
        env,
        family=family,
        observation_builder=adapter_observation_builder,
        state_builder=adapter_state_builder,
        reward_builder=adapter_reward_builder,
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
            backend_mode=config.backend_mode,
            backend_registry=config.backend_registry,
            logic_switches=config.logic_switches,
            native_options=config.native_options,
            bridge_options=config.bridge_options,
            observation_builder=config.observation_builder,
            state_builder=config.state_builder,
            reward_builder=config.reward_builder,
            action_builder=config.action_builder,
            opponent_runtime=config.opponent_runtime,
            opponent_config=config.opponent_config,
            **config.resolved_env_kwargs(),
        )


def _has_callable_attr(value: Any, name: str) -> bool:
    return value is not None and callable(getattr(value, name, None))
