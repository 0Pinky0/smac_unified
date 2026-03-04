from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Mapping, Optional

from .adapters import NormalizedEnvAdapter
from .config import VariantSwitches, default_switches
from .core import SMACEnv
from .players import (
    EngineBotOpponentRuntime,
    OpponentRuntime,
    build_scripted_runtime_from_config,
)

_TRANSPORT_PROFILE_OPTIONS: dict[str, dict[str, bool]] = {
    'B0': {},
    'B1': {'reuse_step_observe_requests': True},
    'B2': {
        'reuse_step_observe_requests': True,
        'pipeline_step_and_observe': True,
    },
    'B3': {
        'reuse_step_observe_requests': True,
        'pipeline_step_and_observe': True,
        'pipeline_actions_and_step': True,
    },
    'B4': {
        'reuse_step_observe_requests': True,
        'pipeline_step_and_observe': True,
        'pipeline_actions_and_step': True,
        'ensure_available_actions': False,
    },
}


@dataclass
class EnvFactoryConfig:
    family: str
    map_name: str = '8m'
    normalized_api: bool = True
    capability_config: Optional[Dict[str, Any]] = None
    # Kept for backward compatibility with earlier smac_unified config style.
    kwargs: Dict[str, Any] = field(default_factory=dict)
    # Preferred explicit env kwargs.
    env_kwargs: Dict[str, Any] = field(default_factory=dict)
    source_root: str | None = None
    transport_profile: Literal['B0', 'B1', 'B2', 'B3', 'B4'] | None = None
    allow_experimental_transport: bool = False
    logic_switches: Dict[str, str] = field(default_factory=dict)
    native_options: Dict[str, Any] = field(default_factory=dict)
    observation_handler: Any | None = None
    state_handler: Any | None = None
    reward_handler: Any | None = None
    action_handler: Any | None = None
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
    del family
    if switches.opponent_mode == 'scripted_pool':
        return build_scripted_runtime_from_config(opponent_config or {})
    return EngineBotOpponentRuntime()


def _normalize_transport_profile(value: str | None) -> str:
    if value is None:
        return 'B0'
    profile = str(value).strip().upper()
    if profile not in _TRANSPORT_PROFILE_OPTIONS:
        raise ValueError(f'Unsupported transport_profile: {value}')
    return profile


def _merge_native_transport_options(
    *,
    transport_profile: str | None,
    native_options: Mapping[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    profile = _normalize_transport_profile(transport_profile)
    merged: dict[str, Any] = dict(_TRANSPORT_PROFILE_OPTIONS[profile])
    merged.update(dict(native_options or {}))
    return profile, merged


def make_env(
    *,
    family: str,
    map_name: str = '8m',
    normalized_api: bool = True,
    capability_config: Optional[Dict[str, Any]] = None,
    source_root: str | None = None,
    transport_profile: Literal['B0', 'B1', 'B2', 'B3', 'B4'] | None = None,
    allow_experimental_transport: bool = False,
    logic_switches: Mapping[str, str] | None = None,
    native_options: Mapping[str, Any] | None = None,
    observation_handler: Any | None = None,
    state_handler: Any | None = None,
    reward_handler: Any | None = None,
    action_handler: Any | None = None,
    opponent_runtime: OpponentRuntime | None = None,
    opponent_config: Mapping[str, Any] | None = None,
    **kwargs,
):
    for removed_key in ('backend_mode', 'backend_registry', 'bridge_options'):
        if removed_key in kwargs:
            raise TypeError(
                f"make_env() got an unexpected keyword argument '{removed_key}'"
            )
    switches = default_switches(family)
    if logic_switches:
        switches = VariantSwitches(
            variant=switches.variant,
            action_mode=logic_switches.get('action_mode', switches.action_mode),
            opponent_mode=logic_switches.get(
                'opponent_mode', switches.opponent_mode
            ),
            capability_mode=logic_switches.get(
                'capability_mode', switches.capability_mode
            ),
            reward_positive_mode=logic_switches.get(
                'reward_positive_mode', switches.reward_positive_mode
            ),
            team_init_mode=logic_switches.get(
                'team_init_mode', switches.team_init_mode
            ),
        )
    handler_overrides: Dict[str, Any] = {}
    if action_handler is not None:
        handler_overrides['action_handler'] = action_handler
    if _has_callable_attr(observation_handler, 'build_agent_obs'):
        handler_overrides['observation_handler'] = observation_handler
    if _has_callable_attr(state_handler, 'build_state'):
        handler_overrides['state_handler'] = state_handler
    if _has_callable_attr(reward_handler, 'build_step_reward'):
        handler_overrides['reward_handler'] = reward_handler

    resolved_transport_profile, resolved_native_options = _merge_native_transport_options(
        transport_profile=transport_profile,
        native_options=native_options,
    )

    env_kwargs = dict(kwargs)
    if switches is not None:
        env_kwargs.setdefault('logic_switches', switches)
    handler_map = {
        'action_handler': 'action_handler',
        'observation_handler': 'observation_handler',
        'state_handler': 'state_handler',
        'reward_handler': 'reward_handler',
    }
    for src_key, dst_key in handler_map.items():
        value = handler_overrides.get(src_key)
        if value is not None:
            env_kwargs[dst_key] = value

    env = SMACEnv(
        variant=family,
        map_name=map_name,
        capability_config=capability_config,
        env_kwargs=env_kwargs,
        source_root=source_root,
        native_options=resolved_native_options,
        transport_profile=resolved_transport_profile,
        allow_experimental_transport=bool(allow_experimental_transport),
    )
    resolved_opponent_config: dict[str, Any] = dict(opponent_config or {})
    if 'seed' not in resolved_opponent_config and 'seed' in env_kwargs:
        resolved_opponent_config['seed'] = env_kwargs['seed']
    runtime = opponent_runtime or _default_opponent_runtime(
        family,
        switches,
        resolved_opponent_config,
    )
    if not normalized_api:
        if hasattr(env, 'set_opponent_runtime'):
            env.set_opponent_runtime(runtime)
        if hasattr(env, 'set_runtime_lifecycle_owner'):
            env.set_runtime_lifecycle_owner('env')
        return env

    return NormalizedEnvAdapter(
        env,
        family=family,
        opponent_runtime=runtime,
        opponent_config=resolved_opponent_config,
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
            transport_profile=config.transport_profile,
            allow_experimental_transport=config.allow_experimental_transport,
            logic_switches=config.logic_switches,
            native_options=config.native_options,
            observation_handler=config.observation_handler,
            state_handler=config.state_handler,
            reward_handler=config.reward_handler,
            action_handler=config.action_handler,
            opponent_runtime=config.opponent_runtime,
            opponent_config=config.opponent_config,
            **config.resolved_env_kwargs(),
        )


def _has_callable_attr(value: Any, name: str) -> bool:
    return value is not None and callable(getattr(value, name, None))
