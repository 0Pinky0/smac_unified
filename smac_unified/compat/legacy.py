from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping
import warnings

import numpy as np

from ..env_factory import make_env
from ..players import EngineBotOpponentRuntime

_ACTION_MODES = {'classic', 'conic_fov', 'ability_augmented'}
_OPPONENT_MODES = {'sc2_computer', 'scripted_pool'}
_CAPABILITY_MODES = {'none', 'stochastic_attack', 'stochastic_health', 'team_gen'}
_TEAM_INIT_MODES = {'map_default', 'episode_generated'}
_UNIT_TYPE_ALIAS_NAMES = (
    'marine_id',
    'marauder_id',
    'medivac_id',
    'stalker_id',
    'zealot_id',
    'colossus_id',
    'hydralisk_id',
    'zergling_id',
    'baneling_id',
)


@dataclass(frozen=True)
class LegacyEnvBuild:
    family: str
    map_name: str
    env_kwargs: Dict[str, Any]
    capability_config: Dict[str, Any] | None
    logic_switches: Dict[str, str]
    native_options: Dict[str, Any]
    opponent_config: Dict[str, Any]
    source_root: str | None
    transport_profile: str | None
    allow_experimental_transport: bool
    replay_dir: str
    replay_prefix: str


def translate_legacy_kwargs(*, family: str, kwargs: Mapping[str, Any]) -> LegacyEnvBuild:
    payload = dict(kwargs)
    map_name = str(payload.pop('map_name', '8m'))
    capability_config = payload.pop('capability_config', None)
    if capability_config is not None and not isinstance(capability_config, Mapping):
        raise TypeError('capability_config must be a mapping when provided.')
    capability_config_dict = (
        dict(capability_config) if isinstance(capability_config, Mapping) else None
    )
    source_root = payload.pop('source_root', None)
    transport_profile = payload.pop('transport_profile', None)
    allow_experimental_transport = bool(
        payload.pop('allow_experimental_transport', False)
    )
    native_options = dict(payload.pop('native_options', {}) or {})
    opponent_config = dict(payload.pop('opponent_config', {}) or {})

    logic_switches = dict(payload.pop('logic_switches', {}) or {})
    _validate_logic_switches(logic_switches)

    # Explicit override in legacy shims: opponent_mode, if provided.
    opponent_mode = payload.pop('opponent_mode', None)
    if opponent_mode is not None:
        mode = str(opponent_mode).strip()
        if mode not in _OPPONENT_MODES:
            raise ValueError(f'Unsupported opponent_mode: {opponent_mode!r}')
        logic_switches['opponent_mode'] = mode

    # Family-aware capability/action mode inference for legacy kwargs.
    if bool(payload.get('conic_fov', False)):
        logic_switches.setdefault('action_mode', 'conic_fov')
    if family == 'smac-hard':
        logic_switches.setdefault('action_mode', 'ability_augmented')
    elif family in {'smac', 'smacv2'}:
        logic_switches.setdefault('action_mode', 'classic')

    if family == 'smacv2' and capability_config_dict:
        inferred_mode = _infer_capability_mode(capability_config_dict)
        logic_switches.setdefault('capability_mode', inferred_mode)

    if family == 'smac-hard':
        # Hard-style scripted pool is legacy default unless explicitly overridden.
        logic_switches.setdefault('opponent_mode', 'scripted_pool')

    if capability_config_dict and (
        'start_positions' in capability_config_dict
        or 'team_gen' in capability_config_dict
    ):
        logic_switches.setdefault('team_init_mode', 'episode_generated')

    replay_dir = str(payload.get('replay_dir', '') or '')
    replay_prefix = str(payload.get('replay_prefix', '') or '')

    return LegacyEnvBuild(
        family=family,
        map_name=map_name,
        env_kwargs=payload,
        capability_config=capability_config_dict,
        logic_switches=logic_switches,
        native_options=native_options,
        opponent_config=opponent_config,
        source_root=source_root,
        transport_profile=transport_profile,
        allow_experimental_transport=allow_experimental_transport,
        replay_dir=replay_dir,
        replay_prefix=replay_prefix,
    )


def make_legacy_env(*, family: str, **kwargs) -> 'LegacyEnvAdapter':
    spec = translate_legacy_kwargs(family=family, kwargs=kwargs)
    opponent_runtime = None
    if spec.logic_switches.get('opponent_mode') == 'sc2_computer':
        opponent_runtime = EngineBotOpponentRuntime()
    env = make_env(
        family=spec.family,
        map_name=spec.map_name,
        normalized_api=False,
        capability_config=spec.capability_config,
        source_root=spec.source_root,
        transport_profile=spec.transport_profile,
        allow_experimental_transport=spec.allow_experimental_transport,
        logic_switches=spec.logic_switches,
        native_options=spec.native_options,
        opponent_runtime=opponent_runtime,
        opponent_config=spec.opponent_config,
        **spec.env_kwargs,
    )
    return LegacyEnvAdapter(
        env=env,
        family=spec.family,
        replay_dir=spec.replay_dir,
        replay_prefix=spec.replay_prefix,
    )


class LegacyEnvAdapter:
    """Compatibility facade that exposes legacy SMAC-like API over SMACEnv."""

    _LOCAL_ATTRS = {
        '_env',
        'family',
        'replay_dir',
        'replay_prefix',
        '_obs_feature_names',
        '_state_feature_names',
    }

    def __init__(
        self,
        *,
        env: Any,
        family: str,
        replay_dir: str = '',
        replay_prefix: str = '',
    ):
        object.__setattr__(self, '_env', env)
        object.__setattr__(self, 'family', family)
        object.__setattr__(self, 'replay_dir', replay_dir)
        object.__setattr__(self, 'replay_prefix', replay_prefix)
        object.__setattr__(self, '_obs_feature_names', None)
        object.__setattr__(self, '_state_feature_names', None)

    def __getattr__(self, name: str):
        if name in _UNIT_TYPE_ALIAS_NAMES:
            ids = getattr(self._env, '_unit_ids', None)
            return int(getattr(ids, name, 0) or 0)
        if name == 'map_type':
            params = getattr(self._env, 'map_params', None)
            return str(getattr(params, 'map_type', ''))
        return getattr(self._env, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self._LOCAL_ATTRS:
            object.__setattr__(self, name, value)
            return
        if hasattr(self._env, name):
            setattr(self._env, name, value)
            return
        object.__setattr__(self, name, value)

    def reset(self, episode_config: Mapping[str, Any] | None = None, **kwargs):
        return self._env.reset(episode_config=episode_config, **kwargs)

    def reset_batch(
        self,
        episode_config: Mapping[str, Any] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        return self._env.reset_batch(episode_config=episode_config, **kwargs)

    def step(self, actions):
        return self._env.step(actions)

    def step_batch(self, actions):
        return self._env.step_batch(actions)

    def full_restart(self):
        if hasattr(self._env, 'force_restarts'):
            self._env.force_restarts = int(getattr(self._env, 'force_restarts', 0)) + 1
        self.close()
        return None

    def get_unit_by_id(self, a_id: int):
        return self._env.agents[a_id]

    def get_capabilities_agent(self, agent_id: int):
        cap_size = self.get_cap_size()
        if cap_size <= 0:
            return np.zeros(0, dtype=np.float32)
        feats = np.zeros(cap_size, dtype=np.float32)
        idx = 0
        cfg = self._capability_config()
        if 'attack' in cfg:
            values = np.asarray(
                getattr(self._env, 'agent_attack_probabilities', []),
                dtype=np.float32,
            )
            if 0 <= agent_id < values.size:
                feats[idx] = float(values[agent_id])
            idx += 1
        if 'health' in cfg:
            values = np.asarray(
                getattr(self._env, 'agent_health_levels', []),
                dtype=np.float32,
            )
            if 0 <= agent_id < values.size:
                feats[idx] = float(values[agent_id])
            idx += 1
        unit_type_bits = int(getattr(self._env, 'unit_type_bits', 0))
        if unit_type_bits > 0:
            unit = self.get_unit_by_id(agent_id)
            bit_idx = self._unit_type_bit_index(getattr(unit, 'unit_type', 0))
            if 0 <= bit_idx < unit_type_bits and idx + bit_idx < feats.size:
                feats[idx + bit_idx] = 1.0
        return feats

    def get_capabilities(self):
        if int(getattr(self._env, 'n_agents', 0)) <= 0:
            return np.zeros(0, dtype=np.float32)
        caps = [
            self.get_capabilities_agent(agent_id)
            for agent_id in range(int(self._env.n_agents))
        ]
        if not caps:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(caps, axis=0).astype(np.float32)

    def get_cap_size(self) -> int:
        cfg = self._capability_config()
        cap_feats = 0
        if 'attack' in cfg:
            cap_feats += 1
        if 'health' in cfg:
            cap_feats += 1
        unit_type_bits = int(getattr(self._env, 'unit_type_bits', 0))
        if unit_type_bits > 0:
            cap_feats += unit_type_bits
        return int(cap_feats)

    def get_obs_feature_names(self):
        names = self._obs_feature_names
        if names is None:
            obs_size = int(self._env.get_obs_size())
            names = [f'obs_{idx}' for idx in range(obs_size)]
            object.__setattr__(self, '_obs_feature_names', names)
        return list(names)

    def get_state_feature_names(self):
        names = self._state_feature_names
        if names is None:
            state_size = int(self._env.get_state_size())
            names = [f'state_{idx}' for idx in range(state_size)]
            object.__setattr__(self, '_state_feature_names', names)
        return list(names)

    def save_replay(self):
        session = getattr(self._env, '_session', None)
        sc2_env = getattr(session, 'env', None) if session is not None else None
        if sc2_env is not None and hasattr(sc2_env, 'save_replay'):
            try:
                return sc2_env.save_replay(
                    self.replay_dir,
                    self.replay_prefix or str(getattr(self._env, 'map_name', 'episode')),
                )
            except TypeError:
                try:
                    return sc2_env.save_replay(
                        self.replay_prefix or str(getattr(self._env, 'map_name', 'episode'))
                    )
                except Exception:
                    pass
            except Exception:
                pass
        warnings.warn(
            'save_replay is unavailable in legacy adapter without active SC2 replay backend.',
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    def render(self, mode: str = 'human'):
        del mode
        return None

    def close(self):
        return self._env.close()

    def get_env_info(self):
        info = dict(self._env.get_env_info())
        info['n_actions'] = int(self.get_total_actions())
        if self.family == 'smacv2':
            info['cap_shape'] = int(self.get_cap_size())
        return info

    def get_total_actions(self):
        return int(self._env.get_total_actions())

    def _capability_config(self) -> Dict[str, Any]:
        cfg = getattr(self._env, 'capability_config', None)
        if isinstance(cfg, Mapping):
            return dict(cfg)
        return {}

    def _unit_type_bit_index(self, unit_type: int) -> int:
        ids = getattr(self._env, '_unit_ids', None)
        if ids is None:
            return -1
        candidates = [int(getattr(ids, name, 0) or 0) for name in _UNIT_TYPE_ALIAS_NAMES]
        candidates = [value for value in candidates if value > 0]
        if not candidates:
            return -1
        base = min(candidates)
        return int(unit_type) - int(base)


def _infer_capability_mode(capability_config: Mapping[str, Any]) -> str:
    if 'team_gen' in capability_config:
        return 'team_gen'
    if 'attack' in capability_config:
        return 'stochastic_attack'
    if 'health' in capability_config:
        return 'stochastic_health'
    return 'none'


def _validate_logic_switches(logic_switches: Mapping[str, str]) -> None:
    action_mode = logic_switches.get('action_mode')
    if action_mode is not None and action_mode not in _ACTION_MODES:
        raise ValueError(f'Unsupported action_mode: {action_mode!r}')
    opponent_mode = logic_switches.get('opponent_mode')
    if opponent_mode is not None and opponent_mode not in _OPPONENT_MODES:
        raise ValueError(f'Unsupported opponent_mode: {opponent_mode!r}')
    capability_mode = logic_switches.get('capability_mode')
    if capability_mode is not None and capability_mode not in _CAPABILITY_MODES:
        raise ValueError(f'Unsupported capability_mode: {capability_mode!r}')
    team_init_mode = logic_switches.get('team_init_mode')
    if team_init_mode is not None and team_init_mode not in _TEAM_INIT_MODES:
        raise ValueError(f'Unsupported team_init_mode: {team_init_mode!r}')
