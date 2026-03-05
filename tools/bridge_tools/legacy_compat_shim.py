from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from smac_unified import make_env


@dataclass(frozen=True)
class LegacyEnvBuild:
    family: str
    map_name: str
    capability_config: dict[str, Any] | None
    logic_switches: dict[str, str]
    env_kwargs: dict[str, Any]
    factory_kwargs: dict[str, Any]


def translate_legacy_kwargs(*, family: str, kwargs: Mapping[str, Any]) -> LegacyEnvBuild:
    payload = dict(kwargs)
    map_name = str(payload.pop('map_name', '8m'))
    capability_config_raw = payload.pop('capability_config', None)
    capability_config = (
        dict(capability_config_raw)
        if isinstance(capability_config_raw, Mapping)
        else None
    )
    logic_switches: dict[str, str] = {}

    if family == 'smacv2':
        if bool(payload.pop('conic_fov', False)):
            logic_switches['action_mode'] = 'conic_fov'
        if capability_config is not None:
            attack_cfg = capability_config.get('attack', {})
            health_cfg = capability_config.get('health', {})
            if isinstance(attack_cfg, Mapping) and bool(attack_cfg.get('observe')):
                logic_switches['capability_mode'] = 'stochastic_attack'
            elif isinstance(health_cfg, Mapping) and bool(health_cfg.get('observe')):
                logic_switches['capability_mode'] = 'stochastic_health'

    if family == 'smac-hard':
        opponent_mode = payload.pop('opponent_mode', None)
        if isinstance(opponent_mode, str) and opponent_mode:
            logic_switches['opponent_mode'] = opponent_mode
        elif bool(payload.get('enable_dual_controller', True)):
            logic_switches.setdefault('opponent_mode', 'scripted_pool')
        else:
            logic_switches.setdefault('opponent_mode', 'sc2_computer')

    factory_kwargs: dict[str, Any] = {}
    for key in (
        'source_root',
        'transport_profile',
        'allow_experimental_transport',
        'native_options',
        'opponent_runtime',
        'opponent_config',
    ):
        if key in payload:
            factory_kwargs[key] = payload.pop(key)

    return LegacyEnvBuild(
        family=str(family),
        map_name=map_name,
        capability_config=capability_config,
        logic_switches=logic_switches,
        env_kwargs=payload,
        factory_kwargs=factory_kwargs,
    )


def make_legacy_env(*, family: str, **kwargs) -> 'LegacyEnvAdapter':
    spec = translate_legacy_kwargs(family=family, kwargs=kwargs)
    env = make_env(
        family=spec.family,
        map_name=spec.map_name,
        normalized_api=False,
        capability_config=spec.capability_config,
        logic_switches=spec.logic_switches or None,
        **spec.factory_kwargs,
        **spec.env_kwargs,
    )
    return LegacyEnvAdapter(env)


class LegacyEnvAdapter:
    """Bridge-only legacy adapter shim for older repository wrappers."""

    _LOCAL_ATTRS = {'_env'}

    def __init__(self, env: Any):
        object.__setattr__(self, '_env', env)

    def __getattr__(self, name: str):
        return getattr(self._env, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self._LOCAL_ATTRS or '_env' not in self.__dict__:
            object.__setattr__(self, name, value)
            return
        if hasattr(self._env, name):
            setattr(self._env, name, value)
            return
        object.__setattr__(self, name, value)

    @property
    def _seed(self) -> int | None:
        return int(self._env.seed()) if self._env.seed() is not None else None

    @_seed.setter
    def _seed(self, value: int | None) -> None:
        if value is None:
            return
        self._env.seed(int(value))

    def seed(self, seed: int | None = None):
        return self._env.seed(seed)

    def reset(self, episode_config: Mapping[str, Any] | None = None, **kwargs):
        return self._env.reset(episode_config=episode_config, **kwargs)

    def reset_batch(self, episode_config: Mapping[str, Any] | None = None, **kwargs):
        return self._env.reset_batch(episode_config=episode_config, **kwargs)

    def get_cap_size(self) -> int:
        return len(self._capability_keys())

    def get_capabilities_agent(self, agent_id: int) -> np.ndarray:
        keys = self._capability_keys()
        if not keys:
            return np.zeros((0,), dtype=np.float32)
        vec = np.zeros((len(keys),), dtype=np.float32)
        attack_levels = np.asarray(
            getattr(self._env, 'agent_attack_probabilities', []),
            dtype=np.float32,
        )
        health_levels = np.asarray(
            getattr(self._env, 'agent_health_levels', []),
            dtype=np.float32,
        )
        for idx, key in enumerate(keys):
            if key == 'attack' and agent_id < attack_levels.shape[0]:
                vec[idx] = float(attack_levels[agent_id])
            elif key == 'health' and agent_id < health_levels.shape[0]:
                vec[idx] = float(health_levels[agent_id])
        return vec

    def get_capabilities(self) -> np.ndarray:
        cap_size = self.get_cap_size()
        if cap_size == 0:
            return np.zeros((0,), dtype=np.float32)
        rows = [
            self.get_capabilities_agent(agent_id)
            for agent_id in range(int(getattr(self._env, 'n_agents', 0)))
        ]
        if not rows:
            return np.zeros((0,), dtype=np.float32)
        return np.concatenate(rows, axis=0).astype(np.float32, copy=False)

    def _capability_keys(self) -> list[str]:
        cfg = getattr(self._env, 'capability_config', None)
        if not isinstance(cfg, Mapping):
            return []
        keys: list[str] = []
        for key in ('attack', 'health'):
            value = cfg.get(key)
            if isinstance(value, Mapping) and bool(value.get('observe')):
                keys.append(str(key))
        if keys:
            return keys
        cap_shape = int(cfg.get('cap_shape', 0) or 0)
        return [f'cap_{idx}' for idx in range(max(cap_shape, 0))]
