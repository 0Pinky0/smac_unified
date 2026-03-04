from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Sequence

from s2clientprotocol import sc2api_pb2 as sc_pb

from ..maps import MapParams, resolve_map_directory, resolve_map_filename


RACES = {
    'R': 'random',
    'P': 'protoss',
    'T': 'terran',
    'Z': 'zerg',
}

DIFFICULTIES = {
    '1': 'very_easy',
    '2': 'easy',
    '3': 'medium',
    '4': 'medium_hard',
    '5': 'hard',
    '6': 'harder',
    '7': 'very_hard',
    '8': 'cheat_vision',
    '9': 'cheat_money',
    'A': 'cheat_insane',
}


def _prepend_path(path: Path) -> None:
    value = str(path)
    if value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)


def _autodetect_source_root() -> Path | None:
    env_value = os.environ.get('SMAC_UNIFIED_SOURCE_ROOT', '').strip()
    if env_value:
        candidate = Path(env_value).expanduser().resolve()
        if candidate.exists():
            return candidate
    here = Path(__file__).resolve()
    for candidate in (here.parents[3], here.parents[2], Path.cwd()):
        if (candidate / 'dependency' / 'pysc2-compat').exists():
            return candidate
    return None


def _ensure_pysc2_compat(source_root: str | None = None) -> None:
    root = (
        Path(source_root).expanduser().resolve()
        if source_root
        else _autodetect_source_root()
    )
    compat = root / 'dependency' / 'pysc2-compat' if root else None
    if compat is not None and compat.exists():
        _prepend_path(compat)

    try:
        import pysc2  # noqa: F401
    except Exception:
        if compat is None or not compat.exists():
            raise
        import pysc2  # noqa: F401  # pylint: disable=unused-import
        return

    if compat is not None and compat.exists():
        import importlib

        import pysc2 as pysc2_pkg

        if 'pysc2-compat' not in str(Path(pysc2_pkg.__file__).resolve()):
            for key in list(sys.modules.keys()):
                if key == 'pysc2' or key.startswith('pysc2.'):
                    del sys.modules[key]
            importlib.invalidate_caches()
            import pysc2  # noqa: F401  # pylint: disable=unused-import


def _safe_map_class_name(value: str) -> str:
    cleaned = ''.join(ch if ch.isalnum() else '_' for ch in str(value))
    return cleaned or 'Map'


def _build_map_spec(config: 'SC2SessionConfig'):
    from pysc2.maps import lib

    directory = resolve_map_directory(
        params=config.map_params,
        opponent_mode=config.opponent_mode,
    )
    filename = resolve_map_filename(
        map_name=config.map_name,
        params=config.map_params,
    )
    class_name = '_'.join(
        [
            'SMACUnifiedMap',
            _safe_map_class_name(config.map_name),
            _safe_map_class_name(directory),
            _safe_map_class_name(filename),
        ]
    )
    attrs = {
        'directory': directory,
        'filename': filename,
        'players': 2,
        'step_mul': int(config.step_mul),
        'game_steps_per_episode': 0,
        'download': 'https://github.com/oxwhirl/smac#smac-maps',
    }
    return type(class_name, (lib.Map,), attrs)()


@dataclass
class SC2SessionConfig:
    map_name: str
    map_params: MapParams
    step_mul: int = 8
    difficulty: str = '7'
    seed: int | None = None
    realtime: bool = False
    ensure_available_actions: bool = True
    pipeline_actions_and_step: bool = False
    pipeline_step_and_observe: bool = False
    reuse_step_observe_requests: bool = False
    transport_profile: str = 'B0'
    allow_experimental_transport: bool = False
    opponent_mode: str = 'sc2_computer'
    enable_dual_controller: bool = False
    game_version: str | None = None
    source_root: str | None = None


class SC2EnvRawSession:
    """Safe SC2Env wrapper using raw InterfaceOptions passthrough."""

    def __init__(self, config: SC2SessionConfig):
        self.config = config
        self._env = None
        self._num_agents = 1

    @property
    def env(self):
        return self._env

    @property
    def num_agents(self) -> int:
        return self._num_agents

    def launch(self) -> None:
        if self._env is not None:
            return
        profile = str(self.config.transport_profile or 'B0').upper()
        scripted_mode = self.config.opponent_mode == 'scripted_pool'
        if (
            profile == 'B4' or not bool(self.config.ensure_available_actions)
        ) and not bool(self.config.allow_experimental_transport):
            raise ValueError(
                'Experimental transport requires allow_experimental_transport=True '
                '(B4 or ensure_available_actions=False).'
            )
        if scripted_mode and not bool(self.config.enable_dual_controller):
            raise ValueError(
                'Scripted opponent mode requires enable_dual_controller=True.'
            )
        _ensure_pysc2_compat(self.config.source_root)

        from pysc2.env import sc2_env

        map_spec = _build_map_spec(self.config)
        race_agent = getattr(sc2_env.Race, RACES[self.config.map_params.a_race])
        race_enemy = getattr(sc2_env.Race, RACES[self.config.map_params.b_race])
        difficulty = getattr(
            sc2_env.Difficulty, DIFFICULTIES.get(self.config.difficulty, 'very_hard')
        )

        interface = sc_pb.InterfaceOptions(
            raw=True,
            score=True,
            raw_affects_selection=True,
            raw_crop_to_playable_area=True,
        )

        players: List[Any] = [sc2_env.Agent(race_agent)]
        if scripted_mode and self.config.enable_dual_controller:
            players.append(sc2_env.Agent(race_enemy))
            self._num_agents = 2
        else:
            players.append(sc2_env.Bot(race_enemy, difficulty))
            self._num_agents = 1

        self._env = sc2_env.SC2Env(
            map_name=map_spec,
            players=players,
            agent_interface_format=interface,
            step_mul=self.config.step_mul,
            realtime=self.config.realtime,
            random_seed=self.config.seed,
            ensure_available_actions=self.config.ensure_available_actions,
            pipeline_actions_and_step=self.config.pipeline_actions_and_step,
            pipeline_step_and_observe=self.config.pipeline_step_and_observe,
            reuse_step_observe_requests=self.config.reuse_step_observe_requests,
            version=self.config.game_version,
        )
        if scripted_mode:
            actual_agents = int(getattr(self._env, '_num_agents', self._num_agents))
            if actual_agents != 2:
                self.close()
                raise RuntimeError(
                    'Scripted opponent mode launch failed to enter '
                    f'dual-controller setup: num_agents={actual_agents}.'
                )

    def reset(self):
        if self._env is None:
            self.launch()
        timesteps = self._env.reset()
        return list(timesteps)

    def step(
        self,
        *,
        agent_actions: Sequence[Any],
        opponent_actions: Sequence[Any] | None = None,
    ):
        if self._env is None:
            raise RuntimeError('SC2 session has not been launched.')

        if self._num_agents == 1:
            payload = [list(agent_actions)]
        else:
            payload = [
                list(agent_actions),
                list(opponent_actions or []),
            ]
        timesteps = self._env.step(payload)
        return list(timesteps)

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None
