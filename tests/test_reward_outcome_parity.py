from __future__ import annotations

from types import MethodType, SimpleNamespace

import numpy as np

from smac_unified.core import SMACEnvCore
from smac_unified.handlers import TrackedUnit, UnitFrame, UnitTeamFrame


class _NoopActionHandler:
    def reset(self, *, frame, context):
        del frame, context

    def build_agent_action(self, *, frame, context, agent_id, action):
        del frame, context, agent_id, action
        return None

    def build_opponent_actions(self, *, frame, context, actions, runtime):
        del frame, context, actions, runtime
        return []

    def get_avail_agent_actions(self, *, frame, context, agent_id):
        del frame, agent_id
        return [1] + [0] * (context.n_actions - 1)


class _FixedRewardHandler:
    def __init__(self, value: float):
        self.value = float(value)

    def reset(self, *, frame, context):
        del frame, context

    def build_step_reward(self, *, frame, context) -> float:
        del frame, context
        return self.value


class _FakeTimestep:
    def __init__(self, *, last: bool):
        self._last = bool(last)
        self.observation = SimpleNamespace(raw_data=SimpleNamespace(units=[]))

    def last(self):
        return self._last


class _FakeSession:
    def __init__(self, *, last: bool):
        self._last = bool(last)

    def step(self, *, agent_actions, opponent_actions=None):
        del agent_actions, opponent_actions
        return [_FakeTimestep(last=self._last)]


class _FakeUnitTracker:
    def __init__(self, frame: UnitFrame):
        self._frame = frame

    def update(self, *, allies, enemies):
        del allies, enemies
        return self._frame

    def raw_units_by_id(self, ally: bool):
        del ally
        return {}


def _tracked(*, unit_id: int, tag: int, alive: bool) -> TrackedUnit:
    return TrackedUnit(
        unit_id=unit_id,
        tag=tag,
        unit_type=1,
        x=0.0,
        y=0.0,
        health=45.0 if alive else 0.0,
        health_max=45.0,
        shield=0.0,
        shield_max=0.0,
        weapon_cooldown=0.0,
        alive=alive,
        owner=1,
        raw=None,
    )


def _team(units):
    return UnitTeamFrame(
        units=tuple(units),
        health=np.asarray([u.health for u in units], dtype=np.float32),
        shield=np.asarray([u.shield for u in units], dtype=np.float32),
        alive=np.asarray([u.alive for u in units], dtype=bool),
        tags=np.asarray([u.tag for u in units], dtype=np.int64),
    )


def _build_env(
    *,
    base_reward: float,
    reward_scale: bool,
    reward_win: float,
    reward_defeat: float,
    max_reward: float,
    reward_scale_rate: float,
    continuing_episode: bool,
    episode_limit: int,
    battle_code: int | None,
    timestep_last: bool,
) -> SMACEnvCore:
    env = SMACEnvCore(
        variant='smac',
        map_name='3m',
        env_kwargs={
            'reward_scale': reward_scale,
            'reward_win': reward_win,
            'reward_defeat': reward_defeat,
            'episode_limit': episode_limit,
        },
    )
    frame = UnitFrame(
        allies=_team([_tracked(unit_id=0, tag=1, alive=True)]),
        enemies=_team([_tracked(unit_id=0, tag=101, alive=(battle_code != 1))]),
        prev_allies_health=np.asarray([45.0], dtype=np.float32),
        prev_allies_shield=np.asarray([0.0], dtype=np.float32),
        prev_enemies_health=np.asarray([45.0], dtype=np.float32),
        prev_enemies_shield=np.asarray([0.0], dtype=np.float32),
        step_token=1,
    )
    env._unit_frame = frame
    env._action_handler = _NoopActionHandler()
    env._reward_handler = _FixedRewardHandler(base_reward)
    env._session = _FakeSession(last=timestep_last)
    env._unit_tracker = _FakeUnitTracker(frame=frame)
    env._opponent_runtime = None
    env.continuing_episode = continuing_episode
    env.reward_sparse = False
    env.reward_scale = reward_scale
    env.reward_win = float(reward_win)
    env.reward_defeat = float(reward_defeat)
    env.max_reward = float(max_reward)
    env.reward_scale_rate = float(reward_scale_rate)
    env.episode_limit = int(episode_limit)
    env.battles_won = 0
    env.battles_game = 0
    env.timeouts = 0
    env.win_counted = False
    env.defeat_counted = False
    env._episode_steps = 0

    def _split_raw_units_stub(self):
        return [], []

    def _sync_stub(self):
        return None

    def _battle_stub(self):
        return battle_code

    env._split_raw_units = MethodType(_split_raw_units_stub, env)
    env._sync_legacy_unit_views = MethodType(_sync_stub, env)
    env._battle_outcome_code = MethodType(_battle_stub, env)
    env._refresh_handler_context()
    return env


def test_step_scales_terminal_reward_after_win_bonus():
    env = _build_env(
        base_reward=5.0,
        reward_scale=True,
        reward_win=15.0,
        reward_defeat=0.0,
        max_reward=100.0,
        reward_scale_rate=20.0,
        continuing_episode=False,
        episode_limit=120,
        battle_code=1,
        timestep_last=False,
    )
    reward, terminated, info = env.step([0])
    assert terminated is True
    assert np.isclose(reward, 4.0)
    assert info['battle_won'] is True
    assert info['dead_enemies'] == 1
    assert env.battles_game == 1
    assert env.battles_won == 1


def test_timeout_counts_and_omits_episode_limit_when_not_continuing():
    env = _build_env(
        base_reward=2.0,
        reward_scale=False,
        reward_win=0.0,
        reward_defeat=0.0,
        max_reward=100.0,
        reward_scale_rate=20.0,
        continuing_episode=False,
        episode_limit=1,
        battle_code=None,
        timestep_last=False,
    )
    reward, terminated, info = env.step([0])
    assert terminated is True
    assert np.isclose(reward, 2.0)
    assert 'episode_limit' not in info
    assert env.timeouts == 1
    assert env.battles_game == 1


def test_timeout_sets_episode_limit_flag_when_continuing():
    env = _build_env(
        base_reward=2.0,
        reward_scale=False,
        reward_win=0.0,
        reward_defeat=0.0,
        max_reward=100.0,
        reward_scale_rate=20.0,
        continuing_episode=True,
        episode_limit=1,
        battle_code=None,
        timestep_last=False,
    )
    _, terminated, info = env.step([0])
    assert terminated is True
    assert info.get('episode_limit') is True
    assert env.timeouts == 1

