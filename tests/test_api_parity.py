from __future__ import annotations

from types import MethodType

import numpy as np

from smac_unified.core import SMACEnv


def test_env_api_exposes_legacy_stats_and_metadata_fields():
    env = SMACEnv(variant='smac', map_name='3m')
    env.battles_won = 3
    env.battles_game = 5
    env.timeouts = 1
    env.force_restarts = 2

    stats = env.get_stats()
    assert stats['battles_won'] == 3
    assert stats['battles_game'] == 5
    assert stats['battles_draw'] == 1
    assert np.isclose(stats['win_rate'], 0.6)
    assert stats['restarts'] == 2

    env_info = env.get_env_info()
    assert 'agent_features' in env_info
    assert 'enemy_features' in env_info
    assert env.get_total_actions() == env.n_actions


def test_step_batch_matches_legacy_payload_contract():
    env = SMACEnv(variant='smac', map_name='3m')

    def _step_stub(self, actions):
        del actions
        return 2.5, True, {'battle_won': True}

    def _obs_stub(self):
        return [np.asarray([1.0], dtype=np.float32)]

    def _state_stub(self):
        return np.asarray([0.5, 0.25], dtype=np.float32)

    def _avail_stub(self):
        return [[1, 0, 1]]

    env.step = MethodType(_step_stub, env)
    env.get_obs = MethodType(_obs_stub, env)
    env.get_state = MethodType(_state_stub, env)
    env.get_avail_actions = MethodType(_avail_stub, env)

    payload = env.step_batch([0])
    assert payload['terminated'] is True
    assert np.isclose(payload['reward'], 2.5)
    assert payload['info']['battle_won'] is True
    assert payload['obs'][0].shape[0] == 1
    assert payload['state'].shape[0] == 2
    assert payload['avail_actions'][0][0] == 1

