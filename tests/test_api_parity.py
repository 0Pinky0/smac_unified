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


def test_smacv2_env_info_includes_cap_shape_field():
    env = SMACEnv(variant='smacv2', map_name='8m')
    env_info = env.get_env_info()
    assert env_info['cap_shape'] == 0


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


def test_reset_batch_matches_legacy_payload_contract():
    env = SMACEnv(variant='smac', map_name='3m')

    def _reset_stub(self, episode_config=None, **kwargs):
        del episode_config, kwargs
        return [np.asarray([1.0], dtype=np.float32)], np.asarray([0.5], dtype=np.float32)

    def _avail_stub(self):
        return [[1, 0, 1]]

    env.reset = MethodType(_reset_stub, env)
    env.get_avail_actions = MethodType(_avail_stub, env)

    payload = env.reset_batch()
    assert payload['terminated'] is False
    assert np.isclose(payload['reward'], 0.0)
    assert payload['info'] == {}
    assert payload['obs'][0].shape[0] == 1
    assert payload['state'].shape[0] == 1
    assert payload['avail_actions'][0][0] == 1


def test_seed_supports_getter_setter_and_rng_reseed():
    env = SMACEnv(variant='smac', map_name='3m')
    assert env.seed() is None
    assert env.seed(123) == 123
    first = env._rng.uniform(size=4)
    assert env.seed() == 123
    env.seed(123)
    second = env._rng.uniform(size=4)
    assert np.allclose(first, second)
    assert env._session.config.seed == 123


def test_last_action_update_is_in_place_one_hot():
    env = SMACEnv(variant='smac', map_name='3m')
    before = env.last_action
    action_ids = [1] * env.n_agents
    env._update_last_action_matrix(action_ids)
    assert env.last_action is before
    assert np.allclose(env.last_action.sum(axis=1), np.ones(env.n_agents))
    assert np.allclose(env.last_action[:, 1], np.ones(env.n_agents))


def test_handler_context_refresh_reuses_instance():
    env = SMACEnv(variant='smac', map_name='3m')
    env._episode_steps = 2
    env._refresh_handler_context()
    ctx = env._handler_context
    assert ctx is not None
    assert ctx.episode_step == 2

    env._episode_steps = 5
    env.max_distance_x = 77.0
    env._update_last_action_matrix([0] * env.n_agents)
    env._refresh_handler_context()
    assert env._handler_context is ctx
    assert ctx.episode_step == 5
    assert np.isclose(ctx.max_distance_x, 77.0)
    assert ctx.last_action is env.last_action

