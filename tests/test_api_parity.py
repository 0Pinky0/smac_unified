from __future__ import annotations

from types import MethodType

import numpy as np

from smac_unified import make_env, merge_switches
from smac_unified.core import SMACEnvCore
from smac_unified.core.sc2session import SC2SessionConfig, _build_map_spec
from smac_unified.maps import get_map_params
from smac_unified.players import ScriptedOpponentRuntime


def test_env_api_exposes_legacy_stats_and_metadata_fields():
    env = SMACEnvCore(variant='smac', map_name='3m')
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
    env = SMACEnvCore(variant='smacv2', map_name='8m')
    env_info = env.get_env_info()
    assert env_info['cap_shape'] == 0


def test_env_info_reports_stable_space_shapes_before_reset():
    env = SMACEnvCore(variant='smac', map_name='3m')
    env_info = env.get_env_info()
    assert int(env_info['obs_shape']) > 0
    assert int(env_info['state_shape']) > 0
    assert env.get_obs_size() == int(env_info['obs_shape'])
    assert env.get_state().shape[0] == int(env_info['state_shape'])


def test_capability_handler_size_contracts_increase_shapes():
    base = SMACEnvCore(variant='smacv2', map_name='8m')
    cap = SMACEnvCore(
        variant='smacv2',
        map_name='8m',
        env_kwargs={
            'logic_switches': merge_switches(
                'smacv2',
                {'capability_mode': 'team_gen'},
            )
        },
    )
    assert cap.get_obs_size() > base.get_obs_size()
    assert cap.get_state_size() > base.get_state_size()


def test_step_batch_matches_legacy_payload_contract():
    env = SMACEnvCore(variant='smac', map_name='3m')

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
    env = SMACEnvCore(variant='smac', map_name='3m')

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
    env = SMACEnvCore(variant='smac', map_name='3m')
    assert env.seed() is None
    assert env.seed(123) == 123
    first = env._rng.uniform(size=4)
    assert env.seed() == 123
    env.seed(123)
    second = env._rng.uniform(size=4)
    assert np.allclose(first, second)
    assert env._session.config.seed == 123


def test_last_action_update_is_in_place_one_hot():
    env = SMACEnvCore(variant='smac', map_name='3m')
    before = env.last_action
    action_ids = [1] * env.n_agents
    env._update_last_action_matrix(action_ids)
    assert env.last_action is before
    assert np.allclose(env.last_action.sum(axis=1), np.ones(env.n_agents))
    assert np.allclose(env.last_action[:, 1], np.ones(env.n_agents))


def test_last_action_contract_tracks_requested_ids_per_agent():
    env = SMACEnvCore(variant='smac', map_name='3m')
    requested = [idx % env.n_actions for idx in range(env.n_agents)]
    env._update_last_action_matrix(requested)
    for agent_id, action_id in enumerate(requested):
        row = env.last_action[agent_id]
        assert np.isclose(row[action_id], 1.0)
        assert np.isclose(np.sum(row), 1.0)


def test_handler_context_refresh_reuses_instance():
    env = SMACEnvCore(variant='smac', map_name='3m')
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


def test_step_pipeline_invokes_encode_submit_collect_decode_in_order():
    env = SMACEnvCore(variant='smac', map_name='3m')
    call_order = []

    def _normalize_stub(self, actions):
        del actions
        call_order.append('normalize')
        return [0] * self.n_agents

    def _encode_stub(self, actions_int):
        del actions_int
        call_order.append('encode')
        return [], []

    def _submit_stub(self, *, ally_sc_actions, opponent_actions):
        del ally_sc_actions, opponent_actions
        call_order.append('submit')

    def _collect_stub(self):
        call_order.append('collect')

    def _decode_stub(self):
        call_order.append('decode')
        return 1.5, False, {'ok': True}

    env._normalize_actions = MethodType(_normalize_stub, env)
    env._encode_step_actions = MethodType(_encode_stub, env)
    env._submit_step_actions = MethodType(_submit_stub, env)
    env._collect_step_timesteps = MethodType(_collect_stub, env)
    env._decode_step_outcome = MethodType(_decode_stub, env)

    reward, terminated, info = env.step([1, 2, 3])
    assert np.isclose(reward, 1.5)
    assert terminated is False
    assert info == {'ok': True}
    assert call_order == ['normalize', 'encode', 'submit', 'collect', 'decode']


def test_transport_profile_b2_propagates_to_native_session_config():
    env = make_env(
        family='smac',
        map_name='3m',
        normalized_api=False,
        transport_profile='B2',
    )
    cfg = env._session.config
    assert cfg.transport_profile == 'B2'
    assert cfg.reuse_step_observe_requests is True
    assert cfg.pipeline_step_and_observe is True
    assert cfg.pipeline_actions_and_step is False
    assert cfg.ensure_available_actions is True
    env.close()


def test_native_options_override_transport_profile_defaults():
    env = make_env(
        family='smac',
        map_name='3m',
        normalized_api=False,
        transport_profile='B2',
        native_options={
            'pipeline_step_and_observe': False,
            'pipeline_actions_and_step': True,
        },
    )
    cfg = env._session.config
    assert cfg.transport_profile == 'B2'
    assert cfg.reuse_step_observe_requests is True
    assert cfg.pipeline_step_and_observe is False
    assert cfg.pipeline_actions_and_step is True
    env.close()


def test_experimental_transport_flag_propagates_to_session_config():
    env = make_env(
        family='smac',
        map_name='3m',
        normalized_api=False,
        transport_profile='B4',
        allow_experimental_transport=True,
    )
    cfg = env._session.config
    assert cfg.transport_profile == 'B4'
    assert cfg.ensure_available_actions is False
    assert cfg.allow_experimental_transport is True
    env.close()


def test_async_step_flag_propagates_to_session_config():
    env = make_env(
        family='smac',
        map_name='3m',
        normalized_api=False,
        native_options={'enable_async_step': True},
    )
    cfg = env._session.config
    assert cfg.enable_async_step is True
    env.close()


def test_scripted_pool_raw_mode_enables_dual_controller_by_default():
    env = make_env(
        family='smac-hard',
        map_name='3m',
        normalized_api=False,
    )
    cfg = env._session.config
    assert cfg.opponent_mode == 'scripted_pool'
    assert cfg.enable_dual_controller is True
    env.close()


def test_scripted_pool_raw_mode_binds_runtime_for_env_lifecycle():
    env = make_env(
        family='smac-hard',
        map_name='3m',
        normalized_api=False,
    )
    assert isinstance(env._opponent_runtime, ScriptedOpponentRuntime)
    assert env._runtime_lifecycle_owner == 'env'
    env.close()


def test_scripted_pool_map_spec_prefers_new_maps_for_overlap_maps():
    params = get_map_params('3m')
    scripted_cfg = SC2SessionConfig(
        map_name='3m',
        map_params=params,
        opponent_mode='scripted_pool',
        enable_dual_controller=True,
    )
    scripted_map = _build_map_spec(scripted_cfg)
    assert scripted_map.directory == 'new_maps'
    assert scripted_map.filename == '3m'

    computer_cfg = SC2SessionConfig(
        map_name='3m',
        map_params=params,
        opponent_mode='sc2_computer',
        enable_dual_controller=False,
    )
    computer_map = _build_map_spec(computer_cfg)
    assert computer_map.directory == 'SMAC_Maps'
    assert computer_map.filename == '3m'


def test_forced_opponent_action_schedule_overrides_runtime_branch():
    env = SMACEnvCore(variant='smac-hard', map_name='3m')

    class _ActionStub:
        def build_agent_action(
            self,
            *,
            frame,
            context,
            agent_id,
            action,
        ):
            del frame, context, agent_id, action
            return None

        def build_opponent_actions(
            self,
            *,
            frame,
            context,
            actions,
            runtime,
        ):
            del frame, context, actions, runtime
            return ['runtime-branch']

    env._action_handler = _ActionStub()
    env._unit_frame = object()
    env._refresh_handler_context()
    env.set_forced_opponent_actions_schedule(
        [
            [
                {
                    'ability_id': 23,
                    'target_unit_tag': 4242,
                    'unit_tags': [1010],
                }
            ]
        ]
    )

    _, opponent_actions = env._encode_step_actions([0] * env.n_agents)
    assert len(opponent_actions) == 1
    command = opponent_actions[0].action_raw.unit_command
    assert int(command.ability_id) == 23
    assert int(command.target_unit_tag) == 4242
    assert list(command.unit_tags) == [1010]

    env._episode_steps = 9
    _, opponent_actions = env._encode_step_actions([0] * env.n_agents)
    assert opponent_actions == []

