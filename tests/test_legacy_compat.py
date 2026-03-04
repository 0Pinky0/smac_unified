from types import SimpleNamespace

import numpy as np

from smac_unified import make_env
from smac_unified.compat import make_legacy_env, translate_legacy_kwargs
from smac_unified.players import EngineBotOpponentRuntime, ScriptedOpponentRuntime


def test_translate_legacy_kwargs_infers_smacv2_modes():
    spec = translate_legacy_kwargs(
        family='smacv2',
        kwargs={
            'map_name': '8m',
            'conic_fov': True,
            'capability_config': {
                'attack': {'observe': True},
                'n_units': 8,
                'n_enemies': 8,
            },
        },
    )
    assert spec.logic_switches['action_mode'] == 'conic_fov'
    assert spec.logic_switches['capability_mode'] == 'stochastic_attack'


def test_make_legacy_env_respects_explicit_hard_opponent_mode_override():
    env = make_legacy_env(
        family='smac-hard',
        map_name='3m',
        opponent_mode='sc2_computer',
    )
    assert env._env.switches.opponent_mode == 'sc2_computer'
    assert isinstance(env._env._opponent_runtime, EngineBotOpponentRuntime)
    env.close()


def test_legacy_adapter_forwards_seed_assignments_to_native_env():
    env = make_legacy_env(
        family='smac',
        map_name='3m',
        seed=7,
    )
    env._seed = 123
    assert env._env._seed == 123
    env.seed(55)
    assert env._env._seed == 55
    env.close()


def test_legacy_adapter_smacv2_capability_helpers_follow_shape_contract():
    env = make_legacy_env(
        family='smacv2',
        map_name='8m',
        capability_config={
            'attack': {'observe': True},
            'health': {'observe': True},
        },
    )
    assert env.get_cap_size() == 2
    env._env.agents[0] = SimpleNamespace(unit_type=0)
    cap_agent = env.get_capabilities_agent(0)
    assert cap_agent.shape[0] == 2
    assert np.isclose(cap_agent[0], 1.0)
    assert np.isclose(cap_agent[1], 0.0)
    cap_all = env.get_capabilities()
    assert cap_all.shape[0] == env.n_agents * 2
    env.close()


def test_make_env_propagates_seed_to_default_scripted_runtime():
    env = make_env(
        family='smac-hard',
        map_name='3m',
        normalized_api=False,
        seed=17,
    )
    runtime = env._opponent_runtime
    assert isinstance(runtime, ScriptedOpponentRuntime)
    assert runtime._config.seed == 17
    env.close()
