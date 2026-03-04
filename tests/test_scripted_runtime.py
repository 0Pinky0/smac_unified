import sys

from smac_unified.players import (
    OpponentEpisodeContext,
    OpponentStepContext,
    ScriptedOpponentConfig,
    ScriptedOpponentRuntime,
)


class _Script5:
    def __init__(self, map_name=None):
        self.map_name = map_name

    def script(self, agents, enemies, agent_ability, visible_matrix, episode_step):
        del agents, enemies, agent_ability, visible_matrix, episode_step
        return ['ok']


class _ScriptA:
    def __init__(self, map_name=None):
        self.map_name = map_name

    def script(self, agents, enemies, agent_ability, visible_matrix, episode_step):
        del agents, enemies, agent_ability, visible_matrix, episode_step
        return ['a']


class _ScriptB:
    def __init__(self, map_name=None):
        self.map_name = map_name

    def script(self, agents, enemies, agent_ability, visible_matrix, episode_step):
        del agents, enemies, agent_ability, visible_matrix, episode_step
        return ['b']


def test_scripted_runtime_computes_actions_from_payload():
    runtime = ScriptedOpponentRuntime(
        script_dict={'3m': [_Script5]},
        config=ScriptedOpponentConfig(strategy='fixed', fixed_index=0),
    )
    runtime.bind_env(object(), 'smac-hard')
    runtime.on_reset(
        OpponentEpisodeContext(
            family='smac-hard',
            map_name='3m',
        )
    )
    actions = runtime.compute_actions(
        OpponentStepContext(
            family='smac-hard',
            episode_step=0,
            actions=[0, 0, 0],
            payload={
                'agents': {},
                'enemies': {},
                'agent_ability': [],
                'visible_matrix': {},
                'episode_step': 0,
            },
        )
    )
    assert actions == ['ok']


def test_scripted_runtime_resolves_local_smac_hard_script_pack():
    for name in list(sys.modules.keys()):
        if name.startswith('smac_hard.env.scripts'):
            sys.modules.pop(name, None)

    runtime = ScriptedOpponentRuntime()
    pool = runtime._resolve_script_pool('3m')

    assert len(pool) > 0
    assert 'smac_unified.players.smac_hard_scripts' in sys.modules
    assert not any(
        name.startswith('smac_hard.env.scripts') for name in sys.modules.keys()
    )


def test_scripted_runtime_keeps_generic_fallback_for_unknown_maps():
    runtime = ScriptedOpponentRuntime()
    pool = runtime._resolve_script_pool('unknown-map-name')
    assert len(pool) > 0


def test_scripted_runtime_reseeds_random_selection_from_episode_seed():
    runtime = ScriptedOpponentRuntime(
        script_dict={'3m': [_ScriptA, _ScriptB]},
        config=ScriptedOpponentConfig(strategy='random'),
    )
    runtime.bind_env(object(), 'smac-hard')
    runtime.on_reset(
        OpponentEpisodeContext(
            family='smac-hard',
            map_name='3m',
            seed=42,
        )
    )
    first = runtime.last_script_name
    runtime.on_reset(
        OpponentEpisodeContext(
            family='smac-hard',
            map_name='3m',
            seed=42,
        )
    )
    second = runtime.last_script_name
    assert first == second
