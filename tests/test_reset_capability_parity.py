from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from smac_unified.core import SMACEnvCore
from smac_unified.handlers import DefaultActionHandler, HandlerContext, TrackedUnit, UnitFrame, UnitTeamFrame


class _Pos:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


class _RawUnit:
    def __init__(
        self,
        *,
        owner: int,
        unit_type: int,
        x: float,
        y: float,
        health: float,
        health_max: float,
        tag: int,
    ):
        self.owner = owner
        self.unit_type = unit_type
        self.pos = _Pos(x=x, y=y)
        self.health = health
        self.health_max = health_max
        self.shield = 0.0
        self.shield_max = 0.0
        self.tag = tag


class _VariantLogic:
    @staticmethod
    def shoot_range_by_type(unit_ids):
        return {
            getattr(unit_ids, 'marine_id', 0): 6.0,
            1: 6.0,
            2: 6.0,
        }


def _tracked(*, unit_id: int, tag: int, x: float, y: float, alive: bool = True):
    return TrackedUnit(
        unit_id=unit_id,
        tag=tag,
        unit_type=1,
        x=x,
        y=y,
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


def test_reset_capabilities_apply_episode_vectors_and_payload():
    env = SMACEnvCore(
        variant='smacv2',
        map_name='8m',
        capability_config={
            'attack': {'item': [0.6, 0.6]},
            'health': {'item': [0.1, 0.1]},
            'enemy_mask': {'item': [1, 1]},
        },
    )
    env._apply_episode_capabilities(
        {
            'attack': {'item': [0.9]},
            'health': {'item': [0.2, 0.3]},
            'enemy_mask': {'item': [1, 0, 1]},
            'start_positions': {
                'ally_start_positions': [[1.0, 2.0], [3.0, 4.0]],
                'enemy_start_positions': [[20.0, 21.0]],
            },
            'team_gen': {
                'ally_team': ['marine', 'marauder'],
                'enemy_team': ['zealot'],
            },
        }
    )

    assert np.isclose(env.agent_attack_probabilities[0], 0.9)
    assert np.isclose(env.agent_attack_probabilities[1], 1.0)  # padded default
    assert np.isclose(env.agent_health_levels[0], 0.2)
    assert np.isclose(env.enemy_mask[1], 0.0)
    assert env.ally_start_positions is not None
    assert env.enemy_start_positions is not None
    assert env.ally_team == ['marine', 'marauder']
    assert env.enemy_team == ['zealot']


def test_split_raw_units_applies_health_and_enemy_masks():
    env = SMACEnvCore(variant='smacv2', map_name='3m')
    env.agent_health_levels = np.asarray([0.8, 0.0, 0.0], dtype=np.float32)
    env.enemy_mask = np.asarray([1.0, 0.0, 1.0], dtype=np.float32)
    units = [
        _RawUnit(owner=1, unit_type=1, x=1, y=1, health=10, health_max=45, tag=1),
        _RawUnit(owner=1, unit_type=1, x=2, y=1, health=45, health_max=45, tag=2),
        _RawUnit(owner=1, unit_type=1, x=3, y=1, health=45, health_max=45, tag=3),
        _RawUnit(owner=2, unit_type=1, x=6, y=1, health=45, health_max=45, tag=101),
        _RawUnit(owner=2, unit_type=1, x=7, y=1, health=45, health_max=45, tag=102),
        _RawUnit(owner=2, unit_type=1, x=8, y=1, health=45, health_max=45, tag=103),
    ]
    env._obs = SimpleNamespace(observation=SimpleNamespace(raw_data=SimpleNamespace(units=units)))

    allies, enemies = env._split_raw_units()
    assert len(allies) == 2
    assert len(enemies) == 2
    assert all(u.tag != 1 for u in allies)
    assert all(u.tag != 102 for u in enemies)
    probe = env._last_split_probe
    assert probe['allies_sorted_tags'] == [1, 2, 3]
    assert probe['allies_filtered_tags'] == [2, 3]
    assert probe['enemies_sorted_tags'] == [101, 102, 103]
    assert probe['enemies_filtered_tags'] == [101, 103]
    assert probe['ally_health_filter'][0]['tag'] == 1
    assert probe['ally_health_filter'][0]['kept'] is False
    assert probe['enemy_mask_filter'][1]['tag'] == 102
    assert probe['enemy_mask_filter'][1]['kept'] is False


def test_stochastic_attack_probability_can_block_attack_command():
    frame = UnitFrame(
        allies=_team([_tracked(unit_id=0, tag=1, x=4.0, y=4.0)]),
        enemies=_team([_tracked(unit_id=0, tag=101, x=8.0, y=4.0)]),
        prev_allies_health=np.asarray([45.0], dtype=np.float32),
        prev_allies_shield=np.asarray([0.0], dtype=np.float32),
        prev_enemies_health=np.asarray([45.0], dtype=np.float32),
        prev_enemies_shield=np.asarray([0.0], dtype=np.float32),
        step_token=1,
    )
    env = SimpleNamespace(
        agent_attack_probabilities=np.asarray([0.0], dtype=np.float32),
        _rng=np.random.default_rng(7),
    )
    context = HandlerContext(
        family='smac',
        map_name='3m',
        episode_step=0,
        n_agents=1,
        n_enemies=1,
        n_actions=7,
        n_actions_no_attack=6,
        attack_slots=1,
        move_amount=2.0,
        map_x=32.0,
        map_y=32.0,
        max_distance_x=32.0,
        max_distance_y=32.0,
        state_last_action=True,
        last_action=np.zeros((1, 7), dtype=np.float32),
        reward_sparse=False,
        reward_only_positive=False,
        reward_death_value=10.0,
        reward_negative_scale=0.5,
        reward_scale=False,
        reward_scale_rate=20.0,
        max_reward=0.0,
        variant_logic=_VariantLogic(),
        unit_type_ids=SimpleNamespace(marine_id=1, medivac_id=99),
        switches=SimpleNamespace(opponent_mode='sc2_computer'),
        env=env,
    )

    handler = DefaultActionHandler()
    cmd = handler.build_agent_action(
        frame=frame,
        context=context,
        agent_id=0,
        action=6,
    )
    assert cmd is None


def test_split_raw_units_preserves_enemy_observed_order_for_slot_mapping():
    env = SMACEnvCore(variant='smac', map_name='3m')
    env.enemy_mask = np.asarray([1.0, 1.0, 1.0], dtype=np.float32)
    units = [
        _RawUnit(owner=1, unit_type=1, x=1, y=1, health=45, health_max=45, tag=1),
        _RawUnit(owner=1, unit_type=1, x=2, y=1, health=45, health_max=45, tag=2),
        _RawUnit(owner=1, unit_type=1, x=3, y=1, health=45, health_max=45, tag=3),
        _RawUnit(owner=2, unit_type=1, x=9, y=1, health=45, health_max=45, tag=103),
        _RawUnit(owner=2, unit_type=1, x=1, y=1, health=45, health_max=45, tag=101),
        _RawUnit(owner=2, unit_type=1, x=5, y=1, health=45, health_max=45, tag=102),
    ]
    env._obs = SimpleNamespace(
        observation=SimpleNamespace(raw_data=SimpleNamespace(units=units))
    )
    _, enemies = env._split_raw_units()
    assert [u.tag for u in enemies] == [103, 101, 102]
    assert env._last_split_probe['enemies_sorted_tags'] == [103, 101, 102]


def test_split_raw_units_prefers_opponent_observation_for_enemy_team():
    env = SMACEnvCore(variant='smac-hard', map_name='3m')
    env.enemy_mask = np.asarray([1.0, 1.0, 1.0], dtype=np.float32)
    ally_obs_units = [
        _RawUnit(owner=1, unit_type=1, x=1, y=1, health=45, health_max=45, tag=1),
        _RawUnit(owner=1, unit_type=1, x=2, y=1, health=45, health_max=45, tag=2),
        _RawUnit(owner=1, unit_type=1, x=3, y=1, health=45, health_max=45, tag=3),
        _RawUnit(owner=2, unit_type=1, x=9, y=1, health=45, health_max=45, tag=999),
    ]
    enemy_obs_units = [
        _RawUnit(owner=2, unit_type=1, x=8, y=1, health=45, health_max=45, tag=103),
        _RawUnit(owner=2, unit_type=1, x=6, y=1, health=45, health_max=45, tag=101),
        _RawUnit(owner=2, unit_type=1, x=7, y=1, health=45, health_max=45, tag=102),
    ]
    env._obs = SimpleNamespace(
        observation=SimpleNamespace(raw_data=SimpleNamespace(units=ally_obs_units))
    )
    env._opponent_obs = SimpleNamespace(
        observation=SimpleNamespace(raw_data=SimpleNamespace(units=enemy_obs_units))
    )
    _, enemies = env._split_raw_units()
    assert [u.tag for u in enemies] == [101, 102, 103]
    assert env._last_split_probe['enemies_sorted_tags'] == [101, 102, 103]

