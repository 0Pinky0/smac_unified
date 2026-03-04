from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from smac_unified.handlers import (
    CapabilityObservationHandler,
    CapabilityStateHandler,
    DefaultObservationHandler,
    DefaultStateHandler,
    HandlerContext,
    TrackedUnit,
    UnitFrame,
    UnitTeamFrame,
)


class _VariantLogic:
    def shoot_range_by_type(self, unit_ids):
        return {
            getattr(unit_ids, 'marine_id', 0): 6.0,
            getattr(unit_ids, 'medivac_id', 0): 4.0,
            1: 6.0,
            2: 6.0,
        }


class _AvailEnv:
    def __init__(self, masks: dict[int, list[int]]):
        self._masks = masks
        self.episode_limit = 120
        self.map_params = SimpleNamespace(map_type='MMM')
        self.agent_attack_probabilities = np.asarray([0.9, 0.8], dtype=np.float32)
        self.agent_health_levels = np.asarray([0.7, 0.6], dtype=np.float32)

    def get_avail_agent_actions(self, agent_id: int) -> list[int]:
        return list(self._masks[agent_id])

    def get_obs(self):
        return []


def _tracked(
    *,
    unit_id: int,
    tag: int,
    unit_type: int,
    x: float,
    y: float,
    health: float,
    health_max: float = 45.0,
    shield: float = 0.0,
    shield_max: float = 0.0,
    alive: bool = True,
):
    return TrackedUnit(
        unit_id=unit_id,
        tag=tag,
        unit_type=unit_type,
        x=x,
        y=y,
        health=health,
        health_max=health_max,
        shield=shield,
        shield_max=shield_max,
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


def _context(*, env):
    return HandlerContext(
        family='smac',
        map_name='MMM',
        episode_step=10,
        n_agents=2,
        n_enemies=1,
        n_actions=12,
        n_actions_no_attack=10,
        attack_slots=2,
        move_amount=2.0,
        map_x=32.0,
        map_y=32.0,
        max_distance_x=32.0,
        max_distance_y=32.0,
        state_last_action=True,
        last_action=np.ones((2, 12), dtype=np.float32),
        reward_sparse=False,
        reward_only_positive=False,
        reward_death_value=10.0,
        reward_negative_scale=0.5,
        reward_scale=False,
        reward_scale_rate=20.0,
        max_reward=0.0,
        variant_logic=_VariantLogic(),
        unit_type_ids=SimpleNamespace(marine_id=1, marauder_id=2, medivac_id=3),
        switches=SimpleNamespace(opponent_mode='sc2_computer'),
        env=env,
        obs_all_health=True,
        obs_own_health=True,
        obs_last_action=True,
        obs_pathing_grid=True,
        obs_terrain_height=True,
        obs_timestep_number=True,
        state_timestep_number=True,
        obs_instead_of_state=False,
        shield_bits_ally=1,
        shield_bits_enemy=0,
        unit_type_bits=3,
        n_fov_actions=4,
        conic_fov_angle=float(np.pi / 2.0),
        fov_directions=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        canonical_fov_directions=np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
            dtype=np.float32,
        ),
        action_mask=True,
        pathing_grid=np.ones((32, 32), dtype=bool),
        terrain_height=np.zeros((32, 32), dtype=np.float32),
    )


def test_observation_handler_restores_structured_feature_toggles():
    env = _AvailEnv(
        masks={
            0: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            1: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        }
    )
    context = _context(env=env)
    frame = UnitFrame(
        allies=_team(
            [
                _tracked(unit_id=0, tag=1, unit_type=2, x=4, y=4, health=40, shield=5, shield_max=5),
                _tracked(unit_id=1, tag=2, unit_type=1, x=6, y=4, health=45, shield=0, shield_max=0),
            ]
        ),
        enemies=_team([_tracked(unit_id=0, tag=101, unit_type=1, x=8, y=4, health=35)]),
        prev_allies_health=np.asarray([40.0, 45.0], dtype=np.float32),
        prev_allies_shield=np.asarray([5.0, 0.0], dtype=np.float32),
        prev_enemies_health=np.asarray([35.0], dtype=np.float32),
        prev_enemies_shield=np.asarray([0.0], dtype=np.float32),
        step_token=2,
    )

    handler = DefaultObservationHandler()
    obs = handler.build_agent_obs(frame=frame, context=context, agent_id=0)

    move_dim = 4 + 8 + 9
    enemy_dim = 4 + 1 + 3
    ally_dim = 4 + 1 + 1 + 3 + context.n_actions
    own_dim = 1 + 1 + 3
    expected = move_dim + context.attack_slots * enemy_dim + (context.n_agents - 1) * ally_dim + own_dim + 1
    assert obs.shape[0] == expected
    assert np.isclose(obs[-1], context.episode_step / env.episode_limit)


def test_capability_handlers_append_capability_channels():
    env = _AvailEnv(
        masks={
            0: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            1: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        }
    )
    context = _context(env=env)
    frame = UnitFrame(
        allies=_team(
            [
                _tracked(unit_id=0, tag=1, unit_type=2, x=4, y=4, health=40, shield=5, shield_max=5),
                _tracked(unit_id=1, tag=2, unit_type=1, x=6, y=4, health=45),
            ]
        ),
        enemies=_team([_tracked(unit_id=0, tag=101, unit_type=1, x=8, y=4, health=35)]),
        prev_allies_health=np.asarray([40.0, 45.0], dtype=np.float32),
        prev_allies_shield=np.asarray([5.0, 0.0], dtype=np.float32),
        prev_enemies_health=np.asarray([35.0], dtype=np.float32),
        prev_enemies_shield=np.asarray([0.0], dtype=np.float32),
        step_token=3,
    )

    base_obs = DefaultObservationHandler().build_agent_obs(frame=frame, context=context, agent_id=0)
    cap_obs = CapabilityObservationHandler().build_agent_obs(frame=frame, context=context, agent_id=0)
    assert cap_obs.shape[0] == base_obs.shape[0] + 4  # fov(2) + attack_prob + health_level

    base_state = DefaultStateHandler().build_state(frame=frame, context=context)
    cap_state = CapabilityStateHandler().build_state(frame=frame, context=context)
    assert cap_state.shape[0] == base_state.shape[0] + 4


def test_state_handler_uses_center_relative_coordinates():
    env = _AvailEnv(
        masks={
            0: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            1: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        }
    )
    context = _context(env=env)
    context.max_distance_x = 20.0
    context.max_distance_y = 16.0
    frame = UnitFrame(
        allies=_team(
            [
                _tracked(unit_id=0, tag=1, unit_type=2, x=18, y=12, health=40, shield=5, shield_max=5),
                _tracked(unit_id=1, tag=2, unit_type=1, x=6, y=4, health=45),
            ]
        ),
        enemies=_team([_tracked(unit_id=0, tag=101, unit_type=1, x=22, y=16, health=35)]),
        prev_allies_health=np.asarray([40.0, 45.0], dtype=np.float32),
        prev_allies_shield=np.asarray([5.0, 0.0], dtype=np.float32),
        prev_enemies_health=np.asarray([35.0], dtype=np.float32),
        prev_enemies_shield=np.asarray([0.0], dtype=np.float32),
        step_token=4,
    )
    state = DefaultStateHandler().build_state(frame=frame, context=context)
    ally_rel_x = state[2]
    ally_rel_y = state[3]
    center_x = context.map_x / 2.0
    center_y = context.map_y / 2.0
    assert np.isclose(ally_rel_x, (18.0 - center_x) / context.max_distance_x)
    assert np.isclose(ally_rel_y, (12.0 - center_y) / context.max_distance_y)


def test_observation_handler_preserves_enemy_feature_layout():
    env = _AvailEnv(
        masks={
            0: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            1: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        }
    )
    context = _context(env=env)
    frame = UnitFrame(
        allies=_team(
            [
                _tracked(unit_id=0, tag=1, unit_type=2, x=4, y=4, health=40, shield=5, shield_max=5),
                _tracked(unit_id=1, tag=2, unit_type=1, x=6, y=4, health=45),
            ]
        ),
        enemies=_team([_tracked(unit_id=0, tag=101, unit_type=1, x=8, y=4, health=35)]),
        prev_allies_health=np.asarray([40.0, 45.0], dtype=np.float32),
        prev_allies_shield=np.asarray([5.0, 0.0], dtype=np.float32),
        prev_enemies_health=np.asarray([35.0], dtype=np.float32),
        prev_enemies_shield=np.asarray([0.0], dtype=np.float32),
        step_token=5,
    )
    obs = DefaultObservationHandler().build_agent_obs(frame=frame, context=context, agent_id=0)

    move_dim = 4 + 8 + 9
    enemy_slot0 = move_dim
    sight_range = 9.0
    assert obs[enemy_slot0] == 1.0
    assert np.isclose(obs[enemy_slot0 + 1], 4.0 / sight_range)
    assert np.isclose(obs[enemy_slot0 + 2], 4.0 / sight_range)
    assert np.isclose(obs[enemy_slot0 + 3], 0.0)


def test_state_handler_preserves_flattened_team_layout():
    env = _AvailEnv(
        masks={
            0: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            1: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        }
    )
    context = _context(env=env)
    frame = UnitFrame(
        allies=_team(
            [
                _tracked(unit_id=0, tag=1, unit_type=2, x=4, y=4, health=40, shield=5, shield_max=5),
                _tracked(unit_id=1, tag=2, unit_type=1, x=6, y=4, health=45),
            ]
        ),
        enemies=_team([_tracked(unit_id=0, tag=101, unit_type=1, x=8, y=4, health=35)]),
        prev_allies_health=np.asarray([40.0, 45.0], dtype=np.float32),
        prev_allies_shield=np.asarray([5.0, 0.0], dtype=np.float32),
        prev_enemies_health=np.asarray([35.0], dtype=np.float32),
        prev_enemies_shield=np.asarray([0.0], dtype=np.float32),
        step_token=6,
    )
    state = DefaultStateHandler().build_state(frame=frame, context=context)

    ally_attr = 4 + context.shield_bits_ally + context.unit_type_bits
    enemy_offset = context.n_agents * ally_attr
    assert np.isclose(state[0], 40.0 / 45.0)
    assert np.isclose(state[enemy_offset], 35.0 / 45.0)


def test_observation_handler_hides_enemy_at_exact_sight_range_boundary():
    env = _AvailEnv(
        masks={
            0: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            1: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        }
    )
    context = _context(env=env)
    frame = UnitFrame(
        allies=_team(
            [
                _tracked(unit_id=0, tag=1, unit_type=2, x=4, y=4, health=40, shield=5, shield_max=5),
                _tracked(unit_id=1, tag=2, unit_type=1, x=6, y=4, health=45),
            ]
        ),
        enemies=_team([_tracked(unit_id=0, tag=101, unit_type=1, x=13, y=4, health=35)]),
        prev_allies_health=np.asarray([40.0, 45.0], dtype=np.float32),
        prev_allies_shield=np.asarray([5.0, 0.0], dtype=np.float32),
        prev_enemies_health=np.asarray([35.0], dtype=np.float32),
        prev_enemies_shield=np.asarray([0.0], dtype=np.float32),
        step_token=10,
    )
    obs = DefaultObservationHandler().build_agent_obs(frame=frame, context=context, agent_id=0)
    move_dim = 4 + 8 + 9
    enemy_slot0 = move_dim
    assert np.allclose(obs[enemy_slot0 : enemy_slot0 + 4], np.zeros(4, dtype=np.float32))


def test_state_handler_zeroes_dead_unit_rows():
    env = _AvailEnv(
        masks={
            0: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            1: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        }
    )
    context = _context(env=env)
    context.n_enemies = 2
    frame = UnitFrame(
        allies=_team(
            [
                _tracked(unit_id=0, tag=1, unit_type=2, x=18, y=12, health=40, shield=5, shield_max=5),
                _tracked(
                    unit_id=1,
                    tag=2,
                    unit_type=1,
                    x=0,
                    y=0,
                    health=0,
                    alive=False,
                ),
            ]
        ),
        enemies=_team(
            [
                _tracked(unit_id=0, tag=101, unit_type=1, x=22, y=16, health=35),
                _tracked(
                    unit_id=1,
                    tag=102,
                    unit_type=1,
                    x=0,
                    y=0,
                    health=0,
                    alive=False,
                ),
            ]
        ),
        prev_allies_health=np.asarray([40.0, 0.0], dtype=np.float32),
        prev_allies_shield=np.asarray([5.0, 0.0], dtype=np.float32),
        prev_enemies_health=np.asarray([35.0, 0.0], dtype=np.float32),
        prev_enemies_shield=np.asarray([0.0, 0.0], dtype=np.float32),
        step_token=11,
    )
    state = DefaultStateHandler().build_state(frame=frame, context=context)
    ally_attr = 4 + context.shield_bits_ally + context.unit_type_bits
    enemy_attr = 3 + context.shield_bits_enemy + context.unit_type_bits
    dead_ally_start = ally_attr
    enemy_start = context.n_agents * ally_attr
    dead_enemy_start = enemy_start + enemy_attr
    assert np.allclose(state[dead_ally_start : dead_ally_start + ally_attr], 0.0)
    assert np.allclose(state[dead_enemy_start : dead_enemy_start + enemy_attr], 0.0)

