from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from smac_unified.handlers import (
    AbilityAugmentedActionHandler,
    ClassicActionHandler,
    ConicFovActionHandler,
    HandlerContext,
    TrackedUnit,
    UnitFrame,
    UnitTeamFrame,
)


class _VariantLogic:
    @staticmethod
    def shoot_range_by_type(unit_ids):
        return {
            getattr(unit_ids, 'marine_id', 0): 6.0,
            getattr(unit_ids, 'medivac_id', 0): 4.0,
            1: 6.0,
            2: 6.0,
        }


class _Pos:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


class _RawUnit:
    def __init__(
        self,
        *,
        tag: int,
        x: float,
        y: float,
        health: float = 10.0,
        health_max: float = 10.0,
    ):
        self.tag = tag
        self.pos = _Pos(x=x, y=y)
        self.health = health
        self.health_max = health_max


class _AbilityProbeHandler(AbilityAugmentedActionHandler):
    @staticmethod
    def _query_agent_abilities(env):
        del env
        return {0: (380,)}


class _AbilityFallbackHandler(AbilityAugmentedActionHandler):
    @staticmethod
    def _query_agent_abilities(env):
        del env
        return {0: (23,)}


class _CountingAbilityHandler(AbilityAugmentedActionHandler):
    def __init__(self):
        super().__init__(use_ability=True)
        self.query_calls = 0

    def _query_agent_abilities(self, env):
        del env
        self.query_calls += 1
        return {0: (380,), 1: (380,)}


def _tracked(
    *,
    unit_id: int,
    tag: int,
    x: float,
    y: float,
    unit_type: int = 1,
    alive: bool = True,
):
    return TrackedUnit(
        unit_id=unit_id,
        tag=tag,
        unit_type=unit_type,
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


def _context(
    *,
    n_agents: int,
    n_enemies: int,
    n_actions: int,
    n_actions_no_attack: int,
    attack_slots: int,
    env=None,
):
    return HandlerContext(
        family='smac',
        map_name='3m',
        episode_step=0,
        n_agents=n_agents,
        n_enemies=n_enemies,
        n_actions=n_actions,
        n_actions_no_attack=n_actions_no_attack,
        attack_slots=attack_slots,
        move_amount=2.0,
        map_x=32.0,
        map_y=32.0,
        max_distance_x=32.0,
        max_distance_y=32.0,
        state_last_action=True,
        last_action=np.zeros((n_agents, n_actions), dtype=np.float32),
        reward_sparse=False,
        reward_only_positive=False,
        reward_death_value=10.0,
        reward_negative_scale=0.5,
        reward_scale=False,
        reward_scale_rate=20.0,
        max_reward=0.0,
        variant_logic=_VariantLogic(),
        unit_type_ids=SimpleNamespace(medivac_id=99, marine_id=1),
        switches=SimpleNamespace(opponent_mode='sc2_computer'),
        env=env,
    )


def test_classic_action_handler_uses_pathing_grid_for_move_checks():
    frame = UnitFrame(
        allies=_team([_tracked(unit_id=0, tag=1, x=1.0, y=1.0)]),
        enemies=_team([_tracked(unit_id=0, tag=101, x=5.0, y=1.0)]),
        prev_allies_health=np.asarray([45.0], dtype=np.float32),
        prev_allies_shield=np.asarray([0.0], dtype=np.float32),
        prev_enemies_health=np.asarray([45.0], dtype=np.float32),
        prev_enemies_shield=np.asarray([0.0], dtype=np.float32),
        step_token=1,
    )
    context = _context(
        n_agents=1,
        n_enemies=1,
        n_actions=7,
        n_actions_no_attack=6,
        attack_slots=1,
    )
    grid = np.ones((32, 32), dtype=bool)
    grid[1, 2] = False
    context.pathing_grid = grid

    handler = ClassicActionHandler()
    avail = handler.get_avail_agent_actions(frame=frame, context=context, agent_id=0)
    assert avail[2] == 0  # north blocked by pathing grid
    assert avail[3] == 1


def test_conic_handler_updates_fov_and_unmasks_targets():
    frame0 = UnitFrame(
        allies=_team([_tracked(unit_id=0, tag=1, x=0.0, y=0.0)]),
        enemies=_team([_tracked(unit_id=0, tag=101, x=0.0, y=4.0)]),
        prev_allies_health=np.asarray([45.0], dtype=np.float32),
        prev_allies_shield=np.asarray([0.0], dtype=np.float32),
        prev_enemies_health=np.asarray([45.0], dtype=np.float32),
        prev_enemies_shield=np.asarray([0.0], dtype=np.float32),
        step_token=1,
    )
    context = _context(
        n_agents=1,
        n_enemies=1,
        n_actions=11,
        n_actions_no_attack=10,  # 6 base + 4 fov
        attack_slots=1,
    )
    context.n_fov_actions = 4
    context.conic_fov_angle = float(np.pi / 2.0)
    context.action_mask = True
    context.fov_directions = np.asarray([[1.0, 0.0]], dtype=np.float32)
    context.canonical_fov_directions = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
        dtype=np.float32,
    )

    handler = ConicFovActionHandler(num_fov_actions=4, action_mask=True)
    avail0 = handler.get_avail_agent_actions(frame=frame0, context=context, agent_id=0)
    assert avail0[10] == 0  # target north not in east-facing cone
    handler.build_agent_action(frame=frame0, context=context, agent_id=0, action=7)

    frame1 = UnitFrame(
        allies=frame0.allies,
        enemies=frame0.enemies,
        prev_allies_health=frame0.prev_allies_health,
        prev_allies_shield=frame0.prev_allies_shield,
        prev_enemies_health=frame0.prev_enemies_health,
        prev_enemies_shield=frame0.prev_enemies_shield,
        step_token=2,
    )
    avail1 = handler.get_avail_agent_actions(frame=frame1, context=context, agent_id=0)
    assert avail1[10] == 1


def test_ability_handler_enables_ability_branch_and_builds_command():
    ally_raw = _RawUnit(tag=1, x=2.0, y=2.0, health=9.0, health_max=10.0)
    enemy_raw = _RawUnit(tag=101, x=5.0, y=2.0)
    fake_env = SimpleNamespace(
        agents={0: ally_raw},
        enemies={0: enemy_raw},
    )
    frame = UnitFrame(
        allies=_team([_tracked(unit_id=0, tag=1, x=2.0, y=2.0)]),
        enemies=_team([_tracked(unit_id=0, tag=101, x=5.0, y=2.0)]),
        prev_allies_health=np.asarray([45.0], dtype=np.float32),
        prev_allies_shield=np.asarray([0.0], dtype=np.float32),
        prev_enemies_health=np.asarray([45.0], dtype=np.float32),
        prev_enemies_shield=np.asarray([0.0], dtype=np.float32),
        step_token=3,
    )
    context = _context(
        n_agents=1,
        n_enemies=1,
        n_actions=24,  # 6 + 9 attack + 9 ability
        n_actions_no_attack=6,
        attack_slots=1,
        env=fake_env,
    )
    context.ability_padding = 9
    context.use_ability = True

    handler = _AbilityProbeHandler(use_ability=True)
    avail = handler.get_avail_agent_actions(frame=frame, context=context, agent_id=0)
    ability_action = context.n_actions_no_attack + context.ability_padding
    assert avail[ability_action] == 1
    cmd = handler.build_agent_action(
        frame=frame,
        context=context,
        agent_id=0,
        action=ability_action,
    )
    assert cmd is not None
    assert cmd.action_raw.unit_command.ability_id == 3675


def test_ability_handler_falls_back_to_legacy_unit_ability():
    ally_raw = _RawUnit(tag=1, x=2.0, y=2.0, health=9.0, health_max=10.0)
    enemy_raw = _RawUnit(tag=101, x=5.0, y=2.0)
    fake_env = SimpleNamespace(
        agents={0: ally_raw},
        enemies={0: enemy_raw},
    )
    frame = UnitFrame(
        allies=_team([_tracked(unit_id=0, tag=1, x=2.0, y=2.0, unit_type=1)]),
        enemies=_team([_tracked(unit_id=0, tag=101, x=5.0, y=2.0)]),
        prev_allies_health=np.asarray([45.0], dtype=np.float32),
        prev_allies_shield=np.asarray([0.0], dtype=np.float32),
        prev_enemies_health=np.asarray([45.0], dtype=np.float32),
        prev_enemies_shield=np.asarray([0.0], dtype=np.float32),
        step_token=4,
    )
    context = _context(
        n_agents=1,
        n_enemies=1,
        n_actions=24,
        n_actions_no_attack=6,
        attack_slots=1,
        env=fake_env,
    )
    context.ability_padding = 9
    context.use_ability = True

    handler = _AbilityFallbackHandler(use_ability=True)
    avail = handler.get_avail_agent_actions(frame=frame, context=context, agent_id=0)
    ability_action = context.n_actions_no_attack + context.ability_padding
    assert avail[ability_action] == 1


def test_ability_query_cache_reuses_same_step_and_invalidates_next_step():
    ally0 = _RawUnit(tag=1, x=2.0, y=2.0, health=9.0, health_max=10.0)
    ally1 = _RawUnit(tag=2, x=3.0, y=2.0, health=9.0, health_max=10.0)
    enemy = _RawUnit(tag=101, x=5.0, y=2.0)
    fake_env = SimpleNamespace(
        agents={0: ally0, 1: ally1},
        enemies={0: enemy},
    )
    frame = UnitFrame(
        allies=_team(
            [
                _tracked(unit_id=0, tag=1, x=2.0, y=2.0, unit_type=1),
                _tracked(unit_id=1, tag=2, x=3.0, y=2.0, unit_type=1),
            ]
        ),
        enemies=_team([_tracked(unit_id=0, tag=101, x=5.0, y=2.0)]),
        prev_allies_health=np.asarray([45.0, 45.0], dtype=np.float32),
        prev_allies_shield=np.asarray([0.0, 0.0], dtype=np.float32),
        prev_enemies_health=np.asarray([45.0], dtype=np.float32),
        prev_enemies_shield=np.asarray([0.0], dtype=np.float32),
        step_token=5,
    )
    context = _context(
        n_agents=2,
        n_enemies=1,
        n_actions=24,
        n_actions_no_attack=6,
        attack_slots=1,
        env=fake_env,
    )
    context.ability_padding = 9
    context.use_ability = True

    handler = _CountingAbilityHandler()
    handler.get_avail_agent_actions(frame=frame, context=context, agent_id=0)
    handler.get_avail_agent_actions(frame=frame, context=context, agent_id=1)
    assert handler.query_calls == 1

    frame_next = UnitFrame(
        allies=frame.allies,
        enemies=frame.enemies,
        prev_allies_health=frame.prev_allies_health,
        prev_allies_shield=frame.prev_allies_shield,
        prev_enemies_health=frame.prev_enemies_health,
        prev_enemies_shield=frame.prev_enemies_shield,
        step_token=6,
    )
    handler.get_avail_agent_actions(frame=frame_next, context=context, agent_id=0)
    assert handler.query_calls == 2

