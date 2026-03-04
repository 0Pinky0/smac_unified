from __future__ import annotations

import numpy as np

from ..types import HandlerContext, TrackedUnit, UnitFrame
from .base import ObservationHandler


class DefaultObservationHandler(ObservationHandler):
    def build_agent_obs(
        self,
        *,
        frame: UnitFrame,
        context: HandlerContext,
        agent_id: int,
    ):
        move_dim, enemy_dim, ally_dim, own_dim = _feature_dims(context)
        total = (
            move_dim
            + context.attack_slots * enemy_dim
            + max(context.n_agents - 1, 1) * ally_dim
            + own_dim
            + (1 if context.obs_timestep_number else 0)
        )
        unit = frame.allies.units[agent_id]
        if not unit.alive:
            obs = np.zeros(total, dtype=np.float32)
            if context.obs_timestep_number:
                obs[-1] = _timestep_ratio(context)
            return obs

        avail = (
            context.env.get_avail_agent_actions(agent_id)
            if context.env is not None
            else [0] * context.n_actions
        )

        move_feats = np.zeros(move_dim, dtype=np.float32)
        move_feats[:4] = np.asarray(avail[2:6], dtype=np.float32)
        idx = 4
        if context.obs_pathing_grid:
            move_feats[idx : idx + 8] = _surrounding_pathing(unit=unit, context=context)
            idx += 8
        if context.obs_terrain_height:
            move_feats[idx : idx + 9] = _surrounding_height(unit=unit, context=context)

        enemy_feats = np.zeros((context.attack_slots, enemy_dim), dtype=np.float32)
        enemies = list(frame.enemies.units)[: context.attack_slots]
        sight_range = _unit_sight_range(agent_id=agent_id, frame=frame, context=context)
        for slot, enemy in enumerate(enemies):
            if not enemy.alive:
                continue
            dist = _distance(unit.x, unit.y, enemy.x, enemy.y)
            if dist > sight_range:
                continue
            cursor = 0
            action_id = context.n_actions_no_attack + slot
            if action_id < len(avail):
                enemy_feats[slot, cursor] = float(avail[action_id])
            cursor += 1
            enemy_feats[slot, cursor] = dist / max(sight_range, 1.0)
            cursor += 1
            enemy_feats[slot, cursor] = (enemy.x - unit.x) / max(sight_range, 1.0)
            cursor += 1
            enemy_feats[slot, cursor] = (enemy.y - unit.y) / max(sight_range, 1.0)
            cursor += 1
            if context.obs_all_health:
                enemy_feats[slot, cursor] = enemy.health / max(enemy.health_max, 1.0)
                cursor += 1
                if context.shield_bits_enemy > 0:
                    enemy_feats[slot, cursor] = enemy.shield / max(enemy.shield_max, 1.0)
                    cursor += 1
            if context.unit_type_bits > 0:
                unit_type_idx = _unit_type_index(unit=enemy, context=context, ally=False)
                if 0 <= unit_type_idx < context.unit_type_bits:
                    enemy_feats[slot, cursor + unit_type_idx] = 1.0

        ally_feats = np.zeros((max(context.n_agents - 1, 1), ally_dim), dtype=np.float32)
        ally_slot = 0
        for ally in frame.allies.units:
            if ally.unit_id == agent_id:
                continue
            if ally_slot >= ally_feats.shape[0]:
                break
            cursor = 0
            if ally.alive:
                dist = _distance(unit.x, unit.y, ally.x, ally.y)
                if dist <= sight_range:
                    ally_feats[ally_slot, cursor] = 1.0
                    cursor += 1
                    ally_feats[ally_slot, cursor] = dist / max(sight_range, 1.0)
                    cursor += 1
                    ally_feats[ally_slot, cursor] = (ally.x - unit.x) / max(
                        sight_range,
                        1.0,
                    )
                    cursor += 1
                    ally_feats[ally_slot, cursor] = (ally.y - unit.y) / max(
                        sight_range,
                        1.0,
                    )
                    cursor += 1
                    if context.obs_all_health:
                        ally_feats[ally_slot, cursor] = ally.health / max(ally.health_max, 1.0)
                        cursor += 1
                        if context.shield_bits_ally > 0:
                            ally_feats[ally_slot, cursor] = ally.shield / max(
                                ally.shield_max,
                                1.0,
                            )
                            cursor += 1
                    if context.unit_type_bits > 0:
                        unit_type_idx = _unit_type_index(unit=ally, context=context, ally=True)
                        if 0 <= unit_type_idx < context.unit_type_bits:
                            ally_feats[ally_slot, cursor + unit_type_idx] = 1.0
                    cursor += context.unit_type_bits
                    if context.obs_last_action and ally.unit_id < context.last_action.shape[0]:
                        ally_feats[ally_slot, cursor : cursor + context.n_actions] = context.last_action[
                            ally.unit_id
                        ]
            ally_slot += 1

        own_feats = np.zeros(own_dim, dtype=np.float32)
        cursor = 0
        if context.obs_own_health:
            own_feats[cursor] = unit.health / max(unit.health_max, 1.0)
            cursor += 1
            if context.shield_bits_ally > 0:
                own_feats[cursor] = unit.shield / max(unit.shield_max, 1.0)
                cursor += 1
        if context.unit_type_bits > 0:
            unit_type_idx = _unit_type_index(unit=unit, context=context, ally=True)
            if 0 <= unit_type_idx < context.unit_type_bits:
                own_feats[cursor + unit_type_idx] = 1.0

        chunks = [
            move_feats.flatten(),
            enemy_feats.flatten(),
            ally_feats.flatten(),
            own_feats.flatten(),
        ]
        if context.obs_timestep_number:
            chunks.append(np.asarray([_timestep_ratio(context)], dtype=np.float32))
        return np.concatenate(chunks, axis=0).astype(np.float32)

    def build_obs(
        self,
        *,
        frame: UnitFrame,
        context: HandlerContext,
    ):
        return [
            self.build_agent_obs(
                frame=frame,
                context=context,
                agent_id=agent_id,
            )
            for agent_id in range(context.n_agents)
        ]


class CapabilityObservationHandler(DefaultObservationHandler):
    """SMACv2 capability-aware observation semantics."""

    def build_agent_obs(
        self,
        *,
        frame: UnitFrame,
        context: HandlerContext,
        agent_id: int,
    ):
        base = super().build_agent_obs(frame=frame, context=context, agent_id=agent_id)
        extras: list[float] = []
        if (
            context.n_fov_actions > 0
            and context.fov_directions is not None
            and agent_id < len(context.fov_directions)
        ):
            extras.extend(
                [
                    float(context.fov_directions[agent_id][0]),
                    float(context.fov_directions[agent_id][1]),
                ]
            )
        env = context.env
        if env is not None:
            attack_probs = getattr(env, 'agent_attack_probabilities', None)
            if attack_probs is not None and agent_id < len(attack_probs):
                extras.append(float(attack_probs[agent_id]))
            health_levels = getattr(env, 'agent_health_levels', None)
            if health_levels is not None and agent_id < len(health_levels):
                extras.append(float(health_levels[agent_id]))
        if not extras:
            return base
        return np.concatenate((base, np.asarray(extras, dtype=np.float32)), axis=0)


def _feature_dims(context: HandlerContext) -> tuple[int, int, int, int]:
    move_dim = 4
    if context.obs_pathing_grid:
        move_dim += 8
    if context.obs_terrain_height:
        move_dim += 9

    enemy_dim = 4
    if context.obs_all_health:
        enemy_dim += 1 + context.shield_bits_enemy
    if context.unit_type_bits > 0:
        enemy_dim += context.unit_type_bits

    ally_dim = 4
    if context.obs_all_health:
        ally_dim += 1 + context.shield_bits_ally
    if context.unit_type_bits > 0:
        ally_dim += context.unit_type_bits
    if context.obs_last_action:
        ally_dim += context.n_actions

    own_dim = 0
    if context.obs_own_health:
        own_dim += 1 + context.shield_bits_ally
    if context.unit_type_bits > 0:
        own_dim += context.unit_type_bits
    return move_dim, enemy_dim, ally_dim, own_dim


def _distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return float(np.hypot(x2 - x1, y2 - y1))


def _unit_sight_range(
    *,
    agent_id: int,
    frame: UnitFrame,
    context: HandlerContext,
) -> float:
    unit = frame.allies.units[agent_id]
    ids = context.variant_logic.shoot_range_by_type(context.unit_type_ids)
    default_range = 6.0
    shoot_range = default_range
    if unit.unit_type in ids and ids[unit.unit_type] > 0:
        shoot_range = max(float(ids[unit.unit_type]), 0.1)
    return max(shoot_range + 3.0, 9.0)


def _unit_type_index(
    *,
    unit: TrackedUnit,
    context: HandlerContext,
    ally: bool,
) -> int:
    del ally
    if context.unit_type_bits <= 0:
        return -1
    ids = context.unit_type_ids
    map_type = ''
    if context.env is not None:
        map_type = str(getattr(getattr(context.env, 'map_params', None), 'map_type', ''))
    if map_type in ('stalkers_and_zealots',):
        if unit.unit_type == getattr(ids, 'stalker_id', -1):
            return 0
        if unit.unit_type == getattr(ids, 'zealot_id', -1):
            return 1
    if map_type in ('colossi_stalkers_zealots',):
        if unit.unit_type == getattr(ids, 'colossus_id', -1):
            return 0
        if unit.unit_type == getattr(ids, 'stalker_id', -1):
            return 1
        if unit.unit_type == getattr(ids, 'zealot_id', -1):
            return 2
    if map_type in ('bane',):
        if unit.unit_type == getattr(ids, 'baneling_id', -1):
            return 0
        if unit.unit_type == getattr(ids, 'zergling_id', -1):
            return 1
    if map_type in ('MMM', 'terran_gen'):
        if unit.unit_type == getattr(ids, 'marauder_id', -1):
            return 0
        if unit.unit_type == getattr(ids, 'marine_id', -1):
            return 1
        if unit.unit_type == getattr(ids, 'medivac_id', -1):
            return 2

    candidates = [
        value
        for value in (
            getattr(ids, 'marine_id', 0),
            getattr(ids, 'marauder_id', 0),
            getattr(ids, 'medivac_id', 0),
            getattr(ids, 'stalker_id', 0),
            getattr(ids, 'zealot_id', 0),
            getattr(ids, 'colossus_id', 0),
            getattr(ids, 'hydralisk_id', 0),
            getattr(ids, 'zergling_id', 0),
            getattr(ids, 'baneling_id', 0),
        )
        if value > 0
    ]
    if unit.unit_type in candidates:
        return candidates.index(unit.unit_type) % context.unit_type_bits
    return -1


def _surrounding_points(
    *,
    unit: TrackedUnit,
    move_amount: float,
    include_self: bool,
) -> list[tuple[int, int]]:
    x = int(unit.x)
    y = int(unit.y)
    ma = move_amount
    points = [
        (x, int(y + 2 * ma)),
        (x, int(y - 2 * ma)),
        (int(x + 2 * ma), y),
        (int(x - 2 * ma), y),
        (int(x + ma), int(y + ma)),
        (int(x - ma), int(y - ma)),
        (int(x + ma), int(y - ma)),
        (int(x - ma), int(y + ma)),
    ]
    if include_self:
        points.append((x, y))
    return points


def _surrounding_pathing(*, unit: TrackedUnit, context: HandlerContext) -> np.ndarray:
    points = _surrounding_points(unit=unit, move_amount=context.move_amount, include_self=False)
    vals = np.ones(len(points), dtype=np.float32)
    grid = context.pathing_grid
    if grid is None:
        return vals
    for idx, (x, y) in enumerate(points):
        if _check_bounds(x=x, y=y, context=context):
            vals[idx] = float(grid[x, y])
    return vals


def _surrounding_height(*, unit: TrackedUnit, context: HandlerContext) -> np.ndarray:
    points = _surrounding_points(unit=unit, move_amount=context.move_amount, include_self=True)
    vals = np.ones(len(points), dtype=np.float32)
    grid = context.terrain_height
    if grid is None:
        return vals
    for idx, (x, y) in enumerate(points):
        if _check_bounds(x=x, y=y, context=context):
            vals[idx] = float(grid[x, y])
    return vals


def _check_bounds(*, x: int, y: int, context: HandlerContext) -> bool:
    if context.pathing_grid is not None:
        return 0 <= x < context.pathing_grid.shape[0] and 0 <= y < context.pathing_grid.shape[1]
    return 0 <= x < int(context.map_x) and 0 <= y < int(context.map_y)


def _timestep_ratio(context: HandlerContext) -> float:
    env = context.env
    episode_limit = float(getattr(env, 'episode_limit', 0) or 0.0) if env is not None else 0.0
    if episode_limit <= 0.0:
        return 0.0
    return float(context.episode_step) / episode_limit
