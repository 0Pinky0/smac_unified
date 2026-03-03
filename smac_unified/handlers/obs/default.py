from __future__ import annotations

import numpy as np

from ..types import HandlerContext, UnitFrame
from .base import ObservationHandler


class DefaultObservationHandler(ObservationHandler):
    def build_agent_obs(
        self,
        *,
        frame: UnitFrame,
        context: HandlerContext,
        agent_id: int,
    ):
        unit = frame.allies.units[agent_id]
        if not unit.alive:
            return np.zeros(_obs_vector_size(context), dtype=np.float32)

        avail = (
            context.env.get_avail_agent_actions(agent_id)
            if context.env is not None
            else [0] * context.n_actions
        )
        move_feats = np.asarray(avail[2:6], dtype=np.float32)

        enemy_feats = np.zeros((context.attack_slots, 6), dtype=np.float32)
        enemies = list(frame.enemies.units)[: context.attack_slots]
        sight_range = _unit_sight_range(agent_id=agent_id, frame=frame, context=context)
        for slot, enemy in enumerate(enemies):
            if not enemy.alive:
                continue
            dist = _distance(unit.x, unit.y, enemy.x, enemy.y)
            enemy_feats[slot, 0] = 1.0
            enemy_feats[slot, 1] = float(avail[context.n_actions_no_attack + slot])
            enemy_feats[slot, 2] = dist / max(sight_range, 1.0)
            enemy_feats[slot, 3] = (enemy.x - unit.x) / max(context.max_distance_x, 1.0)
            enemy_feats[slot, 4] = (enemy.y - unit.y) / max(context.max_distance_y, 1.0)
            enemy_feats[slot, 5] = enemy.health / max(enemy.health_max, 1.0)

        ally_dim = max(context.n_agents - 1, 1)
        ally_feats = np.zeros((ally_dim, 6), dtype=np.float32)
        ally_slot = 0
        for ally in frame.allies.units:
            if ally.unit_id == agent_id:
                continue
            if ally_slot >= ally_dim:
                break
            if ally.alive:
                dist = _distance(unit.x, unit.y, ally.x, ally.y)
                ally_feats[ally_slot, 0] = 1.0
                ally_feats[ally_slot, 1] = dist / max(sight_range, 1.0)
                ally_feats[ally_slot, 2] = (ally.x - unit.x) / max(
                    context.max_distance_x, 1.0
                )
                ally_feats[ally_slot, 3] = (ally.y - unit.y) / max(
                    context.max_distance_y, 1.0
                )
                ally_feats[ally_slot, 4] = ally.health / max(ally.health_max, 1.0)
                ally_feats[ally_slot, 5] = ally.shield / max(ally.shield_max, 1.0)
            ally_slot += 1

        own_feats = np.asarray(
            [
                unit.health / max(unit.health_max, 1.0),
                unit.shield / max(unit.shield_max, 1.0),
                unit.x / max(context.map_x, 1.0),
                unit.y / max(context.map_y, 1.0),
            ],
            dtype=np.float32,
        )
        return np.concatenate(
            (
                move_feats.flatten(),
                enemy_feats.flatten(),
                ally_feats.flatten(),
                own_feats.flatten(),
            ),
            axis=0,
        )

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
    return max(shoot_range + 3.0, 6.0)


def _obs_vector_size(context: HandlerContext) -> int:
    return 4 + context.attack_slots * 6 + max(context.n_agents - 1, 1) * 6 + 4


class CapabilityObservationHandler(DefaultObservationHandler):
    """SMACv2 capability-aware observation semantics (scaffolded variant)."""
