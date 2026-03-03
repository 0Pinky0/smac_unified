from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from s2clientprotocol import common_pb2 as sc_common
from s2clientprotocol import query_pb2 as q_pb
from s2clientprotocol import raw_pb2 as r_pb
from s2clientprotocol import sc2api_pb2 as sc_pb

from ...players import OpponentStepContext
from ..types import HandlerContext, TrackedUnit, UnitFrame
from .base import ActionHandler

_ACTIONS = {
    'move': 16,
    'attack': 23,
    'stop': 4,
    'heal': 386,
}


class DefaultActionHandler(ActionHandler):
    def __init__(self):
        self._avail_actions_cache: dict[int, list[int]] = {}
        self._cache_step_token = -1

    def reset(
        self,
        *,
        frame: UnitFrame,
        context: HandlerContext,
    ) -> None:
        del frame, context
        self._avail_actions_cache.clear()
        self._cache_step_token = -1

    def get_avail_agent_actions(
        self,
        *,
        frame: UnitFrame,
        context: HandlerContext,
        agent_id: int,
    ) -> list[int]:
        if self._cache_step_token != frame.step_token:
            self._avail_actions_cache.clear()
            self._cache_step_token = frame.step_token

        cached = self._avail_actions_cache.get(agent_id)
        if cached is not None:
            return cached

        unit = frame.allies.units[agent_id]
        if not unit.alive:
            dead = [1] + [0] * (context.n_actions - 1)
            self._avail_actions_cache[agent_id] = dead
            return dead

        avail = [0] * context.n_actions
        avail[1] = 1  # stop
        if self._can_move(unit, context=context, dx=0.0, dy=context.move_amount):
            avail[2] = 1
        if self._can_move(unit, context=context, dx=0.0, dy=-context.move_amount):
            avail[3] = 1
        if self._can_move(unit, context=context, dx=context.move_amount, dy=0.0):
            avail[4] = 1
        if self._can_move(unit, context=context, dx=-context.move_amount, dy=0.0):
            avail[5] = 1

        targets = self._attack_targets(unit=unit, frame=frame, context=context)
        unit_range = self._unit_shoot_range(unit_type=unit.unit_type, context=context)
        for slot in range(context.attack_slots):
            if slot >= len(targets):
                break
            target = targets[slot]
            if not target.alive:
                continue
            dist = _distance(unit.x, unit.y, target.x, target.y)
            if dist <= unit_range:
                action_id = context.n_actions_no_attack + slot
                if action_id < context.n_actions:
                    avail[action_id] = 1

        self._avail_actions_cache[agent_id] = avail
        return avail

    def build_agent_action(
        self,
        *,
        frame: UnitFrame,
        context: HandlerContext,
        agent_id: int,
        action: int,
    ) -> sc_pb.Action | None:
        avail = self.get_avail_agent_actions(
            frame=frame,
            context=context,
            agent_id=agent_id,
        )
        if action < 0 or action >= context.n_actions or avail[action] == 0:
            return None

        unit = frame.allies.units[agent_id]
        if not unit.alive:
            return None

        tag = unit.tag
        x = unit.x
        y = unit.y
        cmd = None
        if action == 0:
            return None
        if action == 1:
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=_ACTIONS['stop'],
                unit_tags=[tag],
                queue_command=False,
            )
        elif action == 2:
            cmd = self._build_move_cmd(tag, x, y + context.move_amount)
        elif action == 3:
            cmd = self._build_move_cmd(tag, x, y - context.move_amount)
        elif action == 4:
            cmd = self._build_move_cmd(tag, x + context.move_amount, y)
        elif action == 5:
            cmd = self._build_move_cmd(tag, x - context.move_amount, y)
        else:
            slot = action - context.n_actions_no_attack
            targets = self._attack_targets(unit=unit, frame=frame, context=context)
            if slot >= len(targets):
                return None
            target = targets[slot]
            ability_id = (
                _ACTIONS['heal']
                if self._is_healer(unit=unit, context=context)
                else _ACTIONS['attack']
            )
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=ability_id,
                target_unit_tag=target.tag,
                unit_tags=[tag],
                queue_command=False,
            )
        return sc_pb.Action(action_raw=r_pb.ActionRaw(unit_command=cmd))

    def build_opponent_actions(
        self,
        *,
        frame: UnitFrame,
        context: HandlerContext,
        actions: Sequence[int],
        runtime: Any | None,
    ) -> Sequence[Any]:
        if (
            runtime is None
            or getattr(context.switches, 'opponent_mode', '') != 'scripted_pool'
        ):
            return []

        payload = {
            'agents': _raw_unit_dict(frame.enemies.units),
            'enemies': _raw_unit_dict(frame.allies.units),
            'agent_ability': self._query_enemy_abilities(context.env),
            'visible_matrix': self._fog_visibility_matrix(frame),
            'episode_step': context.episode_step,
        }
        runtime_ctx = OpponentStepContext(
            family=context.family,
            episode_step=context.episode_step,
            actions=list(actions),
            terminated=False,
            info={},
            payload=payload,
        )
        if hasattr(runtime, 'compute_actions'):
            try:
                return runtime.compute_actions(runtime_ctx)
            except Exception:
                return []
        return []

    @staticmethod
    def _build_move_cmd(tag: int, x: float, y: float):
        return r_pb.ActionRawUnitCommand(
            ability_id=_ACTIONS['move'],
            target_world_space_pos=sc_common.Point2D(x=x, y=y),
            unit_tags=[tag],
            queue_command=False,
        )

    @staticmethod
    def _query_enemy_abilities(env: Any):
        if env is None:
            return []
        inner_env = getattr(env, '_session', None)
        inner_env = getattr(inner_env, 'env', None)
        controllers = (
            getattr(inner_env, '_controllers', None)
            if inner_env is not None
            else None
        )
        if not controllers or len(controllers) < 2:
            return []
        try:
            query = q_pb.RequestQuery()
            enemies = getattr(env, 'enemies', {})
            for unit in enemies.values():
                ability = query.abilities.add()
                ability.unit_tag = unit.tag
            if len(query.abilities) == 0:
                return []
            result = controllers[1].query(query)
            return list(result.abilities)
        except Exception:
            return []

    @staticmethod
    def _fog_visibility_matrix(frame: UnitFrame):
        red_visible = [u.tag for u in frame.enemies.units if u.alive]
        blue_visible = [u.tag for u in frame.allies.units if u.alive]
        return {'red': red_visible, 'blue': blue_visible}

    @staticmethod
    def _can_move(
        unit: TrackedUnit,
        *,
        context: HandlerContext,
        dx: float,
        dy: float,
    ) -> bool:
        nx = unit.x + dx
        ny = unit.y + dy
        return 0 <= nx <= context.map_x and 0 <= ny <= context.map_y

    @staticmethod
    def _is_healer(unit: TrackedUnit, *, context: HandlerContext) -> bool:
        medivac_id = getattr(context.unit_type_ids, 'medivac_id', 0)
        return medivac_id > 0 and unit.unit_type == medivac_id

    def _attack_targets(
        self,
        *,
        unit: TrackedUnit,
        frame: UnitFrame,
        context: HandlerContext,
    ) -> list[TrackedUnit]:
        if self._is_healer(unit, context=context):
            medivac_id = getattr(context.unit_type_ids, 'medivac_id', 0)
            targets = [
                ally
                for ally in frame.allies.units
                if ally.alive and ally.unit_type != medivac_id
            ]
        else:
            targets = [enemy for enemy in frame.enemies.units if enemy.alive]
        return targets[: context.attack_slots]

    @staticmethod
    def _unit_shoot_range(*, unit_type: int, context: HandlerContext) -> float:
        ids = context.variant_logic.shoot_range_by_type(context.unit_type_ids)
        default_range = 6.0
        if unit_type in ids and ids[unit_type] > 0:
            return max(float(ids[unit_type]), 0.1)
        return default_range


def _distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return float(np.hypot(x2 - x1, y2 - y1))


def _raw_unit_dict(units: Sequence[TrackedUnit]) -> dict[int, Any]:
    payload = {}
    for unit in units:
        payload[unit.unit_id] = unit.raw if unit.raw is not None else unit
    return payload


class ClassicActionHandler(DefaultActionHandler):
    """Legacy SMAC action semantics (classic stop/move/attack indexing)."""


class ConicFovActionHandler(DefaultActionHandler):
    """SMACv2 conic-FOV action semantics (scaffolded variant)."""

    def __init__(
        self,
        *,
        num_fov_actions: int = 12,
        action_mask: bool = True,
    ):
        super().__init__()
        self.num_fov_actions = int(max(0, num_fov_actions))
        self.action_mask = bool(action_mask)


class AbilityAugmentedActionHandler(DefaultActionHandler):
    """SMAC-Hard ability-augmented action semantics (scaffolded variant)."""

    def __init__(self, *, use_ability: bool = True):
        super().__init__()
        self.use_ability = bool(use_ability)
