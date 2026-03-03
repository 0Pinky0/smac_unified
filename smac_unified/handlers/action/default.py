from __future__ import annotations

import math
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

_ABILITY_DICT = {
    380: ('self', 3675, 0.0),  # stim
    1664: ('ally', 1664, 7.0),  # transfusion
    2393: ('self', 2393, 0.0),  # prismatic alignment
    1998: ('self', 1998, 0.0),  # morph hellbat
    1978: ('self', 1978, 0.0),  # morph hellion
    2588: ('point', 2588, 5.0),  # KD8
    1442: ('point', 3687, 8.0),  # blink
    388: ('self', 388, 0.0),  # siege
    253: ('self', 3675, 0.0),  # marauder stim
    2116: ('self', 2116, 0.0),  # medivac boost
    390: ('self', 390, 0.0),  # unsiege
}


class DefaultActionHandler(ActionHandler):
    """Classic SMAC action semantics with unit-frame inputs."""

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
        avail[1] = 1
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
            if self._can_shoot_target(
                agent_id=agent_id,
                unit=unit,
                target=target,
                frame=frame,
                context=context,
                unit_range=unit_range,
            ):
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
            if slot < 0 or slot >= len(targets):
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
        if context.pathing_grid is not None:
            mx = int(unit.x + dx * 0.5)
            my = int(unit.y + dy * 0.5)
            if not _in_bounds(mx, my, context.pathing_grid.shape[0], context.pathing_grid.shape[1]):
                return False
            return bool(context.pathing_grid[mx, my])
        return 0 <= nx <= context.map_x and 0 <= ny <= context.map_y

    @staticmethod
    def _is_healer(unit: TrackedUnit, *, context: HandlerContext) -> bool:
        medivac_id = getattr(context.unit_type_ids, 'medivac_id', 0)
        return medivac_id > 0 and unit.unit_type == medivac_id

    def _can_shoot_target(
        self,
        *,
        agent_id: int,
        unit: TrackedUnit,
        target: TrackedUnit,
        frame: UnitFrame,
        context: HandlerContext,
        unit_range: float,
    ) -> bool:
        del agent_id, frame, context
        dist = _distance(unit.x, unit.y, target.x, target.y)
        return dist <= unit_range

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


class ClassicActionHandler(DefaultActionHandler):
    """Legacy SMAC action semantics (classic stop/move/attack indexing)."""


class ConicFovActionHandler(DefaultActionHandler):
    """SMACv2 conic-FOV action semantics for core training paths."""

    def __init__(
        self,
        *,
        num_fov_actions: int = 12,
        action_mask: bool = True,
    ):
        super().__init__()
        self.num_fov_actions = int(max(0, num_fov_actions))
        self.action_mask = bool(action_mask)

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
        avail[1] = 1
        if self._can_move(unit, context=context, dx=0.0, dy=context.move_amount):
            avail[2] = 1
        if self._can_move(unit, context=context, dx=0.0, dy=-context.move_amount):
            avail[3] = 1
        if self._can_move(unit, context=context, dx=context.move_amount, dy=0.0):
            avail[4] = 1
        if self._can_move(unit, context=context, dx=-context.move_amount, dy=0.0):
            avail[5] = 1

        fov_actions = int(max(context.n_fov_actions, self.num_fov_actions))
        fov_stop = min(6 + fov_actions, context.n_actions_no_attack)
        for action_id in range(6, fov_stop):
            avail[action_id] = 1

        targets = self._attack_targets(unit=unit, frame=frame, context=context)
        unit_range = self._unit_shoot_range(unit_type=unit.unit_type, context=context)
        for slot in range(context.attack_slots):
            if slot >= len(targets):
                break
            target = targets[slot]
            if not target.alive:
                continue
            can_shoot = (
                True
                if not (self.action_mask and context.action_mask)
                else self._is_position_in_cone(
                    agent_id=agent_id,
                    frame=frame,
                    context=context,
                    target=target,
                    unit_range=unit_range,
                )
            )
            if can_shoot:
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

        if action in range(6, context.n_actions_no_attack):
            if (
                context.fov_directions is not None
                and context.canonical_fov_directions is not None
                and len(context.canonical_fov_directions) > 0
            ):
                fov_idx = action - 6
                if 0 <= fov_idx < len(context.canonical_fov_directions):
                    context.fov_directions[agent_id] = context.canonical_fov_directions[fov_idx]
            return None

        return super().build_agent_action(
            frame=frame,
            context=context,
            agent_id=agent_id,
            action=action,
        )

    def _is_position_in_cone(
        self,
        *,
        agent_id: int,
        frame: UnitFrame,
        context: HandlerContext,
        target: TrackedUnit,
        unit_range: float,
    ) -> bool:
        source = frame.allies.units[agent_id]
        dx = target.x - source.x
        dy = target.y - source.y
        dist = _distance(source.x, source.y, target.x, target.y)
        if dist > unit_range:
            return False
        if dist <= 1e-9:
            return True
        if (
            context.fov_directions is None
            or agent_id >= len(context.fov_directions)
            or context.conic_fov_angle <= 0.0
        ):
            return True
        fov_vec = context.fov_directions[agent_id]
        fov_norm = float(np.linalg.norm(fov_vec))
        if fov_norm <= 1e-9:
            return True
        cos_theta = float((dx * fov_vec[0] + dy * fov_vec[1]) / (dist * fov_norm))
        cos_threshold = math.cos(context.conic_fov_angle / 2.0)
        return cos_theta >= cos_threshold


class AbilityAugmentedActionHandler(DefaultActionHandler):
    """SMAC-Hard ability-augmented action semantics for core training paths."""

    def __init__(self, *, use_ability: bool = True):
        super().__init__()
        self.use_ability = bool(use_ability)
        self._agent_ability_cache: dict[int, tuple[str, int, float]] = {}

    def reset(
        self,
        *,
        frame: UnitFrame,
        context: HandlerContext,
    ) -> None:
        super().reset(frame=frame, context=context)
        self._agent_ability_cache.clear()

    def get_avail_agent_actions(
        self,
        *,
        frame: UnitFrame,
        context: HandlerContext,
        agent_id: int,
    ) -> list[int]:
        if self._cache_step_token != frame.step_token:
            self._avail_actions_cache.clear()
            self._agent_ability_cache.clear()
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
        avail[1] = 1
        if self._can_move(unit, context=context, dx=0.0, dy=context.move_amount):
            avail[2] = 1
        if self._can_move(unit, context=context, dx=0.0, dy=-context.move_amount):
            avail[3] = 1
        if self._can_move(unit, context=context, dx=context.move_amount, dy=0.0):
            avail[4] = 1
        if self._can_move(unit, context=context, dx=-context.move_amount, dy=0.0):
            avail[5] = 1

        padding = int(max(context.ability_padding, context.attack_slots))
        targets = self._attack_targets(unit=unit, frame=frame, context=context)
        unit_range = self._unit_shoot_range(unit_type=unit.unit_type, context=context)
        for slot in range(min(padding, len(targets))):
            target = targets[slot]
            if not target.alive:
                continue
            if self._can_shoot_target(
                agent_id=agent_id,
                unit=unit,
                target=target,
                frame=frame,
                context=context,
                unit_range=unit_range,
            ):
                action_id = context.n_actions_no_attack + slot
                if action_id < context.n_actions:
                    avail[action_id] = 1

        if self.use_ability and context.use_ability:
            abilities_by_agent = self._query_agent_abilities(context.env)
            for ability_id in abilities_by_agent.get(agent_id, ()):
                if ability_id not in _ABILITY_DICT:
                    continue
                perform, ability_code, ability_range = _ABILITY_DICT[ability_id]
                self._agent_ability_cache[agent_id] = (perform, ability_code, ability_range)
                mask = self._ability_mask(
                    frame=frame,
                    context=context,
                    agent_id=agent_id,
                    perform=perform,
                    ability_range=ability_range,
                    padding=padding,
                )
                start = context.n_actions_no_attack + padding
                for offset, flag in enumerate(mask):
                    idx = start + offset
                    if idx < context.n_actions:
                        avail[idx] = flag
                break

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

        padding = int(max(context.ability_padding, context.attack_slots))
        ability_start = context.n_actions_no_attack + padding
        if action < ability_start:
            return super().build_agent_action(
                frame=frame,
                context=context,
                agent_id=agent_id,
                action=action,
            )

        ability = self._agent_ability_cache.get(agent_id)
        if ability is None:
            return None
        perform, ability_code, ability_range = ability
        unit = frame.allies.units[agent_id]
        tag = unit.tag
        x = unit.x
        y = unit.y
        target_id = action - ability_start

        if perform == 'self':
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=ability_code,
                unit_tags=[tag],
                queue_command=False,
            )
            return sc_pb.Action(action_raw=r_pb.ActionRaw(unit_command=cmd))

        env = context.env
        if env is None:
            return None

        if perform == 'ally':
            target_unit = getattr(env, 'agents', {}).get(target_id)
            if target_unit is None:
                return None
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=ability_code,
                target_unit_tag=target_unit.tag,
                unit_tags=[tag],
                queue_command=False,
            )
            return sc_pb.Action(action_raw=r_pb.ActionRaw(unit_command=cmd))

        if perform == 'enemy':
            target_unit = getattr(env, 'enemies', {}).get(target_id)
            if target_unit is None:
                return None
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=ability_code,
                target_unit_tag=target_unit.tag,
                unit_tags=[tag],
                queue_command=False,
            )
            return sc_pb.Action(action_raw=r_pb.ActionRaw(unit_command=cmd))

        if perform == 'point':
            radius = int(math.ceil(float(ability_range) / 2.0))
            points = _legacy_point_targets(x=x, y=y, radius=radius)
            if target_id < 0 or target_id >= len(points):
                return None
            tx, ty = points[target_id]
            if not _in_bounds(int(tx), int(ty), int(context.map_x), int(context.map_y)):
                return None
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=ability_code,
                target_world_space_pos=sc_common.Point2D(x=float(tx), y=float(ty)),
                unit_tags=[tag],
                queue_command=False,
            )
            return sc_pb.Action(action_raw=r_pb.ActionRaw(unit_command=cmd))

        return None

    @staticmethod
    def _query_agent_abilities(env: Any) -> dict[int, tuple[int, ...]]:
        if env is None:
            return {}
        inner_env = getattr(env, '_session', None)
        inner_env = getattr(inner_env, 'env', None)
        controllers = (
            getattr(inner_env, '_controllers', None)
            if inner_env is not None
            else None
        )
        if not controllers:
            return {}
        try:
            query = q_pb.RequestQuery()
            agents = getattr(env, 'agents', {})
            tag_to_agent = {}
            for agent_id, unit in agents.items():
                ability = query.abilities.add()
                ability.unit_tag = unit.tag
                tag_to_agent[unit.tag] = agent_id
            if len(query.abilities) == 0:
                return {}
            result = controllers[0].query(query)
            mapping: dict[int, tuple[int, ...]] = {}
            for payload in result.abilities:
                agent_id = tag_to_agent.get(payload.unit_tag)
                if agent_id is None:
                    continue
                ability_ids = tuple(item.ability_id for item in payload.abilities)
                mapping[agent_id] = ability_ids
            return mapping
        except Exception:
            return {}

    @staticmethod
    def _ability_mask(
        *,
        frame: UnitFrame,
        context: HandlerContext,
        agent_id: int,
        perform: str,
        ability_range: float,
        padding: int,
    ) -> list[int]:
        del frame
        mask = [0] * max(padding, 0)
        if len(mask) == 0:
            return mask
        env = context.env
        if env is None:
            return mask
        unit = getattr(env, 'agents', {}).get(agent_id)
        if unit is None:
            return mask
        ux, uy = _unit_xy(unit)
        if perform == 'self':
            mask[0] = 1
            return mask
        if perform == 'ally':
            for ally_id, ally in getattr(env, 'agents', {}).items():
                if ally_id >= len(mask):
                    continue
                if ally.tag == unit.tag:
                    continue
                health = float(getattr(ally, 'health', 0.0))
                health_max = float(getattr(ally, 'health_max', 0.0))
                ax, ay = _unit_xy(ally)
                if health < health_max and _distance(ux, uy, ax, ay) < ability_range:
                    mask[ally_id] = 1
            return mask
        if perform == 'enemy':
            for enemy_id, enemy in getattr(env, 'enemies', {}).items():
                if enemy_id >= len(mask):
                    continue
                ex, ey = _unit_xy(enemy)
                if _distance(ux, uy, ex, ey) < ability_range:
                    mask[enemy_id] = 1
            return mask
        if perform == 'point':
            radius = int(math.ceil(float(ability_range) / 2.0))
            points = _legacy_point_targets(x=ux, y=uy, radius=radius)
            for idx, (px, py) in enumerate(points):
                if idx >= len(mask):
                    break
                if _in_bounds(int(px), int(py), int(context.map_x), int(context.map_y)):
                    mask[idx] = 1
            return mask
        return mask


def _distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return float(np.hypot(x2 - x1, y2 - y1))


def _raw_unit_dict(units: Sequence[TrackedUnit]) -> dict[int, Any]:
    payload = {}
    for unit in units:
        payload[unit.unit_id] = unit.raw if unit.raw is not None else unit
    return payload


def _unit_xy(unit: Any) -> tuple[float, float]:
    if hasattr(unit, 'x') and hasattr(unit, 'y'):
        return float(unit.x), float(unit.y)
    pos = getattr(unit, 'pos', None)
    if pos is not None and hasattr(pos, 'x') and hasattr(pos, 'y'):
        return float(pos.x), float(pos.y)
    return 0.0, 0.0


def _in_bounds(x: int, y: int, map_x: int, map_y: int) -> bool:
    return 0 <= x < map_x and 0 <= y < map_y


def _legacy_point_targets(*, x: float, y: float, radius: int) -> list[tuple[float, float]]:
    r = float(radius)
    half = r / 2.0
    return [
        (x, y),
        (x, y + r),
        (x + half, y + half),
        (x + r, y),
        (x + half, y - half),
        (x, y - half),
        (x - half, y - half),
        (x - r, y),
        (x - half, y + half),
    ]
