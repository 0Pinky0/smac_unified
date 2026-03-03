from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from s2clientprotocol import common_pb2 as sc_common
from s2clientprotocol import query_pb2 as q_pb
from s2clientprotocol import raw_pb2 as r_pb
from s2clientprotocol import sc2api_pb2 as sc_pb

from ..players import OpponentStepContext
from .base import (
    BuilderContext,
    BuildContext,
    NativeActionBuilder,
    NativeObservationBuilder,
    NativeRewardBuilder,
    NativeStateBuilder,
    ObservationBuilder,
    RewardBuilder,
    RewardContext,
    StateBuilder,
    TrackedUnit,
    UnitFrame,
)

_ACTIONS = {
    "move": 16,
    "attack": 23,
    "stop": 4,
    "heal": 386,
}


class DefaultObservationBuilder(ObservationBuilder):
    def build(
        self,
        *,
        raw_obs: Sequence[Any],
        context: BuildContext,
    ) -> np.ndarray:
        del context
        return np.asarray(raw_obs, dtype=np.float32)


class DefaultStateBuilder(StateBuilder):
    def build(
        self,
        *,
        raw_state: Sequence[Any],
        context: BuildContext,
    ) -> np.ndarray:
        del context
        return np.asarray(raw_state, dtype=np.float32)


class DefaultRewardBuilder(RewardBuilder):
    def __init__(self, scale_from_env: bool = False):
        self.scale_from_env = scale_from_env

    def build(
        self,
        *,
        raw_reward: float,
        context: RewardContext,
    ) -> float:
        reward = float(raw_reward)
        if not self.scale_from_env:
            return reward

        max_reward = float(getattr(context.env, "max_reward", 0.0) or 0.0)
        scale_rate = float(
            getattr(context.env, "reward_scale_rate", 0.0) or 0.0
        )
        if max_reward <= 0.0 or scale_rate <= 0.0:
            return reward
        return reward / (max_reward / scale_rate)


def builder_bundle(
    *,
    observation_builder: ObservationBuilder | None = None,
    state_builder: StateBuilder | None = None,
    reward_builder: RewardBuilder | None = None,
) -> Mapping[str, Any]:
    return {
        "observation_builder": observation_builder or DefaultObservationBuilder(),
        "state_builder": state_builder or DefaultStateBuilder(),
        "reward_builder": reward_builder or DefaultRewardBuilder(),
    }


class DefaultNativeActionBuilder(NativeActionBuilder):
    def __init__(self):
        self._avail_actions_cache: dict[int, list[int]] = {}
        self._cache_step_token = -1

    def reset(
        self,
        *,
        frame: UnitFrame,
        context: BuilderContext,
    ) -> None:
        del frame, context
        self._avail_actions_cache.clear()
        self._cache_step_token = -1

    def get_avail_agent_actions(
        self,
        *,
        frame: UnitFrame,
        context: BuilderContext,
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
        context: BuilderContext,
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
                ability_id=_ACTIONS["stop"],
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
                _ACTIONS["heal"]
                if self._is_healer(unit=unit, context=context)
                else _ACTIONS["attack"]
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
        context: BuilderContext,
        actions: Sequence[int],
        runtime: Any | None,
    ) -> Sequence[Any]:
        if (
            runtime is None
            or getattr(context.switches, "opponent_mode", "") != "scripted_pool"
        ):
            return []

        payload = {
            "agents": _raw_unit_dict(frame.enemies.units),
            "enemies": _raw_unit_dict(frame.allies.units),
            "agent_ability": self._query_enemy_abilities(context.env),
            "visible_matrix": self._fog_visibility_matrix(frame),
            "episode_step": context.episode_step,
        }
        runtime_ctx = OpponentStepContext(
            family=context.family,
            episode_step=context.episode_step,
            actions=list(actions),
            terminated=False,
            info={},
            payload=payload,
        )
        if hasattr(runtime, "compute_actions"):
            try:
                return runtime.compute_actions(runtime_ctx)
            except Exception:
                return []
        return []

    @staticmethod
    def _build_move_cmd(tag: int, x: float, y: float):
        return r_pb.ActionRawUnitCommand(
            ability_id=_ACTIONS["move"],
            target_world_space_pos=sc_common.Point2D(x=x, y=y),
            unit_tags=[tag],
            queue_command=False,
        )

    @staticmethod
    def _query_enemy_abilities(env: Any):
        if env is None:
            return []
        inner_env = getattr(env, "_session", None)
        inner_env = getattr(inner_env, "env", None)
        controllers = (
            getattr(inner_env, "_controllers", None)
            if inner_env is not None
            else None
        )
        if not controllers or len(controllers) < 2:
            return []
        try:
            query = q_pb.RequestQuery()
            enemies = getattr(env, "enemies", {})
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
        return {"red": red_visible, "blue": blue_visible}

    @staticmethod
    def _can_move(
        unit: TrackedUnit,
        *,
        context: BuilderContext,
        dx: float,
        dy: float,
    ) -> bool:
        nx = unit.x + dx
        ny = unit.y + dy
        return 0 <= nx <= context.map_x and 0 <= ny <= context.map_y

    @staticmethod
    def _is_healer(unit: TrackedUnit, *, context: BuilderContext) -> bool:
        medivac_id = getattr(context.unit_type_ids, "medivac_id", 0)
        return medivac_id > 0 and unit.unit_type == medivac_id

    def _attack_targets(
        self,
        *,
        unit: TrackedUnit,
        frame: UnitFrame,
        context: BuilderContext,
    ) -> list[TrackedUnit]:
        if self._is_healer(unit, context=context):
            medivac_id = getattr(context.unit_type_ids, "medivac_id", 0)
            targets = [
                ally
                for ally in frame.allies.units
                if ally.alive and ally.unit_type != medivac_id
            ]
        else:
            targets = [enemy for enemy in frame.enemies.units if enemy.alive]
        return targets[: context.attack_slots]

    @staticmethod
    def _unit_shoot_range(*, unit_type: int, context: BuilderContext) -> float:
        ids = context.variant_logic.shoot_range_by_type(context.unit_type_ids)
        default_range = 6.0
        if unit_type in ids and ids[unit_type] > 0:
            return max(float(ids[unit_type]), 0.1)
        return default_range


class DefaultNativeObservationBuilder(NativeObservationBuilder):
    def build_agent_obs(
        self,
        *,
        frame: UnitFrame,
        context: BuilderContext,
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
        context: BuilderContext,
    ):
        return [
            self.build_agent_obs(
                frame=frame,
                context=context,
                agent_id=agent_id,
            )
            for agent_id in range(context.n_agents)
        ]


class DefaultNativeStateBuilder(NativeStateBuilder):
    def build_state(
        self,
        *,
        frame: UnitFrame,
        context: BuilderContext,
    ):
        ally_state = np.zeros((context.n_agents, 5), dtype=np.float32)
        for unit in frame.allies.units:
            ally_state[unit.unit_id, 0] = unit.health / max(unit.health_max, 1.0)
            ally_state[unit.unit_id, 1] = unit.shield / max(unit.shield_max, 1.0)
            ally_state[unit.unit_id, 2] = unit.weapon_cooldown / 30.0
            ally_state[unit.unit_id, 3] = unit.x / max(context.map_x, 1.0)
            ally_state[unit.unit_id, 4] = unit.y / max(context.map_y, 1.0)

        enemy_state = np.zeros((context.n_enemies, 5), dtype=np.float32)
        for unit in frame.enemies.units:
            enemy_state[unit.unit_id, 0] = unit.health / max(unit.health_max, 1.0)
            enemy_state[unit.unit_id, 1] = unit.shield / max(unit.shield_max, 1.0)
            enemy_state[unit.unit_id, 2] = unit.weapon_cooldown / 30.0
            enemy_state[unit.unit_id, 3] = unit.x / max(context.map_x, 1.0)
            enemy_state[unit.unit_id, 4] = unit.y / max(context.map_y, 1.0)

        chunks = [ally_state.flatten(), enemy_state.flatten()]
        if context.state_last_action:
            chunks.append(np.asarray(context.last_action, dtype=np.float32).flatten())
        return np.concatenate(chunks, axis=0).astype(np.float32)


class DefaultNativeRewardBuilder(NativeRewardBuilder):
    def __init__(self):
        self._death_tracker_ally: np.ndarray | None = None
        self._death_tracker_enemy: np.ndarray | None = None

    def reset(
        self,
        *,
        frame: UnitFrame,
        context: BuilderContext,
    ) -> None:
        del context
        n_agents = len(frame.allies.units)
        n_enemies = len(frame.enemies.units)
        self._death_tracker_ally = np.zeros(n_agents, dtype=np.int8)
        self._death_tracker_enemy = np.zeros(n_enemies, dtype=np.int8)

    def build_step_reward(
        self,
        *,
        frame: UnitFrame,
        context: BuilderContext,
    ) -> float:
        if context.reward_sparse:
            return 0.0
        if self._death_tracker_ally is None or self._death_tracker_enemy is None:
            self.reset(frame=frame, context=context)

        assert self._death_tracker_ally is not None
        assert self._death_tracker_enemy is not None

        delta_deaths = 0.0
        delta_ally = 0.0
        delta_enemy = 0.0
        neg_scale = context.reward_negative_scale

        for agent_id, unit in enumerate(frame.allies.units):
            prev_health = frame.prev_allies_health[agent_id] + frame.prev_allies_shield[agent_id]
            if not unit.alive:
                if self._death_tracker_ally[agent_id] == 0 and prev_health > 0:
                    self._death_tracker_ally[agent_id] = 1
                    if not context.reward_only_positive:
                        delta_deaths -= context.reward_death_value * neg_scale
                    delta_ally += prev_health * neg_scale
            else:
                if self._death_tracker_ally[agent_id] == 0:
                    cur_health = unit.health + unit.shield
                    delta_ally += max(prev_health - cur_health, 0.0) * neg_scale

        for enemy_id, unit in enumerate(frame.enemies.units):
            prev_health = frame.prev_enemies_health[enemy_id] + frame.prev_enemies_shield[enemy_id]
            if not unit.alive:
                if self._death_tracker_enemy[enemy_id] == 0 and prev_health > 0:
                    self._death_tracker_enemy[enemy_id] = 1
                    delta_deaths += context.reward_death_value
                    delta_enemy += prev_health
            else:
                if self._death_tracker_enemy[enemy_id] == 0:
                    cur_health = unit.health + unit.shield
                    delta_enemy += max(prev_health - cur_health, 0.0)

        if context.reward_only_positive:
            reward = context.variant_logic.reward_positive_transform(
                delta_enemy + delta_deaths
            )
        else:
            reward = delta_enemy + delta_deaths - delta_ally
        if context.reward_scale and context.max_reward > 0 and context.reward_scale_rate > 0:
            reward /= context.max_reward / context.reward_scale_rate
        return float(reward)


def _distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return float(np.hypot(x2 - x1, y2 - y1))


def _unit_sight_range(
    *,
    agent_id: int,
    frame: UnitFrame,
    context: BuilderContext,
) -> float:
    unit = frame.allies.units[agent_id]
    ids = context.variant_logic.shoot_range_by_type(context.unit_type_ids)
    default_range = 6.0
    shoot_range = default_range
    if unit.unit_type in ids and ids[unit.unit_type] > 0:
        shoot_range = max(float(ids[unit.unit_type]), 0.1)
    return max(shoot_range + 3.0, 6.0)


def _obs_vector_size(context: BuilderContext) -> int:
    return 4 + context.attack_slots * 6 + max(context.n_agents - 1, 1) * 6 + 4


def _raw_unit_dict(units: Sequence[TrackedUnit]) -> dict[int, Any]:
    payload = {}
    for unit in units:
        payload[unit.unit_id] = unit.raw if unit.raw is not None else unit
    return payload
