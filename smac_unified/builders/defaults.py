from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from s2clientprotocol import common_pb2 as sc_common
from s2clientprotocol import query_pb2 as q_pb
from s2clientprotocol import raw_pb2 as r_pb
from s2clientprotocol import sc2api_pb2 as sc_pb

from ..opponents import OpponentStepContext
from .base import (
    BuildContext,
    NativeActionBuilder,
    NativeObservationBuilder,
    NativeRewardBuilder,
    NativeStateBuilder,
    ObservationBuilder,
    RewardBuilder,
    RewardContext,
    StateBuilder,
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

    def reset(self, env: Any = None, *args, **kwargs) -> None:
        del env, args, kwargs
        self._avail_actions_cache.clear()

    def get_avail_agent_actions(self, env: Any, agent_id: int) -> list[int]:
        cached = self._avail_actions_cache.get(agent_id)
        if cached is not None:
            return cached

        unit = env.agents.get(agent_id)
        if unit is None or unit.health <= 0:
            dead = [1] + [0] * (env.n_actions - 1)
            self._avail_actions_cache[agent_id] = dead
            return dead

        avail = [0] * env.n_actions
        avail[1] = 1  # stop
        if env._can_move(unit, dx=0.0, dy=env._move_amount):
            avail[2] = 1
        if env._can_move(unit, dx=0.0, dy=-env._move_amount):
            avail[3] = 1
        if env._can_move(unit, dx=env._move_amount, dy=0.0):
            avail[4] = 1
        if env._can_move(unit, dx=-env._move_amount, dy=0.0):
            avail[5] = 1

        targets = env._attack_targets_for_unit(unit)
        unit_range = env._unit_shoot_range(unit)
        for slot in range(env._attack_slots):
            if slot >= len(targets):
                break
            target = targets[slot]
            if target.health <= 0:
                continue
            dist = env._distance(unit.pos.x, unit.pos.y, target.pos.x, target.pos.y)
            if dist <= unit_range:
                action_id = env.n_actions_no_attack + slot
                if action_id < env.n_actions:
                    avail[action_id] = 1

        self._avail_actions_cache[agent_id] = avail
        return avail

    def build_agent_action(
        self,
        env: Any,
        agent_id: int,
        action: int,
    ) -> sc_pb.Action | None:
        avail = self.get_avail_agent_actions(env, agent_id)
        if action < 0 or action >= env.n_actions or avail[action] == 0:
            return None
        unit = env.agents.get(agent_id)
        if unit is None or unit.health <= 0:
            return None

        tag = unit.tag
        x = unit.pos.x
        y = unit.pos.y
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
            cmd = self._build_move_cmd(tag, x, y + env._move_amount)
        elif action == 3:
            cmd = self._build_move_cmd(tag, x, y - env._move_amount)
        elif action == 4:
            cmd = self._build_move_cmd(tag, x + env._move_amount, y)
        elif action == 5:
            cmd = self._build_move_cmd(tag, x - env._move_amount, y)
        else:
            slot = action - env.n_actions_no_attack
            targets = env._attack_targets_for_unit(unit)
            if slot >= len(targets):
                return None
            target = targets[slot]
            ability_id = (
                _ACTIONS["heal"]
                if env._is_healer(unit)
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
        env: Any,
        actions: Sequence[int],
        runtime: Any | None,
    ) -> Sequence[Any]:
        if (
            runtime is None
            or env.switches.opponent_mode != "scripted_pool"
            or env._session.num_agents < 2
        ):
            return []

        payload = {
            "agents": env.enemies,
            "enemies": env.agents,
            "agent_ability": self._query_enemy_abilities(env),
            "visible_matrix": self._fog_visibility_matrix(env),
            "episode_step": env._episode_steps,
        }
        context = OpponentStepContext(
            family=env.variant,
            episode_step=env._episode_steps,
            actions=list(actions),
            terminated=False,
            info={},
            payload=payload,
        )
        if hasattr(runtime, "compute_actions"):
            try:
                return runtime.compute_actions(context)
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
        inner_env = env._session.env
        controllers = (
            getattr(inner_env, "_controllers", None)
            if inner_env is not None
            else None
        )
        if not controllers or len(controllers) < 2:
            return []
        try:
            query = q_pb.RequestQuery()
            for unit in env.enemies.values():
                ability = query.abilities.add()
                ability.unit_tag = unit.tag
            if len(query.abilities) == 0:
                return []
            result = controllers[1].query(query)
            return list(result.abilities)
        except Exception:
            return []

    @staticmethod
    def _fog_visibility_matrix(env: Any):
        red_visible = [unit.tag for unit in env.enemies.values() if unit.health > 0]
        blue_visible = [unit.tag for unit in env.agents.values() if unit.health > 0]
        return {"red": red_visible, "blue": blue_visible}


class DefaultNativeObservationBuilder(NativeObservationBuilder):
    def build_agent_obs(self, env: Any, agent_id: int):
        unit = env.agents.get(agent_id)
        if unit is None:
            return np.zeros(env.get_obs_size(), dtype=np.float32)

        avail = env.get_avail_agent_actions(agent_id)
        move_feats = np.asarray(avail[2:6], dtype=np.float32)

        enemy_feats = np.zeros((env._attack_slots, 6), dtype=np.float32)
        enemy_items = list(env.enemies.items())[: env._attack_slots]
        sight_range = env._unit_sight_range(agent_id)
        for slot, (_, enemy) in enumerate(enemy_items):
            if enemy.health <= 0:
                continue
            dist = env._distance(unit.pos.x, unit.pos.y, enemy.pos.x, enemy.pos.y)
            enemy_feats[slot, 0] = 1.0
            enemy_feats[slot, 1] = float(avail[env.n_actions_no_attack + slot])
            enemy_feats[slot, 2] = dist / max(sight_range, 1.0)
            enemy_feats[slot, 3] = (enemy.pos.x - unit.pos.x) / max(
                env.max_distance_x, 1.0
            )
            enemy_feats[slot, 4] = (enemy.pos.y - unit.pos.y) / max(
                env.max_distance_y, 1.0
            )
            enemy_feats[slot, 5] = enemy.health / max(enemy.health_max, 1.0)

        ally_dim = max(env.n_agents - 1, 1)
        ally_feats = np.zeros((ally_dim, 6), dtype=np.float32)
        ally_slot = 0
        for ally_id, ally in env.agents.items():
            if ally_id == agent_id:
                continue
            if ally_slot >= ally_dim:
                break
            if ally.health > 0:
                dist = env._distance(unit.pos.x, unit.pos.y, ally.pos.x, ally.pos.y)
                ally_feats[ally_slot, 0] = 1.0
                ally_feats[ally_slot, 1] = dist / max(sight_range, 1.0)
                ally_feats[ally_slot, 2] = (ally.pos.x - unit.pos.x) / max(
                    env.max_distance_x, 1.0
                )
                ally_feats[ally_slot, 3] = (ally.pos.y - unit.pos.y) / max(
                    env.max_distance_y, 1.0
                )
                ally_feats[ally_slot, 4] = ally.health / max(ally.health_max, 1.0)
                ally_feats[ally_slot, 5] = ally.shield / max(ally.shield_max, 1.0)
            ally_slot += 1

        own_feats = np.asarray(
            [
                unit.health / max(unit.health_max, 1.0),
                unit.shield / max(unit.shield_max, 1.0),
                unit.pos.x / max(env.map_x, 1.0),
                unit.pos.y / max(env.map_y, 1.0),
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

    def build_obs(self, env: Any):
        return [
            self.build_agent_obs(env, agent_id)
            for agent_id in range(env.n_agents)
        ]


class DefaultNativeStateBuilder(NativeStateBuilder):
    def build_state(self, env: Any):
        ally_state = np.zeros((env.n_agents, 5), dtype=np.float32)
        for agent_id, unit in env.agents.items():
            ally_state[agent_id, 0] = unit.health / max(unit.health_max, 1.0)
            ally_state[agent_id, 1] = unit.shield / max(unit.shield_max, 1.0)
            ally_state[agent_id, 2] = unit.weapon_cooldown / 30.0
            ally_state[agent_id, 3] = unit.pos.x / max(env.map_x, 1.0)
            ally_state[agent_id, 4] = unit.pos.y / max(env.map_y, 1.0)

        enemy_state = np.zeros((env.n_enemies, 5), dtype=np.float32)
        for enemy_id, unit in env.enemies.items():
            enemy_state[enemy_id, 0] = unit.health / max(unit.health_max, 1.0)
            enemy_state[enemy_id, 1] = unit.shield / max(unit.shield_max, 1.0)
            enemy_state[enemy_id, 2] = unit.weapon_cooldown / 30.0
            enemy_state[enemy_id, 3] = unit.pos.x / max(env.map_x, 1.0)
            enemy_state[enemy_id, 4] = unit.pos.y / max(env.map_y, 1.0)

        chunks = [ally_state.flatten(), enemy_state.flatten()]
        if env.state_last_action:
            chunks.append(env.last_action.flatten())
        return np.concatenate(chunks, axis=0).astype(np.float32)


class DefaultNativeRewardBuilder(NativeRewardBuilder):
    def __init__(self):
        self._prev_ally_health: np.ndarray | None = None
        self._prev_ally_shield: np.ndarray | None = None
        self._prev_enemy_health: np.ndarray | None = None
        self._prev_enemy_shield: np.ndarray | None = None
        self._death_tracker_ally: np.ndarray | None = None
        self._death_tracker_enemy: np.ndarray | None = None

    def reset(self, env: Any, *args, **kwargs) -> None:
        del args, kwargs
        self._prev_ally_health = np.zeros(env.n_agents, dtype=np.float32)
        self._prev_ally_shield = np.zeros(env.n_agents, dtype=np.float32)
        self._prev_enemy_health = np.zeros(env.n_enemies, dtype=np.float32)
        self._prev_enemy_shield = np.zeros(env.n_enemies, dtype=np.float32)
        self._death_tracker_ally = np.zeros(env.n_agents, dtype=np.int8)
        self._death_tracker_enemy = np.zeros(env.n_enemies, dtype=np.int8)
        for agent_id, unit in env.agents.items():
            self._prev_ally_health[agent_id] = unit.health
            self._prev_ally_shield[agent_id] = unit.shield
        for enemy_id, unit in env.enemies.items():
            self._prev_enemy_health[enemy_id] = unit.health
            self._prev_enemy_shield[enemy_id] = unit.shield

    def build_step_reward(self, env: Any) -> float:
        if env.reward_sparse:
            return 0.0

        assert self._prev_ally_health is not None
        assert self._prev_ally_shield is not None
        assert self._prev_enemy_health is not None
        assert self._prev_enemy_shield is not None
        assert self._death_tracker_ally is not None
        assert self._death_tracker_enemy is not None

        delta_deaths = 0.0
        delta_ally = 0.0
        delta_enemy = 0.0
        neg_scale = env.reward_negative_scale

        for agent_id in range(env.n_agents):
            unit = env.agents.get(agent_id)
            prev_health = (
                self._prev_ally_health[agent_id] + self._prev_ally_shield[agent_id]
            )
            if unit is None or unit.health <= 0:
                if self._death_tracker_ally[agent_id] == 0 and prev_health > 0:
                    self._death_tracker_ally[agent_id] = 1
                    if not env.reward_only_positive:
                        delta_deaths -= env.reward_death_value * neg_scale
                    delta_ally += prev_health * neg_scale
            else:
                if self._death_tracker_ally[agent_id] == 0:
                    cur_health = unit.health + unit.shield
                    delta_ally += max(prev_health - cur_health, 0.0) * neg_scale

            if unit is not None:
                self._prev_ally_health[agent_id] = unit.health
                self._prev_ally_shield[agent_id] = unit.shield
            else:
                self._prev_ally_health[agent_id] = 0.0
                self._prev_ally_shield[agent_id] = 0.0

        for enemy_id in range(env.n_enemies):
            unit = env.enemies.get(enemy_id)
            prev_health = (
                self._prev_enemy_health[enemy_id] + self._prev_enemy_shield[enemy_id]
            )
            if unit is None or unit.health <= 0:
                if self._death_tracker_enemy[enemy_id] == 0 and prev_health > 0:
                    self._death_tracker_enemy[enemy_id] = 1
                    delta_deaths += env.reward_death_value
                    delta_enemy += prev_health
            else:
                if self._death_tracker_enemy[enemy_id] == 0:
                    cur_health = unit.health + unit.shield
                    delta_enemy += max(prev_health - cur_health, 0.0)

            if unit is not None:
                self._prev_enemy_health[enemy_id] = unit.health
                self._prev_enemy_shield[enemy_id] = unit.shield
            else:
                self._prev_enemy_health[enemy_id] = 0.0
                self._prev_enemy_shield[enemy_id] = 0.0

        if env.reward_only_positive:
            reward = env._variant_logic.reward_positive_transform(
                delta_enemy + delta_deaths
            )
        else:
            reward = delta_enemy + delta_deaths - delta_ally
        return float(reward)
