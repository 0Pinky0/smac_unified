from __future__ import annotations

import math
from operator import attrgetter
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
from s2clientprotocol import common_pb2 as sc_common
from s2clientprotocol import query_pb2 as q_pb
from s2clientprotocol import raw_pb2 as r_pb
from s2clientprotocol import sc2api_pb2 as sc_pb

from ..config import VariantSwitches, merge_switches
from ..maps import MapParams, get_map_params
from ..opponents import OpponentEpisodeContext, OpponentRuntime, OpponentStepContext
from .session_sc2env import SC2EnvRawSession, SC2SessionConfig
from .variants import UnitTypeIds, VariantLogic, build_variant_logic


ACTIONS = {
    "move": 16,
    "attack": 23,
    "stop": 4,
    "heal": 386,
}


class NativeStarCraft2Env:
    """Standalone native unified SMAC-family environment.

    This native implementation is intentionally modular and conservative:
    - backend transport/session logic comes from pysc2-compat SC2Env raw mode,
    - family differences are routed through variant logic switches,
    - unsupported long-tail variant details are safely degraded to defaults.
    """

    def __init__(
        self,
        *,
        variant: str,
        map_name: str,
        capability_config: Mapping[str, Any] | None = None,
        env_kwargs: Mapping[str, Any] | None = None,
        source_root: str | None = None,
        native_options: Mapping[str, Any] | None = None,
    ):
        self.variant = variant
        self.map_name = map_name
        self.capability_config = dict(capability_config or {})
        self._env_kwargs = dict(env_kwargs or {})
        self._native_options = dict(native_options or {})
        self._source_root = source_root

        self.map_params: MapParams = get_map_params(map_name)
        raw_switches = self._env_kwargs.get("logic_switches")
        if isinstance(raw_switches, VariantSwitches):
            self.switches = raw_switches
        else:
            self.switches = merge_switches(variant, raw_switches)
        self._variant_logic: VariantLogic = build_variant_logic(
            self.switches,
            self.map_params,
        )

        self.n_agents = self.map_params.n_agents
        self.n_enemies = self.map_params.n_enemies
        self.episode_limit = int(
            self._env_kwargs.get("episode_limit", self.map_params.limit)
        )
        self._move_amount = float(self._env_kwargs.get("move_amount", 2.0))

        self.reward_sparse = bool(self._env_kwargs.get("reward_sparse", False))
        self.reward_only_positive = bool(
            self._env_kwargs.get("reward_only_positive", True)
        )
        self.reward_death_value = float(
            self._env_kwargs.get("reward_death_value", 10.0)
        )
        self.reward_win = float(self._env_kwargs.get("reward_win", 200.0))
        self.reward_defeat = float(self._env_kwargs.get("reward_defeat", 0.0))
        self.reward_negative_scale = float(
            self._env_kwargs.get("reward_negative_scale", 0.5)
        )
        self.reward_scale = bool(self._env_kwargs.get("reward_scale", True))
        self.reward_scale_rate = float(
            self._env_kwargs.get("reward_scale_rate", 20.0)
        )
        self.state_last_action = bool(
            self._env_kwargs.get("state_last_action", True)
        )
        self._seed = self._env_kwargs.get("seed")

        self._attack_slots = max(self.n_agents, self.n_enemies)
        self.n_actions = self._variant_logic.n_actions(
            n_agents=self.n_agents,
            n_enemies=self.n_enemies,
        )
        self.n_actions_no_attack = 6

        self._session = self._build_session()
        self._opponent_runtime: OpponentRuntime | None = None

        self._episode_steps = 0
        self._total_steps = 0
        self._episode_count = 0
        self._obs = None
        self._opponent_obs = None
        self._latest_timesteps = []

        self.map_x = 32.0
        self.map_y = 32.0
        self.max_distance_x = 32.0
        self.max_distance_y = 32.0

        self.agents: Dict[int, Any] = {}
        self.enemies: Dict[int, Any] = {}
        self._unit_ids = UnitTypeIds()
        self.max_reward = 0.0
        self.reward = 0.0
        self.last_action = np.zeros((self.n_agents, self.n_actions), dtype=np.float32)
        self._action_eye = np.eye(self.n_actions, dtype=np.float32)

        self.death_tracker_ally = np.zeros(self.n_agents, dtype=np.int8)
        self.death_tracker_enemy = np.zeros(self.n_enemies, dtype=np.int8)
        self.prev_ally_health = np.zeros(self.n_agents, dtype=np.float32)
        self.prev_ally_shield = np.zeros(self.n_agents, dtype=np.float32)
        self.prev_enemy_health = np.zeros(self.n_enemies, dtype=np.float32)
        self.prev_enemy_shield = np.zeros(self.n_enemies, dtype=np.float32)
        self._avail_actions_cache: Dict[int, List[int]] = {}

    def _build_session(self) -> SC2EnvRawSession:
        cfg = SC2SessionConfig(
            map_name=self.map_name,
            map_params=self.map_params,
            step_mul=int(self._env_kwargs.get("step_mul", 8)),
            difficulty=str(self._env_kwargs.get("difficulty", "7")),
            seed=self._seed,
            realtime=bool(self._env_kwargs.get("realtime", False)),
            ensure_available_actions=bool(
                self._native_options.get("ensure_available_actions", True)
            ),
            pipeline_actions_and_step=bool(
                self._native_options.get("pipeline_actions_and_step", False)
            ),
            pipeline_step_and_observe=bool(
                self._native_options.get("pipeline_step_and_observe", False)
            ),
            reuse_step_observe_requests=bool(
                self._native_options.get("reuse_step_observe_requests", False)
            ),
            opponent_mode=self.switches.opponent_mode,
            enable_dual_controller=bool(
                self._native_options.get("enable_dual_controller", False)
            ),
            game_version=self._env_kwargs.get("game_version"),
            source_root=self._source_root,
        )
        return SC2EnvRawSession(cfg)

    def set_opponent_runtime(self, runtime: OpponentRuntime) -> None:
        self._opponent_runtime = runtime
        runtime.bind_env(self, self.variant)

    def seed(self, seed: int | None = None):
        self._seed = seed
        if self._session is not None:
            self._session.config.seed = seed

    def reset(self, episode_config: Mapping[str, Any] | None = None, **kwargs):
        del kwargs
        if episode_config and isinstance(episode_config, Mapping):
            self.capability_config.update(dict(episode_config))

        self._episode_steps = 0
        self.reward = 0.0
        self.death_tracker_ally.fill(0)
        self.death_tracker_enemy.fill(0)
        self.last_action.fill(0.0)
        self._avail_actions_cache.clear()

        self._latest_timesteps = self._session.reset()
        self._obs = self._latest_timesteps[0].observation
        self._opponent_obs = (
            self._latest_timesteps[1].observation
            if len(self._latest_timesteps) > 1
            else None
        )
        self._sync_map_geometry()
        self._rebuild_units()
        self._init_unit_type_ids()
        self._init_max_reward()
        self._snapshot_previous_health()

        if self._opponent_runtime is not None:
            self._opponent_runtime.on_reset(
                OpponentEpisodeContext(
                    family=self.variant,
                    map_name=self.map_name,
                    seed=self._seed,
                    episode_config=self.capability_config,
                )
            )

        return self.get_obs(), self.get_state()

    def step(self, actions: Sequence[int]):
        actions_int = [int(a) for a in actions]
        if len(actions_int) < self.n_agents:
            actions_int.extend([0] * (self.n_agents - len(actions_int)))
        actions_int = actions_int[: self.n_agents]
        self.last_action = self._action_eye[np.asarray(actions_int, dtype=np.int64)]
        self._avail_actions_cache.clear()

        ally_sc_actions: List[sc_pb.Action] = []
        for agent_id, action in enumerate(actions_int):
            sc_action = self._build_agent_action(agent_id, action)
            if sc_action is not None:
                ally_sc_actions.append(sc_action)

        opponent_actions = self._build_opponent_actions(actions_int)
        self._latest_timesteps = self._session.step(
            agent_actions=ally_sc_actions,
            opponent_actions=opponent_actions,
        )
        self._obs = self._latest_timesteps[0].observation
        self._opponent_obs = (
            self._latest_timesteps[1].observation
            if len(self._latest_timesteps) > 1
            else None
        )

        self._total_steps += 1
        self._episode_steps += 1
        self._rebuild_units()

        terminated = bool(self._latest_timesteps[0].last())
        reward = self._compute_reward()
        info = {"battle_won": False}
        battle_code = self._battle_outcome_code()
        if battle_code is not None:
            terminated = True
            if battle_code == 1:
                info["battle_won"] = True
                if not self.reward_sparse:
                    reward += self.reward_win
                else:
                    reward = 1.0
            elif battle_code == -1:
                if not self.reward_sparse:
                    reward += self.reward_defeat
                else:
                    reward = -1.0
        elif self._episode_steps >= self.episode_limit:
            terminated = True
            info["episode_limit"] = True

        if self.reward_scale and self.max_reward > 0 and self.reward_scale_rate > 0:
            reward /= self.max_reward / self.reward_scale_rate

        if terminated:
            self._episode_count += 1
        self.reward = float(reward)
        return float(reward), bool(terminated), info

    def close(self) -> None:
        self._session.close()

    def get_episode_step(self) -> int:
        return int(self._episode_steps)

    def get_obs(self):
        return [self.get_obs_agent(agent_id) for agent_id in range(self.n_agents)]

    def get_obs_agent(self, agent_id: int):
        unit = self.agents.get(agent_id)
        if unit is None:
            return np.zeros(self.get_obs_size(), dtype=np.float32)

        avail = self.get_avail_agent_actions(agent_id)
        move_feats = np.asarray(avail[2:6], dtype=np.float32)

        enemy_feats = np.zeros((self._attack_slots, 6), dtype=np.float32)
        enemy_items = list(self.enemies.items())[: self._attack_slots]
        sight_range = self._unit_sight_range(agent_id)
        for slot, (_, enemy) in enumerate(enemy_items):
            if enemy.health <= 0:
                continue
            dist = self._distance(unit.pos.x, unit.pos.y, enemy.pos.x, enemy.pos.y)
            enemy_feats[slot, 0] = 1.0
            enemy_feats[slot, 1] = float(avail[self.n_actions_no_attack + slot])
            enemy_feats[slot, 2] = dist / max(sight_range, 1.0)
            enemy_feats[slot, 3] = (enemy.pos.x - unit.pos.x) / max(self.max_distance_x, 1.0)
            enemy_feats[slot, 4] = (enemy.pos.y - unit.pos.y) / max(self.max_distance_y, 1.0)
            enemy_feats[slot, 5] = enemy.health / max(enemy.health_max, 1.0)

        ally_dim = max(self.n_agents - 1, 1)
        ally_feats = np.zeros((ally_dim, 6), dtype=np.float32)
        ally_slot = 0
        for ally_id, ally in self.agents.items():
            if ally_id == agent_id:
                continue
            if ally_slot >= ally_dim:
                break
            if ally.health > 0:
                dist = self._distance(unit.pos.x, unit.pos.y, ally.pos.x, ally.pos.y)
                ally_feats[ally_slot, 0] = 1.0
                ally_feats[ally_slot, 1] = dist / max(sight_range, 1.0)
                ally_feats[ally_slot, 2] = (ally.pos.x - unit.pos.x) / max(
                    self.max_distance_x, 1.0
                )
                ally_feats[ally_slot, 3] = (ally.pos.y - unit.pos.y) / max(
                    self.max_distance_y, 1.0
                )
                ally_feats[ally_slot, 4] = ally.health / max(ally.health_max, 1.0)
                ally_feats[ally_slot, 5] = ally.shield / max(ally.shield_max, 1.0)
            ally_slot += 1

        own_feats = np.asarray(
            [
                unit.health / max(unit.health_max, 1.0),
                unit.shield / max(unit.shield_max, 1.0),
                unit.pos.x / max(self.map_x, 1.0),
                unit.pos.y / max(self.map_y, 1.0),
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

    def get_state(self):
        ally_state = np.zeros((self.n_agents, 5), dtype=np.float32)
        for agent_id, unit in self.agents.items():
            ally_state[agent_id, 0] = unit.health / max(unit.health_max, 1.0)
            ally_state[agent_id, 1] = unit.shield / max(unit.shield_max, 1.0)
            ally_state[agent_id, 2] = unit.weapon_cooldown / 30.0
            ally_state[agent_id, 3] = unit.pos.x / max(self.map_x, 1.0)
            ally_state[agent_id, 4] = unit.pos.y / max(self.map_y, 1.0)

        enemy_state = np.zeros((self.n_enemies, 5), dtype=np.float32)
        for enemy_id, unit in self.enemies.items():
            enemy_state[enemy_id, 0] = unit.health / max(unit.health_max, 1.0)
            enemy_state[enemy_id, 1] = unit.shield / max(unit.shield_max, 1.0)
            enemy_state[enemy_id, 2] = unit.weapon_cooldown / 30.0
            enemy_state[enemy_id, 3] = unit.pos.x / max(self.map_x, 1.0)
            enemy_state[enemy_id, 4] = unit.pos.y / max(self.map_y, 1.0)

        chunks = [ally_state.flatten(), enemy_state.flatten()]
        if self.state_last_action:
            chunks.append(self.last_action.flatten())
        return np.concatenate(chunks, axis=0).astype(np.float32)

    def get_state_size(self):
        return int(self.get_state().shape[0])

    def get_obs_size(self):
        return int(self.get_obs_agent(0).shape[0])

    def get_avail_actions(self):
        return [
            self.get_avail_agent_actions(agent_id)
            for agent_id in range(self.n_agents)
        ]

    def get_avail_agent_actions(self, agent_id: int):
        cached = self._avail_actions_cache.get(agent_id)
        if cached is not None:
            return cached

        unit = self.agents.get(agent_id)
        if unit is None or unit.health <= 0:
            dead = [1] + [0] * (self.n_actions - 1)
            self._avail_actions_cache[agent_id] = dead
            return dead

        avail = [0] * self.n_actions
        avail[1] = 1  # stop
        if self._can_move(unit, dx=0.0, dy=self._move_amount):
            avail[2] = 1
        if self._can_move(unit, dx=0.0, dy=-self._move_amount):
            avail[3] = 1
        if self._can_move(unit, dx=self._move_amount, dy=0.0):
            avail[4] = 1
        if self._can_move(unit, dx=-self._move_amount, dy=0.0):
            avail[5] = 1

        targets = self._attack_targets_for_unit(unit)
        unit_range = self._unit_shoot_range(unit)
        for slot in range(self._attack_slots):
            if slot >= len(targets):
                break
            target = targets[slot]
            if target.health <= 0:
                continue
            dist = self._distance(unit.pos.x, unit.pos.y, target.pos.x, target.pos.y)
            if dist <= unit_range:
                action_id = self.n_actions_no_attack + slot
                if action_id < self.n_actions:
                    avail[action_id] = 1

        self._avail_actions_cache[agent_id] = avail
        return avail

    def get_env_info(self):
        return {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.n_actions,
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit,
            "variant": self.variant,
            "switches": {
                "action_mode": self.switches.action_mode,
                "opponent_mode": self.switches.opponent_mode,
                "capability_mode": self.switches.capability_mode,
                "reward_positive_mode": self.switches.reward_positive_mode,
                "team_init_mode": self.switches.team_init_mode,
            },
            "scripted_dual_controller_active": bool(
                self.switches.opponent_mode == "scripted_pool"
                and self._session.num_agents == 2
            ),
            "native_backend": True,
        }

    def _sync_map_geometry(self) -> None:
        env = self._session.env
        if env is None:
            return
        try:
            game_info = env.game_info[0]
            map_size = game_info.start_raw.map_size
            self.map_x = float(map_size.x)
            self.map_y = float(map_size.y)
            self.max_distance_x = self.map_x
            self.max_distance_y = self.map_y
        except Exception:
            return

    def _rebuild_units(self) -> None:
        if self._obs is None:
            self.agents = {}
            self.enemies = {}
            return

        raw_units = list(self._obs.observation.raw_data.units)
        allies = [u for u in raw_units if u.owner == 1]
        enemies = [u for u in raw_units if u.owner == 2]
        allies_sorted = sorted(allies, key=attrgetter("unit_type", "pos.x", "pos.y"))
        enemies_sorted = sorted(
            enemies,
            key=attrgetter("unit_type", "pos.x", "pos.y"),
        )
        self.agents = {
            idx: allies_sorted[idx]
            for idx in range(min(self.n_agents, len(allies_sorted)))
        }
        self.enemies = {
            idx: enemies_sorted[idx]
            for idx in range(min(self.n_enemies, len(enemies_sorted)))
        }

    def _init_unit_type_ids(self) -> None:
        if not self.agents:
            return
        min_unit_type = min(unit.unit_type for unit in self.agents.values())
        self._unit_ids = self._variant_logic.infer_unit_type_ids(min_unit_type)

    def _init_max_reward(self) -> None:
        if self.max_reward > 0:
            return
        self.max_reward = sum(
            (unit.health_max + unit.shield_max) for unit in self.enemies.values()
        )
        self.max_reward += self.reward_win

    def _snapshot_previous_health(self) -> None:
        self.prev_ally_health.fill(0.0)
        self.prev_ally_shield.fill(0.0)
        self.prev_enemy_health.fill(0.0)
        self.prev_enemy_shield.fill(0.0)
        for agent_id, unit in self.agents.items():
            self.prev_ally_health[agent_id] = unit.health
            self.prev_ally_shield[agent_id] = unit.shield
        for enemy_id, unit in self.enemies.items():
            self.prev_enemy_health[enemy_id] = unit.health
            self.prev_enemy_shield[enemy_id] = unit.shield

    def _compute_reward(self) -> float:
        if self.reward_sparse:
            return 0.0

        reward = 0.0
        delta_deaths = 0.0
        delta_ally = 0.0
        delta_enemy = 0.0
        neg_scale = self.reward_negative_scale

        for agent_id in range(self.n_agents):
            unit = self.agents.get(agent_id)
            prev_health = self.prev_ally_health[agent_id] + self.prev_ally_shield[agent_id]
            if unit is None or unit.health <= 0:
                if self.death_tracker_ally[agent_id] == 0 and prev_health > 0:
                    self.death_tracker_ally[agent_id] = 1
                    if not self.reward_only_positive:
                        delta_deaths -= self.reward_death_value * neg_scale
                    delta_ally += prev_health * neg_scale
            else:
                if self.death_tracker_ally[agent_id] == 0:
                    cur_health = unit.health + unit.shield
                    delta_ally += max(prev_health - cur_health, 0.0) * neg_scale

            if unit is not None:
                self.prev_ally_health[agent_id] = unit.health
                self.prev_ally_shield[agent_id] = unit.shield
            else:
                self.prev_ally_health[agent_id] = 0.0
                self.prev_ally_shield[agent_id] = 0.0

        for enemy_id in range(self.n_enemies):
            unit = self.enemies.get(enemy_id)
            prev_health = self.prev_enemy_health[enemy_id] + self.prev_enemy_shield[enemy_id]
            if unit is None or unit.health <= 0:
                if self.death_tracker_enemy[enemy_id] == 0 and prev_health > 0:
                    self.death_tracker_enemy[enemy_id] = 1
                    delta_deaths += self.reward_death_value
                    delta_enemy += prev_health
            else:
                if self.death_tracker_enemy[enemy_id] == 0:
                    cur_health = unit.health + unit.shield
                    delta_enemy += max(prev_health - cur_health, 0.0)

            if unit is not None:
                self.prev_enemy_health[enemy_id] = unit.health
                self.prev_enemy_shield[enemy_id] = unit.shield
            else:
                self.prev_enemy_health[enemy_id] = 0.0
                self.prev_enemy_shield[enemy_id] = 0.0

        if self.reward_only_positive:
            reward = self._variant_logic.reward_positive_transform(
                delta_enemy + delta_deaths
            )
        else:
            reward = delta_enemy + delta_deaths - delta_ally
        return float(reward)

    def _battle_outcome_code(self):
        n_ally_alive = sum(1 for unit in self.agents.values() if unit.health > 0)
        n_enemy_alive = sum(1 for unit in self.enemies.values() if unit.health > 0)

        if (n_ally_alive == 0 and n_enemy_alive > 0) or self._only_medivac_left(True):
            return -1
        if (n_ally_alive > 0 and n_enemy_alive == 0) or self._only_medivac_left(False):
            return 1
        if n_ally_alive == 0 and n_enemy_alive == 0:
            return 0
        return None

    def _only_medivac_left(self, ally: bool) -> bool:
        medivac_id = self._unit_ids.medivac_id
        if self.map_params.map_type != "MMM" or medivac_id <= 0:
            return False
        units = self.agents if ally else self.enemies
        non_medivac_alive = [
            unit
            for unit in units.values()
            if unit.health > 0 and unit.unit_type != medivac_id
        ]
        return len(non_medivac_alive) == 0

    def _attack_targets_for_unit(self, unit) -> List[Any]:
        is_healer = self._is_healer(unit)
        if is_healer:
            targets = [
                ally
                for ally in self.agents.values()
                if ally.health > 0 and ally.unit_type != self._unit_ids.medivac_id
            ]
        else:
            targets = [enemy for enemy in self.enemies.values() if enemy.health > 0]
        return targets[: self._attack_slots]

    def _is_healer(self, unit) -> bool:
        medivac_id = self._unit_ids.medivac_id
        return medivac_id > 0 and unit.unit_type == medivac_id

    def _build_agent_action(self, agent_id: int, action: int):
        avail = self.get_avail_agent_actions(agent_id)
        if action < 0 or action >= self.n_actions or avail[action] == 0:
            return None
        unit = self.agents.get(agent_id)
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
                ability_id=ACTIONS["stop"],
                unit_tags=[tag],
                queue_command=False,
            )
        elif action == 2:
            cmd = self._build_move_cmd(tag, x, y + self._move_amount)
        elif action == 3:
            cmd = self._build_move_cmd(tag, x, y - self._move_amount)
        elif action == 4:
            cmd = self._build_move_cmd(tag, x + self._move_amount, y)
        elif action == 5:
            cmd = self._build_move_cmd(tag, x - self._move_amount, y)
        else:
            slot = action - self.n_actions_no_attack
            targets = self._attack_targets_for_unit(unit)
            if slot >= len(targets):
                return None
            target = targets[slot]
            ability_id = ACTIONS["heal"] if self._is_healer(unit) else ACTIONS["attack"]
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=ability_id,
                target_unit_tag=target.tag,
                unit_tags=[tag],
                queue_command=False,
            )

        return sc_pb.Action(action_raw=r_pb.ActionRaw(unit_command=cmd))

    def _build_move_cmd(self, tag: int, x: float, y: float):
        return r_pb.ActionRawUnitCommand(
            ability_id=ACTIONS["move"],
            target_world_space_pos=sc_common.Point2D(x=x, y=y),
            unit_tags=[tag],
            queue_command=False,
        )

    def _build_opponent_actions(self, actions: Sequence[int]) -> Sequence[Any]:
        runtime = self._opponent_runtime
        if (
            runtime is None
            or self.switches.opponent_mode != "scripted_pool"
            or self._session.num_agents < 2
        ):
            return []

        payload = {
            "agents": self.enemies,
            "enemies": self.agents,
            "agent_ability": self._query_enemy_abilities(),
            "visible_matrix": self._fog_visibility_matrix(),
            "episode_step": self._episode_steps,
        }
        context = OpponentStepContext(
            family=self.variant,
            episode_step=self._episode_steps,
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

    def _query_enemy_abilities(self):
        env = self._session.env
        controllers = getattr(env, "_controllers", None) if env is not None else None
        if not controllers or len(controllers) < 2:
            return []
        try:
            query = q_pb.RequestQuery()
            for unit in self.enemies.values():
                ability = query.abilities.add()
                ability.unit_tag = unit.tag
            if len(query.abilities) == 0:
                return []
            result = controllers[1].query(query)
            return list(result.abilities)
        except Exception:
            return []

    def _fog_visibility_matrix(self):
        red_visible = [
            unit.tag for unit in self.enemies.values() if unit.health > 0
        ]
        blue_visible = [
            unit.tag for unit in self.agents.values() if unit.health > 0
        ]
        return {"red": red_visible, "blue": blue_visible}

    def _unit_shoot_range(self, unit) -> float:
        ids = self._variant_logic.shoot_range_by_type(self._unit_ids)
        unit_type = unit.unit_type
        default_range = 6.0
        if unit_type in ids and ids[unit_type] > 0:
            return max(ids[unit_type], 0.1)
        return default_range

    def _unit_sight_range(self, agent_id: int) -> float:
        unit = self.agents.get(agent_id)
        if unit is None:
            return 9.0
        return max(self._unit_shoot_range(unit) + 3.0, 6.0)

    def _can_move(self, unit, *, dx: float, dy: float) -> bool:
        nx = unit.pos.x + dx
        ny = unit.pos.y + dy
        return 0 <= nx <= self.map_x and 0 <= ny <= self.map_y

    @staticmethod
    def _distance(x1, y1, x2, y2):
        return math.hypot(x2 - x1, y2 - y1)
