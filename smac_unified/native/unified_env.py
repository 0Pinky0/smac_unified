from __future__ import annotations

import math
from operator import attrgetter
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from ..config import VariantSwitches, merge_switches
from ..maps import MapParams, get_map_params
from ..opponents import OpponentEpisodeContext, OpponentRuntime
from .builders import (
    DefaultNativeActionBuilder,
    DefaultNativeObservationBuilder,
    DefaultNativeRewardBuilder,
    DefaultNativeStateBuilder,
    NativeActionBuilder,
    NativeObservationBuilder,
    NativeRewardBuilder,
    NativeStateBuilder,
)
from .session_sc2env import SC2EnvRawSession, SC2SessionConfig
from .variants import UnitTypeIds, VariantLogic, build_variant_logic


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
        self._action_builder: NativeActionBuilder = (
            self._env_kwargs.get("native_action_builder")
            or DefaultNativeActionBuilder()
        )
        self._observation_builder: NativeObservationBuilder = (
            self._env_kwargs.get("native_observation_builder")
            or DefaultNativeObservationBuilder()
        )
        self._state_builder: NativeStateBuilder = (
            self._env_kwargs.get("native_state_builder")
            or DefaultNativeStateBuilder()
        )
        self._reward_builder: NativeRewardBuilder = (
            self._env_kwargs.get("native_reward_builder")
            or DefaultNativeRewardBuilder()
        )

        self._attack_slots = max(self.n_agents, self.n_enemies)
        self.n_actions = self._variant_logic.n_actions(
            n_agents=self.n_agents,
            n_enemies=self.n_enemies,
        )
        self.n_actions_no_attack = 6
        self._obs_vector_size = (
            4
            + self._attack_slots * 6
            + max(self.n_agents - 1, 1) * 6
            + 4
        )

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
        self.last_action.fill(0.0)
        self._action_builder.reset(self)

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
        self._reward_builder.reset(self)

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
        self._action_builder.reset(self)

        ally_sc_actions: List[Any] = []
        for agent_id, action in enumerate(actions_int):
            sc_action = self._action_builder.build_agent_action(
                self,
                agent_id,
                action,
            )
            if sc_action is not None:
                ally_sc_actions.append(sc_action)

        opponent_actions = self._action_builder.build_opponent_actions(
            self,
            actions_int,
            self._opponent_runtime,
        )
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
        reward = self._reward_builder.build_step_reward(self)
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
        return self._observation_builder.build_obs(self)

    def get_obs_agent(self, agent_id: int):
        return self._observation_builder.build_agent_obs(self, agent_id)

    def get_state(self):
        return self._state_builder.build_state(self)

    def get_state_size(self):
        return int(self.get_state().shape[0])

    def get_obs_size(self):
        return int(self._obs_vector_size)

    def get_avail_actions(self):
        return [
            self.get_avail_agent_actions(agent_id)
            for agent_id in range(self.n_agents)
        ]

    def get_avail_agent_actions(self, agent_id: int):
        return self._action_builder.get_avail_agent_actions(self, agent_id)

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
