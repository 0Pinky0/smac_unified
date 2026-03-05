from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from operator import attrgetter
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Sequence

import numpy as np
from s2clientprotocol import common_pb2 as sc_common
from s2clientprotocol import raw_pb2 as r_pb
from s2clientprotocol import sc2api_pb2 as sc_pb

from ..handlers import (
    ActionHandler,
    HandlerContext,
    ObservationHandler,
    RewardHandler,
    StateHandler,
    UnitFrame,
    build_default_handler_bundle,
)
from ..maps import MapParams, get_map_params
from ..players import (
    EngineBotOpponentRuntime,
    OpponentEpisodeContext,
    OpponentRuntime,
    OpponentStepContext,
    build_scripted_runtime_from_config,
)
from ..types import StepBatch
from .sc2session import SC2EnvRawSession, SC2SessionConfig
from .switches import VariantSwitches
from .unit_tracker import UnitTracker
from .variants import (
    UnitTypeIds,
    VariantLogic,
    build_variant_logic,
    merge_switches,
)


_TRANSPORT_PROFILE_OPTIONS: dict[str, dict[str, bool]] = {
    'B0': {},
    'B1': {'reuse_step_observe_requests': True},
    'B2': {
        'reuse_step_observe_requests': True,
        'pipeline_step_and_observe': True,
    },
    'B3': {
        'reuse_step_observe_requests': True,
        'pipeline_step_and_observe': True,
        'pipeline_actions_and_step': True,
    },
    'B4': {
        'reuse_step_observe_requests': True,
        'pipeline_step_and_observe': True,
        'pipeline_actions_and_step': True,
        'ensure_available_actions': False,
    },
}


@dataclass
class EnvFactoryConfig:
    family: str
    map_name: str = '8m'
    normalized_api: bool = True
    capability_config: Optional[Dict[str, Any]] = None
    # Kept for backward compatibility with earlier smac_unified config style.
    kwargs: Dict[str, Any] = field(default_factory=dict)
    # Preferred explicit env kwargs.
    env_kwargs: Dict[str, Any] = field(default_factory=dict)
    source_root: str | None = None
    transport_profile: Literal['B0', 'B1', 'B2', 'B3', 'B4'] | None = None
    allow_experimental_transport: bool = False
    logic_switches: Dict[str, str] = field(default_factory=dict)
    native_options: Dict[str, Any] = field(default_factory=dict)
    observation_handler: Any | None = None
    state_handler: Any | None = None
    reward_handler: Any | None = None
    action_handler: Any | None = None
    opponent_runtime: OpponentRuntime | None = None
    opponent_config: Dict[str, Any] = field(default_factory=dict)

    def resolved_env_kwargs(self) -> Dict[str, Any]:
        merged = dict(self.kwargs)
        merged.update(self.env_kwargs)
        return merged


class SMACEnvCore:
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
        transport_profile: str = 'B0',
        allow_experimental_transport: bool = False,
    ):
        self.variant = variant
        self.map_name = map_name
        self.capability_config = dict(capability_config or {})
        self._env_kwargs = dict(env_kwargs or {})
        self._native_options = dict(native_options or {})
        self._transport_profile = str(transport_profile or 'B0').upper()
        self._allow_experimental_transport = bool(allow_experimental_transport)
        self._source_root = source_root

        self.map_params: MapParams = get_map_params(map_name)
        raw_switches = self._env_kwargs.get('logic_switches')
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
            self._env_kwargs.get('episode_limit', self.map_params.limit)
        )
        self._move_amount = float(self._env_kwargs.get('move_amount', 2.0))

        self.reward_sparse = bool(self._env_kwargs.get('reward_sparse', False))
        self.reward_only_positive = bool(
            self._env_kwargs.get('reward_only_positive', True)
        )
        self.reward_death_value = float(
            self._env_kwargs.get('reward_death_value', 10.0)
        )
        self.reward_win = float(self._env_kwargs.get('reward_win', 200.0))
        self.reward_defeat = float(self._env_kwargs.get('reward_defeat', 0.0))
        self.reward_negative_scale = float(
            self._env_kwargs.get('reward_negative_scale', 0.5)
        )
        self.reward_scale = bool(self._env_kwargs.get('reward_scale', True))
        self.reward_scale_rate = float(
            self._env_kwargs.get('reward_scale_rate', 20.0)
        )
        self.state_last_action = bool(
            self._env_kwargs.get('state_last_action', True)
        )
        self.state_timestep_number = bool(
            self._env_kwargs.get('state_timestep_number', False)
        )
        self.obs_all_health = bool(self._env_kwargs.get('obs_all_health', True))
        self.obs_own_health = bool(self._env_kwargs.get('obs_own_health', True))
        if self.obs_all_health:
            self.obs_own_health = True
        self.obs_last_action = bool(self._env_kwargs.get('obs_last_action', False))
        self.obs_pathing_grid = bool(
            self._env_kwargs.get('obs_pathing_grid', False)
        )
        self.obs_terrain_height = bool(
            self._env_kwargs.get('obs_terrain_height', False)
        )
        self.obs_timestep_number = bool(
            self._env_kwargs.get('obs_timestep_number', False)
        )
        self.obs_instead_of_state = bool(
            self._env_kwargs.get('obs_instead_of_state', False)
        )
        self.continuing_episode = bool(
            self._env_kwargs.get('continuing_episode', False)
        )
        self.shield_bits_ally = 1 if self.map_params.a_race == 'P' else 0
        self.shield_bits_enemy = 1 if self.map_params.b_race == 'P' else 0
        self.unit_type_bits = int(self.map_params.unit_type_bits)
        self.ally_state_attr_names = ['health', 'energy/cooldown', 'rel_x', 'rel_y']
        self.enemy_state_attr_names = ['health', 'rel_x', 'rel_y']
        if self.shield_bits_ally > 0:
            self.ally_state_attr_names.append('shield')
        if self.shield_bits_enemy > 0:
            self.enemy_state_attr_names.append('shield')
        if self.switches.action_mode == 'conic_fov':
            self.ally_state_attr_names.extend(['fov_x', 'fov_y'])
        if self.unit_type_bits > 0:
            type_names = [f'type_{idx}' for idx in range(self.unit_type_bits)]
            self.ally_state_attr_names.extend(type_names)
            self.enemy_state_attr_names.extend(type_names)
        self._seed = self._env_kwargs.get('seed')
        self._rng = np.random.default_rng(self._seed)
        default_handlers = build_default_handler_bundle(
            switches=self.switches,
            map_params=self.map_params,
            env_kwargs=self._env_kwargs,
        )
        self._action_handler: ActionHandler = (
            self._env_kwargs.get('action_handler')
            or default_handlers.action_handler
        )
        self._observation_handler: ObservationHandler = (
            self._env_kwargs.get('observation_handler')
            or default_handlers.observation_handler
        )
        self._state_handler: StateHandler = (
            self._env_kwargs.get('state_handler')
            or default_handlers.state_handler
        )
        self._reward_handler: RewardHandler = (
            self._env_kwargs.get('reward_handler')
            or default_handlers.reward_handler
        )

        self._n_fov_actions = int(
            self._env_kwargs.get('num_fov_actions', 12)
            if self.switches.action_mode == 'conic_fov'
            else 0
        )
        self._action_mask = bool(self._env_kwargs.get('action_mask', True))
        self._use_ability = bool(
            self._env_kwargs.get(
                'use_ability',
                self.switches.action_mode == 'ability_augmented',
            )
        )
        self._ability_padding = max(self.n_agents, self.n_enemies, 9)
        (
            self.n_actions_no_attack,
            self.n_actions,
            self._attack_slots,
        ) = self._action_handler.resolve_action_space(
            n_agents=self.n_agents,
            n_enemies=self.n_enemies,
            env_kwargs=self._env_kwargs,
        )
        self._obs_vector_size = 0
        self._state_vector_size = 0

        self._session = self._build_session()
        self._opponent_runtime: OpponentRuntime | None = None
        self._runtime_lifecycle_owner = 'env'

        self._episode_steps = 0
        self._total_steps = 0
        self._episode_count = 0
        self.battles_won = 0
        self.battles_game = 0
        self.timeouts = 0
        self.force_restarts = 0
        self.win_counted = False
        self.defeat_counted = False
        self._obs = None
        self._opponent_obs = None
        self._latest_timesteps = []
        self._pending_step_timesteps: list[Any] | None = None

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
        self._agent_row_index = np.arange(self.n_agents, dtype=np.int64)
        self.pathing_grid: np.ndarray | None = None
        self.terrain_height: np.ndarray | None = None
        self.agent_attack_probabilities = np.ones(self.n_agents, dtype=np.float32)
        self.agent_health_levels = np.zeros(self.n_agents, dtype=np.float32)
        self.enemy_mask = np.ones(self.n_enemies, dtype=np.float32)
        self.ally_start_positions: np.ndarray | None = None
        self.enemy_start_positions: np.ndarray | None = None
        self.ally_team: Sequence[Any] | None = None
        self.enemy_team: Sequence[Any] | None = None
        self.fov_directions = np.zeros((self.n_agents, 2), dtype=np.float32)
        self.fov_directions[:, 0] = 1.0
        if self._n_fov_actions > 0:
            angles = np.linspace(
                0.0,
                2.0 * np.pi,
                num=self._n_fov_actions,
                endpoint=False,
                dtype=np.float32,
            )
            self.canonical_fov_directions = np.stack(
                (np.cos(angles), np.sin(angles)),
                axis=1,
            ).astype(np.float32)
            self._conic_fov_angle = float((2.0 * np.pi) / self._n_fov_actions)
        else:
            self.canonical_fov_directions = np.zeros((0, 2), dtype=np.float32)
            self._conic_fov_angle = 0.0
        self._unit_tracker = UnitTracker(self.n_agents, self.n_enemies)
        self._unit_frame: UnitFrame | None = None
        self._handler_context: HandlerContext | None = None
        self._last_split_probe: dict[str, Any] = {}
        self._last_ally_sc_actions: list[Any] = []
        self._last_opponent_actions: list[Any] = []
        self._forced_opponent_actions_schedule: list[list[dict[str, Any]]] | None = None
        self._refresh_handler_context()
        self._refresh_space_sizes()

    def _build_session(self) -> SC2EnvRawSession:
        dual_controller_default = self.switches.opponent_mode == 'scripted_pool'
        cfg = SC2SessionConfig(
            map_name=self.map_name,
            map_params=self.map_params,
            step_mul=int(self._env_kwargs.get('step_mul', 8)),
            difficulty=str(self._env_kwargs.get('difficulty', '7')),
            seed=self._seed,
            realtime=bool(self._env_kwargs.get('realtime', False)),
            ensure_available_actions=bool(
                self._native_options.get('ensure_available_actions', True)
            ),
            pipeline_actions_and_step=bool(
                self._native_options.get('pipeline_actions_and_step', False)
            ),
            pipeline_step_and_observe=bool(
                self._native_options.get('pipeline_step_and_observe', False)
            ),
            reuse_step_observe_requests=bool(
                self._native_options.get('reuse_step_observe_requests', False)
            ),
            transport_profile=self._transport_profile,
            allow_experimental_transport=self._allow_experimental_transport,
            opponent_mode=self.switches.opponent_mode,
            enable_dual_controller=bool(
                self._native_options.get(
                    'enable_dual_controller',
                    dual_controller_default,
                )
            ),
            enable_async_step=bool(
                self._native_options.get('enable_async_step', False)
            ),
            game_version=self._env_kwargs.get('game_version'),
            source_root=self._source_root,
        )
        return SC2EnvRawSession(cfg)

    def set_opponent_runtime(self, runtime: OpponentRuntime) -> None:
        self._opponent_runtime = runtime
        runtime.bind_env(self, self.variant)

    def set_runtime_lifecycle_owner(self, owner: str) -> None:
        self._runtime_lifecycle_owner = owner

    def set_forced_opponent_actions_schedule(
        self,
        schedule: Sequence[Sequence[Mapping[str, Any]]] | None,
    ) -> None:
        if not schedule:
            self._forced_opponent_actions_schedule = None
            return
        self._forced_opponent_actions_schedule = [
            [dict(item) for item in list(step or []) if isinstance(item, Mapping)]
            for step in list(schedule)
        ]

    def seed(self, seed: int | None = None):
        if seed is None:
            return self._seed
        self._seed = int(seed)
        self._rng = np.random.default_rng(self._seed)
        if self._session is not None:
            self._session.config.seed = self._seed
        return self._seed

    def reset(self, episode_config: Mapping[str, Any] | None = None, **kwargs):
        del kwargs
        if episode_config and isinstance(episode_config, Mapping):
            self.capability_config.update(dict(episode_config))
        self._apply_episode_capabilities(episode_config)

        self._episode_steps = 0
        self.reward = 0.0
        self.last_action.fill(0.0)
        self.win_counted = False
        self.defeat_counted = False
        if self.fov_directions.shape[0] == self.n_agents:
            self.fov_directions.fill(0.0)
            self.fov_directions[:, 0] = 1.0

        self._latest_timesteps = self._session.reset()
        self._obs = self._latest_timesteps[0].observation
        self._opponent_obs = (
            self._latest_timesteps[1].observation
            if len(self._latest_timesteps) > 1
            else None
        )
        self._sync_map_geometry()
        allies, enemies = self._split_raw_units()
        self._unit_frame = self._unit_tracker.reset(
            allies=allies,
            enemies=enemies,
        )
        self._sync_legacy_unit_views()
        self._init_unit_type_ids()
        self._init_max_reward()
        self._refresh_handler_context()
        self._refresh_space_sizes()
        self._action_handler.reset(
            frame=self._unit_frame,
            context=self._handler_context,
        )
        self._reward_handler.reset(
            frame=self._unit_frame,
            context=self._handler_context,
        )

        if (
            self._opponent_runtime is not None
            and self._runtime_lifecycle_owner == 'env'
        ):
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
        actions_int = self._normalize_actions(actions)
        ally_sc_actions, opponent_actions = self._encode_step_actions(actions_int)
        self._submit_step_actions(
            ally_sc_actions=ally_sc_actions,
            opponent_actions=opponent_actions,
        )
        self._collect_step_timesteps()
        reward, terminated, info = self._decode_step_outcome()
        return float(reward), bool(terminated), info

    def _normalize_actions(self, actions: Sequence[int]) -> list[int]:
        actions_int = [int(a) for a in actions]
        if len(actions_int) < self.n_agents:
            actions_int.extend([0] * (self.n_agents - len(actions_int)))
        return actions_int[: self.n_agents]

    def _encode_step_actions(
        self,
        actions_int: Sequence[int],
    ) -> tuple[List[Any], Sequence[Any]]:
        self._update_last_action_matrix(actions_int)
        self._refresh_handler_context()

        ally_sc_actions: List[Any] = []
        for agent_id, action in enumerate(actions_int):
            sc_action = self._action_handler.build_agent_action(
                frame=self._unit_frame,
                context=self._handler_context,
                agent_id=agent_id,
                action=action,
            )
            if sc_action is not None:
                ally_sc_actions.append(sc_action)

        forced_opponent = self._forced_opponent_actions_for_step(self._episode_steps)
        if forced_opponent is not None:
            opponent_actions = forced_opponent
        else:
            opponent_actions = self._action_handler.build_opponent_actions(
                frame=self._unit_frame,
                context=self._handler_context,
                actions=actions_int,
                runtime=self._opponent_runtime,
            )
        self._last_ally_sc_actions = list(ally_sc_actions)
        self._last_opponent_actions = list(opponent_actions or [])
        return ally_sc_actions, opponent_actions

    def _submit_step_actions(
        self,
        *,
        ally_sc_actions: Sequence[Any],
        opponent_actions: Sequence[Any] | None,
    ) -> None:
        if hasattr(self._session, 'submit_step'):
            self._session.submit_step(
                agent_actions=ally_sc_actions,
                opponent_actions=opponent_actions,
            )
            return
        # Backward-compatible fallback for stub/test sessions with only step().
        self._pending_step_timesteps = self._session.step(
            agent_actions=ally_sc_actions,
            opponent_actions=opponent_actions,
        )

    def _collect_step_timesteps(self) -> None:
        if self._pending_step_timesteps is not None:
            self._latest_timesteps = self._pending_step_timesteps
            self._pending_step_timesteps = None
        else:
            self._latest_timesteps = self._session.collect_step()
        self._obs = self._latest_timesteps[0].observation
        self._opponent_obs = (
            self._latest_timesteps[1].observation
            if len(self._latest_timesteps) > 1
            else None
        )

        self._total_steps += 1
        self._episode_steps += 1
        allies, enemies = self._split_raw_units()
        self._unit_frame = self._unit_tracker.update(
            allies=allies,
            enemies=enemies,
        )
        self._sync_legacy_unit_views()
        self._refresh_handler_context()

    def _decode_step_outcome(self) -> tuple[float, bool, dict[str, Any]]:
        terminated = bool(self._latest_timesteps[0].last())
        reward = self._reward_handler.build_step_reward(
            frame=self._unit_frame,
            context=self._handler_context,
        )
        dead_allies = int(np.sum(~self._unit_frame.allies.alive))
        dead_enemies = int(np.sum(~self._unit_frame.enemies.alive))
        info = {
            'battle_won': False,
            'dead_allies': dead_allies,
            'dead_enemies': dead_enemies,
        }
        battle_code = self._battle_outcome_code()
        if battle_code is not None:
            terminated = True
            self.battles_game += 1
            if battle_code == 1 and not self.win_counted:
                self.battles_won += 1
                self.win_counted = True
                info['battle_won'] = True
                if not self.reward_sparse:
                    reward += self.reward_win
                else:
                    reward = 1.0
            elif battle_code == -1 and not self.defeat_counted:
                self.defeat_counted = True
                if not self.reward_sparse:
                    reward += self.reward_defeat
                else:
                    reward = -1.0
        elif self._episode_steps >= self.episode_limit:
            terminated = True
            if self.continuing_episode:
                info['episode_limit'] = True
            self.battles_game += 1
            self.timeouts += 1

        if terminated:
            self._episode_count += 1
        if self.reward_scale and self.max_reward > 0 and self.reward_scale_rate > 0:
            reward /= self.max_reward / self.reward_scale_rate
        self.reward = float(reward)
        return float(reward), bool(terminated), info

    def close(self) -> None:
        if (
            self._opponent_runtime is not None
            and self._runtime_lifecycle_owner == 'env'
        ):
            self._opponent_runtime.close()
        self._session.close()

    def step_batch(self, actions: Sequence[int]) -> dict[str, Any]:
        reward, terminated, info = self.step(actions)
        return {
            'obs': self.get_obs(),
            'state': self.get_state(),
            'avail_actions': self.get_avail_actions(),
            'reward': reward,
            'terminated': terminated,
            'info': info,
        }

    def reset_batch(
        self,
        episode_config: Mapping[str, Any] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        obs, state = self.reset(episode_config=episode_config, **kwargs)
        return {
            'obs': obs,
            'state': state,
            'avail_actions': self.get_avail_actions(),
            'reward': 0.0,
            'terminated': False,
            'info': {},
        }

    def get_episode_step(self) -> int:
        return int(self._episode_steps)

    def get_total_actions(self) -> int:
        return int(self.n_actions)

    def get_obs(self):
        if self._unit_frame is None or self._handler_context is None:
            return []
        return self._observation_handler.build_obs(
            frame=self._unit_frame,
            context=self._handler_context,
        )

    def get_obs_agent(self, agent_id: int):
        if self._unit_frame is None or self._handler_context is None:
            return np.zeros(self.get_obs_size(), dtype=np.float32)
        return self._observation_handler.build_agent_obs(
            frame=self._unit_frame,
            context=self._handler_context,
            agent_id=agent_id,
        )

    def get_state(self):
        if self._unit_frame is None or self._handler_context is None:
            return np.zeros(self.get_state_size(), dtype=np.float32)
        return self._state_handler.build_state(
            frame=self._unit_frame,
            context=self._handler_context,
        )

    def get_state_size(self):
        if self._state_vector_size > 0:
            return int(self._state_vector_size)
        if self._unit_frame is not None and self._handler_context is not None:
            return int(
                self._state_handler.build_state(
                    frame=self._unit_frame,
                    context=self._handler_context,
                ).shape[0]
            )
        return 0

    def get_obs_size(self):
        if self._obs_vector_size > 0:
            return int(self._obs_vector_size)
        if (
            self._unit_frame is not None
            and self._handler_context is not None
            and self.n_agents > 0
        ):
            return int(self.get_obs_agent(0).shape[0])
        return int(self._obs_vector_size)

    def get_avail_actions(self):
        return [
            self.get_avail_agent_actions(agent_id)
            for agent_id in range(self.n_agents)
        ]

    def get_avail_agent_actions(self, agent_id: int):
        if self._unit_frame is None or self._handler_context is None:
            return [1] + [0] * (self.n_actions - 1)
        return self._action_handler.get_avail_agent_actions(
            frame=self._unit_frame,
            context=self._handler_context,
            agent_id=agent_id,
        )

    def get_env_info(self):
        info = {
            'state_shape': self.get_state_size(),
            'obs_shape': self.get_obs_size(),
            'n_actions': self.n_actions,
            'n_agents': self.n_agents,
            'episode_limit': self.episode_limit,
            'variant': self.variant,
            'switches': {
                'action_mode': self.switches.action_mode,
                'opponent_mode': self.switches.opponent_mode,
                'capability_mode': self.switches.capability_mode,
                'reward_positive_mode': self.switches.reward_positive_mode,
                'team_init_mode': self.switches.team_init_mode,
            },
            'scripted_dual_controller_active': bool(
                self.switches.opponent_mode == 'scripted_pool'
                and self._session.num_agents == 2
            ),
            'agent_features': list(self.ally_state_attr_names),
            'enemy_features': list(self.enemy_state_attr_names),
            'native_backend': True,
        }
        if self.variant == 'smacv2':
            cap_shape = 0
            if isinstance(self.capability_config, Mapping):
                cap_shape = int(self.capability_config.get('cap_shape', 0) or 0)
            info['cap_shape'] = cap_shape
        if self.variant == 'smac-hard':
            info['pysc2_backend'] = True
        return info

    def get_stats(self) -> dict[str, float]:
        win_rate = (
            float(self.battles_won) / float(self.battles_game)
            if self.battles_game > 0
            else 0.0
        )
        return {
            'battles_won': int(self.battles_won),
            'battles_game': int(self.battles_game),
            'battles_draw': int(self.timeouts),
            'win_rate': float(win_rate),
            'timeouts': int(self.timeouts),
            'restarts': int(self.force_restarts),
        }

    def _refresh_space_sizes(self) -> None:
        if self._handler_context is None:
            return
        obs_size = int(self._observation_handler.obs_size(context=self._handler_context))
        if obs_size <= 0:
            obs_size = self._estimate_obs_vector_size()
        self._obs_vector_size = int(obs_size)
        state_size = int(self._state_handler.state_size(context=self._handler_context))
        if state_size <= 0 and self._unit_frame is not None:
            state_size = int(
                self._state_handler.build_state(
                    frame=self._unit_frame,
                    context=self._handler_context,
                ).shape[0]
            )
        self._state_vector_size = max(int(state_size), 0)

    def _estimate_obs_vector_size(self) -> int:
        move_feats = 4
        if self.obs_pathing_grid:
            move_feats += 8
        if self.obs_terrain_height:
            move_feats += 9

        enemy_feats = 4
        if self.obs_all_health:
            enemy_feats += 1 + self.shield_bits_enemy
        if self.unit_type_bits > 0:
            enemy_feats += self.unit_type_bits

        ally_feats = 4
        if self.obs_all_health:
            ally_feats += 1 + self.shield_bits_ally
        if self.unit_type_bits > 0:
            ally_feats += self.unit_type_bits
        if self.obs_last_action:
            ally_feats += self.n_actions

        own_feats = 0
        if self.obs_own_health:
            own_feats += 1 + self.shield_bits_ally
        if self.unit_type_bits > 0:
            own_feats += self.unit_type_bits

        total = (
            move_feats
            + self._attack_slots * enemy_feats
            + max(self.n_agents - 1, 1) * ally_feats
            + own_feats
        )
        if self.obs_timestep_number:
            total += 1
        return int(total)

    def _apply_episode_capabilities(
        self,
        episode_config: Mapping[str, Any] | None,
    ) -> None:
        payload: dict[str, Any] = {}
        if isinstance(self.capability_config, Mapping):
            payload.update(dict(self.capability_config))
        if isinstance(episode_config, Mapping):
            payload.update(dict(episode_config))

        self.agent_attack_probabilities = _extract_capability_vector(
            payload=payload.get('attack'),
            size=self.n_agents,
            default=1.0,
        )
        self.agent_health_levels = _extract_capability_vector(
            payload=payload.get('health'),
            size=self.n_agents,
            default=0.0,
        )
        self.enemy_mask = _extract_capability_vector(
            payload=payload.get('enemy_mask'),
            size=self.n_enemies,
            default=1.0,
        )

        start_positions = payload.get('start_positions', {})
        if isinstance(start_positions, Mapping):
            self.ally_start_positions = _extract_positions(
                start_positions.get('ally_start_positions'),
            )
            self.enemy_start_positions = _extract_positions(
                start_positions.get('enemy_start_positions'),
            )
        else:
            self.ally_start_positions = None
            self.enemy_start_positions = None

        team_payload = payload.get('team_gen', {})
        if isinstance(team_payload, Mapping):
            ally_team = team_payload.get('ally_team')
            enemy_team = team_payload.get('enemy_team')
            self.ally_team = list(ally_team) if ally_team is not None else None
            self.enemy_team = list(enemy_team) if enemy_team is not None else None
        else:
            self.ally_team = None
            self.enemy_team = None

    def _sync_map_geometry(self) -> None:
        env = self._session.env
        if env is None:
            return
        try:
            game_info = env.game_info[0]
            map_size = game_info.start_raw.map_size
            self.map_x = float(map_size.x)
            self.map_y = float(map_size.y)
            playable = getattr(game_info.start_raw, 'playable_area', None)
            if playable is not None:
                p0 = getattr(playable, 'p0', None)
                p1 = getattr(playable, 'p1', None)
                if p0 is not None and p1 is not None:
                    self.max_distance_x = max(float(p1.x - p0.x), 1.0)
                    self.max_distance_y = max(float(p1.y - p0.y), 1.0)
                else:
                    self.max_distance_x = self.map_x
                    self.max_distance_y = self.map_y
            else:
                self.max_distance_x = self.map_x
                self.max_distance_y = self.map_y
            self.pathing_grid = _decode_pathing_grid(
                map_info=game_info.start_raw,
                map_x=int(self.map_x),
                map_y=int(self.map_y),
            )
            self.terrain_height = _decode_terrain_height(
                map_info=game_info.start_raw,
                map_x=int(self.map_x),
                map_y=int(self.map_y),
            )
        except Exception:
            return

    def _update_last_action_matrix(self, actions: Sequence[int]) -> None:
        action_idx = np.asarray(actions, dtype=np.int64)
        self.last_action.fill(0.0)
        self.last_action[self._agent_row_index, action_idx] = 1.0

    def _refresh_handler_context(self) -> None:
        if self._handler_context is None:
            self._handler_context = HandlerContext(
                family=self.variant,
                map_name=self.map_name,
                episode_step=self._episode_steps,
                n_agents=self.n_agents,
                n_enemies=self.n_enemies,
                n_actions=self.n_actions,
                n_actions_no_attack=self.n_actions_no_attack,
                attack_slots=self._attack_slots,
                move_amount=self._move_amount,
                map_x=self.map_x,
                map_y=self.map_y,
                max_distance_x=self.max_distance_x,
                max_distance_y=self.max_distance_y,
                state_last_action=self.state_last_action,
                last_action=self.last_action,
                reward_sparse=self.reward_sparse,
                reward_only_positive=self.reward_only_positive,
                reward_death_value=self.reward_death_value,
                reward_negative_scale=self.reward_negative_scale,
                reward_scale=self.reward_scale,
                reward_scale_rate=self.reward_scale_rate,
                max_reward=self.max_reward,
                variant_logic=self._variant_logic,
                unit_type_ids=self._unit_ids,
                switches=self.switches,
                env=self,
                pathing_grid=self.pathing_grid,
                terrain_height=self.terrain_height,
                n_fov_actions=self._n_fov_actions,
                conic_fov_angle=self._conic_fov_angle,
                fov_directions=self.fov_directions,
                canonical_fov_directions=self.canonical_fov_directions,
                action_mask=self._action_mask,
                ability_padding=self._ability_padding,
                use_ability=self._use_ability,
                obs_all_health=self.obs_all_health,
                obs_own_health=self.obs_own_health,
                obs_last_action=self.obs_last_action,
                obs_pathing_grid=self.obs_pathing_grid,
                obs_terrain_height=self.obs_terrain_height,
                obs_timestep_number=self.obs_timestep_number,
                state_timestep_number=self.state_timestep_number,
                obs_instead_of_state=self.obs_instead_of_state,
                shield_bits_ally=self.shield_bits_ally,
                shield_bits_enemy=self.shield_bits_enemy,
                unit_type_bits=self.unit_type_bits,
            )
            return

        ctx = self._handler_context
        ctx.episode_step = self._episode_steps
        ctx.map_x = self.map_x
        ctx.map_y = self.map_y
        ctx.max_distance_x = self.max_distance_x
        ctx.max_distance_y = self.max_distance_y
        ctx.last_action = self.last_action
        ctx.max_reward = self.max_reward
        ctx.unit_type_ids = self._unit_ids
        ctx.env = self
        ctx.pathing_grid = self.pathing_grid
        ctx.terrain_height = self.terrain_height
        ctx.fov_directions = self.fov_directions
        ctx.canonical_fov_directions = self.canonical_fov_directions

    def _forced_opponent_actions_for_step(
        self,
        step_idx: int,
    ) -> list[Any] | None:
        schedule = self._forced_opponent_actions_schedule
        if not schedule:
            return None
        if int(step_idx) < 0 or int(step_idx) >= len(schedule):
            return []
        return _decode_forced_opponent_actions(schedule[int(step_idx)])

    def _split_raw_units(self) -> tuple[list[Any], list[Any]]:
        if self._obs is None:
            self._last_split_probe = {}
            return [], []
        ally_units = list(self._obs.observation.raw_data.units)
        enemy_units = []
        if self._opponent_obs is not None:
            enemy_units = list(self._opponent_obs.observation.raw_data.units)

        allies = [u for u in ally_units if u.owner == 1]
        # In dual-controller scripted mode, legacy SMAC-Hard tracks enemy slots from
        # the second controller observation; keep the same source preference.
        enemies = [u for u in enemy_units if u.owner == 2]
        if not enemies:
            enemies = [u for u in ally_units if u.owner == 2]
        allies_sorted = sorted(allies, key=attrgetter('unit_type', 'pos.x', 'pos.y'))
        if self.switches.opponent_mode == 'scripted_pool':
            # Scripted dual-controller observations may emit different raw enemy
            # ordering across runs with the same seed. Canonicalize by tag so
            # slot-to-tag assignment is deterministic and parity-stable.
            enemies_ordered = sorted(
                enemies,
                key=lambda unit: int(getattr(unit, 'tag', -1)),
            )
        else:
            # Legacy non-scripted families preserve observed enemy order.
            enemies_ordered = list(enemies)
        probe: dict[str, Any] = {
            'allies_sorted_tags': [int(getattr(u, 'tag', -1)) for u in allies_sorted],
            'enemies_sorted_tags': [int(getattr(u, 'tag', -1)) for u in enemies_ordered],
            'ally_health_filter': [],
            'enemy_mask_filter': [],
            'agent_health_levels': self.agent_health_levels.astype(float).tolist(),
            'enemy_mask': self.enemy_mask.astype(float).tolist(),
        }

        if self.agent_health_levels.size > 0:
            filtered_allies = []
            for idx, unit in enumerate(allies_sorted):
                unit_tag = int(getattr(unit, 'tag', -1))
                if idx >= self.agent_health_levels.size:
                    filtered_allies.append(unit)
                    probe['ally_health_filter'].append(
                        {
                            'index': int(idx),
                            'tag': unit_tag,
                            'threshold': None,
                            'health_ratio': None,
                            'kept': True,
                            'reason': 'out_of_threshold_vector',
                        }
                    )
                    continue
                health_max = float(getattr(unit, 'health_max', 0.0) or 0.0)
                if health_max <= 0.0:
                    filtered_allies.append(unit)
                    probe['ally_health_filter'].append(
                        {
                            'index': int(idx),
                            'tag': unit_tag,
                            'threshold': float(self.agent_health_levels[idx]),
                            'health_ratio': None,
                            'kept': True,
                            'reason': 'invalid_health_max',
                        }
                    )
                    continue
                health = float(getattr(unit, 'health', 0.0))
                health_ratio = float(health / health_max)
                threshold = float(self.agent_health_levels[idx])
                keep = bool(health_ratio >= threshold)
                if not keep:
                    probe['ally_health_filter'].append(
                        {
                            'index': int(idx),
                            'tag': unit_tag,
                            'threshold': threshold,
                            'health_ratio': health_ratio,
                            'kept': False,
                            'reason': 'below_threshold',
                        }
                    )
                    continue
                filtered_allies.append(unit)
                probe['ally_health_filter'].append(
                    {
                        'index': int(idx),
                        'tag': unit_tag,
                        'threshold': threshold,
                        'health_ratio': health_ratio,
                        'kept': True,
                        'reason': 'pass',
                    }
                )
            allies_sorted = filtered_allies

        if self.enemy_mask.size > 0:
            filtered_enemies = []
            for idx, unit in enumerate(enemies_ordered):
                unit_tag = int(getattr(unit, 'tag', -1))
                if idx >= self.enemy_mask.size:
                    filtered_enemies.append(unit)
                    probe['enemy_mask_filter'].append(
                        {
                            'index': int(idx),
                            'tag': unit_tag,
                            'mask': None,
                            'kept': True,
                            'reason': 'out_of_mask_vector',
                        }
                    )
                    continue
                mask_value = float(self.enemy_mask[idx])
                keep = bool(mask_value > 0.0)
                if keep:
                    filtered_enemies.append(unit)
                probe['enemy_mask_filter'].append(
                    {
                        'index': int(idx),
                        'tag': unit_tag,
                        'mask': mask_value,
                        'kept': keep,
                        'reason': 'pass' if keep else 'masked',
                    }
                )
            enemies_ordered = filtered_enemies
        probe['allies_filtered_tags'] = [int(getattr(u, 'tag', -1)) for u in allies_sorted]
        probe['enemies_filtered_tags'] = [int(getattr(u, 'tag', -1)) for u in enemies_ordered]
        self._last_split_probe = probe
        return allies_sorted, enemies_ordered

    def debug_step_probe(self) -> dict[str, Any]:
        frame = self._unit_frame
        return {
            'episode_step': int(self._episode_steps),
            'last_split_probe': dict(self._last_split_probe or {}),
            'tracker_probe': self._unit_tracker.debug_probe(),
            'ally_frame_tags': (
                frame.allies.tags.astype(int).tolist() if frame is not None else []
            ),
            'enemy_frame_tags': (
                frame.enemies.tags.astype(int).tolist() if frame is not None else []
            ),
            'ally_actions_probe': _summarize_sc_actions(self._last_ally_sc_actions),
            'opponent_actions_probe': _summarize_sc_actions(self._last_opponent_actions),
        }

    def _sync_legacy_unit_views(self) -> None:
        self.agents = self._unit_tracker.raw_units_by_id(ally=True)
        self.enemies = self._unit_tracker.raw_units_by_id(ally=False)

    def _init_unit_type_ids(self) -> None:
        if self._unit_frame is None:
            return
        ally_types = [
            unit.unit_type
            for unit in self._unit_frame.allies.units
            if unit.unit_type > 0
        ]
        if not ally_types:
            return
        min_unit_type = min(ally_types)
        self._unit_ids = self._variant_logic.infer_unit_type_ids(min_unit_type)

    def _init_max_reward(self) -> None:
        if self.max_reward > 0 or self._unit_frame is None:
            return
        self.max_reward = sum(
            (unit.health_max + unit.shield_max)
            for unit in self._unit_frame.enemies.units
        )
        self.max_reward += self.reward_win

    def _battle_outcome_code(self):
        if self._unit_frame is None:
            return None
        n_ally_alive = int(np.sum(self._unit_frame.allies.alive))
        n_enemy_alive = int(np.sum(self._unit_frame.enemies.alive))

        if (n_ally_alive == 0 and n_enemy_alive > 0) or self._only_medivac_left(True):
            return -1
        if (n_ally_alive > 0 and n_enemy_alive == 0) or self._only_medivac_left(False):
            return 1
        if n_ally_alive == 0 and n_enemy_alive == 0:
            return 0
        return None

    def _only_medivac_left(self, ally: bool) -> bool:
        medivac_id = self._unit_ids.medivac_id
        if self.map_params.map_type != 'MMM' or medivac_id <= 0:
            return False
        if self._unit_frame is None:
            return False
        units = (
            self._unit_frame.allies.units
            if ally
            else self._unit_frame.enemies.units
        )
        non_medivac_alive = [
            unit
            for unit in units
            if unit.alive and unit.unit_type != medivac_id
        ]
        return len(non_medivac_alive) == 0


def _decode_pathing_grid(*, map_info: Any, map_x: int, map_y: int) -> np.ndarray | None:
    pathing = getattr(map_info, 'pathing_grid', None)
    if pathing is None:
        return None
    try:
        if pathing.bits_per_pixel == 1:
            vals = np.array(list(pathing.data), dtype=np.uint8).reshape(
                map_x,
                int(map_y / 8),
            )
            unpacked = np.array(
                [[(byte >> bit) & 1 for byte in row for bit in range(7, -1, -1)] for row in vals],
                dtype=bool,
            )
            return np.transpose(unpacked)
        arr = np.array(list(pathing.data), dtype=bool).reshape(map_x, map_y)
        return np.invert(np.flip(np.transpose(arr), axis=1))
    except Exception:
        return None


def _extract_capability_vector(
    *,
    payload: Any,
    size: int,
    default: float,
) -> np.ndarray:
    if isinstance(payload, Mapping):
        values = payload.get('item', None)
    else:
        values = payload
    if values is None:
        return np.full(size, default, dtype=np.float32)
    arr = np.asarray(values, dtype=np.float32).flatten()
    if arr.size >= size:
        return arr[:size].astype(np.float32)
    padded = np.full(size, default, dtype=np.float32)
    if arr.size > 0:
        padded[: arr.size] = arr
    return padded


def _extract_positions(values: Any) -> np.ndarray | None:
    if values is None:
        return None
    try:
        arr = np.asarray(values, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 2:
            return None
        return arr
    except Exception:
        return None


def _decode_terrain_height(
    *,
    map_info: Any,
    map_x: int,
    map_y: int,
) -> np.ndarray | None:
    terrain = getattr(map_info, 'terrain_height', None)
    if terrain is None:
        return None
    try:
        arr = np.array(list(terrain.data), dtype=np.float32).reshape(map_x, map_y)
        return np.flip(np.transpose(arr), axis=1) / 255.0
    except Exception:
        return None


def _summarize_sc_actions(actions: Sequence[Any]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for idx, action in enumerate(list(actions)[:16]):
        payload: dict[str, Any] = {
            'idx': int(idx),
            'type': type(action).__name__,
        }
        command = getattr(getattr(action, 'action_raw', None), 'unit_command', None)
        if command is None:
            summary.append(payload)
            continue
        payload['ability_id'] = int(getattr(command, 'ability_id', 0))
        payload['target_unit_tag'] = int(getattr(command, 'target_unit_tag', 0))
        world_pos = getattr(command, 'target_world_space_pos', None)
        if world_pos is not None and hasattr(world_pos, 'x') and hasattr(world_pos, 'y'):
            payload['target_xy'] = [float(world_pos.x), float(world_pos.y)]
        payload['unit_tags'] = [int(tag) for tag in list(getattr(command, 'unit_tags', []))]
        summary.append(payload)
    return summary


def _decode_forced_opponent_actions(
    step_actions: Sequence[Mapping[str, Any]],
) -> list[Any]:
    actions: list[Any] = []
    for payload in list(step_actions or []):
        if not isinstance(payload, Mapping):
            continue
        ability_id = int(payload.get('ability_id', 0) or 0)
        unit_tags = [int(tag) for tag in list(payload.get('unit_tags', []) or [])]
        if ability_id <= 0 or not unit_tags:
            continue
        cmd = r_pb.ActionRawUnitCommand(
            ability_id=ability_id,
            unit_tags=unit_tags,
            queue_command=False,
        )
        target_unit_tag = int(payload.get('target_unit_tag', 0) or 0)
        if target_unit_tag > 0:
            cmd.target_unit_tag = target_unit_tag
        target_xy = payload.get('target_xy')
        if (
            isinstance(target_xy, Sequence)
            and len(target_xy) >= 2
            and target_unit_tag <= 0
        ):
            cmd.target_world_space_pos.CopyFrom(
                sc_common.Point2D(
                    x=float(target_xy[0]),
                    y=float(target_xy[1]),
                )
            )
        actions.append(sc_pb.Action(action_raw=r_pb.ActionRaw(unit_command=cmd)))
    return actions


class NormalizedEnvAdapter:
    """Adapter that exposes a normalized batched API over backend envs."""

    def __init__(
        self,
        env: Any,
        family: str,
        *,
        opponent_runtime: OpponentRuntime | None = None,
        opponent_config: Mapping[str, Any] | None = None,
    ):
        self._env = env
        self.family = family
        self._opponent_config = dict(opponent_config or {})
        self._opponent_runtime = opponent_runtime or EngineBotOpponentRuntime()
        self._opponent_runtime.bind_env(self._env, self.family)
        if hasattr(self._env, 'set_opponent_runtime'):
            self._env.set_opponent_runtime(self._opponent_runtime)
        if hasattr(self._env, 'set_runtime_lifecycle_owner'):
            self._env.set_runtime_lifecycle_owner('adapter')

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> StepBatch:
        if seed is not None:
            if hasattr(self._env, 'seed') and callable(self._env.seed):
                try:
                    self._env.seed(seed)
                except Exception:
                    if hasattr(self._env, '_seed'):
                        self._env._seed = seed
            elif hasattr(self._env, '_seed'):
                self._env._seed = seed

        reset_kwargs = dict(options or {})
        try:
            reset_result = self._env.reset(**reset_kwargs)
        except TypeError:
            if 'episode_config' in reset_kwargs:
                reset_result = self._env.reset(
                    episode_config=reset_kwargs['episode_config']
                )
            else:
                reset_result = self._env.reset()

        if isinstance(reset_result, tuple) and len(reset_result) >= 2:
            obs, state = reset_result[0], reset_result[1]
        else:
            obs = self._env.get_obs()
            state = self._env.get_state()

        episode_context = OpponentEpisodeContext(
            family=self.family,
            map_name=str(getattr(self._env, 'map_name', '')),
            seed=seed,
            episode_config=self._opponent_config,
        )
        self._opponent_runtime.on_reset(episode_context)
        return self._collect_batch(
            obs=obs,
            state=state,
            reward=0.0,
            terminated=False,
            info={},
        )

    def step(self, actions: Sequence[int]) -> StepBatch:
        before_ctx = OpponentStepContext(
            family=self.family,
            episode_step=self._episode_step(),
            actions=list(actions),
        )
        self._opponent_runtime.before_step(before_ctx)
        reward, terminated, info = self._env.step(list(actions))
        after_ctx = OpponentStepContext(
            family=self.family,
            episode_step=self._episode_step(),
            actions=list(actions),
            terminated=bool(terminated),
            info=dict(info),
        )
        self._opponent_runtime.after_step(after_ctx)
        return self._collect_batch(
            obs=self._env.get_obs(),
            state=self._env.get_state(),
            reward=reward,
            terminated=terminated,
            info=info,
        )

    def step_batch(self, actions: Sequence[int]) -> StepBatch:
        return self.step(actions)

    def sample_random_actions(
        self, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        rng = rng or np.random.default_rng()
        actions = []
        for agent_id in range(self._env.n_agents):
            avail = np.asarray(self._env.get_avail_agent_actions(agent_id))
            valid = np.flatnonzero(avail)
            if valid.size == 0:
                actions.append(0)
            else:
                actions.append(int(valid[rng.integers(0, valid.size)]))
        return np.asarray(actions, dtype=np.int64)

    def close(self) -> None:
        self._opponent_runtime.close()
        self._env.close()

    def get_env_info(self) -> dict:
        return self._env.get_env_info()

    def _collect_batch(
        self,
        *,
        obs,
        state,
        reward: float,
        terminated: bool,
        info: Mapping[str, Any],
    ) -> StepBatch:
        episode_step = self._episode_step()
        obs_array = np.asarray(obs, dtype=np.float32)
        state_array = np.asarray(state, dtype=np.float32)
        avail_actions = np.asarray(self._env.get_avail_actions(), dtype=np.int8)
        return StepBatch.from_components(
            obs=obs_array,
            state=state_array,
            avail_actions=avail_actions,
            reward=float(reward),
            terminated=terminated,
            info=dict(info),
            episode_step=episode_step,
        )

    def _episode_step(self) -> int:
        if hasattr(self._env, 'get_episode_step'):
            return int(self._env.get_episode_step())
        return int(getattr(self._env, '_episode_steps', 0))

    def __getattr__(self, name: str):
        return getattr(self._env, name)


class VectorEnvPool:
    """Simple pooled execution wrapper for multiple environment instances."""

    def __init__(
        self,
        *,
        env_fns: Sequence[Callable[[], Any]],
        mode: str = 'sync',
        max_workers: int | None = None,
    ):
        self.mode = str(mode or 'sync').strip().lower()
        if self.mode not in {'sync', 'thread'}:
            raise ValueError(f'Unsupported vector pool mode: {mode!r}')
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        workers = max_workers or max(self.num_envs, 1)
        self._executor: ThreadPoolExecutor | None = None
        if self.mode == 'thread' and self.num_envs > 0:
            self._executor = ThreadPoolExecutor(max_workers=workers)

    def reset(
        self,
        *,
        seeds: Sequence[int | None] | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> list[Any]:
        seeds_seq = list(seeds or [])
        return self._map_envs(
            lambda idx, env: _safe_reset_env(
                env=env,
                seed=seeds_seq[idx] if idx < len(seeds_seq) else None,
                options=options,
            )
        )

    def step(self, actions_batch: Sequence[Sequence[int]]) -> list[Any]:
        actions_seq = list(actions_batch or [])
        if len(actions_seq) != self.num_envs:
            raise ValueError(
                f'actions_batch length {len(actions_seq)} must match num_envs {self.num_envs}.'
            )
        return self._map_envs(
            lambda idx, env: env.step(list(actions_seq[idx])),
        )

    def step_batch(self, actions_batch: Sequence[Sequence[int]]) -> list[Any]:
        actions_seq = list(actions_batch or [])
        if len(actions_seq) != self.num_envs:
            raise ValueError(
                f'actions_batch length {len(actions_seq)} must match num_envs {self.num_envs}.'
            )
        return self._map_envs(
            lambda idx, env: _safe_step_batch_env(env=env, actions=actions_seq[idx]),
        )

    def sample_random_actions(
        self,
        rng: np.random.Generator | None = None,
    ) -> list[np.ndarray]:
        rng = rng or np.random.default_rng()
        actions: list[np.ndarray] = []
        for env in self.envs:
            if hasattr(env, 'sample_random_actions'):
                actions.append(np.asarray(env.sample_random_actions(rng=rng), dtype=np.int64))
                continue
            n_agents = int(getattr(env, 'n_agents', 0))
            actions.append(np.zeros(n_agents, dtype=np.int64))
        return actions

    def close(self) -> None:
        errors = []
        for env in self.envs:
            try:
                env.close()
            except Exception as exc:
                errors.append(exc)
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
        if errors:
            raise RuntimeError(
                f'One or more vector pool environments failed to close: {errors}'
            )

    def get_env_info(self) -> list[dict[str, Any]]:
        infos = []
        for env in self.envs:
            if hasattr(env, 'get_env_info'):
                infos.append(dict(env.get_env_info()))
            else:
                infos.append({})
        return infos

    def _map_envs(self, fn: Callable[[int, Any], Any]) -> list[Any]:
        if self.mode == 'thread' and self._executor is not None:
            futures = [
                self._executor.submit(fn, idx, env)
                for idx, env in enumerate(self.envs)
            ]
            return [future.result() for future in futures]
        return [fn(idx, env) for idx, env in enumerate(self.envs)]


def _default_opponent_runtime(
    family: str,
    switches: VariantSwitches,
    opponent_config: Mapping[str, Any] | None,
) -> OpponentRuntime:
    del family
    if switches.opponent_mode == 'scripted_pool':
        return build_scripted_runtime_from_config(opponent_config or {})
    return EngineBotOpponentRuntime()


def _normalize_transport_profile(value: str | None) -> str:
    if value is None:
        return 'B0'
    profile = str(value).strip().upper()
    if profile not in _TRANSPORT_PROFILE_OPTIONS:
        raise ValueError(f'Unsupported transport_profile: {value}')
    return profile


def _merge_native_transport_options(
    *,
    transport_profile: str | None,
    native_options: Mapping[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    profile = _normalize_transport_profile(transport_profile)
    merged: dict[str, Any] = dict(_TRANSPORT_PROFILE_OPTIONS[profile])
    merged.update(dict(native_options or {}))
    return profile, merged


def make_env(
    *,
    family: str,
    map_name: str = '8m',
    normalized_api: bool = True,
    capability_config: Optional[Dict[str, Any]] = None,
    source_root: str | None = None,
    transport_profile: Literal['B0', 'B1', 'B2', 'B3', 'B4'] | None = None,
    allow_experimental_transport: bool = False,
    logic_switches: Mapping[str, str] | None = None,
    native_options: Mapping[str, Any] | None = None,
    observation_handler: Any | None = None,
    state_handler: Any | None = None,
    reward_handler: Any | None = None,
    action_handler: Any | None = None,
    opponent_runtime: OpponentRuntime | None = None,
    opponent_config: Mapping[str, Any] | None = None,
    **kwargs,
):
    for removed_key in ('backend_mode', 'backend_registry', 'bridge_options'):
        if removed_key in kwargs:
            raise TypeError(
                f"make_env() got an unexpected keyword argument '{removed_key}'"
            )
    switches = merge_switches(family, dict(logic_switches or {}))
    handler_overrides: Dict[str, Any] = {}
    if action_handler is not None:
        handler_overrides['action_handler'] = action_handler
    if _has_callable_attr(observation_handler, 'build_agent_obs'):
        handler_overrides['observation_handler'] = observation_handler
    if _has_callable_attr(state_handler, 'build_state'):
        handler_overrides['state_handler'] = state_handler
    if _has_callable_attr(reward_handler, 'build_step_reward'):
        handler_overrides['reward_handler'] = reward_handler

    resolved_transport_profile, resolved_native_options = _merge_native_transport_options(
        transport_profile=transport_profile,
        native_options=native_options,
    )

    env_kwargs = dict(kwargs)
    if switches is not None:
        env_kwargs.setdefault('logic_switches', switches)
    handler_map = {
        'action_handler': 'action_handler',
        'observation_handler': 'observation_handler',
        'state_handler': 'state_handler',
        'reward_handler': 'reward_handler',
    }
    for src_key, dst_key in handler_map.items():
        value = handler_overrides.get(src_key)
        if value is not None:
            env_kwargs[dst_key] = value

    env = SMACEnvCore(
        variant=family,
        map_name=map_name,
        capability_config=capability_config,
        env_kwargs=env_kwargs,
        source_root=source_root,
        native_options=resolved_native_options,
        transport_profile=resolved_transport_profile,
        allow_experimental_transport=bool(allow_experimental_transport),
    )
    resolved_opponent_config: dict[str, Any] = dict(opponent_config or {})
    if 'seed' not in resolved_opponent_config and 'seed' in env_kwargs:
        resolved_opponent_config['seed'] = env_kwargs['seed']
    runtime = opponent_runtime or _default_opponent_runtime(
        family,
        switches,
        resolved_opponent_config,
    )
    if not normalized_api:
        if hasattr(env, 'set_opponent_runtime'):
            env.set_opponent_runtime(runtime)
        if hasattr(env, 'set_runtime_lifecycle_owner'):
            env.set_runtime_lifecycle_owner('env')
        return env

    return NormalizedEnvAdapter(
        env,
        family=family,
        opponent_runtime=runtime,
        opponent_config=resolved_opponent_config,
    )


class UnifiedFactory:
    """Unified entry point for smac/smacv2/smac-hard environments."""

    @staticmethod
    def make_env(config: EnvFactoryConfig):
        return make_env(
            family=config.family,
            map_name=config.map_name,
            normalized_api=config.normalized_api,
            capability_config=config.capability_config,
            source_root=config.source_root,
            transport_profile=config.transport_profile,
            allow_experimental_transport=config.allow_experimental_transport,
            logic_switches=config.logic_switches,
            native_options=config.native_options,
            observation_handler=config.observation_handler,
            state_handler=config.state_handler,
            reward_handler=config.reward_handler,
            action_handler=config.action_handler,
            opponent_runtime=config.opponent_runtime,
            opponent_config=config.opponent_config,
            **config.resolved_env_kwargs(),
        )


def make_env_pool(
    *,
    num_envs: int,
    family: str,
    map_name: str = '8m',
    normalized_api: bool = True,
    capability_config: Optional[Dict[str, Any]] = None,
    source_root: str | None = None,
    transport_profile: Literal['B0', 'B1', 'B2', 'B3', 'B4'] | None = None,
    allow_experimental_transport: bool = False,
    logic_switches: Mapping[str, str] | None = None,
    native_options: Mapping[str, Any] | None = None,
    observation_handler: Any | None = None,
    state_handler: Any | None = None,
    reward_handler: Any | None = None,
    action_handler: Any | None = None,
    opponent_runtime: OpponentRuntime | None = None,
    opponent_config: Mapping[str, Any] | None = None,
    pool_mode: Literal['sync', 'thread'] = 'sync',
    pool_max_workers: int | None = None,
    **kwargs,
):
    env_count = int(num_envs)
    if env_count <= 0:
        raise ValueError('num_envs must be >= 1')

    base_kwargs = dict(kwargs)
    base_seed = base_kwargs.get('seed')

    def _build_env(index: int):
        env_kwargs = dict(base_kwargs)
        if base_seed is not None:
            env_kwargs['seed'] = int(base_seed) + int(index)
        return make_env(
            family=family,
            map_name=map_name,
            normalized_api=normalized_api,
            capability_config=capability_config,
            source_root=source_root,
            transport_profile=transport_profile,
            allow_experimental_transport=allow_experimental_transport,
            logic_switches=logic_switches,
            native_options=native_options,
            observation_handler=observation_handler,
            state_handler=state_handler,
            reward_handler=reward_handler,
            action_handler=action_handler,
            opponent_runtime=opponent_runtime,
            opponent_config=opponent_config,
            **env_kwargs,
        )

    env_fns = [lambda idx=idx: _build_env(idx) for idx in range(env_count)]
    return VectorEnvPool(
        env_fns=env_fns,
        mode=pool_mode,
        max_workers=pool_max_workers,
    )


def _has_callable_attr(value: Any, name: str) -> bool:
    return value is not None and callable(getattr(value, name, None))


def _safe_reset_env(
    *,
    env: Any,
    seed: int | None,
    options: Mapping[str, Any] | None,
):
    kwargs: dict[str, Any] = {}
    if seed is not None:
        kwargs['seed'] = int(seed)
    if options is not None:
        kwargs['options'] = dict(options)
    if kwargs:
        try:
            return env.reset(**kwargs)
        except TypeError:
            pass
    if seed is not None and hasattr(env, 'seed') and callable(env.seed):
        env.seed(int(seed))
    if options and isinstance(options, Mapping):
        try:
            return env.reset(episode_config=options.get('episode_config', options))
        except TypeError:
            pass
    return env.reset()


def _safe_step_batch_env(*, env: Any, actions: Sequence[int]):
    if hasattr(env, 'step_batch'):
        try:
            return env.step_batch(list(actions))
        except TypeError:
            pass
    reward, terminated, info = env.step(list(actions))
    return {
        'obs': env.get_obs(),
        'state': env.get_state(),
        'avail_actions': env.get_avail_actions(),
        'reward': reward,
        'terminated': terminated,
        'info': info,
    }
