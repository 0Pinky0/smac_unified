from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass
class BuildContext:
    family: str
    env: Any
    episode_step: int = 0


@dataclass
class RewardContext:
    family: str
    env: Any
    episode_step: int
    terminated: bool
    info: Mapping[str, Any]


@dataclass(frozen=True)
class UnitPosition:
    x: float
    y: float


@dataclass(frozen=True)
class TrackedUnit:
    unit_id: int
    tag: int
    unit_type: int
    x: float
    y: float
    health: float
    health_max: float
    shield: float
    shield_max: float
    weapon_cooldown: float
    alive: bool
    owner: int = 0
    raw: Any | None = None

    @property
    def pos(self) -> UnitPosition:
        return UnitPosition(x=self.x, y=self.y)


@dataclass(frozen=True)
class UnitTeamFrame:
    units: tuple[TrackedUnit, ...]
    health: np.ndarray
    shield: np.ndarray
    alive: np.ndarray
    tags: np.ndarray


@dataclass(frozen=True)
class UnitFrame:
    allies: UnitTeamFrame
    enemies: UnitTeamFrame
    prev_allies_health: np.ndarray
    prev_allies_shield: np.ndarray
    prev_enemies_health: np.ndarray
    prev_enemies_shield: np.ndarray
    step_token: int


@dataclass
class BuilderContext:
    family: str
    map_name: str
    episode_step: int
    n_agents: int
    n_enemies: int
    n_actions: int
    n_actions_no_attack: int
    attack_slots: int
    move_amount: float
    map_x: float
    map_y: float
    max_distance_x: float
    max_distance_y: float
    state_last_action: bool
    last_action: np.ndarray
    reward_sparse: bool
    reward_only_positive: bool
    reward_death_value: float
    reward_negative_scale: float
    reward_scale: bool
    reward_scale_rate: float
    max_reward: float
    variant_logic: Any
    unit_type_ids: Any
    switches: Any
    env: Any | None = None
