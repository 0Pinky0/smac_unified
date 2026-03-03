from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

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


class ObservationBuilder(ABC):
    """Builds normalized batched observations from raw backend output."""

    @abstractmethod
    def build(
        self,
        *,
        raw_obs: Sequence[Any],
        context: BuildContext,
    ) -> np.ndarray:
        raise NotImplementedError


class StateBuilder(ABC):
    """Builds normalized global state from raw backend output."""

    @abstractmethod
    def build(
        self,
        *,
        raw_state: Sequence[Any],
        context: BuildContext,
    ) -> np.ndarray:
        raise NotImplementedError


class RewardBuilder(ABC):
    """Builds normalized scalar reward from raw backend output."""

    @abstractmethod
    def build(
        self,
        *,
        raw_reward: float,
        context: RewardContext,
    ) -> float:
        raise NotImplementedError


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


class ActionBuilder(ABC):
    """Unified action builder contract for native and bridge runtimes."""

    def reset(
        self,
        *,
        frame: UnitFrame | None = None,
        context: BuilderContext | None = None,
    ) -> None:
        del frame, context

    @abstractmethod
    def get_avail_agent_actions(
        self,
        *args,
        **kwargs,
    ) -> list[int]:
        raise NotImplementedError

    @abstractmethod
    def build_agent_action(
        self,
        *args,
        **kwargs,
    ) -> Any | None:
        raise NotImplementedError

    def build_opponent_actions(
        self,
        *args,
        **kwargs,
    ) -> Sequence[Any]:
        del args, kwargs
        return []


class NativeActionBuilder(ActionBuilder):
    """Compatibility alias for previous native-only action builder naming."""


class FrameObservationBuilder(ABC):
    """Builds per-agent and batched observations from tracked unit frames."""

    @abstractmethod
    def build_agent_obs(
        self,
        *args,
        **kwargs,
    ) -> np.ndarray:
        raise NotImplementedError

    def build_obs(
        self,
        *args,
        **kwargs,
    ) -> list[np.ndarray]:
        frame = kwargs.get("frame")
        context = kwargs.get("context")
        if frame is None or context is None:
            # Backward-compatible implementations may override this method.
            raise NotImplementedError("build_obs requires frame and context.")
        return [
            self.build_agent_obs(frame=frame, context=context, agent_id=agent_id)
            for agent_id in range(context.n_agents)
        ]


class NativeObservationBuilder(FrameObservationBuilder):
    """Compatibility alias for previous native-only observation builder naming."""


class FrameStateBuilder(ABC):
    """Builds global state vectors from tracked unit frames."""

    @abstractmethod
    def build_state(
        self,
        *args,
        **kwargs,
    ) -> np.ndarray:
        raise NotImplementedError


class NativeStateBuilder(FrameStateBuilder):
    """Compatibility alias for previous native-only state builder naming."""


class FrameRewardBuilder(ABC):
    """Builds step rewards from tracked unit frames."""

    def reset(
        self,
        *args,
        **kwargs,
    ) -> None:
        del args, kwargs

    @abstractmethod
    def build_step_reward(
        self,
        *args,
        **kwargs,
    ) -> float:
        raise NotImplementedError


class NativeRewardBuilder(FrameRewardBuilder):
    """Compatibility alias for previous native-only reward builder naming."""
