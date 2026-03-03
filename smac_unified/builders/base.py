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
