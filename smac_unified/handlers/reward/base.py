from __future__ import annotations

from abc import ABC, abstractmethod

from ..types import BuilderContext, RewardContext, UnitFrame


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


class FrameRewardBuilder(ABC):
    """Builds step rewards from tracked unit frames."""

    def reset(
        self,
        *,
        frame: UnitFrame,
        context: BuilderContext,
    ) -> None:
        del frame, context

    @abstractmethod
    def build_step_reward(
        self,
        *,
        frame: UnitFrame,
        context: BuilderContext,
    ) -> float:
        raise NotImplementedError


class NativeRewardBuilder(FrameRewardBuilder):
    """Compatibility alias for previous native-only reward handler naming."""
