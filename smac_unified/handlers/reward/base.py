from __future__ import annotations

from abc import ABC, abstractmethod

from ..types import HandlerContext, UnitFrame


class RewardHandler(ABC):
    """Builds step rewards from tracked unit frames."""

    def reset(
        self,
        *,
        frame: UnitFrame,
        context: HandlerContext,
    ) -> None:
        del frame, context

    @abstractmethod
    def build_step_reward(
        self,
        *,
        frame: UnitFrame,
        context: HandlerContext,
    ) -> float:
        raise NotImplementedError
