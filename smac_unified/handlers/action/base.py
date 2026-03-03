from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

from ..types import BuilderContext, UnitFrame


class ActionBuilder(ABC):
    """Unified action handler contract for native and bridge runtimes."""

    def reset(
        self,
        *,
        frame: UnitFrame,
        context: BuilderContext,
    ) -> None:
        del frame, context

    @abstractmethod
    def get_avail_agent_actions(
        self,
        *,
        frame: UnitFrame,
        context: BuilderContext,
        agent_id: int,
    ) -> list[int]:
        raise NotImplementedError

    @abstractmethod
    def build_agent_action(
        self,
        *,
        frame: UnitFrame,
        context: BuilderContext,
        agent_id: int,
        action: int,
    ) -> Any | None:
        raise NotImplementedError

    def build_opponent_actions(
        self,
        *,
        frame: UnitFrame,
        context: BuilderContext,
        actions: Sequence[int],
        runtime: Any | None,
    ) -> Sequence[Any]:
        del frame, context, actions, runtime
        return []


class NativeActionBuilder(ActionBuilder):
    """Compatibility alias for previous native-only action handler naming."""
