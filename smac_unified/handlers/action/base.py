from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence

from ..types import HandlerContext, UnitFrame


class ActionHandler(ABC):
    """Unified action handler contract for native and bridge runtimes."""

    def reset(
        self,
        *,
        frame: UnitFrame,
        context: HandlerContext,
    ) -> None:
        del frame, context

    def resolve_action_space(
        self,
        *,
        n_agents: int,
        n_enemies: int,
        env_kwargs: Mapping[str, Any],
    ) -> tuple[int, int, int]:
        del env_kwargs
        attack_slots = max(int(n_agents), int(n_enemies))
        n_actions_no_attack = 6
        n_actions = n_actions_no_attack + attack_slots
        return n_actions_no_attack, n_actions, attack_slots

    @abstractmethod
    def get_avail_agent_actions(
        self,
        *,
        frame: UnitFrame,
        context: HandlerContext,
        agent_id: int,
    ) -> list[int]:
        raise NotImplementedError

    @abstractmethod
    def build_agent_action(
        self,
        *,
        frame: UnitFrame,
        context: HandlerContext,
        agent_id: int,
        action: int,
    ) -> Any | None:
        raise NotImplementedError

    def build_opponent_actions(
        self,
        *,
        frame: UnitFrame,
        context: HandlerContext,
        actions: Sequence[int],
        runtime: Any | None,
    ) -> Sequence[Any]:
        del frame, context, actions, runtime
        return []
