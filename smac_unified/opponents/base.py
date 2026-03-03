from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Sequence


@dataclass
class OpponentEpisodeContext:
    family: str
    map_name: str
    seed: int | None = None
    episode_config: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class OpponentStepContext:
    family: str
    episode_step: int
    actions: Sequence[int]
    terminated: bool = False
    info: Mapping[str, Any] = field(default_factory=dict)
    payload: Mapping[str, Any] = field(default_factory=dict)


class OpponentRuntime(ABC):
    """Lifecycle hooks for opponent behavior integration."""

    def bind_env(self, env: Any, family: str) -> None:
        del env, family

    def on_reset(self, context: OpponentEpisodeContext) -> None:
        del context

    def before_step(self, context: OpponentStepContext) -> None:
        del context

    def after_step(self, context: OpponentStepContext) -> None:
        del context

    def compute_actions(self, context: OpponentStepContext):
        del context
        return []

    def close(self) -> None:
        return None


class OpponentPolicy(ABC):
    """Optional policy abstraction for scripted opponents."""

    def reset(self, context: OpponentEpisodeContext) -> None:
        del context

    def act(self, context: Dict[str, Any]):
        del context
        return []
