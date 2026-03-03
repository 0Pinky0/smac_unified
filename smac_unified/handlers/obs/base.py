from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

import numpy as np

from ..types import BuildContext, BuilderContext, UnitFrame


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


class FrameObservationBuilder(ABC):
    """Builds per-agent and batched observations from tracked unit frames."""

    @abstractmethod
    def build_agent_obs(
        self,
        *,
        frame: UnitFrame,
        context: BuilderContext,
        agent_id: int,
    ) -> np.ndarray:
        raise NotImplementedError

    def build_obs(
        self,
        *,
        frame: UnitFrame,
        context: BuilderContext,
    ) -> list[np.ndarray]:
        return [
            self.build_agent_obs(frame=frame, context=context, agent_id=agent_id)
            for agent_id in range(context.n_agents)
        ]


class NativeObservationBuilder(FrameObservationBuilder):
    """Compatibility alias for previous native-only observation handler naming."""
