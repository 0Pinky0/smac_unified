from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..types import HandlerContext, UnitFrame


class ObservationHandler(ABC):
    """Builds per-agent and batched observations from tracked unit frames."""

    @abstractmethod
    def build_agent_obs(
        self,
        *,
        frame: UnitFrame,
        context: HandlerContext,
        agent_id: int,
    ) -> np.ndarray:
        raise NotImplementedError

    def build_obs(
        self,
        *,
        frame: UnitFrame,
        context: HandlerContext,
    ) -> list[np.ndarray]:
        return [
            self.build_agent_obs(frame=frame, context=context, agent_id=agent_id)
            for agent_id in range(context.n_agents)
        ]
