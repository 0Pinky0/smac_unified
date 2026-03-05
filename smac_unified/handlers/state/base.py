from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..types import HandlerContext, UnitFrame


class StateHandler(ABC):
    """Builds global state vectors from tracked unit frames."""

    @abstractmethod
    def build_state(
        self,
        *,
        frame: UnitFrame,
        context: HandlerContext,
    ) -> np.ndarray:
        raise NotImplementedError

    def state_size(self, *, context: HandlerContext) -> int:
        del context
        return -1
