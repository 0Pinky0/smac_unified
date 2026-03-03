from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

import numpy as np

from ..types import BuildContext, BuilderContext, UnitFrame


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


class FrameStateBuilder(ABC):
    """Builds global state vectors from tracked unit frames."""

    @abstractmethod
    def build_state(
        self,
        *,
        frame: UnitFrame,
        context: BuilderContext,
    ) -> np.ndarray:
        raise NotImplementedError


class NativeStateBuilder(FrameStateBuilder):
    """Compatibility alias for previous native-only state handler naming."""
