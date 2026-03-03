from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Sequence

import numpy as np


def _as_float_array(values: Sequence[Any]) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def _as_int_array(values: Sequence[Any]) -> np.ndarray:
    return np.asarray(values, dtype=np.int8)


@dataclass
class StepBatch:
    """Normalized batched payload returned by reset/step."""

    obs: np.ndarray
    state: np.ndarray
    avail_actions: np.ndarray
    reward: float = 0.0
    terminated: bool = False
    info: Dict[str, Any] = field(default_factory=dict)
    episode_step: int = 0

    @classmethod
    def from_components(
        cls,
        *,
        obs: Sequence[Any],
        state: Sequence[Any],
        avail_actions: Sequence[Any],
        reward: float = 0.0,
        terminated: bool = False,
        info: Dict[str, Any] | None = None,
        episode_step: int = 0,
    ) -> "StepBatch":
        return cls(
            obs=_as_float_array(obs),
            state=_as_float_array(state),
            avail_actions=_as_int_array(avail_actions),
            reward=float(reward),
            terminated=bool(terminated),
            info=dict(info or {}),
            episode_step=int(episode_step),
        )

    @classmethod
    def from_legacy(
        cls,
        *,
        obs: Sequence[Any],
        state: Sequence[Any],
        avail_actions: Sequence[Any],
        reward: float = 0.0,
        terminated: bool = False,
        info: Dict[str, Any] | None = None,
        episode_step: int = 0,
    ) -> "StepBatch":
        # Backward-compatible alias while adopting explicit components API.
        return cls.from_components(
            obs=obs,
            state=state,
            avail_actions=avail_actions,
            reward=reward,
            terminated=terminated,
            info=info,
            episode_step=episode_step,
        )
