from __future__ import annotations

from typing import Sequence

import numpy as np


def build_batched_obs(obs_list: Sequence[np.ndarray]) -> np.ndarray:
    return np.asarray(obs_list, dtype=np.float32)


def build_batched_state(state_vector: Sequence[float]) -> np.ndarray:
    return np.asarray(state_vector, dtype=np.float32)
