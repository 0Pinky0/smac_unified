from __future__ import annotations

from typing import Callable, Dict

import numpy as np


class ActionMaskCache:
    """Per-timestep cache for available-action masks."""

    def __init__(self, n_agents: int):
        self.n_agents = n_agents
        self._token = -1
        self._masks: Dict[int, np.ndarray] = {}

    def invalidate(self, token: int) -> None:
        if token != self._token:
            self._token = token
            self._masks.clear()

    def get(
        self,
        agent_id: int,
        build_fn: Callable[[int], np.ndarray],
    ) -> np.ndarray:
        if agent_id not in self._masks:
            self._masks[agent_id] = np.asarray(build_fn(agent_id), dtype=np.int8)
        return self._masks[agent_id]

    def get_all(self, build_fn: Callable[[int], np.ndarray]) -> np.ndarray:
        return np.asarray(
            [self.get(agent_id, build_fn) for agent_id in range(self.n_agents)],
            dtype=np.int8,
        )
