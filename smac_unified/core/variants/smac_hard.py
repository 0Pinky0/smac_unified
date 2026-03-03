from __future__ import annotations

from .base import VariantLogic


class SmacHardVariantLogic(VariantLogic):
    variant = 'smac-hard'

    def n_actions(self, *, n_agents: int, n_enemies: int) -> int:
        # Match the hard-style padded target branch sizing.
        padding = max(n_agents, n_enemies, 9)
        return 6 + padding

    def reward_positive_transform(self, value: float) -> float:
        return max(value, 0.0)
