from __future__ import annotations

from .base import VariantLogic


class SmacV2VariantLogic(VariantLogic):
    variant = 'smacv2'

    def reward_positive_transform(self, value: float) -> float:
        # SMACv2 reward-positive branch clamps at zero.
        return max(value, 0.0)
