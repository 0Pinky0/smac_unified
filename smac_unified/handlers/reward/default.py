from __future__ import annotations

import numpy as np

from ..types import HandlerContext, UnitFrame
from .base import RewardHandler


class DefaultRewardHandler(RewardHandler):
    def __init__(self):
        self._death_tracker_ally: np.ndarray | None = None
        self._death_tracker_enemy: np.ndarray | None = None

    def reset(
        self,
        *,
        frame: UnitFrame,
        context: HandlerContext,
    ) -> None:
        del context
        n_agents = len(frame.allies.units)
        n_enemies = len(frame.enemies.units)
        self._death_tracker_ally = np.zeros(n_agents, dtype=np.int8)
        self._death_tracker_enemy = np.zeros(n_enemies, dtype=np.int8)

    def build_step_reward(
        self,
        *,
        frame: UnitFrame,
        context: HandlerContext,
    ) -> float:
        if context.reward_sparse:
            return 0.0
        if self._death_tracker_ally is None or self._death_tracker_enemy is None:
            self.reset(frame=frame, context=context)

        assert self._death_tracker_ally is not None
        assert self._death_tracker_enemy is not None

        delta_deaths = 0.0
        delta_ally = 0.0
        delta_enemy = 0.0
        neg_scale = context.reward_negative_scale

        for agent_id, unit in enumerate(frame.allies.units):
            prev_health = frame.prev_allies_health[agent_id] + frame.prev_allies_shield[agent_id]
            if not unit.alive:
                if self._death_tracker_ally[agent_id] == 0 and prev_health > 0:
                    self._death_tracker_ally[agent_id] = 1
                    if not context.reward_only_positive:
                        delta_deaths -= context.reward_death_value * neg_scale
                    delta_ally += prev_health * neg_scale
            else:
                if self._death_tracker_ally[agent_id] == 0:
                    cur_health = unit.health + unit.shield
                    delta_ally += max(prev_health - cur_health, 0.0) * neg_scale

        for enemy_id, unit in enumerate(frame.enemies.units):
            prev_health = frame.prev_enemies_health[enemy_id] + frame.prev_enemies_shield[enemy_id]
            if not unit.alive:
                if self._death_tracker_enemy[enemy_id] == 0 and prev_health > 0:
                    self._death_tracker_enemy[enemy_id] = 1
                    delta_deaths += context.reward_death_value
                    delta_enemy += prev_health
            else:
                if self._death_tracker_enemy[enemy_id] == 0:
                    cur_health = unit.health + unit.shield
                    delta_enemy += max(prev_health - cur_health, 0.0)

        if context.reward_only_positive:
            reward = context.variant_logic.reward_positive_transform(
                delta_enemy + delta_deaths
            )
        else:
            reward = delta_enemy + delta_deaths - delta_ally
        return float(reward)


class AbsolutePositiveRewardHandler(DefaultRewardHandler):
    """Legacy SMAC reward-positive semantics (abs transform)."""


class ClampPositiveRewardHandler(DefaultRewardHandler):
    """SMACv2/SMAC-Hard reward-positive semantics (clamp-at-zero)."""
