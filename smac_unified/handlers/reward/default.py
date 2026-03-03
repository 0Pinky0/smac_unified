from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..types import BuilderContext, RewardContext, UnitFrame
from .base import NativeRewardBuilder, RewardBuilder


class DefaultRewardBuilder(RewardBuilder):
    def __init__(self, scale_from_env: bool = False):
        self.scale_from_env = scale_from_env

    def build(
        self,
        *,
        raw_reward: float,
        context: RewardContext,
    ) -> float:
        reward = float(raw_reward)
        if not self.scale_from_env:
            return reward

        max_reward = float(getattr(context.env, "max_reward", 0.0) or 0.0)
        scale_rate = float(
            getattr(context.env, "reward_scale_rate", 0.0) or 0.0
        )
        if max_reward <= 0.0 or scale_rate <= 0.0:
            return reward
        return reward / (max_reward / scale_rate)


class DefaultNativeRewardBuilder(NativeRewardBuilder):
    def __init__(self):
        self._death_tracker_ally: np.ndarray | None = None
        self._death_tracker_enemy: np.ndarray | None = None

    def reset(
        self,
        *,
        frame: UnitFrame,
        context: BuilderContext,
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
        context: BuilderContext,
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
        if context.reward_scale and context.max_reward > 0 and context.reward_scale_rate > 0:
            reward /= context.max_reward / context.reward_scale_rate
        return float(reward)


def builder_bundle(
    *,
    observation_builder: Any | None = None,
    state_builder: Any | None = None,
    reward_builder: Any | None = None,
) -> Mapping[str, Any]:
    from ..obs.default import DefaultObservationBuilder
    from ..state.default import DefaultStateBuilder

    return {
        "observation_builder": observation_builder or DefaultObservationBuilder(),
        "state_builder": state_builder or DefaultStateBuilder(),
        "reward_builder": reward_builder or DefaultRewardBuilder(),
    }
