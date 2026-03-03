from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .base import BuildContext, ObservationBuilder, RewardBuilder, RewardContext, StateBuilder


class DefaultObservationBuilder(ObservationBuilder):
    def build(
        self,
        *,
        raw_obs: Sequence[Any],
        context: BuildContext,
    ) -> np.ndarray:
        del context
        return np.asarray(raw_obs, dtype=np.float32)


class DefaultStateBuilder(StateBuilder):
    def build(
        self,
        *,
        raw_state: Sequence[Any],
        context: BuildContext,
    ) -> np.ndarray:
        del context
        return np.asarray(raw_state, dtype=np.float32)


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


def builder_bundle(
    *,
    observation_builder: ObservationBuilder | None = None,
    state_builder: StateBuilder | None = None,
    reward_builder: RewardBuilder | None = None,
) -> Mapping[str, Any]:
    return {
        "observation_builder": observation_builder or DefaultObservationBuilder(),
        "state_builder": state_builder or DefaultStateBuilder(),
        "reward_builder": reward_builder or DefaultRewardBuilder(),
    }
