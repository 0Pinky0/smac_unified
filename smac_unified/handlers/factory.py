from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ..config import VariantSwitches
from ..maps import MapParams
from .action import (
    AbilityAugmentedActionHandler,
    ActionHandler,
    ClassicActionHandler,
    ConicFovActionHandler,
)
from .obs import CapabilityObservationHandler, ObservationHandler
from .reward import (
    AbsolutePositiveRewardHandler,
    ClampPositiveRewardHandler,
    RewardHandler,
)
from .state import CapabilityStateHandler, StateHandler
from .obs import DefaultObservationHandler
from .state import DefaultStateHandler


@dataclass(frozen=True)
class HandlerBundle:
    action_handler: ActionHandler
    observation_handler: ObservationHandler
    state_handler: StateHandler
    reward_handler: RewardHandler


def build_default_handler_bundle(
    *,
    switches: VariantSwitches,
    map_params: MapParams,
    env_kwargs: Mapping[str, object],
) -> HandlerBundle:
    del map_params
    action_handler = _build_action_handler(switches=switches, env_kwargs=env_kwargs)
    observation_handler = _build_observation_handler(
        switches=switches,
        env_kwargs=env_kwargs,
    )
    state_handler = _build_state_handler(
        switches=switches,
        env_kwargs=env_kwargs,
    )
    reward_handler = _build_reward_handler(
        switches=switches,
        env_kwargs=env_kwargs,
    )
    return HandlerBundle(
        action_handler=action_handler,
        observation_handler=observation_handler,
        state_handler=state_handler,
        reward_handler=reward_handler,
    )


def _build_action_handler(
    *,
    switches: VariantSwitches,
    env_kwargs: Mapping[str, object],
) -> ActionHandler:
    if switches.action_mode == 'classic':
        return ClassicActionHandler()
    if switches.action_mode == 'conic_fov':
        return ConicFovActionHandler(
            num_fov_actions=int(env_kwargs.get('num_fov_actions', 12)),
            action_mask=bool(env_kwargs.get('action_mask', True)),
        )
    if switches.action_mode == 'ability_augmented':
        return AbilityAugmentedActionHandler(
            use_ability=bool(env_kwargs.get('use_ability', True)),
        )
    return ClassicActionHandler()


def _build_observation_handler(
    *,
    switches: VariantSwitches,
    env_kwargs: Mapping[str, object],
) -> ObservationHandler:
    del env_kwargs
    if switches.capability_mode != 'none':
        return CapabilityObservationHandler()
    return DefaultObservationHandler()


def _build_state_handler(
    *,
    switches: VariantSwitches,
    env_kwargs: Mapping[str, object],
) -> StateHandler:
    del env_kwargs
    if switches.capability_mode != 'none':
        return CapabilityStateHandler()
    return DefaultStateHandler()


def _build_reward_handler(
    *,
    switches: VariantSwitches,
    env_kwargs: Mapping[str, object],
) -> RewardHandler:
    del env_kwargs
    if switches.reward_positive_mode == 'clamp_zero':
        return ClampPositiveRewardHandler()
    return AbsolutePositiveRewardHandler()

