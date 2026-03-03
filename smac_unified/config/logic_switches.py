from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal


VariantName = Literal["smac", "smacv2", "smac-hard"]
ActionMode = Literal["classic", "conic_fov", "ability_augmented"]
OpponentMode = Literal["sc2_computer", "scripted_pool"]
CapabilityMode = Literal[
    "none",
    "stochastic_attack",
    "stochastic_health",
    "team_gen",
]
RewardPositiveMode = Literal["abs", "clamp_zero"]
TeamInitMode = Literal["map_default", "episode_generated"]


@dataclass(frozen=True)
class VariantSwitches:
    variant: VariantName
    action_mode: ActionMode
    opponent_mode: OpponentMode
    capability_mode: CapabilityMode
    reward_positive_mode: RewardPositiveMode
    team_init_mode: TeamInitMode


def default_switches(variant: VariantName) -> VariantSwitches:
    if variant == "smac":
        return VariantSwitches(
            variant="smac",
            action_mode="classic",
            opponent_mode="sc2_computer",
            capability_mode="none",
            reward_positive_mode="abs",
            team_init_mode="map_default",
        )
    if variant == "smacv2":
        return VariantSwitches(
            variant="smacv2",
            action_mode="conic_fov",
            opponent_mode="sc2_computer",
            capability_mode="team_gen",
            reward_positive_mode="clamp_zero",
            team_init_mode="episode_generated",
        )
    if variant == "smac-hard":
        return VariantSwitches(
            variant="smac-hard",
            action_mode="ability_augmented",
            opponent_mode="scripted_pool",
            capability_mode="none",
            reward_positive_mode="clamp_zero",
            team_init_mode="map_default",
        )
    raise ValueError(f"Unsupported variant: {variant!r}")


def merge_switches(
    variant: VariantName,
    overrides: Dict[str, str] | None = None,
) -> VariantSwitches:
    base = default_switches(variant)
    if not overrides:
        return base
    payload = {
        "variant": base.variant,
        "action_mode": base.action_mode,
        "opponent_mode": base.opponent_mode,
        "capability_mode": base.capability_mode,
        "reward_positive_mode": base.reward_positive_mode,
        "team_init_mode": base.team_init_mode,
    }
    payload.update(overrides)
    return VariantSwitches(**payload)
