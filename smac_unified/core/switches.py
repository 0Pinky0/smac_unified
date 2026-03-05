from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


VariantName = Literal['smac', 'smacv2', 'smac-hard']
ActionMode = Literal['classic', 'conic_fov', 'ability_augmented']
OpponentMode = Literal['sc2_computer', 'scripted_pool']
CapabilityMode = Literal[
    'none',
    'stochastic_attack',
    'stochastic_health',
    'team_gen',
]
RewardPositiveMode = Literal['abs', 'clamp_zero']
TeamInitMode = Literal['map_default', 'episode_generated']


@dataclass(frozen=True)
class VariantSwitches:
    variant: VariantName
    action_mode: ActionMode
    opponent_mode: OpponentMode
    capability_mode: CapabilityMode
    reward_positive_mode: RewardPositiveMode
    team_init_mode: TeamInitMode
