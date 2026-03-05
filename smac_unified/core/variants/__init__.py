from __future__ import annotations

from ..switches import VariantName, VariantSwitches
from ...maps import MapParams
from .base import UnitTypeIds, VariantLogic
from .smac import SmacVariantLogic
from .smac_hard import SmacHardVariantLogic
from .smacv2 import SmacV2VariantLogic


def build_variant_logic(
    switches: VariantSwitches,
    map_params: MapParams,
) -> VariantLogic:
    if switches.variant == 'smac':
        return SmacVariantLogic(switches, map_params)
    if switches.variant == 'smacv2':
        return SmacV2VariantLogic(switches, map_params)
    if switches.variant == 'smac-hard':
        return SmacHardVariantLogic(switches, map_params)
    raise ValueError(f'Unsupported variant {switches.variant!r}')


def default_switches(variant: VariantName) -> VariantSwitches:
    if variant == 'smac':
        return VariantSwitches(
            variant='smac',
            action_mode='classic',
            opponent_mode='sc2_computer',
            capability_mode='none',
            reward_positive_mode='abs',
            team_init_mode='map_default',
        )
    if variant == 'smacv2':
        return VariantSwitches(
            variant='smacv2',
            action_mode='classic',
            opponent_mode='sc2_computer',
            capability_mode='none',
            reward_positive_mode='clamp_zero',
            team_init_mode='map_default',
        )
    if variant == 'smac-hard':
        return VariantSwitches(
            variant='smac-hard',
            action_mode='ability_augmented',
            opponent_mode='scripted_pool',
            capability_mode='none',
            reward_positive_mode='clamp_zero',
            team_init_mode='map_default',
        )
    raise ValueError(f'Unsupported variant: {variant!r}')


_VARIANT_LOCKED_FIELDS: dict[VariantName, tuple[str, ...]] = {
    'smac': (),
    'smacv2': ('reward_positive_mode',),
    'smac-hard': ('action_mode', 'opponent_mode', 'reward_positive_mode'),
}


def merge_switches(
    variant: VariantName,
    overrides: dict[str, str] | None = None,
) -> VariantSwitches:
    base = default_switches(variant)
    if not overrides:
        return base
    payload = {
        'variant': base.variant,
        'action_mode': base.action_mode,
        'opponent_mode': base.opponent_mode,
        'capability_mode': base.capability_mode,
        'reward_positive_mode': base.reward_positive_mode,
        'team_init_mode': base.team_init_mode,
    }
    payload.update(dict(overrides))
    for field in _VARIANT_LOCKED_FIELDS.get(variant, ()):
        payload[field] = getattr(base, field)
    return VariantSwitches(**payload)


__all__ = [
    'default_switches',
    'merge_switches',
    'UnitTypeIds',
    'VariantLogic',
    'SmacVariantLogic',
    'SmacV2VariantLogic',
    'SmacHardVariantLogic',
    'build_variant_logic',
]
