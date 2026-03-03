from __future__ import annotations

from ...config import VariantSwitches
from ...maps import MapParams
from .base import UnitTypeIds, VariantLogic
from .smac import SmacVariantLogic
from .smac_hard import SmacHardVariantLogic
from .smacv2 import SmacV2VariantLogic


def build_variant_logic(
    switches: VariantSwitches,
    map_params: MapParams,
) -> VariantLogic:
    if switches.variant == "smac":
        return SmacVariantLogic(switches, map_params)
    if switches.variant == "smacv2":
        return SmacV2VariantLogic(switches, map_params)
    if switches.variant == "smac-hard":
        return SmacHardVariantLogic(switches, map_params)
    raise ValueError(f"Unsupported variant {switches.variant!r}")


__all__ = [
    "UnitTypeIds",
    "VariantLogic",
    "SmacVariantLogic",
    "SmacV2VariantLogic",
    "SmacHardVariantLogic",
    "build_variant_logic",
]
