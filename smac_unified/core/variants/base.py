from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from ...maps import MapParams
from ..switches import VariantSwitches


@dataclass
class UnitTypeIds:
    marine_id: int = 0
    marauder_id: int = 0
    medivac_id: int = 0
    stalker_id: int = 0
    zealot_id: int = 0
    colossus_id: int = 0
    hydralisk_id: int = 0
    zergling_id: int = 0
    baneling_id: int = 0


class VariantLogic:
    variant = 'smac'

    def __init__(self, switches: VariantSwitches, map_params: MapParams):
        self.switches = switches
        self.map_params = map_params

    def n_actions(self, *, n_agents: int, n_enemies: int) -> int:
        attack_slots = max(n_agents, n_enemies)
        return 6 + attack_slots

    def reward_positive_transform(self, value: float) -> float:
        if self.switches.reward_positive_mode == 'clamp_zero':
            return max(value, 0.0)
        return abs(value)

    def infer_unit_type_ids(self, min_unit_type: int) -> UnitTypeIds:
        ids = UnitTypeIds()
        map_type = self.map_params.map_type
        if map_type == 'marines':
            ids.marine_id = min_unit_type
        elif map_type == 'stalkers_and_zealots':
            ids.stalker_id = min_unit_type
            ids.zealot_id = min_unit_type + 1
        elif map_type == 'colossi_stalkers_zealots':
            ids.colossus_id = min_unit_type
            ids.stalker_id = min_unit_type + 1
            ids.zealot_id = min_unit_type + 2
        elif map_type == 'MMM':
            ids.marauder_id = min_unit_type
            ids.marine_id = min_unit_type + 1
            ids.medivac_id = min_unit_type + 2
        elif map_type == 'zealots':
            ids.zealot_id = min_unit_type
        elif map_type == 'hydralisks':
            ids.hydralisk_id = min_unit_type
        elif map_type == 'stalkers':
            ids.stalker_id = min_unit_type
        elif map_type == 'colossus':
            ids.colossus_id = min_unit_type
        elif map_type == 'bane':
            ids.baneling_id = min_unit_type
            ids.zergling_id = min_unit_type + 1
        return ids

    def shoot_range_by_type(self, ids: UnitTypeIds) -> Dict[int, float]:
        return {
            ids.stalker_id: 6.0,
            ids.zealot_id: 0.1,
            ids.colossus_id: 7.0,
            ids.zergling_id: 0.1,
            ids.baneling_id: 0.25,
            ids.hydralisk_id: 5.0,
            ids.marine_id: 5.0,
            ids.marauder_id: 6.0,
            ids.medivac_id: 4.0,
        }

    @property
    def scripted_opponent_enabled(self) -> bool:
        return self.switches.opponent_mode == 'scripted_pool'

    @property
    def ability_mode_enabled(self) -> bool:
        return self.switches.action_mode == 'ability_augmented'
