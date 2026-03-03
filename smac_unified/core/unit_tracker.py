from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np


@dataclass
class UnitValueSnapshot:
    health: np.ndarray
    shield: np.ndarray
    alive: np.ndarray


class UnitTracker:
    """Track health/shield snapshots and update unit dicts by tag."""

    def __init__(self, n_units: int):
        self.n_units = n_units

    def snapshot(self, units: Dict[int, object]) -> UnitValueSnapshot:
        health = np.zeros(self.n_units, dtype=np.float32)
        shield = np.zeros(self.n_units, dtype=np.float32)
        alive = np.zeros(self.n_units, dtype=bool)
        for unit_id, unit in units.items():
            health[unit_id] = getattr(unit, "health", 0.0)
            shield[unit_id] = getattr(unit, "shield", 0.0)
            alive[unit_id] = getattr(unit, "health", 0.0) > 0
        return UnitValueSnapshot(health=health, shield=shield, alive=alive)

    @staticmethod
    def tag_lookup(observed_units: Iterable[object]) -> Dict[int, object]:
        return {unit.tag: unit for unit in observed_units}

    def update_units(
        self,
        units: Dict[int, object],
        observed_units: Iterable[object],
    ) -> Tuple[int, Dict[int, object]]:
        by_tag = self.tag_lookup(observed_units)
        alive_count = 0
        for unit_id, unit in units.items():
            updated = by_tag.get(unit.tag)
            if updated is None:
                unit.health = 0
                continue
            units[unit_id] = updated
            if updated.health > 0:
                alive_count += 1
        return alive_count, units
