from __future__ import annotations

from dataclasses import dataclass
from operator import attrgetter
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from ..handlers import TrackedUnit, UnitFrame, UnitTeamFrame


@dataclass
class UnitValueSnapshot:
    health: np.ndarray
    shield: np.ndarray
    alive: np.ndarray


class UnitTracker:
    """Track stable unit identities and emit frame snapshots."""

    def __init__(self, n_units: int, n_enemy_units: int | None = None):
        self.n_units = n_units
        self.n_enemy_units = n_enemy_units if n_enemy_units is not None else n_units
        self._ally_units: list[TrackedUnit] = []
        self._enemy_units: list[TrackedUnit] = []
        self._prev_ally_health = np.zeros(self.n_units, dtype=np.float32)
        self._prev_ally_shield = np.zeros(self.n_units, dtype=np.float32)
        self._prev_enemy_health = np.zeros(self.n_enemy_units, dtype=np.float32)
        self._prev_enemy_shield = np.zeros(self.n_enemy_units, dtype=np.float32)
        self._step_token = 0

    def snapshot(self, units: Dict[int, object]) -> UnitValueSnapshot:
        health = np.zeros(self.n_units, dtype=np.float32)
        shield = np.zeros(self.n_units, dtype=np.float32)
        alive = np.zeros(self.n_units, dtype=bool)
        for unit_id, unit in units.items():
            health[unit_id] = getattr(unit, 'health', 0.0)
            shield[unit_id] = getattr(unit, 'shield', 0.0)
            alive[unit_id] = getattr(unit, 'health', 0.0) > 0
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

    def reset(
        self,
        *,
        allies: Sequence[object],
        enemies: Sequence[object],
    ) -> UnitFrame:
        self._step_token = 0
        self._ally_units = self._seed_team_units(
            units=allies,
            n_slots=self.n_units,
            owner=1,
        )
        self._enemy_units = self._seed_team_units(
            units=enemies,
            n_slots=self.n_enemy_units,
            owner=2,
        )
        self._prev_ally_health = self._team_health(self._ally_units)
        self._prev_ally_shield = self._team_shield(self._ally_units)
        self._prev_enemy_health = self._team_health(self._enemy_units)
        self._prev_enemy_shield = self._team_shield(self._enemy_units)
        return self.frame()

    def update(
        self,
        *,
        allies: Sequence[object],
        enemies: Sequence[object],
    ) -> UnitFrame:
        self._prev_ally_health = self._team_health(self._ally_units)
        self._prev_ally_shield = self._team_shield(self._ally_units)
        self._prev_enemy_health = self._team_health(self._enemy_units)
        self._prev_enemy_shield = self._team_shield(self._enemy_units)

        self._ally_units = self._update_team_units(
            current=self._ally_units,
            observed=allies,
            owner=1,
        )
        self._enemy_units = self._update_team_units(
            current=self._enemy_units,
            observed=enemies,
            owner=2,
        )
        self._step_token += 1
        return self.frame()

    def frame(self) -> UnitFrame:
        return UnitFrame(
            allies=self._team_frame(self._ally_units),
            enemies=self._team_frame(self._enemy_units),
            prev_allies_health=self._prev_ally_health.copy(),
            prev_allies_shield=self._prev_ally_shield.copy(),
            prev_enemies_health=self._prev_enemy_health.copy(),
            prev_enemies_shield=self._prev_enemy_shield.copy(),
            step_token=self._step_token,
        )

    def raw_units_by_id(self, *, ally: bool) -> Dict[int, object]:
        team = self._ally_units if ally else self._enemy_units
        return {
            idx: unit.raw
            for idx, unit in enumerate(team)
            if unit.raw is not None
        }

    @staticmethod
    def _team_health(units: Sequence[TrackedUnit]) -> np.ndarray:
        return np.asarray([u.health for u in units], dtype=np.float32)

    @staticmethod
    def _team_shield(units: Sequence[TrackedUnit]) -> np.ndarray:
        return np.asarray([u.shield for u in units], dtype=np.float32)

    @staticmethod
    def _sorted_units(units: Sequence[object]) -> list[object]:
        return sorted(units, key=attrgetter('unit_type', 'pos.x', 'pos.y'))

    def _seed_team_units(
        self,
        *,
        units: Sequence[object],
        n_slots: int,
        owner: int,
    ) -> list[TrackedUnit]:
        ordered = self._sorted_units(units)
        tracked: list[TrackedUnit] = []
        for unit_id in range(n_slots):
            raw_unit = ordered[unit_id] if unit_id < len(ordered) else None
            tracked.append(
                self._to_tracked_unit(
                    unit_id=unit_id,
                    raw_unit=raw_unit,
                    owner=owner,
                    fallback_tag=-1,
                    fallback_unit_type=0,
                )
            )
        return tracked

    def _update_team_units(
        self,
        *,
        current: Sequence[TrackedUnit],
        observed: Sequence[object],
        owner: int,
    ) -> list[TrackedUnit]:
        by_tag = self.tag_lookup(observed)
        consumed: set[int] = set()
        updated: list[TrackedUnit] = []

        for prev in current:
            raw_unit = by_tag.get(prev.tag)
            if raw_unit is not None:
                consumed.add(prev.tag)
            updated.append(
                self._to_tracked_unit(
                    unit_id=prev.unit_id,
                    raw_unit=raw_unit,
                    owner=owner,
                    fallback_tag=prev.tag,
                    fallback_unit_type=prev.unit_type,
                )
            )

        extras = [u for u in self._sorted_units(observed) if u.tag not in consumed]
        if extras:
            for idx, prev in enumerate(updated):
                if prev.tag >= 0:
                    continue
                if not extras:
                    break
                raw_unit = extras.pop(0)
                updated[idx] = self._to_tracked_unit(
                    unit_id=prev.unit_id,
                    raw_unit=raw_unit,
                    owner=owner,
                    fallback_tag=prev.tag,
                    fallback_unit_type=prev.unit_type,
                )
        return updated

    @staticmethod
    def _to_tracked_unit(
        *,
        unit_id: int,
        raw_unit: object | None,
        owner: int,
        fallback_tag: int,
        fallback_unit_type: int,
    ) -> TrackedUnit:
        if raw_unit is None:
            return TrackedUnit(
                unit_id=unit_id,
                tag=int(fallback_tag),
                unit_type=int(fallback_unit_type),
                x=0.0,
                y=0.0,
                health=0.0,
                health_max=1.0,
                shield=0.0,
                shield_max=1.0,
                weapon_cooldown=0.0,
                alive=False,
                owner=owner,
                raw=None,
            )
        return TrackedUnit(
            unit_id=unit_id,
            tag=int(getattr(raw_unit, 'tag', fallback_tag)),
            unit_type=int(getattr(raw_unit, 'unit_type', fallback_unit_type)),
            x=float(getattr(getattr(raw_unit, 'pos', None), 'x', 0.0)),
            y=float(getattr(getattr(raw_unit, 'pos', None), 'y', 0.0)),
            health=float(getattr(raw_unit, 'health', 0.0)),
            health_max=float(getattr(raw_unit, 'health_max', 1.0) or 1.0),
            shield=float(getattr(raw_unit, 'shield', 0.0)),
            shield_max=float(getattr(raw_unit, 'shield_max', 1.0) or 1.0),
            weapon_cooldown=float(getattr(raw_unit, 'weapon_cooldown', 0.0)),
            alive=float(getattr(raw_unit, 'health', 0.0)) > 0,
            owner=owner,
            raw=raw_unit,
        )

    @staticmethod
    def _team_frame(units: Sequence[TrackedUnit]) -> UnitTeamFrame:
        return UnitTeamFrame(
            units=tuple(units),
            health=np.asarray([u.health for u in units], dtype=np.float32),
            shield=np.asarray([u.shield for u in units], dtype=np.float32),
            alive=np.asarray([u.alive for u in units], dtype=bool),
            tags=np.asarray([u.tag for u in units], dtype=np.int64),
        )
