from __future__ import annotations

from smac_unified.core import UnitTracker


class _Pos:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


class _Unit:
    def __init__(
        self,
        *,
        tag: int,
        unit_type: int,
        x: float,
        y: float,
        health: float,
        health_max: float,
        shield: float,
        shield_max: float,
        weapon_cooldown: float = 0.0,
    ):
        self.tag = tag
        self.unit_type = unit_type
        self.pos = _Pos(x=x, y=y)
        self.health = health
        self.health_max = health_max
        self.shield = shield
        self.shield_max = shield_max
        self.weapon_cooldown = weapon_cooldown


def _unit(
    *,
    tag: int,
    unit_type: int,
    x: float,
    y: float,
    health: float = 45.0,
    health_max: float = 45.0,
    shield: float = 0.0,
    shield_max: float = 0.0,
) -> _Unit:
    return _Unit(
        tag=tag,
        unit_type=unit_type,
        x=x,
        y=y,
        health=health,
        health_max=health_max,
        shield=shield,
        shield_max=shield_max,
    )


def test_unit_tracker_keeps_stable_ids_across_updates():
    tracker = UnitTracker(n_units=3, n_enemy_units=2)
    frame0 = tracker.reset(
        allies=[
            _unit(tag=30, unit_type=2, x=8, y=8),
            _unit(tag=20, unit_type=1, x=3, y=3),
            _unit(tag=10, unit_type=1, x=1, y=1),
        ],
        enemies=[
            _unit(tag=101, unit_type=4, x=20, y=20),
            _unit(tag=100, unit_type=4, x=18, y=18),
        ],
    )

    assert [u.tag for u in frame0.allies.units] == [10, 20, 30]
    assert [u.tag for u in frame0.enemies.units] == [101, 100]

    frame1 = tracker.update(
        allies=[
            _unit(tag=10, unit_type=1, x=2, y=1.5),
            _unit(tag=30, unit_type=2, x=8.5, y=8.2),
        ],
        enemies=[
            _unit(tag=100, unit_type=4, x=18.2, y=18.1),
            _unit(tag=101, unit_type=4, x=20.5, y=20.0),
        ],
    )

    assert frame1.step_token == 1
    assert frame1.allies.units[1].tag == 20
    assert frame1.allies.units[1].alive is False
    assert frame1.prev_allies_health[1] > 0.0


def test_unit_tracker_debug_probe_exposes_slot_tags_and_alive_flags():
    tracker = UnitTracker(n_units=2, n_enemy_units=2)
    tracker.reset(
        allies=[
            _unit(tag=20, unit_type=1, x=2, y=2),
            _unit(tag=10, unit_type=1, x=1, y=1),
        ],
        enemies=[
            _unit(tag=200, unit_type=2, x=8, y=8),
            _unit(tag=100, unit_type=2, x=7, y=7),
        ],
    )
    probe0 = tracker.debug_probe()
    assert probe0['step_token'] == 0
    assert [row['tag'] for row in probe0['ally_slots']] == [10, 20]
    assert [row['tag'] for row in probe0['enemy_slots']] == [200, 100]

    tracker.update(
        allies=[_unit(tag=10, unit_type=1, x=1.5, y=1.0)],
        enemies=[_unit(tag=100, unit_type=2, x=7.5, y=7.2)],
    )
    probe1 = tracker.debug_probe()
    assert probe1['step_token'] == 1
    assert probe1['ally_slots'][1]['tag'] == 20
    assert probe1['ally_slots'][1]['alive'] is False
