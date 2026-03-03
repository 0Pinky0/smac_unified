from __future__ import annotations

import numpy as np

from ..types import HandlerContext, UnitFrame
from .base import StateHandler


class DefaultStateHandler(StateHandler):
    def build_state(
        self,
        *,
        frame: UnitFrame,
        context: HandlerContext,
    ):
        ally_state = np.zeros((context.n_agents, 5), dtype=np.float32)
        for unit in frame.allies.units:
            ally_state[unit.unit_id, 0] = unit.health / max(unit.health_max, 1.0)
            ally_state[unit.unit_id, 1] = unit.shield / max(unit.shield_max, 1.0)
            ally_state[unit.unit_id, 2] = unit.weapon_cooldown / 30.0
            ally_state[unit.unit_id, 3] = unit.x / max(context.map_x, 1.0)
            ally_state[unit.unit_id, 4] = unit.y / max(context.map_y, 1.0)

        enemy_state = np.zeros((context.n_enemies, 5), dtype=np.float32)
        for unit in frame.enemies.units:
            enemy_state[unit.unit_id, 0] = unit.health / max(unit.health_max, 1.0)
            enemy_state[unit.unit_id, 1] = unit.shield / max(unit.shield_max, 1.0)
            enemy_state[unit.unit_id, 2] = unit.weapon_cooldown / 30.0
            enemy_state[unit.unit_id, 3] = unit.x / max(context.map_x, 1.0)
            enemy_state[unit.unit_id, 4] = unit.y / max(context.map_y, 1.0)

        chunks = [ally_state.flatten(), enemy_state.flatten()]
        if context.state_last_action:
            chunks.append(np.asarray(context.last_action, dtype=np.float32).flatten())
        return np.concatenate(chunks, axis=0).astype(np.float32)


class CapabilityStateHandler(DefaultStateHandler):
    """SMACv2 capability-aware global-state semantics (scaffolded variant)."""
