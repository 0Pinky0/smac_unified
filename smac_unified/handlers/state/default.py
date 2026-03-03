from __future__ import annotations

import numpy as np

from ..types import HandlerContext, TrackedUnit, UnitFrame
from .base import StateHandler


class DefaultStateHandler(StateHandler):
    def build_state(
        self,
        *,
        frame: UnitFrame,
        context: HandlerContext,
    ):
        if context.obs_instead_of_state and context.env is not None:
            obs = context.env.get_obs()
            return np.asarray(obs, dtype=np.float32).flatten()

        ally_attr = 4 + context.shield_bits_ally + context.unit_type_bits
        enemy_attr = 3 + context.shield_bits_enemy + context.unit_type_bits
        ally_state = np.zeros((context.n_agents, ally_attr), dtype=np.float32)
        enemy_state = np.zeros((context.n_enemies, enemy_attr), dtype=np.float32)

        for unit in frame.allies.units:
            ally_state[unit.unit_id, :] = _ally_state_row(unit=unit, context=context)
        for unit in frame.enemies.units:
            enemy_state[unit.unit_id, :] = _enemy_state_row(unit=unit, context=context)

        chunks = [ally_state.flatten(), enemy_state.flatten()]
        if context.state_last_action:
            chunks.append(np.asarray(context.last_action, dtype=np.float32).flatten())
        if context.state_timestep_number:
            chunks.append(np.asarray([_timestep_ratio(context)], dtype=np.float32))
        return np.concatenate(chunks, axis=0).astype(np.float32)


class CapabilityStateHandler(DefaultStateHandler):
    """SMACv2 capability-aware global-state semantics."""

    def build_state(
        self,
        *,
        frame: UnitFrame,
        context: HandlerContext,
    ):
        base = super().build_state(frame=frame, context=context)
        extras: list[float] = []
        env = context.env
        if env is not None:
            attack_probs = getattr(env, 'agent_attack_probabilities', None)
            if attack_probs is not None:
                extras.extend([float(x) for x in np.asarray(attack_probs).flatten()])
            health_levels = getattr(env, 'agent_health_levels', None)
            if health_levels is not None:
                extras.extend([float(x) for x in np.asarray(health_levels).flatten()])
        if not extras:
            return base
        return np.concatenate((base, np.asarray(extras, dtype=np.float32)), axis=0)


def _ally_state_row(*, unit: TrackedUnit, context: HandlerContext) -> np.ndarray:
    row = np.zeros(4 + context.shield_bits_ally + context.unit_type_bits, dtype=np.float32)
    row[0] = unit.health / max(unit.health_max, 1.0)
    row[1] = unit.weapon_cooldown / 30.0
    row[2] = unit.x / max(context.map_x, 1.0)
    row[3] = unit.y / max(context.map_y, 1.0)
    cursor = 4
    if context.shield_bits_ally > 0:
        row[cursor] = unit.shield / max(unit.shield_max, 1.0)
        cursor += 1
    if context.unit_type_bits > 0:
        idx = _unit_type_index(unit=unit, context=context)
        if 0 <= idx < context.unit_type_bits:
            row[cursor + idx] = 1.0
    return row


def _enemy_state_row(*, unit: TrackedUnit, context: HandlerContext) -> np.ndarray:
    row = np.zeros(3 + context.shield_bits_enemy + context.unit_type_bits, dtype=np.float32)
    row[0] = unit.health / max(unit.health_max, 1.0)
    row[1] = unit.x / max(context.map_x, 1.0)
    row[2] = unit.y / max(context.map_y, 1.0)
    cursor = 3
    if context.shield_bits_enemy > 0:
        row[cursor] = unit.shield / max(unit.shield_max, 1.0)
        cursor += 1
    if context.unit_type_bits > 0:
        idx = _unit_type_index(unit=unit, context=context)
        if 0 <= idx < context.unit_type_bits:
            row[cursor + idx] = 1.0
    return row


def _unit_type_index(*, unit: TrackedUnit, context: HandlerContext) -> int:
    ids = context.unit_type_ids
    map_type = ''
    if context.env is not None:
        map_type = str(getattr(getattr(context.env, 'map_params', None), 'map_type', ''))
    if map_type in ('stalkers_and_zealots',):
        if unit.unit_type == getattr(ids, 'stalker_id', -1):
            return 0
        if unit.unit_type == getattr(ids, 'zealot_id', -1):
            return 1
    if map_type in ('colossi_stalkers_zealots',):
        if unit.unit_type == getattr(ids, 'colossus_id', -1):
            return 0
        if unit.unit_type == getattr(ids, 'stalker_id', -1):
            return 1
        if unit.unit_type == getattr(ids, 'zealot_id', -1):
            return 2
    if map_type in ('bane',):
        if unit.unit_type == getattr(ids, 'baneling_id', -1):
            return 0
        if unit.unit_type == getattr(ids, 'zergling_id', -1):
            return 1
    if map_type in ('MMM', 'terran_gen'):
        if unit.unit_type == getattr(ids, 'marauder_id', -1):
            return 0
        if unit.unit_type == getattr(ids, 'marine_id', -1):
            return 1
        if unit.unit_type == getattr(ids, 'medivac_id', -1):
            return 2

    candidates = [
        value
        for value in (
            getattr(ids, 'marine_id', 0),
            getattr(ids, 'marauder_id', 0),
            getattr(ids, 'medivac_id', 0),
            getattr(ids, 'stalker_id', 0),
            getattr(ids, 'zealot_id', 0),
            getattr(ids, 'colossus_id', 0),
            getattr(ids, 'hydralisk_id', 0),
            getattr(ids, 'zergling_id', 0),
            getattr(ids, 'baneling_id', 0),
        )
        if value > 0
    ]
    if unit.unit_type in candidates and context.unit_type_bits > 0:
        return candidates.index(unit.unit_type) % context.unit_type_bits
    return -1


def _timestep_ratio(context: HandlerContext) -> float:
    env = context.env
    episode_limit = float(getattr(env, 'episode_limit', 0) or 0.0) if env is not None else 0.0
    if episode_limit <= 0.0:
        return 0.0
    return float(context.episode_step) / episode_limit
