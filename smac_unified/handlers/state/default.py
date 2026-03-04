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
        ally_size = context.n_agents * ally_attr
        enemy_size = context.n_enemies * enemy_attr
        last_action_size = (
            context.n_agents * context.n_actions if context.state_last_action else 0
        )
        timestep_size = 1 if context.state_timestep_number else 0
        state = np.zeros(
            ally_size + enemy_size + last_action_size + timestep_size,
            dtype=np.float32,
        )
        center_x = context.map_x / 2.0
        center_y = context.map_y / 2.0

        for unit in frame.allies.units:
            if unit.unit_id < 0 or unit.unit_id >= context.n_agents:
                continue
            row_start = unit.unit_id * ally_attr
            _write_ally_state_row(
                out=state,
                start=row_start,
                unit=unit,
                context=context,
                center_x=center_x,
                center_y=center_y,
            )
        for unit in frame.enemies.units:
            if unit.unit_id < 0 or unit.unit_id >= context.n_enemies:
                continue
            row_start = ally_size + unit.unit_id * enemy_attr
            _write_enemy_state_row(
                out=state,
                start=row_start,
                unit=unit,
                context=context,
                center_x=center_x,
                center_y=center_y,
            )

        cursor = ally_size + enemy_size
        if context.state_last_action:
            state[cursor : cursor + last_action_size] = np.asarray(
                context.last_action,
                dtype=np.float32,
            ).reshape(-1)
            cursor += last_action_size
        if context.state_timestep_number:
            state[cursor] = _timestep_ratio(context)
        return state


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


def _write_ally_state_row(
    *,
    out: np.ndarray,
    start: int,
    unit: TrackedUnit,
    context: HandlerContext,
    center_x: float,
    center_y: float,
) -> None:
    out[start] = unit.health / max(unit.health_max, 1.0)
    out[start + 1] = unit.weapon_cooldown / max(
        _unit_max_cooldown(unit=unit, context=context),
        1.0,
    )
    out[start + 2] = (unit.x - center_x) / max(context.max_distance_x, 1.0)
    out[start + 3] = (unit.y - center_y) / max(context.max_distance_y, 1.0)
    cursor = 4
    if context.shield_bits_ally > 0:
        out[start + cursor] = unit.shield / max(unit.shield_max, 1.0)
        cursor += 1
    if context.unit_type_bits > 0:
        idx = _unit_type_index(unit=unit, context=context)
        if 0 <= idx < context.unit_type_bits:
            out[start + cursor + idx] = 1.0


def _write_enemy_state_row(
    *,
    out: np.ndarray,
    start: int,
    unit: TrackedUnit,
    context: HandlerContext,
    center_x: float,
    center_y: float,
) -> None:
    out[start] = unit.health / max(unit.health_max, 1.0)
    out[start + 1] = (unit.x - center_x) / max(context.max_distance_x, 1.0)
    out[start + 2] = (unit.y - center_y) / max(context.max_distance_y, 1.0)
    cursor = 3
    if context.shield_bits_enemy > 0:
        out[start + cursor] = unit.shield / max(unit.shield_max, 1.0)
        cursor += 1
    if context.unit_type_bits > 0:
        idx = _unit_type_index(unit=unit, context=context)
        if 0 <= idx < context.unit_type_bits:
            out[start + cursor + idx] = 1.0


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


def _unit_max_cooldown(*, unit: TrackedUnit, context: HandlerContext) -> float:
    ids = context.unit_type_ids
    lookup = {
        getattr(ids, 'marine_id', 0): 15.0,
        getattr(ids, 'marauder_id', 0): 25.0,
        getattr(ids, 'medivac_id', 0): 200.0,
        getattr(ids, 'stalker_id', 0): 35.0,
        getattr(ids, 'zealot_id', 0): 22.0,
        getattr(ids, 'colossus_id', 0): 24.0,
        getattr(ids, 'hydralisk_id', 0): 10.0,
        getattr(ids, 'zergling_id', 0): 11.0,
        getattr(ids, 'baneling_id', 0): 1.0,
    }
    value = lookup.get(unit.unit_type, 15.0)
    return max(float(value), 1.0)
