from __future__ import annotations

import math
from typing import Any, Iterable, List, Mapping

from s2clientprotocol import raw_pb2 as r_pb
from s2clientprotocol import sc2api_pb2 as sc_pb


ATTACK_ABILITY_ID = 23


class BaseScriptPolicy:
    def __init__(self, map_name: str | None = None):
        self.map_name = map_name or ""

    def script(self, agents, enemies, agent_ability, visible_matrix=None, iteration=0):
        del agents, enemies, agent_ability, visible_matrix, iteration
        return []


class NoopScriptPolicy(BaseScriptPolicy):
    pass


class AttackNearestScriptPolicy(BaseScriptPolicy):
    def script(self, agents, enemies, agent_ability, visible_matrix=None, iteration=0):
        del agent_ability, visible_matrix, iteration
        return _attack_actions_by_metric(
            controlled_units=_alive_units(agents.values()),
            target_units=_alive_units(enemies.values()),
            metric=lambda src, dst: _distance(src, dst),
        )


class AttackWeakestScriptPolicy(BaseScriptPolicy):
    def script(self, agents, enemies, agent_ability, visible_matrix=None, iteration=0):
        del agent_ability, visible_matrix, iteration
        return _attack_actions_by_metric(
            controlled_units=_alive_units(agents.values()),
            target_units=_alive_units(enemies.values()),
            metric=lambda src, dst: dst.health,
        )


def default_script_pool(map_name: str) -> List[type]:
    del map_name
    return [AttackNearestScriptPolicy, AttackWeakestScriptPolicy, NoopScriptPolicy]


def _alive_units(units: Iterable[Any]) -> List[Any]:
    return [unit for unit in units if getattr(unit, "health", 0) > 0]


def _distance(src, dst) -> float:
    return math.hypot(dst.pos.x - src.pos.x, dst.pos.y - src.pos.y)


def _attack_actions_by_metric(
    *,
    controlled_units: List[Any],
    target_units: List[Any],
    metric,
) -> List[sc_pb.Action]:
    if not controlled_units or not target_units:
        return []

    actions: List[sc_pb.Action] = []
    for unit in controlled_units:
        target = min(target_units, key=lambda dst: metric(unit, dst))
        cmd = r_pb.ActionRawUnitCommand(
            ability_id=ATTACK_ABILITY_ID,
            target_unit_tag=target.tag,
            unit_tags=[unit.tag],
            queue_command=False,
        )
        actions.append(sc_pb.Action(action_raw=r_pb.ActionRaw(unit_command=cmd)))
    return actions
