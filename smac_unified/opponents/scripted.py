from __future__ import annotations

import inspect
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

from .base import OpponentEpisodeContext, OpponentRuntime


@dataclass
class ScriptedOpponentConfig:
    strategy: str = "random"  # random|fixed
    fixed_index: int = 0
    seed: int | None = None


class HardScriptCompatibilityWrapper:
    """Compatibility shim for varying SMAC-Hard script signatures."""

    def __init__(self, script_object: Any):
        self._script_object = script_object
        self._script_fn = script_object.script
        params = inspect.signature(self._script_fn).parameters
        self._arity = len(params)

    def script(
        self,
        enemies,
        agents,
        agent_ability,
        visible_matrix=None,
        episode_step=0,
    ):
        if self._arity >= 5:
            return self._script_fn(
                enemies,
                agents,
                agent_ability,
                visible_matrix,
                episode_step,
            )
        if self._arity == 4:
            # Legacy variants may treat arg4 as iteration.
            try:
                return self._script_fn(
                    enemies,
                    agents,
                    agent_ability,
                    episode_step,
                )
            except TypeError:
                return self._script_fn(
                    enemies,
                    agents,
                    agent_ability,
                    visible_matrix,
                )
        if self._arity == 3:
            return self._script_fn(enemies, agents, agent_ability)
        return self._script_fn(enemies, agents, agent_ability, episode_step)


class ScriptedOpponentRuntime(OpponentRuntime):
    """Runtime that injects map-specific script opponents for SMAC-Hard."""

    def __init__(
        self,
        *,
        script_dict: Mapping[str, Sequence[Any]] | None = None,
        config: ScriptedOpponentConfig | None = None,
    ):
        self._script_dict = dict(script_dict or {})
        self._config = config or ScriptedOpponentConfig()
        self._rng = random.Random(self._config.seed)
        self._env = None
        self._family = ""
        self._last_script_name = ""

    @property
    def last_script_name(self) -> str:
        return self._last_script_name

    def bind_env(self, env: Any, family: str) -> None:
        self._env = env
        self._family = family

    def on_reset(self, context: OpponentEpisodeContext) -> None:
        if self._family != "smac-hard" or self._env is None:
            return
        script_pool = self._resolve_script_pool(context.map_name)
        if not script_pool:
            return
        selected = self._select_script(script_pool)
        script_instance = self._instantiate_script(selected, context.map_name)
        wrapped = HardScriptCompatibilityWrapper(script_instance)
        self._env.dts_script = wrapped
        self._last_script_name = type(script_instance).__name__

    def _resolve_script_pool(self, map_name: str) -> List[Any]:
        if self._script_dict:
            return list(self._script_dict.get(map_name, []))
        try:
            from smac_hard.env.scripts import SCRIPT_DICT
        except Exception:
            return []
        return list(SCRIPT_DICT.get(map_name, []))

    def _select_script(self, pool: Sequence[Any]):
        if self._config.strategy == "fixed":
            index = self._config.fixed_index % len(pool)
            return pool[index]
        return self._rng.choice(list(pool))

    @staticmethod
    def _instantiate_script(script_class_or_obj: Any, map_name: str):
        if not isinstance(script_class_or_obj, type):
            return script_class_or_obj
        try:
            return script_class_or_obj(map_name)
        except TypeError:
            return script_class_or_obj()


def build_scripted_runtime_from_config(
    config: Mapping[str, Any] | None,
) -> ScriptedOpponentRuntime:
    cfg = config or {}
    runtime_cfg = ScriptedOpponentConfig(
        strategy=str(cfg.get("strategy", "random")),
        fixed_index=int(cfg.get("fixed_index", 0)),
        seed=cfg.get("seed"),
    )
    return ScriptedOpponentRuntime(config=runtime_cfg)
