from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any

from .base import BackendConfig


def _autodetect_source_root() -> Path | None:
    env_value = os.environ.get("SMAC_UNIFIED_SOURCE_ROOT", "").strip()
    if env_value:
        path = Path(env_value).expanduser().resolve()
        if path.exists():
            return path

    here = Path(__file__).resolve()
    candidates = [here.parents[3], here.parents[2], Path.cwd()]
    for candidate in candidates:
        if (candidate / "implementation" / "smac").exists():
            return candidate
    return None


def _prepend_path(path: Path) -> None:
    value = str(path)
    if value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)


def _bootstrap_source_paths(source_root: Path | None) -> None:
    if source_root is None:
        return

    candidates = [
        source_root,
        source_root / "implementation" / "smac",
        source_root / "implementation" / "smacv2",
        source_root / "implementation" / "smac-hard",
        source_root / "dependency" / "pysc2-compat",
    ]
    for path in candidates:
        if path.exists():
            _prepend_path(path)


def _import_symbol(
    module_name: str,
    symbol_name: str,
    *,
    source_root: str | None,
) -> Any:
    try:
        module = importlib.import_module(module_name)
        return getattr(module, symbol_name)
    except Exception as first_exc:
        root = Path(source_root).expanduser().resolve() if source_root else _autodetect_source_root()
        _bootstrap_source_paths(root)
        try:
            module = importlib.import_module(module_name)
            return getattr(module, symbol_name)
        except Exception as second_exc:
            raise ImportError(
                f"Unable to import {symbol_name} from {module_name}. "
                "Install legacy SMAC backends or provide sources via "
                "`source_root` / SMAC_UNIFIED_SOURCE_ROOT."
            ) from second_exc


class SmacBridgeBackend:
    family = "smac"
    kind = "bridge"
    priority = 100

    def is_available(self, config: BackendConfig) -> bool:
        del config
        return True

    def make_env(self, config: BackendConfig):
        constructor = _import_symbol(
            "smac.env",
            "StarCraft2Env",
            source_root=config.source_root,
        )
        kwargs = dict(config.env_kwargs)
        kwargs.setdefault("map_name", config.map_name)
        return constructor(**kwargs)


class SmacV2BridgeBackend:
    family = "smacv2"
    kind = "bridge"
    priority = 100

    def is_available(self, config: BackendConfig) -> bool:
        del config
        return True

    def make_env(self, config: BackendConfig):
        kwargs = dict(config.env_kwargs)
        kwargs.setdefault("map_name", config.map_name)
        if config.capability_config is not None:
            constructor = _import_symbol(
                "smacv2.env.starcraft2.wrapper",
                "StarCraftCapabilityEnvWrapper",
                source_root=config.source_root,
            )
            kwargs["capability_config"] = config.capability_config
            return constructor(**kwargs)

        constructor = _import_symbol(
            "smacv2.env",
            "StarCraft2Env",
            source_root=config.source_root,
        )
        return constructor(**kwargs)


class SmacHardBridgeBackend:
    family = "smac-hard"
    kind = "bridge"
    priority = 100

    def is_available(self, config: BackendConfig) -> bool:
        del config
        return True

    def make_env(self, config: BackendConfig):
        constructor = _import_symbol(
            "smac_hard.env",
            "StarCraft2Env",
            source_root=config.source_root,
        )
        kwargs = dict(config.env_kwargs)
        kwargs.setdefault("map_name", config.map_name)
        kwargs.setdefault("debug", False)
        return constructor(**kwargs)
