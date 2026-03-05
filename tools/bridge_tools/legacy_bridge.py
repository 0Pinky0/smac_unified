from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path
from typing import Any, Mapping


def _autodetect_source_root() -> Path | None:
    env_value = os.environ.get('SMAC_UNIFIED_SOURCE_ROOT', '').strip()
    if env_value:
        path = Path(env_value).expanduser().resolve()
        if path.exists():
            return path

    here = Path(__file__).resolve()
    candidates = [here.parents[3], here.parents[2], Path.cwd()]
    for candidate in candidates:
        if (candidate / 'implementation' / 'smac').exists():
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
        source_root / 'implementation' / 'smac',
        source_root / 'implementation' / 'smacv2',
        source_root / 'implementation' / 'smac-hard',
        source_root / 'dependency' / 'pysc2-compat',
    ]
    for path in candidates:
        if path.exists():
            _prepend_path(path)


def _install_compat_shim() -> None:
    module_name = 'smac_unified.compat'
    existing = sys.modules.get(module_name)
    if existing is not None and hasattr(existing, 'make_legacy_env'):
        return
    from .legacy_compat_shim import (
        LegacyEnvAdapter,
        make_legacy_env,
        translate_legacy_kwargs,
    )

    compat_module = types.ModuleType(module_name)
    compat_module.LegacyEnvAdapter = LegacyEnvAdapter
    compat_module.make_legacy_env = make_legacy_env
    compat_module.translate_legacy_kwargs = translate_legacy_kwargs
    compat_module.__all__ = [
        'LegacyEnvAdapter',
        'make_legacy_env',
        'translate_legacy_kwargs',
    ]
    sys.modules[module_name] = compat_module

    parent = sys.modules.get('smac_unified')
    if parent is not None:
        setattr(parent, 'compat', compat_module)


def _import_symbol(
    module_name: str,
    symbol_name: str,
    *,
    source_root: str | None,
) -> Any:
    _install_compat_shim()
    try:
        module = importlib.import_module(module_name)
        return getattr(module, symbol_name)
    except Exception:
        root = (
            Path(source_root).expanduser().resolve()
            if source_root
            else _autodetect_source_root()
        )
        _bootstrap_source_paths(root)
        module = importlib.import_module(module_name)
        return getattr(module, symbol_name)


def make_bridge_env(
    *,
    family: str,
    map_name: str,
    source_root: str | None = None,
    capability_config: Mapping[str, Any] | None = None,
    env_kwargs: Mapping[str, Any] | None = None,
):
    kwargs = dict(env_kwargs or {})
    kwargs.setdefault('map_name', map_name)

    if family == 'smac':
        constructor = _import_symbol(
            'smac.env',
            'StarCraft2Env',
            source_root=source_root,
        )
        return constructor(**kwargs)

    if family == 'smacv2':
        if capability_config is not None:
            constructor = _import_symbol(
                'smacv2.env.starcraft2.wrapper',
                'StarCraftCapabilityEnvWrapper',
                source_root=source_root,
            )
            kwargs['capability_config'] = dict(capability_config)
            return constructor(**kwargs)

        constructor = _import_symbol(
            'smacv2.env',
            'StarCraft2Env',
            source_root=source_root,
        )
        return constructor(**kwargs)

    if family == 'smac-hard':
        constructor = _import_symbol(
            'smac_hard.env',
            'StarCraft2Env',
            source_root=source_root,
        )
        kwargs.setdefault('debug', False)
        return constructor(**kwargs)

    raise ValueError(f'Unsupported bridge family: {family}')
