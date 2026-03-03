from __future__ import annotations

from pathlib import Path

from .base import BackendConfig


def _native_source_paths(config: BackendConfig):
    if config.source_root:
        root = Path(config.source_root).expanduser().resolve()
        yield root / "dependency" / "pysc2-compat"
        yield root
    else:
        here = Path(__file__).resolve()
        for candidate in (here.parents[3], here.parents[2], Path.cwd()):
            yield candidate / "dependency" / "pysc2-compat"


class NativeUnifiedBackend:
    kind = "native"
    priority = 10

    def __init__(self, family: str):
        self.family = family

    def is_available(self, config: BackendConfig) -> bool:
        try:
            import s2clientprotocol  # noqa: F401
            import pysc2  # noqa: F401
            return True
        except Exception:
            for path in _native_source_paths(config):
                if path.exists():
                    return True
            return False

    def make_env(self, config: BackendConfig):
        from ..native import NativeStarCraft2Env

        env_kwargs = dict(config.env_kwargs)
        if config.logic_switches is not None:
            env_kwargs.setdefault("logic_switches", config.logic_switches)

        return NativeStarCraft2Env(
            variant=config.family,
            map_name=config.map_name,
            capability_config=config.capability_config,
            env_kwargs=env_kwargs,
            source_root=config.source_root,
            native_options=config.native_options,
        )
