from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol


@dataclass
class BackendConfig:
    family: str
    map_name: str = "8m"
    capability_config: Optional[Dict[str, Any]] = None
    env_kwargs: Dict[str, Any] = field(default_factory=dict)
    source_root: str | None = None


class EnvBackend(Protocol):
    family: str

    def make_env(self, config: BackendConfig):
        ...
