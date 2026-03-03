from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Protocol, runtime_checkable


@dataclass
class BackendConfig:
    family: str
    map_name: str = '8m'
    capability_config: Optional[Dict[str, Any]] = None
    env_kwargs: Dict[str, Any] = field(default_factory=dict)
    source_root: str | None = None
    backend_mode: Literal['native', 'bridge', 'auto'] = 'auto'
    native_options: Dict[str, Any] = field(default_factory=dict)
    bridge_options: Dict[str, Any] = field(default_factory=dict)
    logic_switches: Any | None = None
    handler_overrides: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class UnifiedEnv(Protocol):
    n_agents: int
    n_actions: int

    def reset(self, **kwargs):
        ...

    def step(self, actions):
        ...

    def get_obs(self):
        ...

    def get_state(self):
        ...

    def get_avail_actions(self):
        ...

    def get_avail_agent_actions(self, agent_id: int):
        ...

    def get_env_info(self) -> Dict[str, Any]:
        ...

    def close(self) -> None:
        ...


class EnvBackend(Protocol):
    family: str
    kind: Literal['native', 'bridge']
    priority: int

    def is_available(self, config: BackendConfig) -> bool:
        ...

    def make_env(self, config: BackendConfig) -> UnifiedEnv:
        ...
