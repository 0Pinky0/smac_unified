from __future__ import annotations

from typing import Dict

from .base import EnvBackend


class BackendRegistry:
    def __init__(self):
        self._backends: Dict[str, EnvBackend] = {}

    def register(self, backend: EnvBackend) -> None:
        self._backends[backend.family] = backend

    def get(self, family: str) -> EnvBackend:
        backend = self._backends.get(family)
        if backend is None:
            raise ValueError(
                f"Unsupported family={family!r}; "
                f"registered={sorted(self._backends)}"
            )
        return backend

    def has(self, family: str) -> bool:
        return family in self._backends
