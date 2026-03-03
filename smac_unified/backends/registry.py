from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List

from .base import BackendConfig, EnvBackend


class BackendRegistry:
    def __init__(self):
        self._backends: Dict[str, List[EnvBackend]] = defaultdict(list)

    def register(self, backend: EnvBackend) -> None:
        family_backends = self._backends[backend.family]
        family_backends = [
            b for b in family_backends if not (b.kind == backend.kind and type(b) is type(backend))
        ]
        family_backends.append(backend)
        family_backends.sort(key=lambda b: b.priority)
        self._backends[backend.family] = family_backends

    def all_for_family(self, family: str) -> Iterable[EnvBackend]:
        return tuple(self._backends.get(family, []))

    def get(
        self,
        family: str,
        *,
        mode: str = 'auto',
        config: BackendConfig | None = None,
    ) -> EnvBackend:
        candidates = list(self._backends.get(family, []))
        if not candidates:
            raise ValueError(
                f'Unsupported family={family!r}; '
                f'registered={sorted(self._backends.keys())}'
            )

        if mode not in {'native', 'bridge', 'auto'}:
            raise ValueError(
                f'Unsupported backend mode={mode!r}, expected native|bridge|auto'
            )

        cfg = config or BackendConfig(family=family)
        ordered: List[EnvBackend]
        if mode == 'native':
            ordered = [b for b in candidates if b.kind == 'native']
        elif mode == 'bridge':
            ordered = [b for b in candidates if b.kind == 'bridge']
        else:
            native = [b for b in candidates if b.kind == 'native']
            bridge = [b for b in candidates if b.kind == 'bridge']
            ordered = native + bridge

        available = [b for b in ordered if b.is_available(cfg)]
        if available:
            return available[0]

        available_kinds = sorted({b.kind for b in candidates})
        raise RuntimeError(
            'No backend implementation is available for '
            f'family={family!r}, mode={mode!r}. '
            f'Registered kinds={available_kinds}.'
        )

    def has(self, family: str) -> bool:
        return family in self._backends
