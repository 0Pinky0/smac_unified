from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SC2Session:
    """Small wrapper around controller lifecycle calls."""

    run_config: Any
    map_obj: Any
    window_size: tuple[int, int]
    sc2_proc: Any = None
    controller: Any = None

    def launch(self) -> None:
        self.sc2_proc = self.run_config.start(
            window_size=self.window_size,
            want_rgb=False,
        )
        self.controller = self.sc2_proc.controller

    def close(self) -> None:
        if self.sc2_proc is not None:
            self.sc2_proc.close()
            self.sc2_proc = None
            self.controller = None
