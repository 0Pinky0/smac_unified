from __future__ import annotations

from .base import OpponentRuntime


class EngineBotOpponentRuntime(OpponentRuntime):
    """No-op runtime for built-in SC2 computer opponents."""

    def compute_actions(self, context):
        del context
        return []
