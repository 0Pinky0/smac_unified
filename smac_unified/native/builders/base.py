from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

from s2clientprotocol import sc2api_pb2 as sc_pb


class NativeActionBuilder(ABC):
    """Builds action masks and translates discrete actions to SC2 actions."""

    def reset(self, env: Any) -> None:
        del env

    @abstractmethod
    def get_avail_agent_actions(self, env: Any, agent_id: int) -> list[int]:
        raise NotImplementedError

    @abstractmethod
    def build_agent_action(
        self,
        env: Any,
        agent_id: int,
        action: int,
    ) -> sc_pb.Action | None:
        raise NotImplementedError

    def build_opponent_actions(
        self,
        env: Any,
        actions: Sequence[int],
        runtime: Any | None,
    ) -> Sequence[Any]:
        del env, actions, runtime
        return []


class NativeObservationBuilder(ABC):
    """Builds per-agent observation vectors."""

    @abstractmethod
    def build_agent_obs(self, env: Any, agent_id: int):
        raise NotImplementedError

    def build_obs(self, env: Any):
        return [
            self.build_agent_obs(env, agent_id)
            for agent_id in range(env.n_agents)
        ]


class NativeStateBuilder(ABC):
    """Builds global state vectors."""

    @abstractmethod
    def build_state(self, env: Any):
        raise NotImplementedError


class NativeRewardBuilder(ABC):
    """Builds environment step rewards from unit delta snapshots."""

    def reset(self, env: Any) -> None:
        del env

    @abstractmethod
    def build_step_reward(self, env: Any) -> float:
        raise NotImplementedError
