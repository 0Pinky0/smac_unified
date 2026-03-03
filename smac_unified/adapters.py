from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .players import (
    EngineBotOpponentRuntime,
    OpponentEpisodeContext,
    OpponentRuntime,
    OpponentStepContext,
)
from .types import StepBatch


class NormalizedEnvAdapter:
    """Adapter that exposes a normalized batched API over backend envs."""

    def __init__(
        self,
        env: Any,
        family: str,
        *,
        opponent_runtime: OpponentRuntime | None = None,
        opponent_config: Mapping[str, Any] | None = None,
    ):
        self._env = env
        self.family = family
        self._opponent_config = dict(opponent_config or {})
        self._opponent_runtime = opponent_runtime or EngineBotOpponentRuntime()
        self._opponent_runtime.bind_env(self._env, self.family)
        if hasattr(self._env, 'set_opponent_runtime'):
            self._env.set_opponent_runtime(self._opponent_runtime)
        if hasattr(self._env, 'set_runtime_lifecycle_owner'):
            self._env.set_runtime_lifecycle_owner('adapter')

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> StepBatch:
        if seed is not None:
            if hasattr(self._env, 'seed') and callable(self._env.seed):
                try:
                    self._env.seed(seed)
                except Exception:
                    if hasattr(self._env, '_seed'):
                        self._env._seed = seed
            elif hasattr(self._env, '_seed'):
                self._env._seed = seed

        reset_kwargs = dict(options or {})
        try:
            reset_result = self._env.reset(**reset_kwargs)
        except TypeError:
            if 'episode_config' in reset_kwargs:
                reset_result = self._env.reset(
                    episode_config=reset_kwargs['episode_config']
                )
            else:
                reset_result = self._env.reset()

        if isinstance(reset_result, tuple) and len(reset_result) >= 2:
            obs, state = reset_result[0], reset_result[1]
        else:
            obs = self._env.get_obs()
            state = self._env.get_state()

        episode_context = OpponentEpisodeContext(
            family=self.family,
            map_name=str(getattr(self._env, 'map_name', '')),
            seed=seed,
            episode_config=self._opponent_config,
        )
        self._opponent_runtime.on_reset(episode_context)
        return self._collect_batch(
            obs=obs,
            state=state,
            reward=0.0,
            terminated=False,
            info={},
        )

    def step(self, actions: Sequence[int]) -> StepBatch:
        before_ctx = OpponentStepContext(
            family=self.family,
            episode_step=self._episode_step(),
            actions=list(actions),
        )
        self._opponent_runtime.before_step(before_ctx)
        reward, terminated, info = self._env.step(list(actions))
        after_ctx = OpponentStepContext(
            family=self.family,
            episode_step=self._episode_step(),
            actions=list(actions),
            terminated=bool(terminated),
            info=dict(info),
        )
        self._opponent_runtime.after_step(after_ctx)
        return self._collect_batch(
            obs=self._env.get_obs(),
            state=self._env.get_state(),
            reward=reward,
            terminated=terminated,
            info=info,
        )

    def step_batch(self, actions: Sequence[int]) -> StepBatch:
        return self.step(actions)

    def sample_random_actions(
        self, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        rng = rng or np.random.default_rng()
        actions = []
        for agent_id in range(self._env.n_agents):
            avail = np.asarray(self._env.get_avail_agent_actions(agent_id))
            valid = np.flatnonzero(avail)
            if valid.size == 0:
                actions.append(0)
            else:
                actions.append(int(valid[rng.integers(0, valid.size)]))
        return np.asarray(actions, dtype=np.int64)

    def close(self) -> None:
        self._opponent_runtime.close()
        self._env.close()

    def get_env_info(self) -> dict:
        return self._env.get_env_info()

    def _collect_batch(
        self,
        *,
        obs,
        state,
        reward: float,
        terminated: bool,
        info: Mapping[str, Any],
    ) -> StepBatch:
        episode_step = self._episode_step()
        obs_array = np.asarray(obs, dtype=np.float32)
        state_array = np.asarray(state, dtype=np.float32)
        avail_actions = np.asarray(self._env.get_avail_actions(), dtype=np.int8)
        return StepBatch.from_components(
            obs=obs_array,
            state=state_array,
            avail_actions=avail_actions,
            reward=float(reward),
            terminated=terminated,
            info=dict(info),
            episode_step=episode_step,
        )

    def _episode_step(self) -> int:
        if hasattr(self._env, 'get_episode_step'):
            return int(self._env.get_episode_step())
        return int(getattr(self._env, '_episode_steps', 0))

    def __getattr__(self, name: str):
        return getattr(self._env, name)
