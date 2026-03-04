from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Mapping, Sequence

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


class VectorEnvPool:
    """Simple pooled execution wrapper for multiple environment instances."""

    def __init__(
        self,
        *,
        env_fns: Sequence[Callable[[], Any]],
        mode: str = 'sync',
        max_workers: int | None = None,
    ):
        self.mode = str(mode or 'sync').strip().lower()
        if self.mode not in {'sync', 'thread'}:
            raise ValueError(f'Unsupported vector pool mode: {mode!r}')
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        workers = max_workers or max(self.num_envs, 1)
        self._executor: ThreadPoolExecutor | None = None
        if self.mode == 'thread' and self.num_envs > 0:
            self._executor = ThreadPoolExecutor(max_workers=workers)

    def reset(
        self,
        *,
        seeds: Sequence[int | None] | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> list[Any]:
        seeds_seq = list(seeds or [])
        return self._map_envs(
            lambda idx, env: _safe_reset_env(
                env=env,
                seed=seeds_seq[idx] if idx < len(seeds_seq) else None,
                options=options,
            )
        )

    def step(self, actions_batch: Sequence[Sequence[int]]) -> list[Any]:
        actions_seq = list(actions_batch or [])
        if len(actions_seq) != self.num_envs:
            raise ValueError(
                f'actions_batch length {len(actions_seq)} must match num_envs {self.num_envs}.'
            )
        return self._map_envs(
            lambda idx, env: env.step(list(actions_seq[idx])),
        )

    def step_batch(self, actions_batch: Sequence[Sequence[int]]) -> list[Any]:
        actions_seq = list(actions_batch or [])
        if len(actions_seq) != self.num_envs:
            raise ValueError(
                f'actions_batch length {len(actions_seq)} must match num_envs {self.num_envs}.'
            )
        return self._map_envs(
            lambda idx, env: _safe_step_batch_env(env=env, actions=actions_seq[idx]),
        )

    def sample_random_actions(
        self,
        rng: np.random.Generator | None = None,
    ) -> list[np.ndarray]:
        rng = rng or np.random.default_rng()
        actions: list[np.ndarray] = []
        for env in self.envs:
            if hasattr(env, 'sample_random_actions'):
                actions.append(np.asarray(env.sample_random_actions(rng=rng), dtype=np.int64))
                continue
            n_agents = int(getattr(env, 'n_agents', 0))
            actions.append(np.zeros(n_agents, dtype=np.int64))
        return actions

    def close(self) -> None:
        errors = []
        for env in self.envs:
            try:
                env.close()
            except Exception as exc:
                errors.append(exc)
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
        if errors:
            raise RuntimeError(
                f'One or more vector pool environments failed to close: {errors}'
            )

    def get_env_info(self) -> list[dict[str, Any]]:
        infos = []
        for env in self.envs:
            if hasattr(env, 'get_env_info'):
                infos.append(dict(env.get_env_info()))
            else:
                infos.append({})
        return infos

    def _map_envs(self, fn: Callable[[int, Any], Any]) -> list[Any]:
        if self.mode == 'thread' and self._executor is not None:
            futures = [
                self._executor.submit(fn, idx, env)
                for idx, env in enumerate(self.envs)
            ]
            return [future.result() for future in futures]
        return [fn(idx, env) for idx, env in enumerate(self.envs)]


def _safe_reset_env(
    *,
    env: Any,
    seed: int | None,
    options: Mapping[str, Any] | None,
):
    kwargs: dict[str, Any] = {}
    if seed is not None:
        kwargs['seed'] = int(seed)
    if options is not None:
        kwargs['options'] = dict(options)
    if kwargs:
        try:
            return env.reset(**kwargs)
        except TypeError:
            pass
    if seed is not None and hasattr(env, 'seed') and callable(env.seed):
        env.seed(int(seed))
    if options and isinstance(options, Mapping):
        try:
            return env.reset(episode_config=options.get('episode_config', options))
        except TypeError:
            pass
    return env.reset()


def _safe_step_batch_env(*, env: Any, actions: Sequence[int]):
    if hasattr(env, 'step_batch'):
        try:
            return env.step_batch(list(actions))
        except TypeError:
            pass
    reward, terminated, info = env.step(list(actions))
    return {
        'obs': env.get_obs(),
        'state': env.get_state(),
        'avail_actions': env.get_avail_actions(),
        'reward': reward,
        'terminated': terminated,
        'info': info,
    }
