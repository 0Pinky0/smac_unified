from smac_unified import make_env_pool
from smac_unified.adapters import VectorEnvPool


class _DummyEnv:
    def __init__(self, env_id: int):
        self.env_id = env_id
        self.closed = False
        self.reset_calls = []

    def reset(self, **kwargs):
        self.reset_calls.append(dict(kwargs))
        return {'env_id': self.env_id, 'seed': kwargs.get('seed')}

    def step(self, actions):
        reward = float(sum(int(a) for a in actions) + self.env_id)
        return reward, False, {'env_id': self.env_id}

    def step_batch(self, actions):
        reward, terminated, info = self.step(actions)
        return {
            'obs': [self.env_id],
            'state': [self.env_id],
            'avail_actions': [[1, 0, 1]],
            'reward': reward,
            'terminated': terminated,
            'info': info,
        }

    def close(self):
        self.closed = True


def test_vector_env_pool_sync_mode_runs_reset_step_and_close():
    pool = VectorEnvPool(
        env_fns=[
            lambda: _DummyEnv(0),
            lambda: _DummyEnv(1),
        ],
        mode='sync',
    )
    resets = pool.reset(seeds=[10, 11], options={'opt': 1})
    assert resets[0]['seed'] == 10
    assert resets[1]['seed'] == 11

    step_rows = pool.step([[1, 2], [3]])
    assert step_rows[0][0] == 3.0
    assert step_rows[1][0] == 4.0

    batch_rows = pool.step_batch([[1], [2]])
    assert batch_rows[0]['reward'] == 1.0
    assert batch_rows[1]['reward'] == 3.0

    pool.close()
    assert all(env.closed for env in pool.envs)


def test_vector_env_pool_thread_mode_runs_step_batch():
    pool = VectorEnvPool(
        env_fns=[
            lambda: _DummyEnv(2),
            lambda: _DummyEnv(3),
        ],
        mode='thread',
    )
    rows = pool.step_batch([[1], [1, 1]])
    assert rows[0]['reward'] == 3.0
    assert rows[1]['reward'] == 5.0
    pool.close()


def test_make_env_pool_offsets_seed_per_env_instance():
    pool = make_env_pool(
        num_envs=3,
        family='smac',
        map_name='3m',
        normalized_api=False,
        seed=21,
        pool_mode='sync',
    )
    seeds = [int(getattr(env, '_seed', -1)) for env in pool.envs]
    assert seeds == [21, 22, 23]
    infos = pool.get_env_info()
    assert len(infos) == 3
    assert all(int(info.get('n_agents', 0)) == 3 for info in infos)
    pool.close()
