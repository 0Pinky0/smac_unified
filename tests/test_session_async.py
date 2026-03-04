from concurrent.futures import ThreadPoolExecutor
import time

from smac_unified.core.sc2session import SC2EnvRawSession, SC2SessionConfig
from smac_unified.maps import get_map_params


class _FakeSC2Env:
    def __init__(self):
        self.payloads = []
        self.closed = False

    def step(self, payload):
        self.payloads.append(payload)
        return ['fake_timestep']

    def close(self):
        self.closed = True


class _SlowFakeSC2Env(_FakeSC2Env):
    def __init__(self, delay_s: float):
        super().__init__()
        self.delay_s = float(delay_s)

    def step(self, payload):
        time.sleep(self.delay_s)
        return super().step(payload)


def _session(*, enable_async_step: bool, num_agents: int = 1):
    cfg = SC2SessionConfig(
        map_name='3m',
        map_params=get_map_params('3m'),
        enable_async_step=enable_async_step,
    )
    session = SC2EnvRawSession(cfg)
    session._env = _FakeSC2Env()
    session._num_agents = num_agents
    if enable_async_step:
        session._step_executor = ThreadPoolExecutor(max_workers=1)
    return session


def test_session_submit_collect_sync_fallback_path():
    session = _session(enable_async_step=False, num_agents=1)
    session.submit_step(agent_actions=[1, 2, 3])
    assert session.has_pending_step is True
    result = session.collect_step()
    assert result == ['fake_timestep']
    assert session.has_pending_step is False
    assert session._env.payloads[-1] == [[1, 2, 3]]
    session.close()


def test_session_submit_collect_async_path():
    session = _session(enable_async_step=True, num_agents=1)
    session.submit_step(agent_actions=[5])
    assert session.has_pending_step is True
    result = session.collect_step()
    assert result == ['fake_timestep']
    assert session.has_pending_step is False
    assert session._env.payloads[-1] == [[5]]
    session.close()


def test_session_step_keeps_ally_and_opponent_payload_layout():
    session = _session(enable_async_step=False, num_agents=2)
    result = session.step(agent_actions=[7], opponent_actions=[9, 10])
    assert result == ['fake_timestep']
    assert session._env.payloads[-1] == [[7], [9, 10]]
    session.close()


def test_session_rejects_submit_when_previous_step_is_pending():
    session = _session(enable_async_step=False, num_agents=1)
    session.submit_step(agent_actions=[1])
    try:
        session.submit_step(agent_actions=[2])
        raise AssertionError('Expected RuntimeError for duplicate pending submit.')
    except RuntimeError as exc:
        assert 'collect_step' in str(exc)
    finally:
        session.collect_step()
        session.close()


def test_session_collect_without_submit_raises_runtime_error():
    session = _session(enable_async_step=False, num_agents=1)
    try:
        session.collect_step()
        raise AssertionError('Expected RuntimeError for collect without submit.')
    except RuntimeError as exc:
        assert 'pending step' in str(exc)
    finally:
        session.close()


def test_session_close_clears_pending_async_step_state():
    session = _session(enable_async_step=True, num_agents=1)
    slow_env = _SlowFakeSC2Env(delay_s=0.1)
    session._env = slow_env
    session.submit_step(agent_actions=[9])
    assert session.has_pending_step is True
    session.close()
    assert session.has_pending_step is False
    assert session._pending_step_future is None
    assert session._step_executor is None
    assert slow_env.closed is True
