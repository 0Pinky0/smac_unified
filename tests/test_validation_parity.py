from tools.native_core_validation import CaseResult, _compare_case_pair


def _case(mode: str, trace):
    return CaseResult(
        profile='quick',
        family='smac',
        map_name='3m',
        backend_mode=mode,
        ok=True,
        elapsed_s=1.0,
        steps=len(trace),
        sps=1.0,
        trace=trace,
    )


def test_parity_compare_passes_for_matching_traces():
    trace = [
        {
            'step': 0,
            'actions': [1, 1],
            'reward': 0.5,
            'terminated': False,
            'battle_won': False,
            'episode_limit': False,
            'dead_allies': 0,
            'dead_enemies': 0,
            'obs_shape': [2, 10],
            'state_shape': [30],
            'obs_head': [0.1, 0.2],
            'state_head': [0.3, 0.4],
            'avail_actions': [[0, 1], [0, 1]],
        }
    ]
    result = _compare_case_pair(
        native=_case('native', trace),
        bridge=_case('bridge', trace),
        atol=1e-6,
        rtol=1e-6,
    )
    assert result['ok'] is True
    assert result['mismatch_count'] == 0


def test_parity_compare_reports_mismatch_details():
    native_trace = [
        {
            'step': 0,
            'actions': [1],
            'reward': 1.0,
            'terminated': False,
            'battle_won': False,
            'episode_limit': False,
            'dead_allies': 0,
            'dead_enemies': 0,
            'obs_shape': [1, 4],
            'state_shape': [6],
            'obs_head': [0.0],
            'state_head': [0.0],
            'avail_actions': [[0, 1]],
        }
    ]
    bridge_trace = [
        {
            'step': 0,
            'actions': [1],
            'reward': 2.0,
            'terminated': False,
            'battle_won': False,
            'episode_limit': False,
            'dead_allies': 0,
            'dead_enemies': 0,
            'obs_shape': [1, 4],
            'state_shape': [6],
            'obs_head': [0.0],
            'state_head': [0.0],
            'avail_actions': [[0, 1]],
        }
    ]
    result = _compare_case_pair(
        native=_case('native', native_trace),
        bridge=_case('bridge', bridge_trace),
        atol=1e-6,
        rtol=1e-6,
    )
    assert result['ok'] is False
    assert result['mismatch_count'] >= 1
    assert any('reward mismatch' in msg for msg in result['mismatches'])


def test_parity_compare_detects_step_alignment_mismatch():
    native_trace = [
        {
            'step': 1,
            'actions': [1],
            'reward': 0.0,
            'terminated': False,
            'battle_won': False,
            'episode_limit': False,
            'dead_allies': 0,
            'dead_enemies': 0,
            'obs_shape': [1, 4],
            'state_shape': [6],
            'obs_head': [0.0],
            'state_head': [0.0],
            'avail_actions': [[0, 1]],
        }
    ]
    bridge_trace = [
        {
            'step': 0,
            'actions': [1],
            'reward': 0.0,
            'terminated': False,
            'battle_won': False,
            'episode_limit': False,
            'dead_allies': 0,
            'dead_enemies': 0,
            'obs_shape': [1, 4],
            'state_shape': [6],
            'obs_head': [0.0],
            'state_head': [0.0],
            'avail_actions': [[0, 1]],
        }
    ]
    result = _compare_case_pair(
        native=_case('native', native_trace),
        bridge=_case('bridge', bridge_trace),
        atol=1e-6,
        rtol=1e-6,
    )
    assert result['ok'] is False
    assert any('step id mismatch' in msg for msg in result['mismatches'])


def test_parity_compare_uses_relative_tolerance():
    native_trace = [
        {
            'step': 0,
            'actions': [1],
            'reward': 10000.0009,
            'terminated': False,
            'battle_won': False,
            'episode_limit': False,
            'dead_allies': 0,
            'dead_enemies': 0,
            'obs_shape': [1, 4],
            'state_shape': [6],
            'obs_head': [10000.0009],
            'state_head': [10000.0009],
            'avail_actions': [[0, 1]],
        }
    ]
    bridge_trace = [
        {
            'step': 0,
            'actions': [1],
            'reward': 10000.0,
            'terminated': False,
            'battle_won': False,
            'episode_limit': False,
            'dead_allies': 0,
            'dead_enemies': 0,
            'obs_shape': [1, 4],
            'state_shape': [6],
            'obs_head': [10000.0],
            'state_head': [10000.0],
            'avail_actions': [[0, 1]],
        }
    ]
    result = _compare_case_pair(
        native=_case('native', native_trace),
        bridge=_case('bridge', bridge_trace),
        atol=1e-8,
        rtol=1e-4,
    )
    assert result['ok'] is True

