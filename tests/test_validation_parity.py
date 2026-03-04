from types import SimpleNamespace

from tools.native_core_validation import (
    CaseResult,
    _build_native_options,
    _compare_case_pair,
    _summarize,
    _summarize_parity_by_profile,
)


def _case(mode: str, trace, *, repeat_idx: int = 0):
    return CaseResult(
        profile='quick',
        family='smac',
        map_name='3m',
        backend_mode=mode,
        repeat_idx=repeat_idx,
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


def test_summary_aggregates_repeats_with_median_fields():
    rows = [
        CaseResult(
            profile='quick',
            family='smac',
            map_name='3m',
            backend_mode='native',
            repeat_idx=0,
            ok=True,
            elapsed_s=1.0,
            steps=3,
            sps=10.0,
            steady_sps=20.0,
            step_latency_ms_p95=2.0,
        ),
        CaseResult(
            profile='quick',
            family='smac',
            map_name='3m',
            backend_mode='native',
            repeat_idx=1,
            ok=True,
            elapsed_s=1.0,
            steps=3,
            sps=30.0,
            steady_sps=40.0,
            step_latency_ms_p95=4.0,
        ),
        CaseResult(
            profile='quick',
            family='smac',
            map_name='3m',
            backend_mode='bridge',
            repeat_idx=0,
            ok=True,
            elapsed_s=1.0,
            steps=3,
            sps=20.0,
            steady_sps=30.0,
            step_latency_ms_p95=3.0,
        ),
        CaseResult(
            profile='quick',
            family='smac',
            map_name='3m',
            backend_mode='bridge',
            repeat_idx=1,
            ok=True,
            elapsed_s=1.0,
            steps=3,
            sps=20.0,
            steady_sps=30.0,
            step_latency_ms_p95=3.0,
        ),
    ]
    summary = _summarize(rows)
    payload = summary['smac']
    assert payload['repeats'] == 2.0
    assert payload['native_sps'] == 20.0
    assert payload['native_sps_mean'] == 20.0
    assert payload['native_step_latency_ms_p95'] == 3.0


def test_parity_summary_aggregates_repeat_results():
    match_trace = [
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
    mismatch_trace = [
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
    rows = [
        _case('bridge', match_trace, repeat_idx=0),
        _case('native', match_trace, repeat_idx=0),
        _case('bridge', match_trace, repeat_idx=1),
        _case('native', mismatch_trace, repeat_idx=1),
    ]
    parity = _summarize_parity_by_profile(results=rows, atol=1e-6, rtol=1e-6)
    payload = parity['quick']['smac']
    assert payload['ok'] is False
    assert len(payload['repeat_results']) == 2
    assert payload['mismatch_count'] >= 1
    assert any(msg.startswith('repeat 1:') for msg in payload['mismatches'])


def test_native_option_builder_applies_cli_overrides():
    args = SimpleNamespace(
        native_options_json='{"foo": 1, "ensure_available_actions": false}',
        ensure_available_actions='true',
        pipeline_actions_and_step='default',
        pipeline_step_and_observe='false',
        reuse_step_observe_requests='default',
    )
    options = _build_native_options(args)
    assert options['foo'] == 1
    assert options['ensure_available_actions'] is True
    assert options['pipeline_step_and_observe'] is False

