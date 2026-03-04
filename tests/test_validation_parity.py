from types import SimpleNamespace

from tools.native_core_validation import (
    CaseResult,
    _aggregate_parity_by_family,
    _build_native_options,
    _build_matrix_cases,
    _compare_case_pair,
    _effective_steady_parity_steps,
    _run_case,
    _summarize,
    _summarize_parity_by_profile,
)


def _case(mode: str, trace, *, repeat_idx: int = 0, profile: str = 'quick'):
    return CaseResult(
        profile=profile,
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


def test_parity_compare_honors_max_steps_window():
    native_trace = [
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
        },
        {
            'step': 1,
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
        },
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
        },
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
        },
    ]
    result = _compare_case_pair(
        native=_case('native', native_trace),
        bridge=_case('bridge', bridge_trace),
        atol=1e-6,
        rtol=1e-6,
        max_steps=1,
    )
    assert result['ok'] is True
    assert result['steps_compared'] == 1


def test_steady_parity_summary_uses_configured_window():
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
        },
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
        },
    ]
    native_trace = [
        dict(bridge_trace[0]),
        {
            **dict(bridge_trace[1]),
            'reward': 3.0,
        },
    ]
    rows = [
        _case('bridge', bridge_trace, profile='steady'),
        _case('native', native_trace, profile='steady'),
    ]
    parity = _summarize_parity_by_profile(
        results=rows,
        atol=1e-6,
        rtol=1e-6,
        steady_parity_steps=1,
    )
    payload = parity['steady']['smac']
    assert payload['ok'] is True
    assert payload['steps_compared'] == 1


def _trace_step(*, step: int, reward: float = 0.0):
    return {
        'step': step,
        'actions': [1],
        'reward': reward,
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


def test_parity_compare_full_trace_detects_late_step_mismatch():
    bridge_trace = [_trace_step(step=idx, reward=0.0) for idx in range(5)]
    native_trace = [
        _trace_step(step=idx, reward=1.0 if idx == 4 else 0.0)
        for idx in range(5)
    ]
    result = _compare_case_pair(
        native=_case('native', native_trace),
        bridge=_case('bridge', bridge_trace),
        atol=1e-6,
        rtol=1e-6,
        max_steps=0,
    )
    assert result['ok'] is False
    assert result['steps_compared'] == 5
    assert result['first_mismatch_field'] == 'reward'
    assert result['first_mismatch_step'] == 4


def test_steady_parity_summary_strict_mode_catches_late_mismatch():
    bridge_trace = [_trace_step(step=idx, reward=0.0) for idx in range(4)]
    native_trace = [
        _trace_step(step=idx, reward=2.0 if idx == 3 else 0.0)
        for idx in range(4)
    ]
    rows = [
        _case('bridge', bridge_trace, profile='steady'),
        _case('native', native_trace, profile='steady'),
    ]
    parity = _summarize_parity_by_profile(
        results=rows,
        atol=1e-6,
        rtol=1e-6,
        steady_parity_steps=0,
    )
    payload = parity['steady']['smac']
    assert payload['ok'] is False
    assert payload['first_mismatch_field'] == 'reward'
    assert payload['first_mismatch_step'] == 3
    assert payload['mismatch_field_counts']['reward'] >= 1


def test_parity_summary_detects_repeat_count_mismatch():
    trace = [_trace_step(step=0, reward=0.0)]
    rows = [
        _case('bridge', trace, repeat_idx=0, profile='steady'),
        _case('bridge', trace, repeat_idx=1, profile='steady'),
        _case('native', trace, repeat_idx=0, profile='steady'),
    ]
    parity = _summarize_parity_by_profile(
        results=rows,
        atol=1e-6,
        rtol=1e-6,
        steady_parity_steps=0,
    )
    payload = parity['steady']['smac']
    assert payload['ok'] is False
    assert payload['first_mismatch_field'] == 'repeat_count'
    assert payload['mismatch_field_counts']['repeat_count'] == 1


def test_effective_steady_parity_steps_strict_forces_full_trace():
    args = SimpleNamespace(
        steady_parity_mode='strict',
        steady_parity_steps=9,
    )
    assert _effective_steady_parity_steps(args) == 0
    args.steady_parity_mode = 'windowed'
    assert _effective_steady_parity_steps(args) == 9


def test_parity_compare_reports_trace_length_and_missing_keys():
    native_trace = [_trace_step(step=0), _trace_step(step=1)]
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
            # state_head intentionally omitted
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
    assert result['first_mismatch_field'] in ('trace_length', 'trace_keys')
    assert result['mismatch_field_counts'].get('trace_length', 0) >= 1
    assert result['mismatch_field_counts'].get('trace_keys', 0) >= 1
    assert result['first_mismatch_detail']


def test_parity_compare_run_failed_includes_error_detail():
    native = _case('native', [_trace_step(step=0)])
    bridge = _case('bridge', [_trace_step(step=0)])
    native.ok = False
    native.error = 'native failed'
    result = _compare_case_pair(
        native=native,
        bridge=bridge,
        atol=1e-6,
        rtol=1e-6,
    )
    assert result['ok'] is False
    assert result['first_mismatch_field'] == 'run_failed'
    assert result['first_mismatch_detail']['native_error'] == 'native failed'
    assert 'native_failure_kind' in result['first_mismatch_detail']
    assert 'native_exit_code' in result['first_mismatch_detail']


def test_parity_compare_records_first_mismatch_detail_values():
    native_trace = [_trace_step(step=0, reward=1.25)]
    bridge_trace = [_trace_step(step=0, reward=0.5)]
    result = _compare_case_pair(
        native=_case('native', native_trace),
        bridge=_case('bridge', bridge_trace),
        atol=1e-6,
        rtol=1e-6,
    )
    assert result['ok'] is False
    detail = result['first_mismatch_detail']
    assert detail['field'] == 'reward'
    assert detail['step'] == 0
    assert detail['native'] == 1.25
    assert detail['bridge'] == 0.5


def test_native_option_builder_can_enable_capture_debug_probe():
    args = SimpleNamespace(
        native_options_json='{}',
        ensure_available_actions='default',
        pipeline_actions_and_step='default',
        pipeline_step_and_observe='default',
        reuse_step_observe_requests='default',
        capture_debug_probe=True,
    )
    options = _build_native_options(args)
    assert options['capture_debug_probe'] is True


def test_native_option_builder_enables_debug_probe_for_forced_opponent_replay():
    args = SimpleNamespace(
        native_options_json='{}',
        ensure_available_actions='default',
        pipeline_actions_and_step='default',
        pipeline_step_and_observe='default',
        reuse_step_observe_requests='default',
        capture_debug_probe=False,
        force_opponent_actions_from_bridge=True,
    )
    options = _build_native_options(args)
    assert options['capture_debug_probe'] is True


def test_parity_compare_reports_action_divergence_step_counts():
    bridge_trace = [_trace_step(step=idx, reward=0.0) for idx in range(5)]
    native_trace = [dict(step_payload) for step_payload in bridge_trace]
    native_trace[4]['actions'] = [2]
    result = _compare_case_pair(
        native=_case('native', native_trace),
        bridge=_case('bridge', bridge_trace),
        atol=1e-6,
        rtol=1e-6,
    )
    assert result['ok'] is False
    assert result['first_mismatch_field'] == 'actions'
    assert result['first_mismatch_step'] == 4
    assert result['mismatch_step_counts'].get('4', 0) >= 1


def test_parity_compare_aggregates_step_counts_for_multiple_mismatch_steps():
    bridge_trace = [_trace_step(step=idx, reward=0.0) for idx in range(4)]
    native_trace = [dict(step_payload) for step_payload in bridge_trace]
    native_trace[1]['reward'] = 1.0
    native_trace[3]['reward'] = 2.0
    result = _compare_case_pair(
        native=_case('native', native_trace),
        bridge=_case('bridge', bridge_trace),
        atol=1e-6,
        rtol=1e-6,
    )
    assert result['ok'] is False
    assert result['mismatch_step_counts'].get('1', 0) >= 1
    assert result['mismatch_step_counts'].get('3', 0) >= 1


def test_build_matrix_cases_resolves_presets_and_logic_lanes():
    args = SimpleNamespace(
        families=['smac-hard'],
        matrix_preset='smac-hard-longtail',
        family_maps_json='{}',
        logic_lane_preset='hard-opponent-bot',
        logic_lanes_json='[]',
        bridge_overridden_lanes=False,
    )
    cases = _build_matrix_cases(args)
    case_ids = sorted(case.case_id for case in cases)
    assert 'smac-hard:3m:default' in case_ids
    assert any(case.case_id.endswith(':hard_bot') for case in cases)
    hard_bot_cases = [case for case in cases if case.lane_id == 'hard_bot']
    assert hard_bot_cases
    assert all(case.logic_switches.get('opponent_mode') == 'sc2_computer' for case in hard_bot_cases)
    assert all(case.bridge_enabled is False for case in hard_bot_cases)


def test_aggregate_parity_by_family_rolls_up_case_payloads():
    payload = {
        'smac:3m:default': {
            'family': 'smac',
            'ok': True,
            'steps_compared': 10,
            'mismatch_count': 0,
            'mismatch_field_counts': {},
            'mismatch_step_counts': {},
            'first_mismatch_field': '',
            'first_mismatch_step': -1,
            'first_mismatch_detail': {},
        },
        'smac:8m:default': {
            'family': 'smac',
            'ok': False,
            'steps_compared': 10,
            'mismatch_count': 2,
            'mismatch_field_counts': {'reward': 2},
            'mismatch_step_counts': {'4': 1, '7': 1},
            'first_mismatch_field': 'reward',
            'first_mismatch_step': 4,
            'first_mismatch_detail': {'field': 'reward', 'step': 4},
        },
    }
    aggregated = _aggregate_parity_by_family(payload)
    smac_payload = aggregated['smac']
    assert smac_payload['ok'] is False
    assert smac_payload['steps_compared'] == 20
    assert smac_payload['mismatch_count'] == 2
    assert smac_payload['mismatch_field_counts']['reward'] == 2
    assert smac_payload['first_mismatch_field'] == 'reward'


def test_run_case_timeout_sets_failure_kind_and_exit_code():
    import tools.native_core_validation as ncv

    original_run = ncv.subprocess.run

    def _raise_timeout(*args, **kwargs):
        del args, kwargs
        raise ncv.subprocess.TimeoutExpired(cmd=['python'], timeout=1.0)

    ncv.subprocess.run = _raise_timeout
    try:
        row = _run_case(
            profile='quick',
            case_id='smac:3m:default',
            family='smac',
            map_name='3m',
            lane_id='default',
            logic_switches={},
            parity_enabled=True,
            backend_mode='native',
            repeat_idx=0,
            steps=1,
            warmup_steps=0,
            seed=7,
            forced_actions=None,
            forced_opponent_actions=None,
            normalized_api=False,
            native_options={},
            parallel_envs=1,
            pool_mode='sync',
            subprocess_timeout_s=1.0,
            make_env_fn=lambda **_: None,
        )
    finally:
        ncv.subprocess.run = original_run
    assert row.ok is False
    assert row.failure_kind == 'timeout'
    assert row.exit_code == 124


def test_run_case_nonzero_sets_failure_kind_and_exit_code():
    import tools.native_core_validation as ncv

    class _Proc:
        returncode = 17
        stdout = ''
        stderr = 'boom\\n'

    original_run = ncv.subprocess.run

    def _return_nonzero(*args, **kwargs):
        del args, kwargs
        return _Proc()

    ncv.subprocess.run = _return_nonzero
    try:
        row = _run_case(
            profile='quick',
            case_id='smac:3m:default',
            family='smac',
            map_name='3m',
            lane_id='default',
            logic_switches={},
            parity_enabled=True,
            backend_mode='bridge',
            repeat_idx=0,
            steps=1,
            warmup_steps=0,
            seed=7,
            forced_actions=None,
            forced_opponent_actions=None,
            normalized_api=False,
            native_options={},
            parallel_envs=1,
            pool_mode='sync',
            subprocess_timeout_s=1.0,
            make_env_fn=lambda **_: None,
        )
    finally:
        ncv.subprocess.run = original_run
    assert row.ok is False
    assert row.failure_kind == 'subprocess_nonzero'
    assert row.exit_code == 17

