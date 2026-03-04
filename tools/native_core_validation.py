#!/usr/bin/env python3
"""Core-first standalone validation for native/bridge backends."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List


def _ensure_project_on_path() -> None:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


DEFAULT_MAPS = {
    'smac': '3m',
    'smacv2': '8m',
    'smac-hard': '3m',
}


@dataclass
class CaseResult:
    profile: str
    family: str
    map_name: str
    backend_mode: str
    repeat_idx: int
    ok: bool
    elapsed_s: float
    steps: int
    sps: float
    startup_s: float = 0.0
    step_elapsed_s: float = 0.0
    step_sps: float = 0.0
    close_s: float = 0.0
    warmup_steps: int = 0
    steady_steps: int = 0
    steady_elapsed_s: float = 0.0
    steady_sps: float = 0.0
    step_latency_ms_p50: float = 0.0
    step_latency_ms_p95: float = 0.0
    step_latency_ms_p99: float = 0.0
    steady_latency_ms_p50: float = 0.0
    steady_latency_ms_p95: float = 0.0
    steady_latency_ms_p99: float = 0.0
    trace: list[dict[str, Any]] | None = None
    error: str = ''


def _run_case(
    *,
    profile: str,
    family: str,
    map_name: str,
    backend_mode: str,
    repeat_idx: int,
    steps: int,
    warmup_steps: int,
    seed: int,
    forced_actions: list[list[int]] | None,
    normalized_api: bool,
    native_options: dict[str, Any],
    make_env_fn,
) -> CaseResult:
    del make_env_fn
    runner = """
import json
import sys
import time
from pathlib import Path

import numpy as np

root = Path(sys.argv[1]).resolve()
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from smac_unified import make_env

family = sys.argv[2]
map_name = sys.argv[3]
backend_mode = sys.argv[4]
steps = int(sys.argv[5])
warmup_steps = int(sys.argv[6])
forced_actions = json.loads(sys.argv[7]) if sys.argv[7] else None
seed = int(sys.argv[8])
normalized_api = bool(int(sys.argv[9]))
native_options = json.loads(sys.argv[10]) if sys.argv[10] else {}
warmup_steps = max(0, min(warmup_steps, steps))

t0 = time.perf_counter()
env = make_env(
    family=family,
    map_name=map_name,
    backend_mode=backend_mode,
    normalized_api=normalized_api,
    native_options=native_options,
    seed=seed,
)
if normalized_api:
    env.reset(seed=seed)
else:
    env.reset()
t_reset = time.perf_counter()
step_elapsed = 0.0
steady_elapsed = 0.0
steady_steps = 0
step_latencies_ms = []
steady_latencies_ms = []
trace = []
for step_idx in range(steps):
    chosen = []
    forced_step = (
        forced_actions[step_idx]
        if forced_actions is not None and step_idx < len(forced_actions)
        else None
    )
    for agent_id in range(env.n_agents):
        avail = list(env.get_avail_agent_actions(agent_id))
        action = 0
        if forced_step is not None and agent_id < len(forced_step):
            forced = int(forced_step[agent_id])
            if 0 <= forced < len(avail) and avail[forced]:
                action = forced
            else:
                for idx, flag in enumerate(avail):
                    if flag and idx != 0:
                        action = idx
                        break
        else:
            for idx, flag in enumerate(avail):
                if flag and idx != 0:
                    action = idx
                    break
        chosen.append(action)
    if normalized_api:
        t_step0 = time.perf_counter()
        batch = env.step(chosen)
        t_step1 = time.perf_counter()
        reward = float(batch.reward)
        terminated = bool(batch.terminated)
        info = dict(batch.info)
        obs = np.asarray(batch.obs, dtype=np.float32)
        state = np.asarray(batch.state, dtype=np.float32)
        avail_actions = np.asarray(batch.avail_actions, dtype=np.int64).tolist()
    else:
        t_step0 = time.perf_counter()
        reward, terminated, info = env.step(chosen)
        t_step1 = time.perf_counter()
        obs = np.asarray(env.get_obs(), dtype=np.float32)
        state = np.asarray(env.get_state(), dtype=np.float32)
        avail_actions = [
            list(map(int, env.get_avail_agent_actions(agent_id)))
            for agent_id in range(env.n_agents)
        ]
    dt = max(t_step1 - t_step0, 0.0)
    step_elapsed += dt
    dt_ms = dt * 1000.0
    step_latencies_ms.append(dt_ms)
    if step_idx >= warmup_steps:
        steady_elapsed += dt
        steady_steps += 1
        steady_latencies_ms.append(dt_ms)
    trace.append({
        "step": int(step_idx),
        "actions": list(map(int, chosen)),
        "reward": float(reward),
        "terminated": bool(terminated),
        "battle_won": bool(info.get("battle_won", False)),
        "episode_limit": bool(info.get("episode_limit", False)),
        "dead_allies": int(info.get("dead_allies", -1)),
        "dead_enemies": int(info.get("dead_enemies", -1)),
        "obs_shape": list(obs.shape),
        "state_shape": list(state.shape),
        "obs_head": obs.flatten()[:8].astype(float).tolist(),
        "state_head": state.flatten()[:8].astype(float).tolist(),
        "avail_actions": avail_actions,
    })
t_close0 = time.perf_counter()
env.close()
t_end = time.perf_counter()
elapsed = max(t_end - t0, 1e-9)
startup = max(t_reset - t0, 0.0)
close_s = max(t_end - t_close0, 0.0)
step_sps = steps / max(step_elapsed, 1e-9)
steady_sps = (
    steady_steps / max(steady_elapsed, 1e-9)
    if steady_steps > 0
    else 0.0
)
def _pct(values, q):
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))
print(json.dumps({
    "elapsed_s": elapsed,
    "steps": steps,
    "sps": steps / max(elapsed, 1e-9),
    "startup_s": startup,
    "step_elapsed_s": step_elapsed,
    "step_sps": step_sps,
    "close_s": close_s,
    "warmup_steps": warmup_steps,
    "steady_steps": steady_steps,
    "steady_elapsed_s": steady_elapsed,
    "steady_sps": steady_sps,
    "step_latency_ms_p50": _pct(step_latencies_ms, 50),
    "step_latency_ms_p95": _pct(step_latencies_ms, 95),
    "step_latency_ms_p99": _pct(step_latencies_ms, 99),
    "steady_latency_ms_p50": _pct(steady_latencies_ms, 50),
    "steady_latency_ms_p95": _pct(steady_latencies_ms, 95),
    "steady_latency_ms_p99": _pct(steady_latencies_ms, 99),
    "trace": trace,
}))
"""
    t0 = time.perf_counter()
    proc = subprocess.run(
        [
            sys.executable,
            '-c',
            runner,
            str(Path(__file__).resolve().parents[1]),
            family,
            map_name,
            backend_mode,
            str(steps),
            str(warmup_steps),
            json.dumps(forced_actions or []),
            str(int(seed)),
            '1' if normalized_api else '0',
            json.dumps(native_options),
        ],
        capture_output=True,
        text=True,
    )
    elapsed_total = max(time.perf_counter() - t0, 1e-9)
    if proc.returncode != 0:
        error = proc.stderr.strip() or proc.stdout.strip() or 'unknown failure'
        return CaseResult(
            profile=profile,
            family=family,
            map_name=map_name,
            backend_mode=backend_mode,
            repeat_idx=repeat_idx,
            ok=False,
            elapsed_s=elapsed_total,
            steps=steps,
            sps=0.0,
            warmup_steps=warmup_steps,
            error=error.splitlines()[-1][:400],
        )

    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    payload = {}
    for line in reversed(lines):
        if line.startswith('{') and line.endswith('}'):
            payload = json.loads(line)
            break
    elapsed = float(payload.get('elapsed_s', elapsed_total))
    cold_sps = float(payload.get('sps', steps / max(elapsed, 1e-9)))
    return CaseResult(
        profile=profile,
        family=family,
        map_name=map_name,
        backend_mode=backend_mode,
        repeat_idx=repeat_idx,
        ok=True,
        elapsed_s=elapsed,
        steps=steps,
        sps=cold_sps,
        startup_s=float(payload.get('startup_s', 0.0)),
        step_elapsed_s=float(payload.get('step_elapsed_s', 0.0)),
        step_sps=float(payload.get('step_sps', 0.0)),
        close_s=float(payload.get('close_s', 0.0)),
        warmup_steps=int(payload.get('warmup_steps', warmup_steps)),
        steady_steps=int(payload.get('steady_steps', max(steps - warmup_steps, 0))),
        steady_elapsed_s=float(payload.get('steady_elapsed_s', 0.0)),
        steady_sps=float(payload.get('steady_sps', 0.0)),
        step_latency_ms_p50=float(payload.get('step_latency_ms_p50', 0.0)),
        step_latency_ms_p95=float(payload.get('step_latency_ms_p95', 0.0)),
        step_latency_ms_p99=float(payload.get('step_latency_ms_p99', 0.0)),
        steady_latency_ms_p50=float(payload.get('steady_latency_ms_p50', 0.0)),
        steady_latency_ms_p95=float(payload.get('steady_latency_ms_p95', 0.0)),
        steady_latency_ms_p99=float(payload.get('steady_latency_ms_p99', 0.0)),
        trace=list(payload.get('trace', []) or []),
    )


def _summarize(results: List[CaseResult]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for family in sorted({row.family for row in results}):
        native_rows = _rows_by_mode(results=results, family=family, mode='native')
        bridge_rows = _rows_by_mode(results=results, family=family, mode='bridge')
        payload: Dict[str, float] = {}
        payload['repeats'] = float(max(len(native_rows), len(bridge_rows)))
        if native_rows:
            _add_mode_stats(payload=payload, prefix='native', rows=native_rows)
        if bridge_rows:
            _add_mode_stats(payload=payload, prefix='bridge', rows=bridge_rows)
        if native_rows and bridge_rows and payload.get('bridge_sps', 0.0) > 0:
            payload['native_vs_bridge_delta_pct'] = (
                (payload['native_sps'] - payload['bridge_sps'])
                * 100.0
                / payload['bridge_sps']
            )
        if (
            native_rows
            and bridge_rows
            and payload.get('bridge_steady_sps', 0.0) > 0
        ):
            payload['native_vs_bridge_steady_delta_pct'] = (
                (payload['native_steady_sps'] - payload['bridge_steady_sps'])
                * 100.0
                / payload['bridge_steady_sps']
            )
        summary[family] = payload
    return summary


def _summarize_by_profile(results: List[CaseResult]) -> Dict[str, Dict[str, Dict[str, float]]]:
    summary_by_profile: Dict[str, Dict[str, Dict[str, float]]] = {}
    for profile in sorted({row.profile for row in results}):
        profile_rows = [row for row in results if row.profile == profile]
        summary_by_profile[profile] = _summarize(profile_rows)
    return summary_by_profile


def _rows_by_mode(
    *,
    results: List[CaseResult],
    family: str,
    mode: str,
) -> list[CaseResult]:
    return sorted(
        [row for row in results if row.family == family and row.backend_mode == mode],
        key=lambda row: row.repeat_idx,
    )


def _metric_stats(rows: list[CaseResult], attr: str) -> dict[str, float]:
    values = [float(getattr(row, attr)) for row in rows]
    if not values:
        return {'mean': 0.0, 'median': 0.0}
    return {
        'mean': float(statistics.fmean(values)),
        'median': float(statistics.median(values)),
    }


def _add_mode_stats(
    *,
    payload: Dict[str, float],
    prefix: str,
    rows: list[CaseResult],
) -> None:
    payload[f'{prefix}_ok'] = float(all(row.ok for row in rows))
    payload[f'{prefix}_repeat_count'] = float(len(rows))
    for metric in (
        'sps',
        'steady_sps',
        'step_sps',
        'startup_s',
        'step_latency_ms_p95',
        'steady_latency_ms_p95',
    ):
        stats = _metric_stats(rows, metric)
        payload[f'{prefix}_{metric}'] = stats['median']
        payload[f'{prefix}_{metric}_mean'] = stats['mean']
        payload[f'{prefix}_{metric}_median'] = stats['median']


def _compare_case_pair(
    *,
    native: CaseResult,
    bridge: CaseResult,
    atol: float,
    rtol: float,
    max_steps: int = 0,
) -> Dict[str, Any]:
    required_trace_keys = {
        'step',
        'actions',
        'reward',
        'terminated',
        'battle_won',
        'episode_limit',
        'dead_allies',
        'dead_enemies',
        'obs_shape',
        'state_shape',
        'obs_head',
        'state_head',
        'avail_actions',
    }
    payload: Dict[str, Any] = {
        'ok': False,
        'steps_compared': 0,
        'mismatch_count': 0,
        'mismatches': [],
        'mismatch_field_counts': {},
        'mismatch_step_counts': {},
        'first_mismatch_step': -1,
        'first_mismatch_field': '',
        'first_mismatch_detail': {},
    }
    field_counts: dict[str, int] = {}
    step_counts: dict[str, int] = {}

    def _preview_value(value: Any) -> Any:
        if isinstance(value, list):
            if value and isinstance(value[0], list):
                outer = min(len(value), 3)
                return [list(row)[:8] for row in value[:outer]]
            return list(value)[:8]
        return value

    def _record_mismatch(
        *,
        field: str,
        message: str,
        step: int = -1,
        native_value: Any = None,
        bridge_value: Any = None,
    ) -> None:
        field_key = str(field or 'unknown')
        field_counts[field_key] = int(field_counts.get(field_key, 0)) + 1
        if int(step) >= 0:
            step_key = str(int(step))
            step_counts[step_key] = int(step_counts.get(step_key, 0)) + 1
        if payload.get('first_mismatch_field', '') == '':
            payload['first_mismatch_field'] = field_key
            payload['first_mismatch_step'] = int(step)
            payload['first_mismatch_detail'] = {
                'field': field_key,
                'step': int(step),
                'native': _preview_value(native_value),
                'bridge': _preview_value(bridge_value),
            }
        mismatches.append(message)

    if not native.ok or not bridge.ok:
        payload['mismatches'] = ['native/bridge run failed; trace compare skipped']
        payload['mismatch_count'] = 1
        payload['mismatch_field_counts'] = {'run_failed': 1}
        payload['mismatch_step_counts'] = {}
        payload['first_mismatch_step'] = -1
        payload['first_mismatch_field'] = 'run_failed'
        payload['first_mismatch_detail'] = {
            'field': 'run_failed',
            'step': -1,
            'native_ok': bool(native.ok),
            'bridge_ok': bool(bridge.ok),
            'native_error': str(native.error or ''),
            'bridge_error': str(bridge.error or ''),
        }
        return payload
    native_trace = native.trace or []
    bridge_trace = bridge.trace or []
    if not native_trace or not bridge_trace:
        payload['mismatches'] = ['native/bridge trace missing']
        payload['mismatch_count'] = 1
        payload['mismatch_field_counts'] = {'trace_missing': 1}
        payload['mismatch_step_counts'] = {}
        payload['first_mismatch_step'] = -1
        payload['first_mismatch_field'] = 'trace_missing'
        payload['first_mismatch_detail'] = {
            'field': 'trace_missing',
            'step': -1,
            'native_trace_len': int(len(native_trace)),
            'bridge_trace_len': int(len(bridge_trace)),
        }
        return payload

    steps = min(len(native_trace), len(bridge_trace))
    if int(max_steps) > 0:
        steps = min(steps, int(max_steps))
    mismatches: list[str] = []
    if len(native_trace) != len(bridge_trace):
        _record_mismatch(
            field='trace_length',
            step=-1,
            message=(
                f'trace length mismatch native={len(native_trace)} '
                f'bridge={len(bridge_trace)}'
            ),
            native_value=len(native_trace),
            bridge_value=len(bridge_trace),
        )
    for step_idx in range(steps):
        lhs = native_trace[step_idx]
        rhs = bridge_trace[step_idx]
        lhs_missing = sorted(required_trace_keys - set(lhs.keys()))
        rhs_missing = sorted(required_trace_keys - set(rhs.keys()))
        if lhs_missing:
            _record_mismatch(
                field='trace_keys',
                step=step_idx,
                message=f'step {step_idx}: native trace missing keys {lhs_missing}',
                native_value=lhs_missing,
                bridge_value=[],
            )
        if rhs_missing:
            _record_mismatch(
                field='trace_keys',
                step=step_idx,
                message=f'step {step_idx}: bridge trace missing keys {rhs_missing}',
                native_value=[],
                bridge_value=rhs_missing,
            )
        if lhs.get('step') != rhs.get('step'):
            _record_mismatch(
                field='step_id',
                step=step_idx,
                message=(
                    f"step {step_idx}: step id mismatch native={lhs.get('step')} "
                    f"bridge={rhs.get('step')}"
                ),
                native_value=lhs.get('step'),
                bridge_value=rhs.get('step'),
            )
        if lhs.get('actions') != rhs.get('actions'):
            _record_mismatch(
                field='actions',
                step=step_idx,
                message=f'step {step_idx}: actions mismatch',
                native_value=lhs.get('actions'),
                bridge_value=rhs.get('actions'),
            )
        if not _float_close(
            lhs.get('reward', 0.0),
            rhs.get('reward', 0.0),
            atol=atol,
            rtol=rtol,
        ):
            _record_mismatch(
                field='reward',
                step=step_idx,
                message=(
                    f"step {step_idx}: reward mismatch native={lhs.get('reward')} "
                    f"bridge={rhs.get('reward')}"
                ),
                native_value=lhs.get('reward'),
                bridge_value=rhs.get('reward'),
            )
        for key in ('terminated', 'battle_won', 'episode_limit', 'dead_allies', 'dead_enemies'):
            if lhs.get(key) != rhs.get(key):
                _record_mismatch(
                    field=key,
                    step=step_idx,
                    message=f'step {step_idx}: {key} mismatch',
                    native_value=lhs.get(key),
                    bridge_value=rhs.get(key),
                )
        if list(lhs.get('obs_shape', [])) != list(rhs.get('obs_shape', [])):
            _record_mismatch(
                field='obs_shape',
                step=step_idx,
                message=f'step {step_idx}: obs_shape mismatch',
                native_value=lhs.get('obs_shape', []),
                bridge_value=rhs.get('obs_shape', []),
            )
        if list(lhs.get('state_shape', [])) != list(rhs.get('state_shape', [])):
            _record_mismatch(
                field='state_shape',
                step=step_idx,
                message=f'step {step_idx}: state_shape mismatch',
                native_value=lhs.get('state_shape', []),
                bridge_value=rhs.get('state_shape', []),
            )
        if lhs.get('avail_actions') != rhs.get('avail_actions'):
            _record_mismatch(
                field='avail_actions',
                step=step_idx,
                message=f'step {step_idx}: avail_actions mismatch',
                native_value=lhs.get('avail_actions', []),
                bridge_value=rhs.get('avail_actions', []),
            )
        if not _vector_close(
            lhs.get('obs_head', []),
            rhs.get('obs_head', []),
            atol=atol,
            rtol=rtol,
        ):
            _record_mismatch(
                field='obs_head',
                step=step_idx,
                message=f'step {step_idx}: obs_head mismatch',
                native_value=lhs.get('obs_head', []),
                bridge_value=rhs.get('obs_head', []),
            )
        if not _vector_close(
            lhs.get('state_head', []),
            rhs.get('state_head', []),
            atol=atol,
            rtol=rtol,
        ):
            _record_mismatch(
                field='state_head',
                step=step_idx,
                message=f'step {step_idx}: state_head mismatch',
                native_value=lhs.get('state_head', []),
                bridge_value=rhs.get('state_head', []),
            )
        if len(mismatches) >= 20:
            mismatches.append('... additional mismatches truncated ...')
            break

    payload['steps_compared'] = steps
    payload['mismatch_count'] = len(mismatches)
    payload['mismatches'] = mismatches
    payload['mismatch_field_counts'] = field_counts
    payload['mismatch_step_counts'] = step_counts
    payload['ok'] = len(mismatches) == 0
    return payload


def _summarize_parity_by_profile(
    *,
    results: List[CaseResult],
    atol: float,
    rtol: float,
    steady_parity_steps: int = 0,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    parity_by_profile: Dict[str, Dict[str, Dict[str, Any]]] = {}
    profile_keys = sorted({row.profile for row in results})
    for profile in profile_keys:
        family_payload: Dict[str, Dict[str, Any]] = {}
        profile_rows = [row for row in results if row.profile == profile]
        for family in sorted({row.family for row in profile_rows}):
            native_rows = _rows_by_mode(
                results=profile_rows,
                family=family,
                mode='native',
            )
            bridge_rows = _rows_by_mode(
                results=profile_rows,
                family=family,
                mode='bridge',
            )
            if not native_rows or not bridge_rows:
                family_payload[family] = {
                    'ok': False,
                    'steps_compared': 0,
                    'mismatch_count': 1,
                    'mismatches': ['missing native or bridge run'],
                    'mismatch_field_counts': {'missing_mode': 1},
                    'mismatch_step_counts': {},
                    'first_mismatch_field': 'missing_mode',
                    'first_mismatch_step': -1,
                    'first_mismatch_detail': {
                        'field': 'missing_mode',
                        'step': -1,
                        'native_repeat_count': int(len(native_rows)),
                        'bridge_repeat_count': int(len(bridge_rows)),
                    },
                    'repeat_results': [],
                }
                continue
            pair_count = min(len(native_rows), len(bridge_rows))
            mismatch_count = 0
            steps_compared = 0
            mismatches: list[str] = []
            mismatch_field_counts: dict[str, int] = {}
            mismatch_step_counts: dict[str, int] = {}
            first_mismatch_field = ''
            first_mismatch_step = -1
            first_mismatch_detail: dict[str, Any] = {}
            repeat_results: list[dict[str, Any]] = []
            for idx in range(pair_count):
                native = native_rows[idx]
                bridge = bridge_rows[idx]
                repeat_cmp = _compare_case_pair(
                    native=native,
                    bridge=bridge,
                    atol=atol,
                    rtol=rtol,
                    max_steps=(
                        steady_parity_steps
                        if profile == 'steady' and int(steady_parity_steps) > 0
                        else 0
                    ),
                )
                repeat_results.append(
                    {
                        'repeat_idx': int(native.repeat_idx),
                        'ok': bool(repeat_cmp.get('ok', False)),
                        'steps_compared': int(repeat_cmp.get('steps_compared', 0)),
                        'mismatch_count': int(repeat_cmp.get('mismatch_count', 0)),
                        'first_mismatch_field': str(
                            repeat_cmp.get('first_mismatch_field', '')
                        ),
                        'first_mismatch_step': int(
                            repeat_cmp.get('first_mismatch_step', -1)
                        ),
                        'first_mismatch_detail': dict(
                            repeat_cmp.get('first_mismatch_detail', {}) or {}
                        ),
                    }
                )
                steps_compared += int(repeat_cmp.get('steps_compared', 0))
                mismatch_count += int(repeat_cmp.get('mismatch_count', 0))
                repeat_field_counts = dict(
                    repeat_cmp.get('mismatch_field_counts', {}) or {}
                )
                for field, count in repeat_field_counts.items():
                    mismatch_field_counts[field] = (
                        int(mismatch_field_counts.get(field, 0)) + int(count)
                    )
                repeat_step_counts = dict(
                    repeat_cmp.get('mismatch_step_counts', {}) or {}
                )
                for step_key, count in repeat_step_counts.items():
                    mismatch_step_counts[step_key] = (
                        int(mismatch_step_counts.get(step_key, 0)) + int(count)
                    )
                if not repeat_cmp.get('ok', False):
                    if first_mismatch_field == '':
                        first_mismatch_field = str(
                            repeat_cmp.get('first_mismatch_field', '')
                        )
                        first_mismatch_step = int(
                            repeat_cmp.get('first_mismatch_step', -1)
                        )
                        first_mismatch_detail = dict(
                            repeat_cmp.get('first_mismatch_detail', {}) or {}
                        )
                    for msg in repeat_cmp.get('mismatches', []):
                        if len(mismatches) >= 40:
                            break
                        mismatches.append(f"repeat {native.repeat_idx}: {msg}")
            if len(native_rows) != len(bridge_rows):
                mismatch_count += 1
                mismatch_field_counts['repeat_count'] = (
                    int(mismatch_field_counts.get('repeat_count', 0)) + 1
                )
                if first_mismatch_field == '':
                    first_mismatch_field = 'repeat_count'
                    first_mismatch_step = -1
                    first_mismatch_detail = {
                        'field': 'repeat_count',
                        'step': -1,
                        'native_repeat_count': int(len(native_rows)),
                        'bridge_repeat_count': int(len(bridge_rows)),
                    }
                mismatches.append(
                    f"repeat-count mismatch native={len(native_rows)} bridge={len(bridge_rows)}"
                )
            if len(mismatches) >= 40:
                mismatches = mismatches[:40] + ['... additional mismatches truncated ...']
            family_payload[family] = {
                'ok': mismatch_count == 0,
                'steps_compared': int(steps_compared),
                'mismatch_count': int(mismatch_count),
                'mismatches': mismatches,
                'mismatch_field_counts': mismatch_field_counts,
                'mismatch_step_counts': mismatch_step_counts,
                'first_mismatch_field': first_mismatch_field,
                'first_mismatch_step': int(first_mismatch_step),
                'first_mismatch_detail': first_mismatch_detail,
                'repeat_results': repeat_results,
            }
        parity_by_profile[profile] = family_payload
    return parity_by_profile


def _float_close(lhs: Any, rhs: Any, *, atol: float, rtol: float) -> bool:
    try:
        lv = float(lhs)
        rv = float(rhs)
        return abs(lv - rv) <= (float(atol) + float(rtol) * max(abs(lv), abs(rv)))
    except Exception:
        return False


def _vector_close(lhs: Any, rhs: Any, *, atol: float, rtol: float) -> bool:
    try:
        lvec = [float(x) for x in list(lhs)]
        rvec = [float(x) for x in list(rhs)]
    except Exception:
        return False
    if len(lvec) != len(rvec):
        return False
    return all(
        abs(a - b) <= (float(atol) + float(rtol) * max(abs(a), abs(b)))
        for a, b in zip(lvec, rvec)
    )


def _choice_to_bool(choice: str) -> bool | None:
    if choice == 'true':
        return True
    if choice == 'false':
        return False
    return None


def _build_native_options(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    raw = str(getattr(args, 'native_options_json', '{}') or '{}')
    if raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                payload.update(parsed)
        except Exception as exc:
            raise ValueError(f'Invalid --native-options-json payload: {exc}') from exc
    overrides = {
        'ensure_available_actions': _choice_to_bool(args.ensure_available_actions),
        'pipeline_actions_and_step': _choice_to_bool(args.pipeline_actions_and_step),
        'pipeline_step_and_observe': _choice_to_bool(args.pipeline_step_and_observe),
        'reuse_step_observe_requests': _choice_to_bool(args.reuse_step_observe_requests),
    }
    for key, value in overrides.items():
        if value is not None:
            payload[key] = value
    return payload


def _effective_steady_parity_steps(args: argparse.Namespace) -> int:
    mode = str(getattr(args, 'steady_parity_mode', 'windowed'))
    if mode == 'strict':
        return 0
    return max(int(getattr(args, 'steady_parity_steps', 0)), 0)


def main() -> int:
    _ensure_project_on_path()
    from smac_unified import make_env

    parser = argparse.ArgumentParser(
        description=(
            'Run core-first standalone validation for native and bridge backends.'
        )
    )
    parser.add_argument(
        '--families',
        nargs='+',
        default=['smac', 'smacv2', 'smac-hard'],
        choices=['smac', 'smacv2', 'smac-hard'],
    )
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument(
        '--repeats',
        type=int,
        default=1,
        help='Number of repeated runs per family/backend/profile case.',
    )
    parser.add_argument(
        '--warmup-steps',
        type=int,
        default=1,
        help='Exclude first N steps from steady-state SPS.',
    )
    parser.add_argument(
        '--profile',
        choices=['quick', 'steady', 'both'],
        default='quick',
        help='Benchmark profile: quick (default), steady, or both.',
    )
    parser.add_argument(
        '--steady-steps',
        type=int,
        default=200,
        help='Step count for steady profile.',
    )
    parser.add_argument(
        '--steady-warmup-steps',
        type=int,
        default=20,
        help='Warmup step count for steady profile.',
    )
    parser.add_argument(
        '--steady-parity-steps',
        type=int,
        default=3,
        help='When profile is steady, compare only the first N trace steps for parity (0 disables).',
    )
    parser.add_argument(
        '--steady-parity-mode',
        choices=['windowed', 'strict'],
        default='windowed',
        help='Steady parity compare mode. strict compares full trace.',
    )
    parser.add_argument(
        '--parity-atol',
        type=float,
        default=1e-4,
        help='Absolute tolerance for native-vs-bridge parity checks.',
    )
    parser.add_argument(
        '--parity-rtol',
        type=float,
        default=1e-5,
        help='Relative tolerance for native-vs-bridge parity checks.',
    )
    parser.add_argument(
        '--parity-tol',
        type=float,
        default=None,
        help='Deprecated absolute tolerance alias. Overrides parity-atol and sets parity-rtol=0.',
    )
    parser.add_argument(
        '--assert-parity',
        action='store_true',
        help='Fail validation if any profile/family parity check mismatches.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=12345,
        help='Deterministic seed used for both bridge/native runs.',
    )
    parser.add_argument(
        '--normalized-api',
        action='store_true',
        help='Benchmark the normalized adapter API path instead of raw env API.',
    )
    parser.add_argument(
        '--native-options-json',
        default='{}',
        help='JSON object forwarded as make_env(native_options=...).',
    )
    parser.add_argument(
        '--ensure-available-actions',
        choices=['default', 'true', 'false'],
        default='default',
        help='Override native option ensure_available_actions.',
    )
    parser.add_argument(
        '--pipeline-actions-and-step',
        choices=['default', 'true', 'false'],
        default='default',
        help='Override native option pipeline_actions_and_step.',
    )
    parser.add_argument(
        '--pipeline-step-and-observe',
        choices=['default', 'true', 'false'],
        default='default',
        help='Override native option pipeline_step_and_observe.',
    )
    parser.add_argument(
        '--reuse-step-observe-requests',
        choices=['default', 'true', 'false'],
        default='default',
        help='Override native option reuse_step_observe_requests.',
    )
    parser.add_argument('--output-json', default='tools/native_core_validation.json')
    args = parser.parse_args()
    repeats = max(int(args.repeats), 1)
    try:
        native_options = _build_native_options(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.profile == 'quick':
        run_profiles = [('quick', args.steps, args.warmup_steps)]
    elif args.profile == 'steady':
        run_profiles = [('steady', args.steady_steps, args.steady_warmup_steps)]
    else:
        run_profiles = [
            ('quick', args.steps, args.warmup_steps),
            ('steady', args.steady_steps, args.steady_warmup_steps),
        ]

    results: List[CaseResult] = []
    for profile_name, steps, warmup_steps in run_profiles:
        for family in args.families:
            map_name = DEFAULT_MAPS[family]
            for repeat_idx in range(repeats):
                bridge_row = _run_case(
                    profile=profile_name,
                    family=family,
                    map_name=map_name,
                    backend_mode='bridge',
                    repeat_idx=repeat_idx,
                    steps=steps,
                    warmup_steps=warmup_steps,
                    seed=int(args.seed),
                    forced_actions=None,
                    normalized_api=bool(args.normalized_api),
                    native_options=native_options,
                    make_env_fn=make_env,
                )
                results.append(bridge_row)
                bridge_status = 'PASS' if bridge_row.ok else 'FAIL'
                print(
                    f'[{bridge_status}] profile={profile_name} family={family} '
                    f'repeat={repeat_idx} mode=bridge map={map_name} '
                    f'cold_sps={bridge_row.sps:.3f} steady_sps={bridge_row.steady_sps:.3f} '
                    f'p95_ms={bridge_row.step_latency_ms_p95:.3f} elapsed={bridge_row.elapsed_s:.3f}s '
                    f'error={bridge_row.error}'
                )
                forced_actions = None
                if bridge_row.ok and bridge_row.trace:
                    forced_actions = [
                        [int(a) for a in step.get('actions', [])]
                        for step in bridge_row.trace
                    ]
                native_row = _run_case(
                    profile=profile_name,
                    family=family,
                    map_name=map_name,
                    backend_mode='native',
                    repeat_idx=repeat_idx,
                    steps=steps,
                    warmup_steps=warmup_steps,
                    seed=int(args.seed),
                    forced_actions=forced_actions,
                    normalized_api=bool(args.normalized_api),
                    native_options=native_options,
                    make_env_fn=make_env,
                )
                results.append(native_row)
                native_status = 'PASS' if native_row.ok else 'FAIL'
                print(
                    f'[{native_status}] profile={profile_name} family={family} '
                    f'repeat={repeat_idx} mode=native map={map_name} '
                    f'cold_sps={native_row.sps:.3f} steady_sps={native_row.steady_sps:.3f} '
                    f'p95_ms={native_row.step_latency_ms_p95:.3f} elapsed={native_row.elapsed_s:.3f}s '
                    f'error={native_row.error}'
                )

    parity_atol = float(args.parity_atol)
    parity_rtol = float(args.parity_rtol)
    if args.parity_tol is not None:
        parity_atol = float(args.parity_tol)
        parity_rtol = 0.0
    summary_by_profile = _summarize_by_profile(results)
    effective_steady_parity_steps = _effective_steady_parity_steps(args)
    parity_by_profile = _summarize_parity_by_profile(
        results=results,
        atol=parity_atol,
        rtol=parity_rtol,
        steady_parity_steps=effective_steady_parity_steps,
    )
    primary_profile = run_profiles[0][0]
    primary_parity = parity_by_profile.get(primary_profile, {})
    for profile_name, family_payload in parity_by_profile.items():
        for family, payload in family_payload.items():
            status = 'PASS' if payload.get('ok', False) else 'FAIL'
            print(
                f"[{status}] parity profile={profile_name} family={family} "
                f"steps={payload.get('steps_compared', 0)} "
                f"mismatches={payload.get('mismatch_count', 0)} "
                f"first_field={payload.get('first_mismatch_field', '')} "
                f"first_step={payload.get('first_mismatch_step', -1)}"
            )
    output = {
        'requested_profile': args.profile,
        'seed': int(args.seed),
        'repeats': repeats,
        'normalized_api': bool(args.normalized_api),
        'native_options': native_options,
        'steady_parity_mode': str(args.steady_parity_mode),
        'steady_parity_steps': max(int(args.steady_parity_steps), 0),
        'steady_parity_steps_effective': int(effective_steady_parity_steps),
        'profiles': [
            {
                'name': profile_name,
                'steps': steps,
                'warmup_steps': warmup_steps,
            }
            for profile_name, steps, warmup_steps in run_profiles
        ],
        'results': [asdict(row) for row in results],
        'summary': summary_by_profile.get(primary_profile, {}),
        'summary_by_profile': summary_by_profile,
        'parity': primary_parity,
        'parity_by_profile': parity_by_profile,
        'parity_tolerance': {
            'atol': parity_atol,
            'rtol': parity_rtol,
        },
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(f'Wrote validation report: {output_path}')

    if not all(row.ok for row in results):
        return 1
    if args.assert_parity:
        all_parity_ok = all(
            payload.get('ok', False)
            for profile_payload in parity_by_profile.values()
            for payload in profile_payload.values()
        )
        if not all_parity_ok:
            return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
