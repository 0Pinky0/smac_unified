#!/usr/bin/env python3
"""Core-first standalone validation for native/bridge backends."""

from __future__ import annotations

import argparse
import json
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
    trace: list[dict[str, Any]] | None = None
    error: str = ''


def _run_case(
    *,
    profile: str,
    family: str,
    map_name: str,
    backend_mode: str,
    steps: int,
    warmup_steps: int,
    seed: int,
    forced_actions: list[list[int]] | None,
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
warmup_steps = max(0, min(warmup_steps, steps))

t0 = time.perf_counter()
env = make_env(
    family=family,
    map_name=map_name,
    backend_mode=backend_mode,
    normalized_api=False,
    seed=seed,
)
env.reset()
t_reset = time.perf_counter()
step_elapsed = 0.0
steady_elapsed = 0.0
steady_steps = 0
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
    t_step0 = time.perf_counter()
    reward, terminated, info = env.step(chosen)
    t_step1 = time.perf_counter()
    dt = max(t_step1 - t_step0, 0.0)
    step_elapsed += dt
    if step_idx >= warmup_steps:
        steady_elapsed += dt
        steady_steps += 1
    obs = np.asarray(env.get_obs(), dtype=np.float32)
    state = np.asarray(env.get_state(), dtype=np.float32)
    avail_actions = [
        list(map(int, env.get_avail_agent_actions(agent_id)))
        for agent_id in range(env.n_agents)
    ]
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
        trace=list(payload.get('trace', []) or []),
    )


def _summarize(results: List[CaseResult]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for family in sorted({row.family for row in results}):
        by_mode = {row.backend_mode: row for row in results if row.family == family}
        native = by_mode.get('native')
        bridge = by_mode.get('bridge')
        payload: Dict[str, float] = {}
        if native:
            payload['native_ok'] = float(native.ok)
            payload['native_sps'] = native.sps
            payload['native_steady_sps'] = native.steady_sps
        if bridge:
            payload['bridge_ok'] = float(bridge.ok)
            payload['bridge_sps'] = bridge.sps
            payload['bridge_steady_sps'] = bridge.steady_sps
        if native and bridge and bridge.sps > 0:
            payload['native_vs_bridge_delta_pct'] = (
                (native.sps - bridge.sps) * 100.0 / bridge.sps
            )
        if native and bridge and bridge.steady_sps > 0:
            payload['native_vs_bridge_steady_delta_pct'] = (
                (native.steady_sps - bridge.steady_sps) * 100.0 / bridge.steady_sps
            )
        summary[family] = payload
    return summary


def _summarize_by_profile(results: List[CaseResult]) -> Dict[str, Dict[str, Dict[str, float]]]:
    summary_by_profile: Dict[str, Dict[str, Dict[str, float]]] = {}
    for profile in sorted({row.profile for row in results}):
        profile_rows = [row for row in results if row.profile == profile]
        summary_by_profile[profile] = _summarize(profile_rows)
    return summary_by_profile


def _compare_case_pair(
    *,
    native: CaseResult,
    bridge: CaseResult,
    atol: float,
    rtol: float,
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
    }
    if not native.ok or not bridge.ok:
        payload['mismatches'] = ['native/bridge run failed; trace compare skipped']
        payload['mismatch_count'] = 1
        return payload
    native_trace = native.trace or []
    bridge_trace = bridge.trace or []
    if not native_trace or not bridge_trace:
        payload['mismatches'] = ['native/bridge trace missing']
        payload['mismatch_count'] = 1
        return payload

    steps = min(len(native_trace), len(bridge_trace))
    mismatches: list[str] = []
    if len(native_trace) != len(bridge_trace):
        mismatches.append(
            f'trace length mismatch native={len(native_trace)} bridge={len(bridge_trace)}'
        )
    for step_idx in range(steps):
        lhs = native_trace[step_idx]
        rhs = bridge_trace[step_idx]
        lhs_missing = sorted(required_trace_keys - set(lhs.keys()))
        rhs_missing = sorted(required_trace_keys - set(rhs.keys()))
        if lhs_missing:
            mismatches.append(
                f'step {step_idx}: native trace missing keys {lhs_missing}'
            )
        if rhs_missing:
            mismatches.append(
                f'step {step_idx}: bridge trace missing keys {rhs_missing}'
            )
        if lhs.get('step') != rhs.get('step'):
            mismatches.append(
                f"step {step_idx}: step id mismatch native={lhs.get('step')} bridge={rhs.get('step')}"
            )
        if lhs.get('actions') != rhs.get('actions'):
            mismatches.append(f'step {step_idx}: actions mismatch')
        if not _float_close(
            lhs.get('reward', 0.0),
            rhs.get('reward', 0.0),
            atol=atol,
            rtol=rtol,
        ):
            mismatches.append(
                f"step {step_idx}: reward mismatch native={lhs.get('reward')} bridge={rhs.get('reward')}"
            )
        for key in ('terminated', 'battle_won', 'episode_limit', 'dead_allies', 'dead_enemies'):
            if lhs.get(key) != rhs.get(key):
                mismatches.append(f'step {step_idx}: {key} mismatch')
        if list(lhs.get('obs_shape', [])) != list(rhs.get('obs_shape', [])):
            mismatches.append(f'step {step_idx}: obs_shape mismatch')
        if list(lhs.get('state_shape', [])) != list(rhs.get('state_shape', [])):
            mismatches.append(f'step {step_idx}: state_shape mismatch')
        if lhs.get('avail_actions') != rhs.get('avail_actions'):
            mismatches.append(f'step {step_idx}: avail_actions mismatch')
        if not _vector_close(
            lhs.get('obs_head', []),
            rhs.get('obs_head', []),
            atol=atol,
            rtol=rtol,
        ):
            mismatches.append(f'step {step_idx}: obs_head mismatch')
        if not _vector_close(
            lhs.get('state_head', []),
            rhs.get('state_head', []),
            atol=atol,
            rtol=rtol,
        ):
            mismatches.append(f'step {step_idx}: state_head mismatch')
        if len(mismatches) >= 20:
            mismatches.append('... additional mismatches truncated ...')
            break

    payload['steps_compared'] = steps
    payload['mismatch_count'] = len(mismatches)
    payload['mismatches'] = mismatches
    payload['ok'] = len(mismatches) == 0
    return payload


def _summarize_parity_by_profile(
    *,
    results: List[CaseResult],
    atol: float,
    rtol: float,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    parity_by_profile: Dict[str, Dict[str, Dict[str, Any]]] = {}
    profile_keys = sorted({row.profile for row in results})
    for profile in profile_keys:
        family_payload: Dict[str, Dict[str, Any]] = {}
        profile_rows = [row for row in results if row.profile == profile]
        for family in sorted({row.family for row in profile_rows}):
            by_mode = {
                row.backend_mode: row
                for row in profile_rows
                if row.family == family
            }
            native = by_mode.get('native')
            bridge = by_mode.get('bridge')
            if native is None or bridge is None:
                family_payload[family] = {
                    'ok': False,
                    'steps_compared': 0,
                    'mismatch_count': 1,
                    'mismatches': ['missing native or bridge run'],
                }
                continue
            family_payload[family] = _compare_case_pair(
                native=native,
                bridge=bridge,
                atol=atol,
                rtol=rtol,
            )
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
    parser.add_argument('--output-json', default='tools/native_core_validation.json')
    args = parser.parse_args()

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
            bridge_row = _run_case(
                profile=profile_name,
                family=family,
                map_name=map_name,
                backend_mode='bridge',
                steps=steps,
                warmup_steps=warmup_steps,
                seed=int(args.seed),
                forced_actions=None,
                make_env_fn=make_env,
            )
            results.append(bridge_row)
            bridge_status = 'PASS' if bridge_row.ok else 'FAIL'
            print(
                f'[{bridge_status}] profile={profile_name} family={family} '
                f'mode=bridge map={map_name} cold_sps={bridge_row.sps:.3f} '
                f'steady_sps={bridge_row.steady_sps:.3f} elapsed={bridge_row.elapsed_s:.3f}s '
                f'startup={bridge_row.startup_s:.3f}s step={bridge_row.step_elapsed_s:.3f}s '
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
                steps=steps,
                warmup_steps=warmup_steps,
                seed=int(args.seed),
                forced_actions=forced_actions,
                make_env_fn=make_env,
            )
            results.append(native_row)
            native_status = 'PASS' if native_row.ok else 'FAIL'
            print(
                f'[{native_status}] profile={profile_name} family={family} '
                f'mode=native map={map_name} cold_sps={native_row.sps:.3f} '
                f'steady_sps={native_row.steady_sps:.3f} elapsed={native_row.elapsed_s:.3f}s '
                f'startup={native_row.startup_s:.3f}s step={native_row.step_elapsed_s:.3f}s '
                f'error={native_row.error}'
            )

    parity_atol = float(args.parity_atol)
    parity_rtol = float(args.parity_rtol)
    if args.parity_tol is not None:
        parity_atol = float(args.parity_tol)
        parity_rtol = 0.0
    summary_by_profile = _summarize_by_profile(results)
    parity_by_profile = _summarize_parity_by_profile(
        results=results,
        atol=parity_atol,
        rtol=parity_rtol,
    )
    primary_profile = run_profiles[0][0]
    primary_parity = parity_by_profile.get(primary_profile, {})
    for profile_name, family_payload in parity_by_profile.items():
        for family, payload in family_payload.items():
            status = 'PASS' if payload.get('ok', False) else 'FAIL'
            print(
                f"[{status}] parity profile={profile_name} family={family} "
                f"steps={payload.get('steps_compared', 0)} "
                f"mismatches={payload.get('mismatch_count', 0)}"
            )
    output = {
        'requested_profile': args.profile,
        'seed': int(args.seed),
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
