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
from typing import Dict, List


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
    family: str
    map_name: str
    backend_mode: str
    ok: bool
    elapsed_s: float
    steps: int
    sps: float
    error: str = ''


def _run_case(
    *,
    family: str,
    map_name: str,
    backend_mode: str,
    steps: int,
    make_env_fn,
) -> CaseResult:
    del make_env_fn
    runner = """
import json
import sys
import time
from pathlib import Path

root = Path(sys.argv[1]).resolve()
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from smac_unified import make_env

family = sys.argv[2]
map_name = sys.argv[3]
backend_mode = sys.argv[4]
steps = int(sys.argv[5])

t0 = time.perf_counter()
env = make_env(
    family=family,
    map_name=map_name,
    backend_mode=backend_mode,
    normalized_api=False,
)
env.reset()
for _ in range(steps):
    chosen = []
    for agent_id in range(env.n_agents):
        avail = list(env.get_avail_agent_actions(agent_id))
        action = 0
        for idx, flag in enumerate(avail):
            if flag and idx != 0:
                action = idx
                break
        chosen.append(action)
    env.step(chosen)
env.close()
elapsed = max(time.perf_counter() - t0, 1e-9)
print(json.dumps({"elapsed_s": elapsed, "steps": steps}))
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
        ],
        capture_output=True,
        text=True,
    )
    elapsed_total = max(time.perf_counter() - t0, 1e-9)
    if proc.returncode != 0:
        error = proc.stderr.strip() or proc.stdout.strip() or 'unknown failure'
        return CaseResult(
            family=family,
            map_name=map_name,
            backend_mode=backend_mode,
            ok=False,
            elapsed_s=elapsed_total,
            steps=steps,
            sps=0.0,
            error=error.splitlines()[-1][:400],
        )

    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    payload = {}
    for line in reversed(lines):
        if line.startswith('{') and line.endswith('}'):
            payload = json.loads(line)
            break
    elapsed = float(payload.get('elapsed_s', elapsed_total))
    return CaseResult(
        family=family,
        map_name=map_name,
        backend_mode=backend_mode,
        ok=True,
        elapsed_s=elapsed,
        steps=steps,
        sps=steps / max(elapsed, 1e-9),
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
        if bridge:
            payload['bridge_ok'] = float(bridge.ok)
            payload['bridge_sps'] = bridge.sps
        if native and bridge and bridge.sps > 0:
            payload['native_vs_bridge_delta_pct'] = (
                (native.sps - bridge.sps) * 100.0 / bridge.sps
            )
        summary[family] = payload
    return summary


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
    parser.add_argument('--output-json', default='tools/native_core_validation.json')
    args = parser.parse_args()

    results: List[CaseResult] = []
    for family in args.families:
        map_name = DEFAULT_MAPS[family]
        for backend_mode in ('native', 'bridge'):
            row = _run_case(
                family=family,
                map_name=map_name,
                backend_mode=backend_mode,
                steps=args.steps,
                make_env_fn=make_env,
            )
            results.append(row)
            status = 'PASS' if row.ok else 'FAIL'
            print(
                f'[{status}] family={family} mode={backend_mode} map={map_name} '
                f'sps={row.sps:.3f} elapsed={row.elapsed_s:.3f}s error={row.error}'
            )

    summary = _summarize(results)
    output = {
        'results': [asdict(row) for row in results],
        'summary': summary,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(f'Wrote validation report: {output_path}')

    if not all(row.ok for row in results):
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
