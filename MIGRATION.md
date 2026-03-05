# Migration: Native-Only Contract

This guide describes migration to the native-only standalone runtime.

Release metadata:
- License: `LICENSE`
- Release notes: `CHANGELOG.md`

## 1) Remove Backend Flags

Production `make_env(...)` no longer accepts backend-selection parameters.
Remove any usage of:

- `backend_mode`
- `backend_registry`
- `bridge_options`

If you still need bridge comparison, use bridge tooling (`tools/bridge_tools`) via
`tools/native_core_validation.py --bridge-lane on`.

## 2) Keep API Surface Stable First

Start with `normalized_api=True` and keep your existing reset/step loops unchanged.

```python
env = make_env(
    family='smac',
    map_name='3m',
    normalized_api=True,
)
```

## 3) Port Variant Differences via Logic Switches

Instead of hardcoding family-specific assumptions, move behavior to switch config:

```python
logic_switches={
    'action_mode': '...',
    'opponent_mode': '...',
    'reward_positive_mode': '...',
}
```

Recommended path:
- start from defaults per family,
- only override when matching old experiment semantics.

## 4) Opponent Migration

- `smac` / `smacv2`: use `opponent_mode='sc2_computer'`.
- `smac-hard`: use `opponent_mode='scripted_pool'` with common scripted runtime.

Native `smac-hard` defaults to scripted dual-controller stepping and fails fast if scripted prerequisites are not met.
To use SC2 built-in bot instead, switch opponent mode explicitly:

```python
logic_switches={'opponent_mode': 'sc2_computer'}
```

## 5) Validate Before Full Rollout

Quick local checks:

```bash
conda run -n smacnt python tools/run_core_tests.py
conda run -n smacnt python tools/native_core_validation.py --steps 3 --bridge-lane off
conda run -n smacnt python tools/native_core_validation.py --families smac smacv2 --matrix-preset critical-core --profile steady --steady-steps 120 --steady-warmup-steps 20 --steady-parity-mode strict --repeats 2 --assert-parity --output-json tools/native_core_validation_critical_core.json
conda run -n smacnt python tools/native_core_validation.py --families smac-hard --matrix-preset smac-hard-longtail --profile steady --steady-steps 60 --steady-warmup-steps 10 --steady-parity-mode strict --repeats 1 --repeat-seed-stride 0 --seed 1 --force-opponent-actions-from-bridge --assert-parity --output-json tools/native_core_validation_smachard_longtail_b0.json
conda run -n smacnt python tools/native_core_validation.py --families smac smacv2 smac-hard --profile quick --steps 5 --warmup-steps 1 --bridge-lane off --parallel-envs 4 --pool-mode thread --output-json tools/native_core_validation_throughput_sanity.json
```

Internal RC one-shot gate:

```bash
tools/run_internal_rc_gate.sh
```

Use generated reports in `tools/` for native SPS and parity diagnostics. For scripted `smac-hard` strict parity gates, keep `--force-opponent-actions-from-bridge` and the fixed parity seed lane (`--seed 1 --repeat-seed-stride 0`) to avoid seed-sensitive scripted drift amplification.
