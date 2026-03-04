# Migration: Native-Only Contract

This guide describes migration to the native-only standalone runtime.

## 1) Remove Backend Flags

Production `make_env(...)` no longer accepts backend-selection parameters.
Remove any usage of:

- `backend_mode`
- `backend_registry`
- `bridge_options`

If you still need bridge comparison, use tests tooling (`tests/bridge_tools`) via
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

Run:

```bash
conda run -n smacnt python tools/run_core_tests.py
conda run -n smacnt python tools/native_core_validation.py --steps 3 --bridge-lane off
conda run -n smacnt python tools/native_core_validation.py --steps 3 --bridge-lane on --assert-parity
```

Use the generated `tools/native_core_validation.json` report for native SPS, and parity diagnostics when bridge lane is enabled.
