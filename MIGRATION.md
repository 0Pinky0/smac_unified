# Migration: Bridge -> Native

This guide describes moving from legacy bridge backends to the native standalone runtime.

## 1) Pick Backend Mode Explicitly

- Old behavior (legacy imports): `backend_mode='bridge'`
- New recommended behavior: `backend_mode='native'`
- Transitional behavior: `backend_mode='auto'` (native-first, bridge fallback)

## 2) Keep API Surface Stable First

Start with `normalized_api=True` and keep your existing reset/step loops unchanged.

```python
env = make_env(
    family='smac',
    map_name='3m',
    backend_mode='auto',
    normalized_api=True,
)
```

Then pin to `native` once your run is stable.

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

Note: native `smac-hard` defaults to safe single-controller stepping. Enable dual-controller scripted stepping only when needed:

```python
native_options={'enable_dual_controller': True}
```

## 5) Validate Before Full Rollout

Run:

```bash
conda run -n smacnt python tools/run_core_tests.py
conda run -n smacnt python tools/native_core_validation.py --steps 3
```

Use the generated `tools/native_core_validation.json` report to compare native and bridge baseline SPS.
