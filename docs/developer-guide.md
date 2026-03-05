# Developer Guide

This guide is for contributors extending or maintaining `smac-unified`.

## 1. Design Goals

- native-first production runtime
- one unified API over `smac`, `smacv2`, `smac-hard`
- modular handlers and opponent runtimes
- strict parity and reliability gates for release confidence

## 2. Project Layout

Main package modules:

- `smac_unified/core/env_core.py`
  - `SMACEnvCore`, `make_env`, `make_env_pool`, transport profile merge, runtime wiring
- `smac_unified/core/`
  - `SMACEnvCore` orchestration
  - `SC2EnvRawSession` low-level SC2 session transport
  - `UnitTracker` stable tracked unit frames
- `smac_unified/handlers/`
  - `action`, `obs`, `state`, `reward` contracts + defaults
- `smac_unified/players/`
  - opponent runtime hooks and scripted/bot implementations
- `smac_unified/maps/`
  - map params/registry helpers
- `smac_unified/types.py`
  - `StepBatch` normalized return payload

Tooling:

- `tools/native_core_validation.py`
- `tools/run_core_tests.py`
- `tools/run_internal_rc_gate.sh`
- `tools/bridge_tools/` (validation-only bridge helpers)

## 3. Runtime Data Flow

High-level per-step flow in native env:

1. session produces raw timestep payloads
2. `UnitTracker` converts to stable `UnitFrame`
3. handlers consume `UnitFrame + HandlerContext`
4. env returns:
   - normalized `StepBatch` (adapter path), or
   - raw-style tuples/dicts (non-normalized path)

`SMACEnvCore` should stay orchestration-focused; feature logic belongs in handlers or player runtime modules.

## 4. Extension Points

### 4.1 Custom Handlers

Implement and inject contracts from:

- `handlers/action/base.py`
- `handlers/obs/base.py`
- `handlers/state/base.py`
- `handlers/reward/base.py`

Usage:

```python
env = make_env(
    family="smac",
    map_name="3m",
    normalized_api=False,
    action_handler=my_action_handler,
    observation_handler=my_obs_handler,
    state_handler=my_state_handler,
    reward_handler=my_reward_handler,
)
```

### 4.2 Custom Opponent Runtime

Implement `OpponentRuntime` from `players/base.py` and pass via `opponent_runtime=...`.

Key hooks:

- `bind_env`
- `on_reset`
- `before_step`
- `after_step`
- `compute_actions`

### 4.3 Logic-Switch Behavior

Use `VariantSwitches`/`merge_switches` to route controlled behavior differences without hardcoding family branches across the codebase.

## 5. Validation and Release Flow

Core checks:

```bash
conda run -n smacnt python tools/run_core_tests.py
```

Strict parity checks:

- critical `smac` + `smacv2` matrix
- `smac-hard` long-tail B0-B3 matrix

One-shot release gate:

```bash
bash tools/run_internal_rc_gate.sh
```

## 6. Bridge Tooling Policy

- Production package remains native-only.
- Bridge support exists only for validation comparisons in `tools/bridge_tools`.
- Legacy bridge wrappers in older repos may still import `smac_unified.compat`; bridge tools provide a local shim for validation subprocesses without restoring production compat API.

## 7. Documentation Update Rule

When changing behavior, update:

- `README.md` for user-facing contract changes
- `MIGRATION.md` for migration-impact changes
- `docs/*` for user/developer workflow and architecture clarity
