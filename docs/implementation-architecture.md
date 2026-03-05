# Implementation Architecture

This document explains how the current implementation is composed and why.

## 1. Native-First Architecture

Production runtime path:

`make_env` -> `SMACEnvCore` -> `SC2EnvRawSession` -> handler/player modules

Validation-only bridge path:

`tools/native_core_validation.py` -> `tools/bridge_tools/*`

Bridge tooling is intentionally outside production package APIs.

## 2. Core Components

### 2.1 `SMACEnvCore`

Responsibilities:

- lifecycle (`reset`, `step`, `close`)
- session coordination
- wiring handlers and opponent runtime
- collecting legacy-compatible info fields

Non-responsibilities:

- detailed action/obs/state/reward feature logic (delegated to handlers)

### 2.2 `SC2EnvRawSession`

Responsibilities:

- launch and close SC2 raw env safely
- step submit/collect flow
- optional async submit/collect handling
- transport option configuration

### 2.3 `UnitTracker`

Responsibilities:

- convert raw units into stable slot-indexed `UnitFrame`
- preserve consistent unit identity and frame-to-frame deltas

## 3. Handler Layer

Handler contracts:

- `ActionHandler`
- `ObservationHandler`
- `StateHandler`
- `RewardHandler`

Shared inputs:

- `UnitFrame`
- `HandlerContext`

Default selection is done by handler factory based on active switches.

## 4. Player/Opponent Layer

`players/` controls opponent behavior:

- engine-bot runtime
- scripted runtime (`smac-hard`)
- compatibility wrappers for script signature variation

This keeps opponent logic independent from core session and handler concerns.

## 5. Switches and Runtime Selection

`VariantSwitches` is the central config shape for family behavior:

- action mode
- opponent mode
- capability mode
- reward-positive mode
- team-init mode

`core.env_core.make_env` merges defaults and overrides, then wires:

- handlers
- runtime
- transport options

## 6. Transport Profiles

Profiles are mapped once in env factory:

- `B0`: defaults
- `B1`: reuse observe requests
- `B2`: B1 + pipeline step/observe
- `B3`: B2 + pipeline actions/step
- `B4`: B3 + disable ensure_available_actions (experimental)

Safety guard:

- `B4` or equivalent requires `allow_experimental_transport=True`

## 7. Validation Stack

`tools/native_core_validation.py` supports:

- matrix presets and custom lanes
- profile modes (`quick`, `steady`, `both`)
- parity compare with mismatch diagnostics
- report generation with schema versioning

Main generated artifacts:

- `tools/native_core_validation_critical_core.json`
- `tools/native_core_validation_smachard_longtail_b{0..3}.json`
- `tools/native_core_validation_throughput_sanity.json`

## 8. Internal RC Gate

`tools/run_internal_rc_gate.sh` executes:

1. core tests
2. critical strict parity
3. smac-hard long-tail strict parity B0-B3
4. native-only throughput sanity

This is the canonical release-facing gate for the current implementation stage.
