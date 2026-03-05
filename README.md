# smac-unified

Standalone, modular, native-first unified entry point for SMAC-family environments:
- `smac`
- `smacv2`
- `smac-hard`

This package provides a native-only production runtime built on `pysc2-compat` `SC2Env` raw mode, with explicit logic-switch presets for family-level behavior differences.

## Install

```bash
pip install -e .
```

Native runtime dependencies are included in the base package install.

If your `pysc2-compat` source is not installed as a package, you can point to the monorepo root:

```bash
export SMAC_UNIFIED_SOURCE_ROOT=/path/to/SoManySMACs
```

## Release Metadata

- License: `LICENSE`
- Release notes: `CHANGELOG.md`

## Quick Start

```python
from smac_unified import make_env

env = make_env(
    family="smac",
    map_name="3m",
    normalized_api=True,
)

batch = env.reset(seed=1)
actions = env.sample_random_actions()
batch = env.step_batch(actions)
env.close()
```

## Runtime Contract

- Production `make_env(...)` is native-only.
- Bridge tooling is validation-only (`tools/bridge_tools`) and used by validation scripts.
- `backend_mode` / backend-registry inputs are hard-removed from production API.

## Logic Switches

Use `logic_switches` to override variant defaults:

```python
env = make_env(
    family="smacv2",
    map_name="8m",
    logic_switches={
        "action_mode": "conic_fov",
        "reward_positive_mode": "clamp_zero",
    },
)
```

Available knobs:
- `action_mode`: `classic` | `conic_fov` | `ability_augmented`
- `opponent_mode`: `sc2_computer` | `scripted_pool`
- `capability_mode`: `none` | `stochastic_attack` | `stochastic_health` | `team_gen`
- `reward_positive_mode`: `abs` | `clamp_zero`
- `team_init_mode`: `map_default` | `episode_generated`

## Native Session Safety Defaults

Native backend starts with conservative defaults:
- `ensure_available_actions=True`
- `pipeline_actions_and_step=False`
- `pipeline_step_and_observe=False`
- `reuse_step_observe_requests=False`

You can also use explicit transport profiles:

- `B0`: defaults
- `B1`: `reuse_step_observe_requests=True`
- `B2`: `B1 + pipeline_step_and_observe=True`
- `B3`: `B2 + pipeline_actions_and_step=True`
- `B4`: `B3 + ensure_available_actions=False` (experimental)

```python
env = make_env(
    family="smac",
    map_name="3m",
    transport_profile="B2",
)
```

`B4` (or any path with `ensure_available_actions=False`) is safety-guarded and requires explicit opt-in:

```python
env = make_env(
    family="smac",
    map_name="3m",
    transport_profile="B4",
    allow_experimental_transport=True,
)
```

Pass `native_options={...}` for expert overrides. Explicit `native_options` values override profile defaults deterministically.

## Unified Handler Model

Native runtime now follows a tracker-centered data flow:
- `UnitTracker` emits stable `UnitFrame` snapshots each step.
- `Action/Observation/State/Reward` handlers consume `UnitFrame + HandlerContext`.
- `SMACEnv` focuses on session/lifecycle/orchestration.

Handler overrides use one shared factory surface:

```python
from smac_unified import make_env
from smac_unified.handlers import (
    DefaultActionHandler,
    DefaultObservationHandler,
    DefaultStateHandler,
    DefaultRewardHandler,
)

env = make_env(
    family="smac",
    map_name="3m",
    normalized_api=False,
    action_handler=DefaultActionHandler(),
    observation_handler=DefaultObservationHandler(),
    state_handler=DefaultStateHandler(),
    reward_handler=DefaultRewardHandler(),
)
```

Reward scaling ownership is centralized in the reward handler path.

### Scripted Opponents (`smac-hard`)

Scripted runtime is integrated through the common opponent interface. Native `smac-hard` now defaults to scripted dual-controller sessions, and launch is fail-fast if scripted prerequisites are not satisfied (no silent fallback to SC2 bot/no-op):

```python
env = make_env(
    family="smac-hard",
    map_name="3m",
)
```

To intentionally use SC2 built-in bot opposition instead of scripted pool, switch opponent mode explicitly:

```python
env = make_env(
    family="smac-hard",
    map_name="3m",
    logic_switches={"opponent_mode": "sc2_computer"},
)
```

## Validation Commands

Run lightweight standalone checks in `smacnt`:

```bash
conda run -n smacnt python tools/run_core_tests.py
conda run -n smacnt python tools/native_core_validation.py --profile quick --steps 3 --warmup-steps 1 --assert-parity
conda run -n smacnt python tools/native_core_validation.py --profile steady --steady-steps 300 --steady-warmup-steps 30 --assert-parity
conda run -n smacnt python tools/native_core_validation.py --profile quick --steps 3 --bridge-lane off
```

These validate:
- native-only factory contract,
- switch presets/overrides,
- scripted runtime compatibility wrapping,
- optional native-vs-bridge core stepping sanity on baseline maps.

`native_core_validation.py` runs native profiling by default and can run a deterministic bridge-first replay lane (`--bridge-lane on`, default) using `tools/bridge_tools` helpers. The generated `tools/native_core_validation.json` report includes SPS metrics (`cold_sps` + `steady_sps`) and parity diagnostics when bridge lane is enabled.

For `steady` profile:
- default mode is windowed parity (`--steady-parity-steps 3`) for fast SPS iteration;
- strict long-horizon mode is first-class via `--steady-parity-mode strict` (full trace compare, equivalent to `--steady-parity-steps 0`).

Matrix support:
- `--matrix-preset critical-core` for critical `smac`/`smacv2` maps.
- `--matrix-preset smac-hard-longtail` for scripted `smac-hard` long-tail maps.
- `--family-maps-json` and `--logic-lanes-json` for custom matrix lanes.

```bash
conda run -n smacnt python tools/native_core_validation.py \
  --families smac smacv2 \
  --matrix-preset critical-core \
  --profile steady \
  --steady-steps 120 \
  --steady-warmup-steps 20 \
  --steady-parity-mode strict \
  --repeats 2 \
  --assert-parity \
  --output-json tools/native_core_validation_critical_core.json
```

Strict scripted `smac-hard` long-tail matrix (`B0`-`B3`):

```bash
conda run -n smacnt python tools/native_core_validation.py --families smac-hard --matrix-preset smac-hard-longtail --profile steady --steady-steps 60 --steady-warmup-steps 10 --steady-parity-mode strict --repeats 1 --repeat-seed-stride 0 --seed 1 --force-opponent-actions-from-bridge --assert-parity --output-json tools/native_core_validation_smachard_longtail_b0.json
conda run -n smacnt python tools/native_core_validation.py --families smac-hard --matrix-preset smac-hard-longtail --profile steady --steady-steps 60 --steady-warmup-steps 10 --steady-parity-mode strict --repeats 1 --repeat-seed-stride 0 --seed 1 --force-opponent-actions-from-bridge --native-options-json '{"reuse_step_observe_requests": true}' --assert-parity --output-json tools/native_core_validation_smachard_longtail_b1.json
conda run -n smacnt python tools/native_core_validation.py --families smac-hard --matrix-preset smac-hard-longtail --profile steady --steady-steps 60 --steady-warmup-steps 10 --steady-parity-mode strict --repeats 1 --repeat-seed-stride 0 --seed 1 --force-opponent-actions-from-bridge --native-options-json '{"reuse_step_observe_requests": true, "pipeline_step_and_observe": true}' --assert-parity --output-json tools/native_core_validation_smachard_longtail_b2.json
conda run -n smacnt python tools/native_core_validation.py --families smac-hard --matrix-preset smac-hard-longtail --profile steady --steady-steps 60 --steady-warmup-steps 10 --steady-parity-mode strict --repeats 1 --repeat-seed-stride 0 --seed 1 --force-opponent-actions-from-bridge --native-options-json '{"reuse_step_observe_requests": true, "pipeline_step_and_observe": true, "pipeline_actions_and_step": true}' --assert-parity --output-json tools/native_core_validation_smachard_longtail_b3.json
```

Native-only throughput sanity lane:

```bash
conda run -n smacnt python tools/native_core_validation.py --families smac smacv2 smac-hard --profile quick --steps 5 --warmup-steps 1 --bridge-lane off --parallel-envs 4 --pool-mode thread --output-json tools/native_core_validation_throughput_sanity.json
```

Internal RC one-shot gate:

```bash
tools/run_internal_rc_gate.sh
```

## SPS Rollout Decisions

Current strict gate outcomes:

- Critical strict parity (`smac` + `smacv2`, critical-core matrix) passes: `tools/native_core_validation_critical_core.json`.
- Scripted `smac-hard` long-tail strict parity (`B0`-`B3`) passes with bridge-opponent replay and fixed parity seed lane: `tools/native_core_validation_smachard_longtail_b{0..3}.json`.
- `B4` remains experimental-only behind the explicit transport safety guard.

Scripted `smac-hard` parity note:

- Independent bridge/native scripted rollouts can diverge for some seeds due upstream SC2 run-to-run floating drift amplification in scripted targeting.
- Use `--force-opponent-actions-from-bridge` plus a fixed parity seed lane (`--seed 1 --repeat-seed-stride 0`) for strict regression gates that isolate core env logic.

Rollout policy:

- Keep conservative runtime defaults unchanged.
- Keep `B2` as the primary expert opt-in profile for broad workloads.
- Use `B3` as an additional expert opt-in for scripted `smac-hard` when validated in your deployment.
- Keep `B4` experimental-only with explicit `allow_experimental_transport=True`.

## Migration

See `MIGRATION.md` for native-only migration guidance.
