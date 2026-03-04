# smac-unified

Standalone, modular, native-first unified entry point for SMAC-family environments:
- `smac`
- `smacv2`
- `smac-hard`

This package now provides a native backend built on `pysc2-compat` `SC2Env` raw mode, with explicit logic-switch presets for family-level behavior differences.

## Install

```bash
pip install -e .
```

Optional runtime dependencies for native backend:

```bash
pip install -e ".[native-backend]"
```

If your `pysc2-compat` source is not installed as a package, you can point to the monorepo root:

```bash
export SMAC_UNIFIED_SOURCE_ROOT=/path/to/SoManySMACs
```

## Quick Start

```python
from smac_unified import make_env

env = make_env(
    family="smac",
    map_name="3m",
    backend_mode="native",   # native | bridge | auto
    normalized_api=True,
)

batch = env.reset(seed=1)
actions = env.sample_random_actions()
batch = env.step_batch(actions)
env.close()
```

## Backend Modes

- `native`: force standalone native implementation (default).
- `bridge`: force legacy bridge backend (`smac` / `smacv2` / `smac_hard` imports).
- `auto`: try native first, fallback to bridge.

## Logic Switches

Use `logic_switches` to override variant defaults:

```python
env = make_env(
    family="smacv2",
    map_name="8m",
    backend_mode="native",
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
    backend_mode="native",
    transport_profile="B2",
)
```

`B4` (or any path with `ensure_available_actions=False`) is safety-guarded and requires explicit opt-in:

```python
env = make_env(
    family="smac",
    map_name="3m",
    backend_mode="native",
    transport_profile="B4",
    allow_experimental_transport=True,
)
```

Pass `native_options={...}` for expert overrides. Explicit `native_options` values override profile defaults deterministically.

## Unified Handler Model

Native runtime now follows a tracker-centered data flow:
- `UnitTracker` emits stable `UnitFrame` snapshots each step.
- `Action/Observation/State/Reward` handlers consume `UnitFrame + HandlerContext`.
- `NativeStarCraft2Env` focuses on session/lifecycle/orchestration.

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
    backend_mode="native",
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
    backend_mode="native",
)
```

To intentionally use SC2 built-in bot opposition instead of scripted pool, switch opponent mode explicitly:

```python
env = make_env(
    family="smac-hard",
    map_name="3m",
    backend_mode="native",
    logic_switches={"opponent_mode": "sc2_computer"},
)
```

## Validation Commands

Run lightweight standalone checks in `smacnt`:

```bash
conda run -n smacnt python tools/run_core_tests.py
conda run -n smacnt python tools/native_core_validation.py --profile quick --steps 3 --warmup-steps 1 --assert-parity
conda run -n smacnt python tools/native_core_validation.py --profile steady --steady-steps 300 --steady-warmup-steps 30 --assert-parity
```

These validate:
- backend selection policy,
- switch presets/overrides,
- scripted runtime compatibility wrapping,
- native vs bridge core stepping sanity on baseline maps.

`native_core_validation.py` runs a deterministic bridge-first action trace and replays it on native mode, then compares traces with combined `atol+rtol` tolerance checks and strict key/step alignment. The generated `tools/native_core_validation.json` report includes parity diagnostics and SPS metrics (`cold_sps` + `steady_sps`) with repeat/latency support.

For `steady` profile:
- default mode is windowed parity (`--steady-parity-steps 3`) for fast SPS iteration;
- strict long-horizon mode is first-class via `--steady-parity-mode strict` (full trace compare, equivalent to `--steady-parity-steps 0`).

Strict baseline gate example:

```bash
conda run -n smacnt python tools/native_core_validation.py \
  --families smac smacv2 \
  --profile steady \
  --steady-steps 300 \
  --steady-warmup-steps 30 \
  --steady-parity-mode strict \
  --repeats 2 \
  --assert-parity \
  --output-json tools/native_core_validation_transport_b0.json
```

Strict transport matrix commands (`B0`-`B4`):

```bash
conda run -n smacnt python tools/native_core_validation.py --profile steady --steady-steps 300 --steady-warmup-steps 30 --steady-parity-mode strict --repeats 2 --output-json tools/native_core_validation_transport_b0.json
conda run -n smacnt python tools/native_core_validation.py --profile steady --steady-steps 300 --steady-warmup-steps 30 --steady-parity-mode strict --repeats 2 --native-options-json '{"reuse_step_observe_requests": true}' --output-json tools/native_core_validation_transport_b1.json
conda run -n smacnt python tools/native_core_validation.py --profile steady --steady-steps 300 --steady-warmup-steps 30 --steady-parity-mode strict --repeats 2 --native-options-json '{"reuse_step_observe_requests": true, "pipeline_step_and_observe": true}' --output-json tools/native_core_validation_transport_b2.json
conda run -n smacnt python tools/native_core_validation.py --profile steady --steady-steps 300 --steady-warmup-steps 30 --steady-parity-mode strict --repeats 2 --native-options-json '{"reuse_step_observe_requests": true, "pipeline_step_and_observe": true, "pipeline_actions_and_step": true}' --output-json tools/native_core_validation_transport_b3.json
```

`B4` remains experimental. In the validator path, `ensure_available_actions=False` is safety-guarded and requires explicit `allow_experimental_transport=True` support.

Strict scripted-native `smac-hard` matrix (`B0`-`B3`, repeats=`2`):

```bash
conda run -n smacnt python tools/native_core_validation.py --families smac-hard --profile steady --steady-steps 300 --steady-warmup-steps 30 --steady-parity-mode strict --repeats 2 --output-json tools/native_core_validation_smachard_b0.json
conda run -n smacnt python tools/native_core_validation.py --families smac-hard --profile steady --steady-steps 300 --steady-warmup-steps 30 --steady-parity-mode strict --repeats 2 --native-options-json '{"reuse_step_observe_requests": true}' --output-json tools/native_core_validation_smachard_b1.json
conda run -n smacnt python tools/native_core_validation.py --families smac-hard --profile steady --steady-steps 300 --steady-warmup-steps 30 --steady-parity-mode strict --repeats 2 --native-options-json '{"reuse_step_observe_requests": true, "pipeline_step_and_observe": true}' --output-json tools/native_core_validation_smachard_b2.json
conda run -n smacnt python tools/native_core_validation.py --families smac-hard --profile steady --steady-steps 300 --steady-warmup-steps 30 --steady-parity-mode strict --repeats 2 --native-options-json '{"reuse_step_observe_requests": true, "pipeline_step_and_observe": true, "pipeline_actions_and_step": true}' --output-json tools/native_core_validation_smachard_b3.json
```

## SPS Rollout Decisions

Strict outcomes after scripted-native alignment:

- Full-family strict baseline gate (`B0`, repeats=`1`) passes for `smac`, `smacv2`, and `smac-hard`.
- Scripted-native `smac-hard` strict matrix (`B0`-`B3`, repeats=`2`) passes on all profiles with zero mismatches.
- `smac-hard` native median steady SPS in this scripted matrix: `B0=192.1`, `B1=238.3`, `B2=257.2`, `B3=275.8`.
- `B4` remains experimental-only behind the explicit transport safety guard.

Rollout policy:

- Keep conservative runtime defaults unchanged.
- Keep `B2` as the primary expert opt-in profile for broad workloads.
- Use `B3` as an additional expert opt-in for scripted `smac-hard` when validated in your deployment.
- Keep `B4` experimental-only with explicit `allow_experimental_transport=True`.

## Migration

See `MIGRATION.md` for bridge-to-native migration guidance.
