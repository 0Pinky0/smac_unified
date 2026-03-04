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

Pass `native_options={...}` for expert overrides.

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

Scripted runtime is integrated through the common opponent interface. For safety, native mode defaults to single-controller bot opposition and leaves dual-controller scripted stepping disabled unless explicitly enabled:

```python
env = make_env(
    family="smac-hard",
    map_name="3m",
    backend_mode="native",
    native_options={"enable_dual_controller": True},
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

`native_core_validation.py` now runs a deterministic bridge-first action trace and replays it on native mode, then compares traces with combined `atol+rtol` tolerance checks and strict key/step alignment. The generated `tools/native_core_validation.json` report includes parity diagnostics and SPS metrics (`cold_sps` + `steady_sps`) with repeat/latency support.

For `steady` profile, parity comparison defaults to an early deterministic window (`--steady-parity-steps 3`) so long-horizon SPS runs remain performance-focused. Set `--steady-parity-steps 0` for strict full-trace parity.

## SPS Rollout Decisions

Transport matrix (steady profile, parity-gated):

- `B0`: defaults
- `B1`: `reuse_step_observe_requests=True`
- `B2`: `B1 + pipeline_step_and_observe=True`
- `B3`: `B2 + pipeline_actions_and_step=True`
- `B4`: `B3 + ensure_available_actions=False` (experimental)

Observed steady native SPS (`smac`, `smacv2`, `smac-hard`):

- `B0`: `1047`, `823`, `926`
- `B1`: `973`, `823`, `851`
- `B2`: `1440`, `858`, `1316`  **best throughput**
- `B3`: `821`, `730`, `894`
- `B4`: `879`, `754`, `794`

Rollout policy:

- Keep conservative runtime defaults unchanged.
- Keep `B2` as an expert opt-in via `native_options` because it has the best SPS but is a higher-performance transport profile.
- Keep `B4` non-default experimental only (`ensure_available_actions=False`).

## Migration

See `MIGRATION.md` for bridge-to-native migration guidance.
