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
conda run -n smacnt python tools/native_core_validation.py --steps 3
```

These validate:
- backend selection policy,
- switch presets/overrides,
- scripted runtime compatibility wrapping,
- native vs bridge core stepping sanity on baseline maps.

## Migration

See `MIGRATION.md` for bridge-to-native migration guidance.
