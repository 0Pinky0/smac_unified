# User Guide

This guide is for running training or evaluation workloads with `smac-unified`.

## 1. What You Get

`smac-unified` provides one native-first API for:

- `smac`
- `smacv2`
- `smac-hard`

Use:

- `make_env(...)` for one environment
- `make_env_pool(...)` for multiple environments

## 2. Install

From the `smac_unified` project root:

```bash
pip install -e .
```

If your monorepo contains local `pysc2-compat`, point source root:

```bash
export SMAC_UNIFIED_SOURCE_ROOT=/path/to/SoManySMACs
```

## 3. Quick Start (Normalized API)

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

`batch` is a `StepBatch` with:

- `obs`
- `state`
- `avail_actions`
- `reward`
- `terminated`
- `info`
- `episode_step`

## 4. Raw-Style API

If you need legacy-style method calls, set `normalized_api=False`:

```python
env = make_env(
    family="smac",
    map_name="3m",
    normalized_api=False,
)

obs, state = env.reset()
reward, terminated, info = env.step([0] * env.n_agents)
env.close()
```

## 5. Parallel Environments

```python
from smac_unified import make_env_pool

pool = make_env_pool(
    num_envs=4,
    family="smacv2",
    map_name="8m",
    normalized_api=True,
    pool_mode="thread",  # "sync" or "thread"
    seed=42,
)

batches = pool.reset(seeds=[42, 43, 44, 45])
pool.close()
```

## 6. Family Behavior with Logic Switches

`logic_switches` changes family-specific behavior in a controlled way.

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

Main switch keys:

- `action_mode`: `classic`, `conic_fov`, `ability_augmented`
- `opponent_mode`: `sc2_computer`, `scripted_pool`
- `capability_mode`: `none`, `stochastic_attack`, `stochastic_health`, `team_gen`
- `reward_positive_mode`: `abs`, `clamp_zero`
- `team_init_mode`: `map_default`, `episode_generated`

## 7. smac-hard Opponent Mode

Default `smac-hard` behavior is scripted opponent runtime.

Use SC2 built-in bot explicitly:

```python
env = make_env(
    family="smac-hard",
    map_name="3m",
    logic_switches={"opponent_mode": "sc2_computer"},
)
```

## 8. Transport Profiles

Use `transport_profile` for runtime transport options:

- `B0`: safe defaults
- `B1`: reuse observe requests
- `B2`: B1 + pipeline step/observe
- `B3`: B2 + pipeline actions/step
- `B4`: experimental (`ensure_available_actions=False`)

`B4` requires:

```python
allow_experimental_transport=True
```

## 9. Important Contract Notes

- `backend_mode`, `backend_registry`, `bridge_options` are removed from production API.
- Bridge comparison stays in tooling only (`tools/native_core_validation.py`).

## 10. Common Issues

- Import/runtime path problems:
  - set `SMAC_UNIFIED_SOURCE_ROOT` to your monorepo root
- Scripted opponent launch issues:
  - verify SC2 + maps are installed
  - test with `tools/run_core_tests.py`
- Throughput sanity:
  - run `tools/native_core_validation.py --bridge-lane off ...`
