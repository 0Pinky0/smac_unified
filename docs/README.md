# Documentation

This folder is the primary documentation entry for `smac-unified`.

## Start Here

- Users: see `user-guide.md`
- Developers: see `developer-guide.md`
- Architecture and validation flow: see `implementation-architecture.md`

## Scope

These docs describe the current native-first implementation:

- production runtime is native-only (`make_env`, `make_env_pool`)
- environment family differences are controlled by logic switches
- bridge behavior is validation-only tooling under `tools/bridge_tools`

For migration notes from older backend-style usage, also see `MIGRATION.md`.
