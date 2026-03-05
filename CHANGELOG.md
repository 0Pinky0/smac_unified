# Changelog

All notable changes to `smac-unified` are documented in this file.

## [Unreleased]

- No unreleased changes.

## [0.1.0] - 2026-03-02

- Native-first runtime for `smac`, `smacv2`, and `smac-hard` with unified `make_env` and `make_env_pool`.
- Modular handler stack (`action`, `obs`, `state`, `reward`) and player subsystem (`sc2_computer`, scripted pool).
- Matrix-driven validation tooling with strict parity diagnostics and internal RC gate script.
- Hard removal of production compat/backend selection surfaces in favor of native-only runtime semantics.
