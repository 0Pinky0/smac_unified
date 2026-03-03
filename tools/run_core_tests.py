#!/usr/bin/env python3
"""Minimal test runner without external pytest dependency."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


TEST_FUNCTIONS = [
    ('tests.test_logic_switches', 'test_default_switches_have_expected_modes'),
    ('tests.test_logic_switches', 'test_switch_overrides_apply_cleanly'),
    ('tests.test_backend_registry', 'test_registry_prefers_native_in_auto_mode'),
    ('tests.test_backend_registry', 'test_registry_can_force_bridge_mode'),
    ('tests.test_scripted_runtime', 'test_scripted_runtime_computes_actions_from_payload'),
    ('tests.test_unit_tracker', 'test_unit_tracker_keeps_stable_ids_across_updates'),
    (
        'tests.test_native_builders',
        'test_action_handler_avail_masks_are_deterministic',
    ),
    (
        'tests.test_native_builders',
        'test_reward_handler_uses_frame_deltas_and_scaling',
    ),
    (
        'tests.test_native_builders',
        'test_observation_and_state_handlers_follow_unified_contract',
    ),
    (
        'tests.test_parity_matrix',
        'test_core_parity_matrix_covers_all_core_blocks',
    ),
    (
        'tests.test_parity_matrix',
        'test_core_parity_matrix_has_legacy_and_unified_symbols',
    ),
    (
        'tests.test_handler_factory',
        'test_factory_selects_classic_bundle_for_smac',
    ),
    (
        'tests.test_handler_factory',
        'test_factory_selects_conic_and_capability_bundle_for_smacv2',
    ),
    (
        'tests.test_handler_factory',
        'test_factory_selects_ability_augmented_bundle_for_smac_hard',
    ),
    (
        'tests.test_action_parity',
        'test_classic_action_handler_uses_pathing_grid_for_move_checks',
    ),
    (
        'tests.test_action_parity',
        'test_conic_handler_updates_fov_and_unmasks_targets',
    ),
    (
        'tests.test_action_parity',
        'test_ability_handler_enables_ability_branch_and_builds_command',
    ),
    (
        'tests.test_obs_state_parity',
        'test_observation_handler_restores_structured_feature_toggles',
    ),
    (
        'tests.test_obs_state_parity',
        'test_capability_handlers_append_capability_channels',
    ),
]


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    for module_name, function_name in TEST_FUNCTIONS:
        module = importlib.import_module(module_name)
        fn = getattr(module, function_name)
        fn()
        print(f'[PASS] {module_name}.{function_name}')
    print(f'All {len(TEST_FUNCTIONS)} core tests passed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
