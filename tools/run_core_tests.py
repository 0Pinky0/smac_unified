#!/usr/bin/env python3
"""Minimal test runner without external pytest dependency."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


TEST_FUNCTIONS = [
    ("tests.test_logic_switches", "test_default_switches_have_expected_modes"),
    ("tests.test_logic_switches", "test_switch_overrides_apply_cleanly"),
    ("tests.test_backend_registry", "test_registry_prefers_native_in_auto_mode"),
    ("tests.test_backend_registry", "test_registry_can_force_bridge_mode"),
    ("tests.test_scripted_runtime", "test_scripted_runtime_computes_actions_from_payload"),
    ("tests.test_unit_tracker", "test_unit_tracker_keeps_stable_ids_across_updates"),
    (
        "tests.test_native_builders",
        "test_native_action_builder_avail_masks_are_deterministic",
    ),
    (
        "tests.test_native_builders",
        "test_native_reward_builder_uses_frame_deltas_and_scaling",
    ),
    (
        "tests.test_native_builders",
        "test_native_observation_and_state_builders_follow_unified_contract",
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
        print(f"[PASS] {module_name}.{function_name}")
    print(f"All {len(TEST_FUNCTIONS)} core tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
