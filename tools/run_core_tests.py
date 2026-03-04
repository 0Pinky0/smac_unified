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
        'tests.test_unit_tracker',
        'test_unit_tracker_debug_probe_exposes_slot_tags_and_alive_flags',
    ),
    (
        'tests.test_native_builders',
        'test_action_handler_avail_masks_are_deterministic',
    ),
    (
        'tests.test_native_builders',
        'test_reward_handler_uses_frame_deltas_without_scaling',
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
        'test_factory_selects_classic_bundle_for_default_smacv2',
    ),
    (
        'tests.test_handler_factory',
        'test_factory_selects_conic_and_capability_bundle_when_overridden',
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
        'tests.test_action_parity',
        'test_ability_handler_falls_back_to_legacy_unit_ability',
    ),
    (
        'tests.test_action_parity',
        'test_ability_query_cache_reuses_same_step_and_invalidates_next_step',
    ),
    (
        'tests.test_action_parity',
        'test_conic_handler_invalidates_same_step_cache_after_fov_change',
    ),
    (
        'tests.test_action_parity',
        'test_ability_selection_is_stable_with_unsupported_order_noise',
    ),
    (
        'tests.test_obs_state_parity',
        'test_observation_handler_restores_structured_feature_toggles',
    ),
    (
        'tests.test_obs_state_parity',
        'test_capability_handlers_append_capability_channels',
    ),
    (
        'tests.test_obs_state_parity',
        'test_state_handler_uses_center_relative_coordinates',
    ),
    (
        'tests.test_obs_state_parity',
        'test_observation_handler_preserves_enemy_feature_layout',
    ),
    (
        'tests.test_obs_state_parity',
        'test_state_handler_preserves_flattened_team_layout',
    ),
    (
        'tests.test_obs_state_parity',
        'test_observation_handler_hides_enemy_at_exact_sight_range_boundary',
    ),
    (
        'tests.test_obs_state_parity',
        'test_state_handler_zeroes_dead_unit_rows',
    ),
    (
        'tests.test_reward_outcome_parity',
        'test_step_scales_terminal_reward_after_win_bonus',
    ),
    (
        'tests.test_reward_outcome_parity',
        'test_timeout_counts_and_omits_episode_limit_when_not_continuing',
    ),
    (
        'tests.test_reward_outcome_parity',
        'test_timeout_sets_episode_limit_flag_when_continuing',
    ),
    (
        'tests.test_reset_capability_parity',
        'test_reset_capabilities_apply_episode_vectors_and_payload',
    ),
    (
        'tests.test_reset_capability_parity',
        'test_split_raw_units_applies_health_and_enemy_masks',
    ),
    (
        'tests.test_reset_capability_parity',
        'test_stochastic_attack_probability_can_block_attack_command',
    ),
    (
        'tests.test_reset_capability_parity',
        'test_split_raw_units_preserves_enemy_observed_order_for_slot_mapping',
    ),
    (
        'tests.test_api_parity',
        'test_env_api_exposes_legacy_stats_and_metadata_fields',
    ),
    (
        'tests.test_api_parity',
        'test_smacv2_env_info_includes_cap_shape_field',
    ),
    (
        'tests.test_api_parity',
        'test_step_batch_matches_legacy_payload_contract',
    ),
    (
        'tests.test_api_parity',
        'test_reset_batch_matches_legacy_payload_contract',
    ),
    (
        'tests.test_api_parity',
        'test_seed_supports_getter_setter_and_rng_reseed',
    ),
    (
        'tests.test_api_parity',
        'test_last_action_update_is_in_place_one_hot',
    ),
    (
        'tests.test_api_parity',
        'test_last_action_contract_tracks_requested_ids_per_agent',
    ),
    (
        'tests.test_api_parity',
        'test_handler_context_refresh_reuses_instance',
    ),
    (
        'tests.test_api_parity',
        'test_transport_profile_b2_propagates_to_native_session_config',
    ),
    (
        'tests.test_api_parity',
        'test_native_options_override_transport_profile_defaults',
    ),
    (
        'tests.test_api_parity',
        'test_experimental_transport_flag_propagates_to_session_config',
    ),
    (
        'tests.test_validation_parity',
        'test_parity_compare_passes_for_matching_traces',
    ),
    (
        'tests.test_validation_parity',
        'test_parity_compare_reports_mismatch_details',
    ),
    (
        'tests.test_validation_parity',
        'test_parity_compare_detects_step_alignment_mismatch',
    ),
    (
        'tests.test_validation_parity',
        'test_parity_compare_uses_relative_tolerance',
    ),
    (
        'tests.test_validation_parity',
        'test_summary_aggregates_repeats_with_median_fields',
    ),
    (
        'tests.test_validation_parity',
        'test_parity_summary_aggregates_repeat_results',
    ),
    (
        'tests.test_validation_parity',
        'test_native_option_builder_applies_cli_overrides',
    ),
    (
        'tests.test_validation_parity',
        'test_parity_compare_honors_max_steps_window',
    ),
    (
        'tests.test_validation_parity',
        'test_steady_parity_summary_uses_configured_window',
    ),
    (
        'tests.test_validation_parity',
        'test_parity_compare_full_trace_detects_late_step_mismatch',
    ),
    (
        'tests.test_validation_parity',
        'test_steady_parity_summary_strict_mode_catches_late_mismatch',
    ),
    (
        'tests.test_validation_parity',
        'test_parity_summary_detects_repeat_count_mismatch',
    ),
    (
        'tests.test_validation_parity',
        'test_effective_steady_parity_steps_strict_forces_full_trace',
    ),
    (
        'tests.test_validation_parity',
        'test_parity_compare_reports_trace_length_and_missing_keys',
    ),
    (
        'tests.test_validation_parity',
        'test_parity_compare_run_failed_includes_error_detail',
    ),
    (
        'tests.test_validation_parity',
        'test_parity_compare_records_first_mismatch_detail_values',
    ),
    (
        'tests.test_validation_parity',
        'test_native_option_builder_can_enable_capture_debug_probe',
    ),
    (
        'tests.test_validation_parity',
        'test_parity_compare_reports_action_divergence_step_counts',
    ),
    (
        'tests.test_validation_parity',
        'test_parity_compare_aggregates_step_counts_for_multiple_mismatch_steps',
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
