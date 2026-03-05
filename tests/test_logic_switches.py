from smac_unified.config import default_switches, merge_switches


def test_default_switches_have_expected_modes():
    smac = default_switches('smac')
    smacv2 = default_switches('smacv2')
    hard = default_switches('smac-hard')

    assert smac.opponent_mode == 'sc2_computer'
    assert smacv2.action_mode == 'classic'
    assert smacv2.capability_mode == 'none'
    assert smacv2.reward_positive_mode == 'clamp_zero'
    assert hard.action_mode == 'ability_augmented'
    assert hard.opponent_mode == 'scripted_pool'


def test_switch_overrides_apply_cleanly():
    switches = merge_switches(
        'smac',
        {
            'action_mode': 'conic_fov',
            'opponent_mode': 'scripted_pool',
        },
    )
    assert switches.action_mode == 'conic_fov'
    assert switches.opponent_mode == 'scripted_pool'
    assert switches.variant == 'smac'


def test_variant_authority_locks_smac_hard_core_modes():
    switches = merge_switches(
        'smac-hard',
        {
            'action_mode': 'classic',
            'opponent_mode': 'sc2_computer',
            'reward_positive_mode': 'abs',
        },
    )
    assert switches.action_mode == 'ability_augmented'
    assert switches.opponent_mode == 'scripted_pool'
    assert switches.reward_positive_mode == 'clamp_zero'


def test_variant_authority_locks_smacv2_reward_mode():
    switches = merge_switches(
        'smacv2',
        {
            'reward_positive_mode': 'abs',
        },
    )
    assert switches.reward_positive_mode == 'clamp_zero'
