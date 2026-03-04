from smac_unified.config import default_switches, merge_switches
from smac_unified.handlers import (
    AbilityAugmentedActionHandler,
    CapabilityObservationHandler,
    CapabilityStateHandler,
    ClampPositiveRewardHandler,
    ClassicActionHandler,
    ConicFovActionHandler,
    DefaultObservationHandler,
    DefaultStateHandler,
    build_default_handler_bundle,
)
from smac_unified.maps import get_map_params


def test_factory_selects_classic_bundle_for_smac():
    bundle = build_default_handler_bundle(
        switches=default_switches('smac'),
        map_params=get_map_params('3m'),
        env_kwargs={},
    )
    assert isinstance(bundle.action_handler, ClassicActionHandler)
    assert isinstance(bundle.observation_handler, DefaultObservationHandler)
    assert isinstance(bundle.state_handler, DefaultStateHandler)


def test_factory_selects_classic_bundle_for_default_smacv2():
    bundle = build_default_handler_bundle(
        switches=default_switches('smacv2'),
        map_params=get_map_params('8m'),
        env_kwargs={'num_fov_actions': 16, 'action_mask': True},
    )
    assert isinstance(bundle.action_handler, ClassicActionHandler)
    assert isinstance(bundle.observation_handler, DefaultObservationHandler)
    assert isinstance(bundle.state_handler, DefaultStateHandler)
    assert isinstance(bundle.reward_handler, ClampPositiveRewardHandler)


def test_factory_selects_conic_and_capability_bundle_when_overridden():
    switches = merge_switches(
        'smacv2',
        {
            'action_mode': 'conic_fov',
            'capability_mode': 'team_gen',
            'team_init_mode': 'episode_generated',
        },
    )
    bundle = build_default_handler_bundle(
        switches=switches,
        map_params=get_map_params('8m'),
        env_kwargs={'num_fov_actions': 16, 'action_mask': True},
    )
    assert isinstance(bundle.action_handler, ConicFovActionHandler)
    assert bundle.action_handler.num_fov_actions == 16
    assert isinstance(bundle.observation_handler, CapabilityObservationHandler)
    assert isinstance(bundle.state_handler, CapabilityStateHandler)
    assert isinstance(bundle.reward_handler, ClampPositiveRewardHandler)


def test_factory_selects_ability_augmented_bundle_for_smac_hard():
    bundle = build_default_handler_bundle(
        switches=default_switches('smac-hard'),
        map_params=get_map_params('3m'),
        env_kwargs={'use_ability': True},
    )
    assert isinstance(bundle.action_handler, AbilityAugmentedActionHandler)
    assert isinstance(bundle.reward_handler, ClampPositiveRewardHandler)

