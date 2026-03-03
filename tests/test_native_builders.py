from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from smac_unified.handlers import (
    BuilderContext,
    DefaultNativeActionBuilder,
    DefaultNativeObservationBuilder,
    DefaultNativeRewardBuilder,
    DefaultNativeStateBuilder,
    TrackedUnit,
    UnitFrame,
    UnitTeamFrame,
)


class _VariantLogic:
    def shoot_range_by_type(self, unit_ids):
        return {
            getattr(unit_ids, "marine_id", 0): 6.0,
            getattr(unit_ids, "medivac_id", 0): 4.0,
            2: 6.0,
            3: 6.0,
        }

    @staticmethod
    def reward_positive_transform(value: float) -> float:
        return abs(value)


class _AvailEnv:
    def __init__(self, masks: dict[int, list[int]]):
        self._masks = masks

    def get_avail_agent_actions(self, agent_id: int) -> list[int]:
        return list(self._masks[agent_id])


def _tracked(
    *,
    unit_id: int,
    tag: int,
    unit_type: int,
    x: float,
    y: float,
    health: float,
    health_max: float = 45.0,
    shield: float = 0.0,
    shield_max: float = 0.0,
    alive: bool = True,
):
    return TrackedUnit(
        unit_id=unit_id,
        tag=tag,
        unit_type=unit_type,
        x=x,
        y=y,
        health=health,
        health_max=health_max,
        shield=shield,
        shield_max=shield_max,
        weapon_cooldown=0.0,
        alive=alive,
        owner=1,
        raw=None,
    )


def _team_frame(units):
    return UnitTeamFrame(
        units=tuple(units),
        health=np.asarray([u.health for u in units], dtype=np.float32),
        shield=np.asarray([u.shield for u in units], dtype=np.float32),
        alive=np.asarray([u.alive for u in units], dtype=bool),
        tags=np.asarray([u.tag for u in units], dtype=np.int64),
    )


def _context(
    *,
    env,
    n_agents: int,
    n_enemies: int,
    n_actions: int,
    n_actions_no_attack: int,
    attack_slots: int,
    reward_scale: bool = False,
    max_reward: float = 0.0,
    reward_scale_rate: float = 20.0,
):
    return BuilderContext(
        family="smac",
        map_name="3m",
        episode_step=1,
        n_agents=n_agents,
        n_enemies=n_enemies,
        n_actions=n_actions,
        n_actions_no_attack=n_actions_no_attack,
        attack_slots=attack_slots,
        move_amount=2.0,
        map_x=32.0,
        map_y=32.0,
        max_distance_x=32.0,
        max_distance_y=32.0,
        state_last_action=True,
        last_action=np.zeros((n_agents, n_actions), dtype=np.float32),
        reward_sparse=False,
        reward_only_positive=False,
        reward_death_value=10.0,
        reward_negative_scale=0.5,
        reward_scale=reward_scale,
        reward_scale_rate=reward_scale_rate,
        max_reward=max_reward,
        variant_logic=_VariantLogic(),
        unit_type_ids=SimpleNamespace(medivac_id=99, marine_id=2),
        switches=SimpleNamespace(opponent_mode="sc2_computer"),
        env=env,
    )


def test_native_action_builder_avail_masks_are_deterministic():
    frame = UnitFrame(
        allies=_team_frame(
            [_tracked(unit_id=0, tag=1, unit_type=2, x=4, y=4, health=45)]
        ),
        enemies=_team_frame(
            [_tracked(unit_id=0, tag=101, unit_type=3, x=8, y=4, health=45)]
        ),
        prev_allies_health=np.asarray([45.0], dtype=np.float32),
        prev_allies_shield=np.asarray([0.0], dtype=np.float32),
        prev_enemies_health=np.asarray([45.0], dtype=np.float32),
        prev_enemies_shield=np.asarray([0.0], dtype=np.float32),
        step_token=3,
    )
    context = _context(
        env=_AvailEnv({0: [1, 1, 1, 1, 1, 1, 1]}),
        n_agents=1,
        n_enemies=1,
        n_actions=7,
        n_actions_no_attack=6,
        attack_slots=1,
    )
    builder = DefaultNativeActionBuilder()
    mask_a = builder.get_avail_agent_actions(frame=frame, context=context, agent_id=0)
    mask_b = builder.get_avail_agent_actions(frame=frame, context=context, agent_id=0)

    assert mask_a == mask_b
    assert mask_a[1] == 1
    assert mask_a[6] == 1


def test_native_reward_builder_uses_frame_deltas_and_scaling():
    frame = UnitFrame(
        allies=_team_frame(
            [_tracked(unit_id=0, tag=1, unit_type=2, x=4, y=4, health=40)]
        ),
        enemies=_team_frame(
            [_tracked(unit_id=0, tag=101, unit_type=3, x=8, y=4, health=15)]
        ),
        prev_allies_health=np.asarray([40.0], dtype=np.float32),
        prev_allies_shield=np.asarray([0.0], dtype=np.float32),
        prev_enemies_health=np.asarray([20.0], dtype=np.float32),
        prev_enemies_shield=np.asarray([0.0], dtype=np.float32),
        step_token=4,
    )
    unscaled_ctx = _context(
        env=None,
        n_agents=1,
        n_enemies=1,
        n_actions=7,
        n_actions_no_attack=6,
        attack_slots=1,
        reward_scale=False,
    )
    scaled_ctx = _context(
        env=None,
        n_agents=1,
        n_enemies=1,
        n_actions=7,
        n_actions_no_attack=6,
        attack_slots=1,
        reward_scale=True,
        max_reward=100.0,
        reward_scale_rate=20.0,
    )

    reward_builder = DefaultNativeRewardBuilder()
    reward_builder.reset(frame=frame, context=unscaled_ctx)
    unscaled = reward_builder.build_step_reward(frame=frame, context=unscaled_ctx)
    reward_builder.reset(frame=frame, context=scaled_ctx)
    scaled = reward_builder.build_step_reward(frame=frame, context=scaled_ctx)

    assert np.isclose(unscaled, 5.0)
    assert np.isclose(scaled, 1.0)


def test_native_observation_and_state_builders_follow_unified_contract():
    frame = UnitFrame(
        allies=_team_frame(
            [
                _tracked(unit_id=0, tag=1, unit_type=2, x=4, y=4, health=45),
                _tracked(unit_id=1, tag=2, unit_type=2, x=6, y=4, health=45),
            ]
        ),
        enemies=_team_frame(
            [_tracked(unit_id=0, tag=101, unit_type=3, x=8, y=4, health=40)]
        ),
        prev_allies_health=np.asarray([45.0, 45.0], dtype=np.float32),
        prev_allies_shield=np.asarray([0.0, 0.0], dtype=np.float32),
        prev_enemies_health=np.asarray([40.0], dtype=np.float32),
        prev_enemies_shield=np.asarray([0.0], dtype=np.float32),
        step_token=2,
    )
    env = _AvailEnv(
        {
            0: [1, 1, 1, 1, 1, 1, 1, 0],
            1: [1, 1, 1, 1, 1, 1, 1, 0],
        }
    )
    context = _context(
        env=env,
        n_agents=2,
        n_enemies=1,
        n_actions=8,
        n_actions_no_attack=6,
        attack_slots=2,
    )

    obs_builder = DefaultNativeObservationBuilder()
    state_builder = DefaultNativeStateBuilder()
    action_builder = DefaultNativeActionBuilder()

    obs = obs_builder.build_obs(frame=frame, context=context)
    state = state_builder.build_state(frame=frame, context=context)
    stop_cmd = action_builder.build_agent_action(
        frame=frame,
        context=context,
        agent_id=0,
        action=1,
    )

    assert len(obs) == 2
    assert obs[0].dtype == np.float32
    assert state.dtype == np.float32
    assert stop_cmd is not None
