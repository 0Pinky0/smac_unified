from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

ParityBlock = Literal[
    'action',
    'observation_state',
    'reward_termination',
    'reset_init',
    'api_contract',
]


@dataclass(frozen=True)
class ParityEntry:
    block: ParityBlock
    legacy_symbol: str
    unified_symbol: str
    rationale: str


CORE_PARITY_MATRIX: tuple[ParityEntry, ...] = (
    ParityEntry(
        block='action',
        legacy_symbol='smac.env.starcraft2.StarCraft2Env.get_avail_agent_actions',
        unified_symbol='smac_unified.handlers.action.ActionHandler.get_avail_agent_actions',
        rationale='Per-agent legal action mask semantics and targetability gates.',
    ),
    ParityEntry(
        block='action',
        legacy_symbol='smac.env.starcraft2.StarCraft2Env.get_agent_action',
        unified_symbol='smac_unified.handlers.action.ActionHandler.build_agent_action',
        rationale='Discrete action id to SC2 raw command encoding.',
    ),
    ParityEntry(
        block='observation_state',
        legacy_symbol='smac.env.starcraft2.StarCraft2Env.get_obs_agent',
        unified_symbol='smac_unified.handlers.obs.ObservationHandler.build_agent_obs',
        rationale='Agent-centric local feature vector construction.',
    ),
    ParityEntry(
        block='observation_state',
        legacy_symbol='smac.env.starcraft2.StarCraft2Env.get_state',
        unified_symbol='smac_unified.handlers.state.StateHandler.build_state',
        rationale='Global state feature layout and optional action append.',
    ),
    ParityEntry(
        block='reward_termination',
        legacy_symbol='smac.env.starcraft2.StarCraft2Env.reward_battle',
        unified_symbol='smac_unified.handlers.reward.RewardHandler.build_step_reward',
        rationale='Dense reward deltas using health/shield and death trackers.',
    ),
    ParityEntry(
        block='reward_termination',
        legacy_symbol='smac.env.starcraft2.StarCraft2Env.step',
        unified_symbol='smac_unified.core.smac_env.SMACEnv.step',
        rationale='Terminal reward bonuses, timeout rules, and info payload.',
    ),
    ParityEntry(
        block='reset_init',
        legacy_symbol='smac.env.starcraft2.StarCraft2Env.reset',
        unified_symbol='smac_unified.core.smac_env.SMACEnv.reset',
        rationale='Episode lifecycle and unit/counter initialization.',
    ),
    ParityEntry(
        block='reset_init',
        legacy_symbol='smac.env.starcraft2.StarCraft2Env.init_units',
        unified_symbol='smac_unified.core.smac_env.SMACEnv._split_raw_units',
        rationale='Stable ally/enemy slot assignment by unit ordering.',
    ),
    ParityEntry(
        block='api_contract',
        legacy_symbol='smac.env.multiagentenv.MultiAgentEnv.get_env_info',
        unified_symbol='smac_unified.core.smac_env.SMACEnv.get_env_info',
        rationale='Runtime metadata contract used by training stacks.',
    ),
    ParityEntry(
        block='api_contract',
        legacy_symbol='smac.env.multiagentenv.MultiAgentEnv.step/reset',
        unified_symbol='smac_unified.adapters.NormalizedEnvAdapter.step/reset',
        rationale='Normalized batch contract over native and bridge backends.',
    ),
)


def entries_for_block(block: ParityBlock) -> list[ParityEntry]:
    return [entry for entry in CORE_PARITY_MATRIX if entry.block == block]


def blocks() -> Sequence[ParityBlock]:
    return (
        'action',
        'observation_state',
        'reward_termination',
        'reset_init',
        'api_contract',
    )

