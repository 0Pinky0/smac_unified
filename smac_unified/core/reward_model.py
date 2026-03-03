from __future__ import annotations


def scale_reward(raw_reward: float, max_reward: float, scale_rate: float) -> float:
    if max_reward <= 0 or scale_rate <= 0:
        return float(raw_reward)
    return float(raw_reward) / (max_reward / scale_rate)
