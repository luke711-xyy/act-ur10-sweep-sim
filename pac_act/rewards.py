"""Shared, already-tested contact rule; no residual controller is imported."""
from residual_rl.rewards import ContactReward,classify_failure

__all__=['ContactReward','classify_failure']
