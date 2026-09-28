"""Shared names for policy input variants."""

ORDINARY_POLICY = "ordinary"
GOAL_TOKEN_POLICY = "goal_token"
SUPPORTED_POLICY_VARIANTS = (ORDINARY_POLICY, GOAL_TOKEN_POLICY)

TASK_ID_FEATURE = "observation.task_id"
GOAL_COUNT_CLASSES = 6
GOAL_TOKEN_DIM = 512


def normalize_policy_variant(value: str | None) -> str:
    variant = ORDINARY_POLICY if value is None else str(value).strip().lower()
    if variant not in SUPPORTED_POLICY_VARIANTS:
        raise ValueError(
            f"unknown ACT policy variant {value!r}; expected one of "
            f"{', '.join(SUPPORTED_POLICY_VARIANTS)}"
        )
    return variant
