from enum import Enum


class CandidateStatus(Enum):
    """Represent the validation status of a generated candidate."""
    VALID_COMPLETE = "valid_complete"
    VALID_PREFIX = "valid_prefix"
    INVALID = "invalid"


def is_valid_prefix(candidate: str, target: str) -> bool:
    """
    Check whether a candidate is a valid prefix of a target string.

    Args:
        candidate: Generated text to validate.
        target: Expected target string.

    Returns:
        True if the target starts with the candidate, otherwise False.
    """
    return target.startswith(candidate)


def check_candidates(
        candidate: str, allowed_names: list[str]
        ) -> CandidateStatus:
    """
    Validate a candidate against the allowed function names.

    Args:
        candidate: Generated function name or partial name.
        allowed_names: List of valid function names.

    Returns:
        The validation status of the candidate.
    """
    has_valid_prefix = False
    for function_name in allowed_names:
        if candidate == function_name:
            return CandidateStatus.VALID_COMPLETE
        if function_name.startswith(candidate):
            has_valid_prefix = True
    if has_valid_prefix:
        return CandidateStatus.VALID_PREFIX
    return CandidateStatus.INVALID
