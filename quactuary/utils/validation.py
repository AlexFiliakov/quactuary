"""
Utility functions for validating inputs or parameters used throughout the quActuary project.
"""


def validate_probability(value: float | int) -> None:
    """
    Validate that a probability value is between 0 and 1.

    Args:
        value (float): The probability value to validate.

    Raises:
        ValueError: If the probability is not in the range [0, 1].
    """
    if not (isinstance(value, float) or isinstance(value, int)) or not (0 <= value <= 1):
        raise ValueError(f"Probability must be between 0 and 1, got {value}.")


def validate_positive_integer(value: int) -> None:
    """
    Validate that a value is a positive integer.

    Args:
        value (int): The value to validate.

    Raises:
        ValueError: If the value is not a positive integer.
    """
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"Value must be a positive integer, got {value}.")


def validate_non_negative_integer(value: int) -> None:
    """
    Validate that a value is a non-negative integer.

    Args:
        value (int): The value to validate.

    Raises:
        ValueError: If the value is not a non-negative integer.
    """
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"Value must be a non-negative integer, got {value}.")
