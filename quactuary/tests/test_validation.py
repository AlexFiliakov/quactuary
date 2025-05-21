import pytest

from quactuary.utils.validation import (validate_non_negative_integer,
                                        validate_positive_integer,
                                        validate_probability)


def test_validate_probability():
    # Test valid probabilities
    validate_probability(0.0)
    validate_probability(0.5)
    validate_probability(1.0)
    validate_probability(1)

    # Test invalid probabilities
    with pytest.raises(ValueError):
        validate_probability(-0.1)
    with pytest.raises(ValueError):
        validate_probability(1.1)
    with pytest.raises(ValueError):
        validate_probability("string")  # type: ignore


def test_validate_positive_integer():
    # Test valid positive integers
    validate_positive_integer(1)
    validate_positive_integer(100)

    # Test invalid positive integers
    with pytest.raises(ValueError):
        validate_positive_integer(0)
    with pytest.raises(ValueError):
        validate_positive_integer(-1)
    with pytest.raises(ValueError):
        validate_positive_integer(1.0)  # type: ignore
    with pytest.raises(ValueError):
        validate_positive_integer(1.5)  # type: ignore
    with pytest.raises(ValueError):
        validate_positive_integer("string")  # type: ignore


def test_validate_non_negative_integer():
    # Test valid non-negative integers
    validate_non_negative_integer(0)
    validate_non_negative_integer(1)
    validate_non_negative_integer(100)

    # Test invalid non-negative integers
    with pytest.raises(ValueError):
        validate_non_negative_integer(-1)
    with pytest.raises(ValueError):
        validate_non_negative_integer(1.5)  # type: ignore
    with pytest.raises(ValueError):
        validate_non_negative_integer("string")  # type: ignore
