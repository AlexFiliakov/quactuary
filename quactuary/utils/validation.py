"""
Parameter validation utilities for the quactuary package.

This module provides a comprehensive set of validation functions to ensure that
inputs to actuarial models meet required constraints. These validators help catch
errors early and provide clear, informative error messages.

The validation functions follow a consistent pattern:
- They validate a specific type or constraint
- They raise ValueError with descriptive messages on failure
- They return None on success (following Python conventions)

Categories of validators:
    - Numeric validators: positive, non-negative, ranges
    - Probability validators: values in [0,1]
    - Integer validators: positive, non-negative integers
    - Array validators: shape, content, numerical properties
    - Distribution parameter validators: specific constraints

Examples:
    Basic parameter validation:
        >>> from quactuary.utils.validation import validate_positive
        >>> validate_positive(alpha=2.5, beta=1.0)  # Success, returns None
        >>> validate_positive(rate=-0.5)  # Raises ValueError
        
    Validating multiple parameters:
        >>> def __init__(self, lambda_, alpha, p):
        ...     validate_positive(lambda_=lambda_, alpha=alpha)
        ...     validate_probability(p)
        ...     # Parameters are now guaranteed valid
        
    Array validation:
        >>> import numpy as np
        >>> data = np.array([1.2, 3.4, 2.1])
        >>> validate_array_like(data, min_length=2)
        >>> validate_all_positive(data)

Notes:
    - Validators can accept keyword arguments for better error messages
    - Use these validators at the start of __init__ methods and functions
    - Consistent validation improves user experience and debugging
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


def validate_positive(**kwargs) -> None:
    """
    Validate that all provided values are positive (> 0).
    
    This function accepts keyword arguments to provide meaningful parameter names
    in error messages. It's particularly useful for validating distribution parameters.
    
    Args:
        **kwargs: Keyword arguments where keys are parameter names and values are
            the numbers to validate. All values must be positive.
    
    Raises:
        ValueError: If any value is not positive, with the parameter name in the message.
        
    Examples:
        >>> validate_positive(alpha=2.5, beta=1.0)  # Success
        >>> validate_positive(lambda_=5.0, scale=0.1)  # Success
        >>> validate_positive(rate=-0.5)  # ValueError: Parameter 'rate' must be positive, got -0.5
        >>> validate_positive(a=1, b=2, c=0)  # ValueError: Parameter 'c' must be positive, got 0
    """
    for name, value in kwargs.items():
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"Parameter '{name}' must be positive, got {value}")


def validate_integer(**kwargs) -> None:
    """
    Validate that all provided values are integers.
    
    Args:
        **kwargs: Keyword arguments where keys are parameter names and values are
            the values to validate as integers.
    
    Raises:
        ValueError: If any value is not an integer.
        
    Examples:
        >>> validate_integer(n=10, k=5)  # Success
        >>> validate_integer(count=3.5)  # ValueError: Parameter 'count' must be an integer, got 3.5
    """
    for name, value in kwargs.items():
        if not isinstance(value, int):
            raise ValueError(f"Parameter '{name}' must be an integer, got {value}")


def validate_array_like(data, min_length: int = 1, max_length: int = None) -> None:
    """
    Validate that data is array-like with optional length constraints.
    
    Args:
        data: The data to validate. Should be list, tuple, numpy array, or pandas Series.
        min_length (int): Minimum required length. Default is 1.
        max_length (int, optional): Maximum allowed length. None means no upper limit.
    
    Raises:
        ValueError: If data is not array-like or doesn't meet length requirements.
        TypeError: If data is not a supported array-like type.
        
    Examples:
        >>> import numpy as np
        >>> validate_array_like([1, 2, 3])  # Success
        >>> validate_array_like(np.array([1.0, 2.0]), min_length=2)  # Success
        >>> validate_array_like([1], min_length=2)  # ValueError: Data must have at least 2 elements
        >>> validate_array_like("not array")  # TypeError: Data must be array-like
    """
    # Check if array-like
    try:
        length = len(data)
    except TypeError:
        raise TypeError(f"Data must be array-like (list, tuple, array, or Series), got {type(data)}")
    
    # Check length constraints
    if length < min_length:
        raise ValueError(f"Data must have at least {min_length} elements, got {length}")
    
    if max_length is not None and length > max_length:
        raise ValueError(f"Data must have at most {max_length} elements, got {length}")
