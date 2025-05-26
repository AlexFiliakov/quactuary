"""
Utility functions and helpers for the quactuary package.

This module provides common utilities, validation functions, and helper classes
used throughout the quactuary framework. It includes tools for parameter validation,
data transformation, numerical utilities, and other supporting functionality.

Key Components:
    - validation: Parameter and input validation utilities
    - numerical: Numerical computation helpers and stability functions
    - data: Data transformation and formatting utilities
    - decorators: Function decorators for caching, timing, etc.

Examples:
    Using validation utilities:
        >>> from quactuary.utils.validation import validate_positive
        >>> validate_positive(alpha=2.5, beta=1.0)  # Passes
        >>> validate_positive(lambda_=-1.0)  # Raises ValueError
        
    Using numerical utilities:
        >>> from quactuary.utils.numerical import logsumexp
        >>> log_probs = [-1000, -1001, -999]
        >>> result = logsumexp(log_probs)

Notes:
    - Utilities are designed to be reusable across all modules
    - Focus on numerical stability and parameter safety
    - See individual submodules for detailed documentation
"""

# Import commonly used utilities for convenience
from .validation import (
    validate_positive,
    validate_probability,
    validate_integer,
    validate_array_like,
)

from .numerical import (
    logsumexp,
    logaddexp,
    stable_exp,
    stable_log,
    check_finite,
    clip_to_valid_range,
    detect_numerical_issues,
    stable_probability_calculation,
    stable_moment_calculation,
    stable_quantile_interpolation,
)

__all__ = [
    # Validation utilities
    'validate_positive',
    'validate_probability', 
    'validate_integer',
    'validate_array_like',
    # Numerical stability utilities
    'logsumexp',
    'logaddexp',
    'stable_exp',
    'stable_log',
    'check_finite',
    'clip_to_valid_range',
    'detect_numerical_issues',
    'stable_probability_calculation',
    'stable_moment_calculation',
    'stable_quantile_interpolation',
]