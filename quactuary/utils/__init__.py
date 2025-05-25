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
        >>> from quactuary.utils.numerical import stable_log_sum_exp
        >>> log_probs = [-1000, -1001, -999]
        >>> result = stable_log_sum_exp(log_probs)

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

__all__ = [
    'validate_positive',
    'validate_probability', 
    'validate_integer',
    'validate_array_like',
]