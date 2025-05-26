"""Numerical stability utilities for actuarial computations.

This module provides utilities for numerically stable calculations, particularly
for compound distributions and other actuarial computations that involve extreme
values. It includes log-space operations, overflow/underflow detection, and
stable implementations of common actuarial calculations.

Key Features:
    - Log-space arithmetic operations (logsumexp, logaddexp, etc.)
    - Overflow/underflow detection and handling
    - Stable probability and moment calculations
    - Numerical range validation utilities

Example:
    >>> import numpy as np
    >>> from quactuary.utils.numerical import logsumexp, stable_exp
    >>> 
    >>> # Stable computation of log(sum(exp(x)))
    >>> x = np.array([1000, 1001, 1002])
    >>> result = logsumexp(x)  # Avoids overflow
    >>> 
    >>> # Stable exponential with overflow protection
    >>> y = stable_exp(700)  # Returns large but finite value
"""

import numpy as np
from typing import Union, Optional, Tuple, Any
import warnings


# Maximum safe exponent for np.exp to avoid overflow
MAX_EXP_ARG = 700.0
# Minimum safe exponent for np.exp to avoid underflow
MIN_EXP_ARG = -700.0
# Machine epsilon for float64
EPSILON = np.finfo(np.float64).eps
# Maximum representable float64
MAX_FLOAT = np.finfo(np.float64).max
# Minimum positive normal float64
MIN_FLOAT = np.finfo(np.float64).tiny


def logsumexp(
    a: np.ndarray,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
    b: Optional[np.ndarray] = None
) -> Union[float, np.ndarray]:
    """Compute log(sum(exp(a))) in a numerically stable way.
    
    This function computes the log of the sum of exponentials of input elements,
    which is useful for working in log-space to avoid numerical overflow/underflow.
    
    Args:
        a: Input array.
        axis: Axis or axes over which the sum is taken. By default, sum over all axes.
        keepdims: If True, the reduced axes are left in the result as dimensions with size one.
        b: Scaling factors for exp(a). Must be broadcastable to the shape of a.
        
    Returns:
        Result of log(sum(b * exp(a))). If b is not provided, returns log(sum(exp(a))).
        
    Example:
        >>> x = np.array([1000, 1001, 1002])
        >>> logsumexp(x)  # Stable computation avoiding overflow
        1002.4076059644444
    """
    a = np.asarray(a)
    
    # Handle empty array case
    if a.size == 0:
        return -np.inf
    
    if axis is None:
        a_max = np.max(a)
    else:
        a_max = np.max(a, axis=axis, keepdims=True)
    
    if b is not None:
        b = np.asarray(b)
        # Compute log(sum(b * exp(a - a_max))) + a_max
        with np.errstate(divide='ignore'):
            sumexp = np.sum(b * np.exp(a - a_max), axis=axis, keepdims=keepdims)
            out = np.log(sumexp)
    else:
        # Compute log(sum(exp(a - a_max))) + a_max
        sumexp = np.sum(np.exp(a - a_max), axis=axis, keepdims=keepdims)
        out = np.log(sumexp)
    
    if not keepdims:
        a_max = np.squeeze(a_max, axis=axis) if axis is not None else a_max
    
    # Handle edge cases
    mask = ~np.isfinite(a_max)
    if np.any(mask):
        out = np.where(mask, a_max, out + a_max)
    else:
        out = out + a_max
    
    return out


def logaddexp(x1: Union[float, np.ndarray], x2: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Compute log(exp(x1) + exp(x2)) element-wise in a numerically stable way.
    
    Args:
        x1: First input array or scalar.
        x2: Second input array or scalar.
        
    Returns:
        Element-wise log(exp(x1) + exp(x2)).
        
    Example:
        >>> logaddexp(1000, 1001)
        1001.3132616875183
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    
    # Use the identity: log(exp(x1) + exp(x2)) = max(x1, x2) + log(1 + exp(-|x1 - x2|))
    max_val = np.maximum(x1, x2)
    min_val = np.minimum(x1, x2)
    
    # Avoid computing exp of large negative numbers
    diff = min_val - max_val
    mask = diff > -37  # exp(-37) is negligible compared to 1
    
    result = max_val.copy() if isinstance(max_val, np.ndarray) else max_val
    if np.any(mask):
        if isinstance(result, np.ndarray):
            np.add(result, np.log1p(np.exp(diff)), where=mask, out=result)
        else:
            # For scalar case
            if mask:
                result = result + np.log1p(np.exp(diff))
    
    return result


def stable_exp(x: Union[float, np.ndarray], clip: bool = True) -> Union[float, np.ndarray]:
    """Compute exp(x) with overflow protection.
    
    Args:
        x: Input value(s).
        clip: If True, clip input to avoid overflow. If False, return inf for large values.
        
    Returns:
        exp(x) with overflow handling.
        
    Example:
        >>> stable_exp(800)  # Would overflow with regular exp
        1.7976931348623157e+308
    """
    x = np.asarray(x)
    
    # Handle empty array case
    if x.size == 0:
        return x  # Return empty array
    
    if clip:
        # Clip to safe range
        x_clipped = np.clip(x, MIN_EXP_ARG, MAX_EXP_ARG)
        if not np.array_equal(x, x_clipped):
            warnings.warn("Input to stable_exp was clipped to avoid overflow/underflow")
        return np.exp(x_clipped)
    else:
        # Let overflow happen naturally
        return np.exp(x)


def stable_log(x: Union[float, np.ndarray], min_value: float = MIN_FLOAT) -> Union[float, np.ndarray]:
    """Compute log(x) with underflow protection.
    
    Args:
        x: Input value(s). Must be positive.
        min_value: Minimum value to use before taking log. Defaults to machine epsilon.
        
    Returns:
        log(max(x, min_value)) to avoid log(0) or log(negative).
        
    Example:
        >>> stable_log(0)  # Would be -inf with regular log
        -708.3964185322641
    """
    x = np.asarray(x)
    
    # Handle empty array case
    if x.size == 0:
        return x  # Return empty array
    
    # Ensure positive values
    if np.any(x < 0):
        raise ValueError("stable_log requires non-negative input")
    
    # Clip to minimum value
    x_clipped = np.maximum(x, min_value)
    if not np.array_equal(x, x_clipped):
        warnings.warn(f"Input to stable_log was clipped to minimum value {min_value}")
    
    return np.log(x_clipped)


def check_finite(
    x: Union[float, np.ndarray],
    name: str = "array",
    allow_nan: bool = False,
    allow_inf: bool = False
) -> Union[float, np.ndarray]:
    """Check that array contains only finite values.
    
    Args:
        x: Input array to check.
        name: Name of the array for error messages.
        allow_nan: If True, allow NaN values.
        allow_inf: If True, allow infinite values.
        
    Returns:
        The input array if all checks pass.
        
    Raises:
        ValueError: If array contains non-finite values that are not allowed.
        
    Example:
        >>> x = np.array([1, 2, 3])
        >>> check_finite(x, "probabilities")
        array([1, 2, 3])
    """
    x = np.asarray(x)
    
    if not allow_nan and np.any(np.isnan(x)):
        raise ValueError(f"{name} contains NaN values")
    
    if not allow_inf and np.any(np.isinf(x)):
        raise ValueError(f"{name} contains infinite values")
    
    if not allow_nan and not allow_inf and not np.all(np.isfinite(x)):
        raise ValueError(f"{name} contains non-finite values")
    
    return x


def clip_to_valid_range(
    x: Union[float, np.ndarray],
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    name: str = "array"
) -> Union[float, np.ndarray]:
    """Clip values to a valid range with informative warnings.
    
    Args:
        x: Input array to clip.
        min_val: Minimum valid value. If None, no lower bound.
        max_val: Maximum valid value. If None, no upper bound.
        name: Name of the array for warning messages.
        
    Returns:
        Clipped array.
        
    Example:
        >>> x = np.array([-0.1, 0.5, 1.2])
        >>> clip_to_valid_range(x, 0, 1, "probabilities")
        array([0. , 0.5, 1. ])
    """
    x = np.asarray(x)
    original_x = x.copy()
    
    if min_val is not None and max_val is not None:
        x = np.clip(x, min_val, max_val)
        out_of_range = np.any(original_x < min_val) or np.any(original_x > max_val)
        if out_of_range:
            warnings.warn(f"{name} contains values outside [{min_val}, {max_val}], clipping to valid range")
    elif min_val is not None:
        x = np.maximum(x, min_val)
        if np.any(original_x < min_val):
            warnings.warn(f"{name} contains values below {min_val}, clipping to valid range")
    elif max_val is not None:
        x = np.minimum(x, max_val)
        if np.any(original_x > max_val):
            warnings.warn(f"{name} contains values above {max_val}, clipping to valid range")
    
    return x


def detect_numerical_issues(
    x: Union[float, np.ndarray],
    name: str = "array",
    check_underflow: bool = True,
    check_overflow: bool = True
) -> Optional[str]:
    """Detect potential numerical issues in array.
    
    Args:
        x: Input array to check.
        name: Name of the array for diagnostic messages.
        check_underflow: Check for values close to underflow.
        check_overflow: Check for values close to overflow.
        
    Returns:
        Diagnostic message if issues found, None otherwise.
        
    Example:
        >>> x = np.array([1e-300, 1e300])
        >>> detect_numerical_issues(x, "values")
        'values: potential underflow risk (1 values < 1e-290), potential overflow risk (1 values > 1e290)'
    """
    x = np.asarray(x)
    issues = []
    
    # Check for NaN/inf
    n_nan = np.sum(np.isnan(x))
    n_inf = np.sum(np.isinf(x))
    if n_nan > 0:
        issues.append(f"{n_nan} NaN values")
    if n_inf > 0:
        issues.append(f"{n_inf} infinite values")
    
    # Check for underflow risk
    if check_underflow:
        underflow_threshold = 1e-290
        n_underflow = np.sum(np.abs(x) < underflow_threshold)
        if n_underflow > 0:
            issues.append(f"potential underflow risk ({n_underflow} values < {underflow_threshold})")
    
    # Check for overflow risk
    if check_overflow:
        overflow_threshold = 1e290
        n_overflow = np.sum(np.abs(x) > overflow_threshold)
        if n_overflow > 0:
            issues.append(f"potential overflow risk ({n_overflow} values > {overflow_threshold})")
    
    if issues:
        return f"{name}: " + ", ".join(issues)
    return None


def stable_probability_calculation(
    log_probs: np.ndarray,
    normalize: bool = True
) -> np.ndarray:
    """Convert log probabilities to probabilities with numerical stability.
    
    Args:
        log_probs: Array of log probabilities.
        normalize: If True, normalize to ensure probabilities sum to 1.
        
    Returns:
        Array of probabilities.
        
    Example:
        >>> log_probs = np.array([-1000, -1001, -1002])
        >>> stable_probability_calculation(log_probs)
        array([0.57611688, 0.31122452, 0.1126586 ])
    """
    log_probs = np.asarray(log_probs)
    
    if normalize:
        # Normalize in log space
        log_sum = logsumexp(log_probs)
        log_probs = log_probs - log_sum
    
    # Convert to probabilities
    probs = stable_exp(log_probs)
    
    # Ensure valid probability range
    probs = clip_to_valid_range(probs, 0.0, 1.0, "probabilities")
    
    # Renormalize if needed due to clipping
    if normalize and not np.isclose(np.sum(probs), 1.0):
        probs = probs / np.sum(probs)
    
    return probs


def stable_moment_calculation(
    values: np.ndarray,
    probabilities: np.ndarray,
    moment: int = 1,
    central: bool = False,
    log_space: bool = False
) -> float:
    """Calculate moments in a numerically stable way.
    
    Args:
        values: Array of values.
        probabilities: Array of probabilities (must sum to 1).
        moment: Order of moment to calculate.
        central: If True, calculate central moment.
        log_space: If True, perform calculations in log space when possible.
        
    Returns:
        The calculated moment.
        
    Example:
        >>> values = np.array([1, 2, 3])
        >>> probs = np.array([0.2, 0.5, 0.3])
        >>> stable_moment_calculation(values, probs, moment=1)
        2.1
    """
    values = np.asarray(values)
    probabilities = np.asarray(probabilities)
    
    # Validate inputs
    check_finite(values, "values")
    check_finite(probabilities, "probabilities")
    probabilities = clip_to_valid_range(probabilities, 0.0, 1.0, "probabilities")
    
    # Normalize probabilities
    prob_sum = np.sum(probabilities)
    if not np.isclose(prob_sum, 1.0):
        warnings.warn(f"Probabilities sum to {prob_sum}, normalizing")
        probabilities = probabilities / prob_sum
    
    if central and moment > 0:
        # Calculate mean first
        mean = stable_moment_calculation(values, probabilities, moment=1, central=False)
        values = values - mean
    
    if moment == 0:
        return 1.0
    
    if log_space and moment > 1 and np.all(values > 0):
        # Work in log space for positive values
        log_values = stable_log(values)
        log_probs = stable_log(probabilities)
        log_moment_terms = moment * log_values + log_probs
        return np.exp(logsumexp(log_moment_terms))
    else:
        # Standard calculation
        moment_values = np.power(values, moment)
        
        # Check for overflow
        if detect_numerical_issues(moment_values, "moment values"):
            warnings.warn("Potential numerical issues in moment calculation")
        
        return np.sum(moment_values * probabilities)


def stable_quantile_interpolation(
    x: np.ndarray,
    cdf: np.ndarray,
    q: Union[float, np.ndarray],
    extrapolate: bool = False
) -> Union[float, np.ndarray]:
    """Perform stable quantile interpolation.
    
    Args:
        x: Sorted array of x values.
        cdf: Corresponding CDF values (must be monotonic).
        q: Quantile(s) to compute (between 0 and 1).
        extrapolate: If True, extrapolate beyond data range.
        
    Returns:
        Interpolated quantile value(s).
        
    Example:
        >>> x = np.array([0, 1, 2, 3, 4])
        >>> cdf = np.array([0, 0.2, 0.5, 0.8, 1.0])
        >>> stable_quantile_interpolation(x, cdf, 0.9)
        3.5
    """
    x = np.asarray(x)
    cdf = np.asarray(cdf)
    q = np.asarray(q)
    
    # Validate inputs
    check_finite(x, "x values")
    check_finite(cdf, "CDF values")
    check_finite(q, "quantiles")
    
    # Ensure CDF is valid
    cdf = clip_to_valid_range(cdf, 0.0, 1.0, "CDF values")
    q = clip_to_valid_range(q, 0.0, 1.0, "quantiles")
    
    # Check monotonicity
    if not np.all(np.diff(cdf) >= 0):
        raise ValueError("CDF values must be non-decreasing")
    
    # Handle edge cases
    if len(x) == 0:
        return np.full_like(q, np.nan)
    
    if len(x) == 1:
        return np.full_like(q, x[0])
    
    # Interpolate
    if extrapolate:
        # Linear extrapolation
        result = np.interp(q, cdf, x)
    else:
        # Clip to data range
        result = np.interp(q, cdf, x)
        result = np.clip(result, x[0], x[-1])
    
    return result if q.shape else float(result)