"""Tests for numerical stability utilities."""

import numpy as np
import pytest
import warnings
from quactuary.utils.numerical import (
    logsumexp, logaddexp, stable_exp, stable_log,
    check_finite, clip_to_valid_range, detect_numerical_issues,
    stable_probability_calculation, stable_moment_calculation,
    stable_quantile_interpolation,
    MAX_EXP_ARG, MIN_EXP_ARG, EPSILON, MAX_FLOAT, MIN_FLOAT
)


class TestLogSpaceOperations:
    """Test log-space arithmetic operations."""
    
    def test_logsumexp_basic(self):
        """Test basic logsumexp functionality."""
        # Simple case
        x = np.array([1, 2, 3])
        result = logsumexp(x)
        expected = np.log(np.sum(np.exp(x)))
        assert np.isclose(result, expected)
        
    def test_logsumexp_overflow_protection(self):
        """Test logsumexp handles large values without overflow."""
        x = np.array([1000, 1001, 1002])
        result = logsumexp(x)
        # Should not overflow or return inf
        assert np.isfinite(result)
        assert result < 1003  # Should be close to max value
        
    def test_logsumexp_axis(self):
        """Test logsumexp with axis parameter."""
        x = np.array([[1, 2], [3, 4]])
        
        # Sum over rows
        result_0 = logsumexp(x, axis=0)
        expected_0 = np.array([np.log(np.exp(1) + np.exp(3)), 
                               np.log(np.exp(2) + np.exp(4))])
        assert np.allclose(result_0, expected_0)
        
        # Sum over columns
        result_1 = logsumexp(x, axis=1)
        expected_1 = np.array([np.log(np.exp(1) + np.exp(2)), 
                               np.log(np.exp(3) + np.exp(4))])
        assert np.allclose(result_1, expected_1)
        
    def test_logsumexp_keepdims(self):
        """Test logsumexp with keepdims parameter."""
        x = np.array([[1, 2], [3, 4]])
        result = logsumexp(x, axis=1, keepdims=True)
        assert result.shape == (2, 1)
        
    def test_logsumexp_with_weights(self):
        """Test logsumexp with weight parameter b."""
        x = np.array([1, 2, 3])
        b = np.array([0.5, 1.0, 0.5])
        result = logsumexp(x, b=b)
        expected = np.log(np.sum(b * np.exp(x)))
        assert np.isclose(result, expected)
        
    def test_logaddexp_basic(self):
        """Test basic logaddexp functionality."""
        x1 = 2.0
        x2 = 3.0
        result = logaddexp(x1, x2)
        expected = np.log(np.exp(x1) + np.exp(x2))
        assert np.isclose(result, expected)
        
    def test_logaddexp_overflow_protection(self):
        """Test logaddexp handles large values."""
        x1 = 1000
        x2 = 1001
        result = logaddexp(x1, x2)
        assert np.isfinite(result)
        assert result < 1002
        
    def test_logaddexp_arrays(self):
        """Test logaddexp with arrays."""
        x1 = np.array([1, 2, 3])
        x2 = np.array([2, 3, 4])
        result = logaddexp(x1, x2)
        expected = np.log(np.exp(x1) + np.exp(x2))
        assert np.allclose(result, expected)
        
    def test_logaddexp_edge_cases(self):
        """Test logaddexp edge cases."""
        # Very different magnitudes
        result = logaddexp(0, -100)
        assert np.isclose(result, 0, atol=1e-10)
        
        # Equal values
        result = logaddexp(5, 5)
        assert np.isclose(result, 5 + np.log(2))


class TestStableExponentials:
    """Test stable exponential and logarithm functions."""
    
    def test_stable_exp_basic(self):
        """Test basic stable_exp functionality."""
        x = 10.0
        result = stable_exp(x)
        expected = np.exp(x)
        assert np.isclose(result, expected)
        
    def test_stable_exp_overflow_protection(self):
        """Test stable_exp prevents overflow."""
        with warnings.catch_warnings(record=True) as w:
            result = stable_exp(800)
            assert len(w) == 1
            assert "clipped" in str(w[0].message)
        assert np.isfinite(result)
        assert result == np.exp(MAX_EXP_ARG)
        
    def test_stable_exp_underflow_protection(self):
        """Test stable_exp prevents underflow."""
        with warnings.catch_warnings(record=True) as w:
            result = stable_exp(-800)
            assert len(w) == 1
        assert result == np.exp(MIN_EXP_ARG)
        
    def test_stable_exp_no_clip(self):
        """Test stable_exp without clipping."""
        result = stable_exp(800, clip=False)
        assert np.isinf(result)
        
    def test_stable_exp_array(self):
        """Test stable_exp with arrays."""
        x = np.array([-1000, 0, 1000])
        with warnings.catch_warnings(record=True):
            result = stable_exp(x)
        assert np.all(np.isfinite(result))
        
    def test_stable_log_basic(self):
        """Test basic stable_log functionality."""
        x = 10.0
        result = stable_log(x)
        expected = np.log(x)
        assert np.isclose(result, expected)
        
    def test_stable_log_zero_protection(self):
        """Test stable_log handles zero."""
        with warnings.catch_warnings(record=True) as w:
            result = stable_log(0)
            assert len(w) == 1
            assert "clipped" in str(w[0].message)
        assert np.isfinite(result)
        
    def test_stable_log_negative_error(self):
        """Test stable_log raises error for negative input."""
        with pytest.raises(ValueError, match="non-negative"):
            stable_log(-1)
            
    def test_stable_log_array(self):
        """Test stable_log with arrays."""
        x = np.array([0, 1e-400, 1, 10])
        with warnings.catch_warnings(record=True):
            result = stable_log(x)
        assert np.all(np.isfinite(result))
        
    def test_stable_log_custom_min(self):
        """Test stable_log with custom minimum value."""
        x = 0
        min_val = 1e-10
        with warnings.catch_warnings(record=True):
            result = stable_log(x, min_value=min_val)
        assert result == np.log(min_val)


class TestNumericalValidation:
    """Test numerical validation and checking utilities."""
    
    def test_check_finite_basic(self):
        """Test basic check_finite functionality."""
        x = np.array([1, 2, 3])
        result = check_finite(x)
        assert np.array_equal(result, x)
        
    def test_check_finite_nan_error(self):
        """Test check_finite raises error for NaN."""
        x = np.array([1, np.nan, 3])
        with pytest.raises(ValueError, match="NaN"):
            check_finite(x)
            
    def test_check_finite_inf_error(self):
        """Test check_finite raises error for inf."""
        x = np.array([1, np.inf, 3])
        with pytest.raises(ValueError, match="infinite"):
            check_finite(x)
            
    def test_check_finite_allow_nan(self):
        """Test check_finite with allow_nan=True."""
        x = np.array([1, np.nan, 3])
        result = check_finite(x, allow_nan=True)
        assert np.array_equal(result, x, equal_nan=True)
        
    def test_check_finite_allow_inf(self):
        """Test check_finite with allow_inf=True."""
        x = np.array([1, np.inf, 3])
        result = check_finite(x, allow_inf=True)
        assert np.array_equal(result, x)
        
    def test_clip_to_valid_range_basic(self):
        """Test basic clip_to_valid_range functionality."""
        x = np.array([-0.5, 0.5, 1.5])
        result = clip_to_valid_range(x, 0, 1)
        expected = np.array([0, 0.5, 1])
        assert np.array_equal(result, expected)
        
    def test_clip_to_valid_range_warning(self):
        """Test clip_to_valid_range generates warning."""
        x = np.array([-0.5, 0.5, 1.5])
        with warnings.catch_warnings(record=True) as w:
            clip_to_valid_range(x, 0, 1, "test_array")
            assert len(w) == 1
            assert "test_array" in str(w[0].message)
            assert "outside [0, 1]" in str(w[0].message)
            
    def test_clip_to_valid_range_min_only(self):
        """Test clip_to_valid_range with only min value."""
        x = np.array([-1, 0, 1])
        result = clip_to_valid_range(x, min_val=0)
        expected = np.array([0, 0, 1])
        assert np.array_equal(result, expected)
        
    def test_clip_to_valid_range_max_only(self):
        """Test clip_to_valid_range with only max value."""
        x = np.array([0, 1, 2])
        result = clip_to_valid_range(x, max_val=1)
        expected = np.array([0, 1, 1])
        assert np.array_equal(result, expected)
        
    def test_detect_numerical_issues_clean(self):
        """Test detect_numerical_issues with clean data."""
        x = np.array([1, 2, 3])
        result = detect_numerical_issues(x)
        assert result is None
        
    def test_detect_numerical_issues_nan(self):
        """Test detect_numerical_issues detects NaN."""
        x = np.array([1, np.nan, 3])
        result = detect_numerical_issues(x, "test")
        assert "test:" in result
        assert "1 NaN" in result
        
    def test_detect_numerical_issues_inf(self):
        """Test detect_numerical_issues detects inf."""
        x = np.array([1, np.inf, -np.inf])
        result = detect_numerical_issues(x, "test")
        assert "2 infinite" in result
        
    def test_detect_numerical_issues_underflow(self):
        """Test detect_numerical_issues detects underflow risk."""
        x = np.array([1e-300, 1, 2])
        result = detect_numerical_issues(x, "test")
        assert "underflow risk" in result
        
    def test_detect_numerical_issues_overflow(self):
        """Test detect_numerical_issues detects overflow risk."""
        x = np.array([1, 2, 1e300])
        result = detect_numerical_issues(x, "test")
        assert "overflow risk" in result


class TestActuarialUtilities:
    """Test actuarial-specific numerical utilities."""
    
    def test_stable_probability_calculation_basic(self):
        """Test basic stable_probability_calculation."""
        log_probs = np.array([-1, -2, -3])
        result = stable_probability_calculation(log_probs)
        assert np.all(result >= 0)
        assert np.all(result <= 1)
        assert np.isclose(np.sum(result), 1.0)
        
    def test_stable_probability_calculation_extreme(self):
        """Test stable_probability_calculation with extreme values."""
        log_probs = np.array([-1000, -1001, -1002])
        result = stable_probability_calculation(log_probs)
        assert np.all(np.isfinite(result))
        assert np.isclose(np.sum(result), 1.0)
        
    def test_stable_probability_calculation_no_normalize(self):
        """Test stable_probability_calculation without normalization."""
        log_probs = np.array([-1, -2, -3])
        result = stable_probability_calculation(log_probs, normalize=False)
        expected = np.exp(log_probs)
        assert np.allclose(result, expected)
        
    def test_stable_moment_calculation_mean(self):
        """Test stable_moment_calculation for mean."""
        values = np.array([1, 2, 3])
        probs = np.array([0.2, 0.5, 0.3])
        result = stable_moment_calculation(values, probs, moment=1)
        expected = np.sum(values * probs)
        assert np.isclose(result, expected)
        
    def test_stable_moment_calculation_variance(self):
        """Test stable_moment_calculation for variance."""
        values = np.array([1, 2, 3])
        probs = np.array([0.2, 0.5, 0.3])
        # Second central moment
        result = stable_moment_calculation(values, probs, moment=2, central=True)
        mean = np.sum(values * probs)
        expected = np.sum((values - mean)**2 * probs)
        assert np.isclose(result, expected)
        
    def test_stable_moment_calculation_unnormalized_probs(self):
        """Test stable_moment_calculation with unnormalized probabilities."""
        values = np.array([1, 2, 3])
        probs = np.array([2, 5, 3])  # Sum = 10
        with warnings.catch_warnings(record=True) as w:
            result = stable_moment_calculation(values, probs)
            assert len(w) == 1
            assert "normalizing" in str(w[0].message)
        expected = 2.1  # (1*0.2 + 2*0.5 + 3*0.3)
        assert np.isclose(result, expected)
        
    def test_stable_moment_calculation_log_space(self):
        """Test stable_moment_calculation in log space."""
        values = np.array([10, 20, 30])
        probs = np.array([0.2, 0.5, 0.3])
        result = stable_moment_calculation(values, probs, moment=2, log_space=True)
        expected = np.sum(values**2 * probs)
        assert np.isclose(result, expected)
        
    def test_stable_moment_calculation_edge_cases(self):
        """Test stable_moment_calculation edge cases."""
        values = np.array([1, 2, 3])
        probs = np.array([0.2, 0.5, 0.3])
        
        # Zero moment
        result = stable_moment_calculation(values, probs, moment=0)
        assert result == 1.0
        
        # High moment with warning
        with warnings.catch_warnings(record=True) as w:
            result = stable_moment_calculation(values * 100, probs, moment=10)
            # May generate warning about numerical issues
            
    def test_stable_quantile_interpolation_basic(self):
        """Test basic stable_quantile_interpolation."""
        x = np.array([0, 1, 2, 3, 4])
        cdf = np.array([0, 0.2, 0.5, 0.8, 1.0])
        
        # Test various quantiles
        assert stable_quantile_interpolation(x, cdf, 0.5) == 2.0
        assert np.isclose(stable_quantile_interpolation(x, cdf, 0.9), 3.5)
        assert stable_quantile_interpolation(x, cdf, 0.0) == 0.0
        assert stable_quantile_interpolation(x, cdf, 1.0) == 4.0
        
    def test_stable_quantile_interpolation_array(self):
        """Test stable_quantile_interpolation with array input."""
        x = np.array([0, 1, 2, 3, 4])
        cdf = np.array([0, 0.2, 0.5, 0.8, 1.0])
        q = np.array([0.1, 0.5, 0.9])
        
        result = stable_quantile_interpolation(x, cdf, q)
        assert len(result) == 3
        assert result[1] == 2.0
        
    def test_stable_quantile_interpolation_edge_cases(self):
        """Test stable_quantile_interpolation edge cases."""
        # Single point
        x = np.array([5])
        cdf = np.array([1])
        result = stable_quantile_interpolation(x, cdf, 0.5)
        assert result == 5
        
        # Empty arrays
        x = np.array([])
        cdf = np.array([])
        result = stable_quantile_interpolation(x, cdf, 0.5)
        assert np.isnan(result)
        
    def test_stable_quantile_interpolation_validation(self):
        """Test stable_quantile_interpolation input validation."""
        x = np.array([0, 1, 2, 3, 4])
        cdf = np.array([0, 0.3, 0.2, 0.8, 1.0])  # Non-monotonic
        
        with pytest.raises(ValueError, match="non-decreasing"):
            stable_quantile_interpolation(x, cdf, 0.5)
            
    def test_stable_quantile_interpolation_clipping(self):
        """Test stable_quantile_interpolation clips invalid inputs."""
        x = np.array([0, 1, 2, 3, 4])
        cdf = np.array([-0.1, 0.2, 0.5, 0.8, 1.1])  # Invalid CDF values
        
        with warnings.catch_warnings(record=True) as w:
            result = stable_quantile_interpolation(x, cdf, 1.5)  # Invalid quantile
            # Should generate warnings about clipping


class TestIntegration:
    """Integration tests combining multiple utilities."""
    
    def test_probability_workflow(self):
        """Test complete probability calculation workflow."""
        # Start with log probabilities
        log_probs = np.array([10, 11, 12]) - 15  # Scaled to avoid issues
        
        # Convert to probabilities
        probs = stable_probability_calculation(log_probs)
        
        # Calculate moments
        values = np.array([100, 200, 300])
        mean = stable_moment_calculation(values, probs, moment=1)
        var = stable_moment_calculation(values, probs, moment=2, central=True)
        
        assert np.isfinite(mean)
        assert np.isfinite(var)
        assert var > 0
        
    def test_extreme_value_workflow(self):
        """Test workflow with extreme values."""
        # Extreme log values
        log_vals = np.array([500, 501, 502])
        
        # Use logsumexp to get normalizer
        log_norm = logsumexp(log_vals)
        
        # Normalized log probabilities
        log_probs = log_vals - log_norm
        
        # Convert to probabilities
        probs = stable_probability_calculation(log_probs)
        
        assert np.all(np.isfinite(probs))
        assert np.isclose(np.sum(probs), 1.0)
        
    def test_numerical_validation_workflow(self):
        """Test complete numerical validation workflow."""
        # Create data with potential issues
        data = np.array([1e-300, 1.0, 1e300, np.inf])
        
        # Detect issues
        issues = detect_numerical_issues(data, "test_data")
        assert issues is not None
        
        # Clean data
        with warnings.catch_warnings(record=True):
            cleaned = clip_to_valid_range(data, 1e-200, 1e200)
            finite_data = check_finite(cleaned, allow_inf=False)
            
        assert np.all(np.isfinite(finite_data))


class TestEdgeCasesAndErrors:
    """Test edge cases and error conditions."""
    
    def test_empty_arrays(self):
        """Test functions with empty arrays."""
        empty = np.array([])
        
        # Should handle gracefully
        result = logsumexp(empty)
        assert np.isneginf(result)
        
        result = stable_exp(empty)
        assert len(result) == 0
        
        result = stable_log(empty)
        assert len(result) == 0
        
    def test_scalar_inputs(self):
        """Test functions with scalar inputs."""
        # Most functions should handle scalars
        assert isinstance(stable_exp(5.0), float)
        assert isinstance(stable_log(5.0), float)
        assert isinstance(logaddexp(1.0, 2.0), float)
        
    def test_type_preservation(self):
        """Test that functions preserve input types appropriately."""
        # Float input
        x_float = 5.0
        assert isinstance(stable_exp(x_float), float)
        
        # Array input
        x_array = np.array([5.0])
        assert isinstance(stable_exp(x_array), np.ndarray)
        
    def test_nan_propagation(self):
        """Test NaN propagation in functions."""
        x = np.array([1, np.nan, 3])
        
        # Some functions should propagate NaN
        result = stable_exp(x, clip=False)
        assert np.isnan(result[1])
        
        # Others should catch it
        with pytest.raises(ValueError):
            check_finite(x)


class TestValidationFunctions:
    """Test validation utility functions."""
    
    def test_validate_probability(self):
        """Test probability validation."""
        from quactuary.utils.validation import validate_probability
        
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

    def test_validate_positive_integer(self):
        """Test positive integer validation."""
        from quactuary.utils.validation import validate_positive_integer
        
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

    def test_validate_non_negative_integer(self):
        """Test non-negative integer validation."""
        from quactuary.utils.validation import validate_non_negative_integer
        
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