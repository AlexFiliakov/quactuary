"""Enhanced statistical validators for integration tests."""

import numpy as np
from scipy import stats
from typing import Dict, Tuple, Optional, List
import warnings


class EnhancedStatisticalValidator:
    """Enhanced statistical validation methods for integration tests."""
    
    def __init__(self, confidence_level: float = 0.95):
        """Initialize validator with confidence level."""
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
    
    def confidence_interval_test(
        self, 
        sample: np.ndarray, 
        expected_value: float,
        method: str = 'bootstrap'
    ) -> Dict:
        """Test if expected value falls within confidence interval.
        
        Args:
            sample: Sample data
            expected_value: Expected value to test
            method: 'bootstrap', 't-test', or 'percentile'
            
        Returns:
            Dict with test results
        """
        if method == 'bootstrap':
            return self._bootstrap_ci_test(sample, expected_value)
        elif method == 't-test':
            return self._t_test_ci(sample, expected_value)
        elif method == 'percentile':
            return self._percentile_ci_test(sample, expected_value)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _bootstrap_ci_test(
        self, 
        sample: np.ndarray, 
        expected_value: float,
        n_bootstrap: int = 1000
    ) -> Dict:
        """Bootstrap confidence interval test."""
        # Bootstrap resampling
        bootstrap_means = []
        n = len(sample)
        
        for _ in range(n_bootstrap):
            resample = np.random.choice(sample, size=n, replace=True)
            bootstrap_means.append(np.mean(resample))
        
        # Calculate confidence interval
        lower = np.percentile(bootstrap_means, (self.alpha/2) * 100)
        upper = np.percentile(bootstrap_means, (1 - self.alpha/2) * 100)
        
        in_interval = lower <= expected_value <= upper
        
        return {
            'method': 'bootstrap',
            'confidence_level': self.confidence_level,
            'interval': (lower, upper),
            'expected_value': expected_value,
            'sample_mean': np.mean(sample),
            'in_interval': in_interval,
            'passes_test': in_interval
        }
    
    def _t_test_ci(self, sample: np.ndarray, expected_value: float) -> Dict:
        """T-test based confidence interval."""
        n = len(sample)
        mean = np.mean(sample)
        std_err = stats.sem(sample)
        
        # T-distribution critical value
        t_critical = stats.t.ppf(1 - self.alpha/2, n - 1)
        
        # Confidence interval
        margin = t_critical * std_err
        lower = mean - margin
        upper = mean + margin
        
        in_interval = lower <= expected_value <= upper
        
        # Also perform one-sample t-test
        t_stat, p_value = stats.ttest_1samp(sample, expected_value)
        
        return {
            'method': 't-test',
            'confidence_level': self.confidence_level,
            'interval': (lower, upper),
            'expected_value': expected_value,
            'sample_mean': mean,
            'in_interval': in_interval,
            't_statistic': t_stat,
            'p_value': p_value,
            'passes_test': p_value > self.alpha
        }
    
    def _percentile_ci_test(
        self, 
        sample: np.ndarray, 
        expected_value: float
    ) -> Dict:
        """Simple percentile-based confidence interval."""
        lower = np.percentile(sample, (self.alpha/2) * 100)
        upper = np.percentile(sample, (1 - self.alpha/2) * 100)
        
        in_interval = lower <= expected_value <= upper
        
        return {
            'method': 'percentile',
            'confidence_level': self.confidence_level,
            'interval': (lower, upper),
            'expected_value': expected_value,
            'sample_median': np.median(sample),
            'in_interval': in_interval,
            'passes_test': in_interval
        }
    
    def equivalence_test(
        self,
        sample1: np.ndarray,
        sample2: np.ndarray,
        margin: float
    ) -> Dict:
        """Two-one-sided t-tests (TOST) for equivalence.
        
        Tests if two samples are equivalent within a specified margin.
        """
        mean1 = np.mean(sample1)
        mean2 = np.mean(sample2)
        
        # Pooled standard error
        n1, n2 = len(sample1), len(sample2)
        var1, var2 = np.var(sample1, ddof=1), np.var(sample2, ddof=1)
        pooled_se = np.sqrt(var1/n1 + var2/n2)
        
        # TOST procedure
        t_lower = (mean1 - mean2 + margin) / pooled_se
        t_upper = (mean1 - mean2 - margin) / pooled_se
        
        df = n1 + n2 - 2
        p_lower = stats.t.cdf(t_lower, df)
        p_upper = 1 - stats.t.cdf(t_upper, df)
        
        # Equivalence if both p-values < alpha
        p_value = max(p_lower, p_upper)
        equivalent = p_value < self.alpha
        
        return {
            'method': 'TOST',
            'mean_difference': mean1 - mean2,
            'margin': margin,
            'pooled_se': pooled_se,
            't_lower': t_lower,
            't_upper': t_upper,
            'p_value': p_value,
            'equivalent': equivalent,
            'passes_test': equivalent
        }
    
    def adaptive_tolerance_test(
        self,
        value1: float,
        value2: float,
        base_tolerance: float,
        scale_factor: float = 1.0
    ) -> Dict:
        """Adaptive tolerance test that scales with value magnitude.
        
        For small values, uses absolute tolerance.
        For large values, uses relative tolerance.
        """
        magnitude = max(abs(value1), abs(value2))
        
        if magnitude < 1.0:
            # Use absolute tolerance for small values
            tolerance = base_tolerance
            error = abs(value1 - value2)
            error_type = 'absolute'
        else:
            # Use relative tolerance for large values
            tolerance = base_tolerance * scale_factor
            error = abs(value1 - value2) / magnitude
            error_type = 'relative'
        
        passes_test = error < tolerance
        
        return {
            'method': 'adaptive_tolerance',
            'value1': value1,
            'value2': value2,
            'magnitude': magnitude,
            'error_type': error_type,
            'error': error,
            'tolerance': tolerance,
            'passes_test': passes_test
        }
    
    def stochastic_dominance_test(
        self,
        sample1: np.ndarray,
        sample2: np.ndarray,
        order: int = 1
    ) -> Dict:
        """Test for stochastic dominance between two samples.
        
        First-order: CDF of sample1 is always below CDF of sample2
        Second-order: Integral of CDF1 is always below integral of CDF2
        """
        # Sort samples
        sorted1 = np.sort(sample1)
        sorted2 = np.sort(sample2)
        
        # Create common support
        min_val = min(sorted1[0], sorted2[0])
        max_val = max(sorted1[-1], sorted2[-1])
        support = np.linspace(min_val, max_val, 1000)
        
        # Compute empirical CDFs
        cdf1 = np.searchsorted(sorted1, support, side='right') / len(sorted1)
        cdf2 = np.searchsorted(sorted2, support, side='right') / len(sorted2)
        
        if order == 1:
            # First-order stochastic dominance
            dominance = np.all(cdf1 <= cdf2)
            max_violation = np.max(cdf1 - cdf2)
        elif order == 2:
            # Second-order stochastic dominance (integral of CDF)
            integral1 = np.cumsum(cdf1) * (support[1] - support[0])
            integral2 = np.cumsum(cdf2) * (support[1] - support[0])
            dominance = np.all(integral1 <= integral2)
            max_violation = np.max(integral1 - integral2)
        else:
            raise ValueError(f"Order {order} not supported")
        
        return {
            'method': f'stochastic_dominance_order_{order}',
            'dominance': dominance,
            'max_violation': max_violation,
            'passes_test': dominance or abs(max_violation) < 0.05
        }
    
    def multiple_comparison_correction(
        self,
        p_values: List[float],
        method: str = 'bonferroni'
    ) -> Dict:
        """Apply multiple comparison correction to p-values.
        
        Methods: 'bonferroni', 'holm', 'fdr'
        """
        n_tests = len(p_values)
        p_array = np.array(p_values)
        
        if method == 'bonferroni':
            adjusted_alpha = self.alpha / n_tests
            adjusted_p = p_array * n_tests
            adjusted_p = np.minimum(adjusted_p, 1.0)
        elif method == 'holm':
            # Holm-Bonferroni method
            sorted_idx = np.argsort(p_array)
            sorted_p = p_array[sorted_idx]
            adjusted_p = np.zeros_like(sorted_p)
            
            for i in range(n_tests):
                adjusted_p[i] = sorted_p[i] * (n_tests - i)
            
            # Ensure monotonicity
            for i in range(1, n_tests):
                adjusted_p[i] = max(adjusted_p[i], adjusted_p[i-1])
            
            # Restore original order
            inv_idx = np.argsort(sorted_idx)
            adjusted_p = adjusted_p[inv_idx]
        elif method == 'fdr':
            # Benjamini-Hochberg FDR
            sorted_idx = np.argsort(p_array)
            sorted_p = p_array[sorted_idx]
            adjusted_p = np.zeros_like(sorted_p)
            
            for i in range(n_tests):
                adjusted_p[i] = sorted_p[i] * n_tests / (i + 1)
            
            # Ensure monotonicity from the end
            for i in range(n_tests - 2, -1, -1):
                adjusted_p[i] = min(adjusted_p[i], adjusted_p[i+1])
            
            # Restore original order
            inv_idx = np.argsort(sorted_idx)
            adjusted_p = adjusted_p[inv_idx]
        else:
            raise ValueError(f"Unknown method: {method}")
        
        significant = adjusted_p < self.alpha
        
        return {
            'method': method,
            'n_tests': n_tests,
            'original_p_values': p_values,
            'adjusted_p_values': adjusted_p.tolist(),
            'adjusted_alpha': adjusted_alpha if method == 'bonferroni' else self.alpha,
            'significant': significant.tolist(),
            'n_significant': np.sum(significant)
        }