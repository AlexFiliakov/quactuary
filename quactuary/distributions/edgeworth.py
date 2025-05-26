"""
Edgeworth expansion for distribution approximation.

This module provides Edgeworth series expansion for approximating
probability distributions using cumulant corrections to the normal distribution.
"""

import numpy as np
from scipy import stats, special
from typing import Union, Optional, Tuple, Dict

from quactuary.utils.numerical import stable_exp, stable_log


class EdgeworthExpansion:
    """
    Edgeworth expansion for approximating distributions.
    
    The Edgeworth series provides corrections to the normal approximation
    based on higher-order cumulants (skewness, kurtosis, etc.).
    """
    
    def __init__(self, mean: float, variance: float, 
                 skewness: float = 0.0, excess_kurtosis: float = 0.0,
                 cumulants: Dict[int, float] = None):
        """
        Initialize Edgeworth expansion.
        
        Args:
            mean: Mean of the distribution
            variance: Variance of the distribution
            skewness: Standardized third cumulant
            excess_kurtosis: Standardized fourth cumulant minus 3
            cumulants: Optional dict of higher-order standardized cumulants {order: value}
        """
        self.mean = mean
        self.variance = variance
        self.std = np.sqrt(variance)
        self.skewness = skewness
        self.excess_kurtosis = excess_kurtosis
        
        # Store standardized cumulants
        self.cumulants = {3: skewness, 4: excess_kurtosis}
        if cumulants:
            self.cumulants.update(cumulants)
    
    def _hermite_polynomial(self, n: int, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Compute probabilist's Hermite polynomial H_n(x).
        
        Uses recurrence relation:
        H_0(x) = 1
        H_1(x) = x
        H_{n+1}(x) = x * H_n(x) - n * H_{n-1}(x)
        """
        if n == 0:
            return np.ones_like(x)
        elif n == 1:
            return x
        elif n == 2:
            return x**2 - 1
        elif n == 3:
            return x**3 - 3*x
        elif n == 4:
            return x**4 - 6*x**2 + 3
        elif n == 5:
            return x**5 - 10*x**3 + 15*x
        elif n == 6:
            return x**6 - 15*x**4 + 45*x**2 - 15
        else:
            # Use recurrence for higher orders
            h_prev_prev = self._hermite_polynomial(n-2, x)
            h_prev = self._hermite_polynomial(n-1, x)
            return x * h_prev - (n-1) * h_prev_prev
    
    def pdf(self, x: Union[float, np.ndarray], order: int = 4) -> Union[float, np.ndarray]:
        """
        Probability density function using Edgeworth expansion.
        
        Args:
            x: Values at which to evaluate PDF
            order: Maximum order of expansion (2, 3, or 4)
            
        Returns:
            PDF values at x
        """
        x_array = np.atleast_1d(x)
        
        # Standardize
        z = (x_array - self.mean) / self.std
        
        # Standard normal PDF
        phi = stats.norm.pdf(z)
        
        # Base approximation
        result = phi / self.std
        
        if order == 2:
            # No corrections beyond normal
            pass
        elif order == 3:
            # Third-order correction only
            if self.skewness != 0:
                h3 = self._hermite_polynomial(3, z)
                result = result * (1 + self.skewness * h3 / 6)
        elif order >= 4:
            # Full expansion with both corrections applied together
            correction = 1.0
            
            if self.skewness != 0:
                h3 = self._hermite_polynomial(3, z)
                correction += self.skewness * h3 / 6
            
            if self.skewness != 0 or self.excess_kurtosis != 0:
                h4 = self._hermite_polynomial(4, z)
                h6 = self._hermite_polynomial(6, z)
                
                correction += (self.excess_kurtosis * h4 / 24 + 
                              self.skewness**2 * h6 / 72)
            
            result = result * correction
        
        # Ensure non-negative
        result = np.maximum(result, 0)
        
        return result[0] if np.isscalar(x) else result
    
    def cdf(self, x: Union[float, np.ndarray], order: int = 4) -> Union[float, np.ndarray]:
        """
        Cumulative distribution function using Edgeworth expansion.
        
        Args:
            x: Values at which to evaluate CDF
            order: Maximum order of expansion (2, 3, or 4)
            
        Returns:
            CDF values at x
        """
        x_array = np.atleast_1d(x)
        
        # Standardize
        z = (x_array - self.mean) / self.std
        
        # Standard normal CDF and PDF
        Phi = stats.norm.cdf(z)
        phi = stats.norm.pdf(z)
        
        # Base approximation
        result = Phi
        
        if order >= 3 and self.skewness != 0:
            # Third-order correction
            h2 = self._hermite_polynomial(2, z)
            result -= self.skewness * phi * h2 / 6
        
        if order >= 4 and (self.skewness != 0 or self.excess_kurtosis != 0):
            # Fourth-order correction
            h3 = self._hermite_polynomial(3, z)
            h5 = self._hermite_polynomial(5, z)
            
            correction4 = (-self.excess_kurtosis * phi * h3 / 24 - 
                          self.skewness**2 * phi * h5 / 72)
            result += correction4
        
        # Ensure valid CDF
        result = np.clip(result, 0, 1)
        
        return result[0] if np.isscalar(x) else result
    
    def ppf(self, q: Union[float, np.ndarray], order: int = 4,
            method: str = 'cornish-fisher') -> Union[float, np.ndarray]:
        """
        Percent point function (quantile function).
        
        Args:
            q: Quantile values in [0, 1]
            order: Maximum order of expansion
            method: 'cornish-fisher' or 'numerical'
            
        Returns:
            Quantile values
        """
        q_array = np.atleast_1d(q)
        
        if method == 'cornish-fisher':
            # Cornish-Fisher expansion
            z_q = stats.norm.ppf(q_array)
            
            # Base approximation
            result = self.mean + self.std * z_q
            
            if order >= 3 and self.skewness != 0:
                # Third-order correction
                correction3 = (self.skewness * (z_q**2 - 1) / 6)
                result += self.std * correction3
            
            if order >= 4 and (self.skewness != 0 or self.excess_kurtosis != 0):
                # Fourth-order correction
                correction4 = (self.excess_kurtosis * z_q * (z_q**2 - 3) / 24 -
                              self.skewness**2 * z_q * (2*z_q**2 - 5) / 36)
                result += self.std * correction4
        
        else:  # numerical inversion
            from scipy.optimize import brentq
            result = np.zeros_like(q_array)
            
            for i, qi in enumerate(q_array):
                # Initial guess from normal approximation
                x0 = self.mean + self.std * stats.norm.ppf(qi)
                
                # Search bounds
                lower = x0 - 10 * self.std
                upper = x0 + 10 * self.std
                
                try:
                    result[i] = brentq(
                        lambda x: self.cdf(x, order=order) - qi,
                        lower, upper,
                        xtol=1e-6
                    )
                except:
                    # Fall back to normal approximation
                    result[i] = x0
        
        return result[0] if np.isscalar(q) else result
    
    def validate_expansion(self, order: int = 4) -> Dict[str, float]:
        """
        Validate the Edgeworth expansion convergence.
        
        Returns:
            Dictionary with convergence diagnostics
        """
        # Check if moments suggest good convergence
        diagnostics = {
            'skewness': self.skewness,
            'excess_kurtosis': self.excess_kurtosis,
            'order': order
        }
        
        # Edgeworth expansion works best when |skewness| < 1 and |excess_kurtosis| < 2
        diagnostics['skewness_ok'] = abs(self.skewness) < 1.0
        diagnostics['kurtosis_ok'] = abs(self.excess_kurtosis) < 2.0
        
        # Check PDF positivity at several points
        test_points = np.linspace(
            self.mean - 4*self.std,
            self.mean + 4*self.std,
            100
        )
        pdf_vals = self.pdf(test_points, order=order)
        diagnostics['min_pdf'] = np.min(pdf_vals)
        diagnostics['pdf_positive'] = np.all(pdf_vals >= 0)
        
        # Check CDF monotonicity
        cdf_vals = self.cdf(test_points, order=order)
        cdf_diffs = np.diff(cdf_vals)
        diagnostics['cdf_monotonic'] = np.all(cdf_diffs >= 0)
        
        # Overall validity
        diagnostics['valid'] = (diagnostics['skewness_ok'] and 
                               diagnostics['kurtosis_ok'] and
                               diagnostics['pdf_positive'] and
                               diagnostics['cdf_monotonic'])
        
        return diagnostics


class CompoundDistributionEdgeworth:
    """
    Edgeworth expansion specifically for compound distributions.
    
    Uses analytical formulas for cumulants of compound distributions
    to construct accurate Edgeworth approximations.
    """
    
    def __init__(self, compound_dist=None, frequency_mean: float = None, 
                 frequency_var: float = None, severity_moments: Dict[int, float] = None):
        """
        Initialize Edgeworth expansion for compound distribution.
        
        Args:
            compound_dist: Optional compound distribution object (if provided, extracts parameters)
            frequency_mean: E[N] (required if compound_dist not provided)
            frequency_var: Var[N] (required if compound_dist not provided)
            severity_moments: Dict mapping order k to E[X^k] (required if compound_dist not provided)
        """
        if compound_dist is not None:
            # Extract parameters from compound distribution
            # Need to check if mean/var are properties or methods
            if hasattr(compound_dist.frequency, 'mean'):
                if callable(compound_dist.frequency.mean):
                    self.freq_mean = compound_dist.frequency.mean()
                else:
                    self.freq_mean = compound_dist.frequency.mean
            else:
                # Use parameter directly from scipy distribution
                if hasattr(compound_dist.frequency, '_dist'):
                    freq_type = type(compound_dist.frequency).__name__
                    if freq_type == 'Poisson':
                        # For Poisson, mu is in args[0]
                        self.freq_mean = compound_dist.frequency._dist.args[0]
                    elif freq_type == 'Binomial':
                        # For Binomial, mean = n * p
                        n = compound_dist.frequency._dist.args[0]
                        p = compound_dist.frequency._dist.args[1]
                        self.freq_mean = n * p
                    elif freq_type == 'NegativeBinomial':
                        # For NegativeBinomial, mean = r * (1-p) / p
                        r = compound_dist.frequency._dist.args[0]
                        p = compound_dist.frequency._dist.args[1]
                        self.freq_mean = r * (1 - p) / p
                    else:
                        # Try to get mean directly
                        self.freq_mean = compound_dist.frequency._dist.mean()
                else:
                    raise ValueError(f"Cannot extract mean from {type(compound_dist.frequency).__name__}")
                
            if hasattr(compound_dist.frequency, 'var'):
                if callable(compound_dist.frequency.var):
                    self.freq_var = compound_dist.frequency.var()
                else:
                    self.freq_var = compound_dist.frequency.var
            else:
                # Calculate variance based on distribution type
                if hasattr(compound_dist.frequency, '_dist'):
                    freq_type = type(compound_dist.frequency).__name__
                    if freq_type == 'Poisson':
                        # For Poisson, var = mean
                        self.freq_var = self.freq_mean
                    elif freq_type == 'Binomial':
                        # For Binomial, var = n * p * (1 - p)
                        n = compound_dist.frequency._dist.args[0]
                        p = compound_dist.frequency._dist.args[1]
                        self.freq_var = n * p * (1 - p)
                    elif freq_type == 'NegativeBinomial':
                        # For NegativeBinomial, var = r * (1-p) / p^2
                        r = compound_dist.frequency._dist.args[0]
                        p = compound_dist.frequency._dist.args[1]
                        self.freq_var = r * (1 - p) / (p ** 2)
                    else:
                        # Try to get variance directly
                        self.freq_var = compound_dist.frequency._dist.var()
                else:
                    self.freq_var = self.freq_mean  # Default fallback
            
            # Compute severity moments
            self.sev_moments = {}
            for k in range(1, 5):  # Moments up to 4th order
                if hasattr(compound_dist.severity, 'moment'):
                    # Use moment method if available
                    self.sev_moments[k] = compound_dist.severity.moment(k)
                else:
                    # Fallback to standard moment calculations
                    if k == 1:
                        if hasattr(compound_dist.severity, 'mean'):
                            if callable(compound_dist.severity.mean):
                                self.sev_moments[k] = compound_dist.severity.mean()
                            else:
                                self.sev_moments[k] = compound_dist.severity.mean
                        else:
                            # Get mean from distribution parameters
                            if hasattr(compound_dist.severity, '_dist'):
                                sev_type = type(compound_dist.severity).__name__
                                if sev_type == 'Exponential':
                                    # For Exponential, mean = scale
                                    self.sev_moments[k] = compound_dist.severity._dist.kwds.get('scale', 1.0)
                                elif sev_type == 'Gamma':
                                    # For Gamma, mean = shape * scale
                                    shape = compound_dist.severity._dist.args[0]
                                    scale = compound_dist.severity._dist.kwds.get('scale', 1.0)
                                    self.sev_moments[k] = shape * scale
                                else:
                                    # Try to use distribution's mean method
                                    self.sev_moments[k] = compound_dist.severity._dist.mean()
                    elif k == 2:
                        mean_val = self.sev_moments[1]
                        if hasattr(compound_dist.severity, 'var'):
                            if callable(compound_dist.severity.var):
                                var_val = compound_dist.severity.var()
                            else:
                                var_val = compound_dist.severity.var
                        else:
                            # Get variance from distribution parameters
                            if hasattr(compound_dist.severity, '_dist'):
                                sev_type = type(compound_dist.severity).__name__
                                if sev_type == 'Exponential':
                                    # For Exponential, var = scale^2
                                    scale = compound_dist.severity._dist.kwds.get('scale', 1.0)
                                    var_val = scale ** 2
                                elif sev_type == 'Gamma':
                                    # For Gamma, var = shape * scale^2
                                    shape = compound_dist.severity._dist.args[0]
                                    scale = compound_dist.severity._dist.kwds.get('scale', 1.0)
                                    var_val = shape * (scale ** 2)
                                else:
                                    # Try to use distribution's var method
                                    var_val = compound_dist.severity._dist.var()
                        self.sev_moments[k] = var_val + mean_val**2
                    else:
                        # Higher moments - use numerical integration or sampling
                        # For now, estimate from samples
                        np.random.seed(42)
                        samples = compound_dist.severity.rvs(size=10000)
                        self.sev_moments[k] = np.mean(samples**k)
        else:
            if frequency_mean is None or frequency_var is None or severity_moments is None:
                raise ValueError("Must provide either compound_dist or all of (frequency_mean, frequency_var, severity_moments)")
            self.freq_mean = frequency_mean
            self.freq_var = frequency_var
            self.sev_moments = severity_moments
        
        # Compute compound distribution moments
        self._compute_compound_moments()
        
        # Create Edgeworth expansion
        self.expansion = EdgeworthExpansion(
            mean=self.mean,
            variance=self.variance,
            skewness=self.skewness,
            excess_kurtosis=self.excess_kurtosis
        )
    
    def _compute_compound_moments(self):
        """Compute moments of compound distribution S = X₁ + ... + X_N."""
        # Severity moments
        m1 = self.sev_moments.get(1, 0)
        m2 = self.sev_moments.get(2, 0)
        m3 = self.sev_moments.get(3, 0)
        m4 = self.sev_moments.get(4, 0)
        
        # Compute severity variance from moments
        sev_var = m2 - m1**2
        
        # Compound distribution moments
        self.mean = self.freq_mean * m1
        
        # For compound Poisson: Var[S] = λ * E[X²]
        # For general compound: Var[S] = E[N] * Var[X] + Var[N] * E[X]²
        # For Poisson, Var[N] = E[N] = λ, so:
        # Var[S] = λ * Var[X] + λ * E[X]² = λ * E[X²]
        if self.freq_mean == self.freq_var:  # Poisson case
            self.variance = self.freq_mean * m2
        else:
            # General compound distribution formula
            self.variance = self.freq_mean * sev_var + self.freq_var * m1**2
        
        # Third moment
        freq_m3 = self.freq_var + self.freq_mean  # For common distributions
        self.m3 = freq_m3 * m1**3 + 3 * self.freq_mean * m1 * m2 + self.freq_mean * m3
        
        # Fourth moment (simplified formula)
        self.m4 = (self.freq_mean * m4 + 
                   4 * self.freq_mean * m1 * m3 +
                   6 * self.freq_mean * m2**2 +
                   6 * self.freq_var * m1**2 * m2 +
                   self.freq_var * m1**4)
        
        # Standardized moments
        self.std = np.sqrt(self.variance)
        self.skewness = self.m3 / (self.std**3)
        self.kurtosis = self.m4 / (self.std**4)
        self.excess_kurtosis = self.kurtosis - 3
    
    def pdf(self, x: Union[float, np.ndarray], order: int = 4) -> Union[float, np.ndarray]:
        """PDF using Edgeworth expansion."""
        return self.expansion.pdf(x, order=order)
    
    def cdf(self, x: Union[float, np.ndarray], order: int = 4) -> Union[float, np.ndarray]:
        """CDF using Edgeworth expansion."""
        return self.expansion.cdf(x, order=order)
    
    def ppf(self, q: Union[float, np.ndarray], order: int = 4) -> Union[float, np.ndarray]:
        """Quantile function using Edgeworth expansion."""
        return self.expansion.ppf(q, order=order)
    
    def compare_with_simulation(self, n_sim: int = 10000,
                              random_state: Optional[int] = None) -> Dict[str, float]:
        """
        Compare Edgeworth approximation with Monte Carlo simulation.
        
        Args:
            n_sim: Number of simulations
            random_state: Random seed
            
        Returns:
            Dictionary with comparison metrics
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        # This is a placeholder - actual implementation would need
        # the frequency and severity distributions
        comparison = {
            'edgeworth_mean': self.mean,
            'edgeworth_std': self.std,
            'edgeworth_skewness': self.skewness,
            'edgeworth_excess_kurtosis': self.excess_kurtosis
        }
        
        return comparison


def automatic_order_selection(skewness: float, excess_kurtosis: float,
                            sample_size: int) -> int:
    """
    Automatically select Edgeworth expansion order based on moments and sample size.
    
    Args:
        skewness: Standardized third moment
        excess_kurtosis: Standardized fourth moment minus 3
        sample_size: Number of observations
        
    Returns:
        Recommended order (2, 3, or 4)
    """
    # For small samples, use lower order
    if sample_size < 50:
        return 2
    
    # Check moment magnitudes
    if abs(skewness) < 0.5 and abs(excess_kurtosis) < 1.0:
        # Moments are small, normal approximation is good
        return 2
    elif abs(skewness) < 1.0 and abs(excess_kurtosis) < 2.0:
        # Moderate departure from normality
        if sample_size >= 100:
            return 4
        else:
            return 3
    else:
        # Large departure from normality
        # Edgeworth may not be appropriate, but use order 3
        return 3


def cornish_fisher_expansion(z: Union[float, np.ndarray], 
                            skewness: float, 
                            excess_kurtosis: float,
                            order: int = 3) -> Union[float, np.ndarray]:
    """
    Cornish-Fisher expansion for quantile transformation.
    
    Transforms standard normal quantiles to distribution quantiles
    using cumulant corrections.
    
    Args:
        z: Standard normal quantile(s)
        skewness: Standardized third cumulant
        excess_kurtosis: Standardized fourth cumulant minus 3
        order: Order of expansion (1, 2, or 3)
        
    Returns:
        Transformed quantile(s)
    """
    z_array = np.atleast_1d(z)
    
    # Order 1: just return z
    if order == 1:
        result = z_array.copy()
    else:
        # Start with z
        result = z_array.copy()
        
        if order >= 2:
            # Second-order correction (skewness)
            result += skewness * (z_array**2 - 1) / 6
        
        if order >= 3:
            # Third-order correction (kurtosis and skewness squared)
            result += (excess_kurtosis * z_array * (z_array**2 - 3) / 24 -
                      skewness**2 * z_array * (2*z_array**2 - 5) / 36)
    
    return result[0] if np.isscalar(z) else result


def edgeworth_gram_charlier_comparison(moments: Dict[int, float],
                                     x_range: Tuple[float, float],
                                     n_points: int = 1000) -> Dict[str, np.ndarray]:
    """
    Compare Edgeworth and Gram-Charlier expansions.
    
    The Gram-Charlier expansion uses a different arrangement of terms
    that can sometimes provide better approximations.
    
    Args:
        moments: Dictionary with mean, variance, skewness, excess_kurtosis
        x_range: Range of x values for comparison
        n_points: Number of points to evaluate
        
    Returns:
        Dictionary with both expansions' PDFs and CDFs
    """
    mean = moments['mean']
    variance = moments['variance']
    skewness = moments.get('skewness', 0)
    excess_kurtosis = moments.get('excess_kurtosis', 0)
    
    # Create Edgeworth expansion
    edgeworth = EdgeworthExpansion(mean, variance, skewness, excess_kurtosis)
    
    # Evaluation points
    x = np.linspace(x_range[0], x_range[1], n_points)
    
    # Compute both expansions
    results = {
        'x': x,
        'edgeworth_pdf': edgeworth.pdf(x, order=4),
        'edgeworth_cdf': edgeworth.cdf(x, order=4),
        'normal_pdf': stats.norm.pdf(x, loc=mean, scale=np.sqrt(variance)),
        'normal_cdf': stats.norm.cdf(x, loc=mean, scale=np.sqrt(variance))
    }
    
    # Gram-Charlier would use different coefficient arrangement
    # (not implemented here, but structure is similar)
    
    return results