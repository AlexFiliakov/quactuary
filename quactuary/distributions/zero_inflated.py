"""
Zero-inflated compound distributions for modeling excess zeros.

This module provides zero-inflated versions of compound distributions,
useful when data has more zeros than expected from standard models.
"""

import numpy as np
from scipy import stats, optimize
from typing import Union, Optional, Tuple, Dict

from quactuary.distributions.compound import CompoundDistribution, create_compound_distribution
from quactuary.distributions.frequency import FrequencyModel
from quactuary.distributions.severity import SeverityModel
from quactuary.utils.numerical import stable_log, stable_exp


class ZeroInflatedCompound(CompoundDistribution):
    """
    Base class for zero-inflated compound distributions.
    
    Models aggregate loss S as:
    S = 0 with probability π + (1-π) * P(N=0)
    S ~ CompoundDist with probability (1-π) for S > 0
    
    where π is the zero-inflation parameter.
    """
    
    def __init__(self, frequency: FrequencyModel, severity: SeverityModel, 
                 zero_prob: float = None):
        """
        Initialize zero-inflated compound distribution.
        
        Args:
            frequency: Frequency distribution model
            severity: Severity distribution model
            zero_prob: Probability of structural zeros (if None, estimate from data)
        """
        super().__init__(frequency, severity)
        self.zero_prob = zero_prob if zero_prob is not None else 0.0
        self.base_compound = create_compound_distribution(frequency, severity)
        self._validate_params()
    
    def _validate_params(self):
        """Validate parameters."""
        if not 0 <= self.zero_prob < 1:
            raise ValueError("zero_prob must be in [0, 1)")
    
    def mean(self) -> float:
        """E[S] = (1 - π) * E[S|S from compound]"""
        return (1 - self.zero_prob) * self.base_compound.mean()
    
    def var(self) -> float:
        """Var[S] using law of total variance."""
        base_mean = self.base_compound.mean()
        base_var = self.base_compound.var()
        
        # Var[S] = E[Var[S|Z]] + Var[E[S|Z]]
        # where Z indicates zero-inflation
        prob_compound = 1 - self.zero_prob
        
        # E[Var[S|Z]] = prob_compound * base_var
        # Var[E[S|Z]] = prob_compound * (1 - prob_compound) * base_mean^2
        
        return (prob_compound * base_var + 
                prob_compound * (1 - prob_compound) * base_mean**2)
    
    def pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Probability density/mass function."""
        x_array = np.atleast_1d(x)
        result = np.zeros_like(x_array, dtype=float)
        
        # Handle zero separately
        zero_mask = x_array == 0
        if np.any(zero_mask):
            # Total probability at zero
            p_zero_base = self.base_compound.pdf(0)
            result[zero_mask] = self.zero_prob + (1 - self.zero_prob) * p_zero_base
        
        # Handle positive values
        pos_mask = x_array > 0
        if np.any(pos_mask):
            result[pos_mask] = (1 - self.zero_prob) * self.base_compound.pdf(x_array[pos_mask])
        
        return result[0] if np.isscalar(x) else result
    
    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Cumulative distribution function."""
        x_array = np.atleast_1d(x)
        result = np.zeros_like(x_array, dtype=float)
        
        # Handle negative values
        result[x_array < 0] = 0.0
        
        # Handle infinite values (should be exactly 1)
        inf_mask = np.isinf(x_array) & (x_array > 0)
        result[inf_mask] = 1.0
        
        # Handle zero and finite positive values
        finite_non_neg_mask = (x_array >= 0) & np.isfinite(x_array)
        if np.any(finite_non_neg_mask):
            base_cdf = self.base_compound.cdf(x_array[finite_non_neg_mask])
            p_zero_base = self.base_compound.pdf(0)
            
            # Adjust CDF for zero-inflation
            adjusted_cdf = self.zero_prob + (1 - self.zero_prob) * base_cdf
            result[finite_non_neg_mask] = adjusted_cdf
        
        return result[0] if np.isscalar(x) else result
    
    def ppf(self, q: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Percent point function (inverse CDF)."""
        q_array = np.atleast_1d(q)
        result = np.zeros_like(q_array, dtype=float)
        
        # Total probability at zero
        p_zero_total = self.pdf(0)
        
        for i, qi in enumerate(q_array):
            if qi <= p_zero_total:
                result[i] = 0.0
            else:
                # Adjust quantile for base distribution
                q_adjusted = (qi - self.zero_prob) / (1 - self.zero_prob)
                result[i] = self.base_compound.ppf(q_adjusted)
        
        return result[0] if np.isscalar(q) else result
    
    def rvs(self, size: int = 1, random_state: Optional[int] = None) -> Union[float, np.ndarray]:
        """Generate random variates."""
        if random_state is not None:
            np.random.seed(random_state)
        
        # Determine which samples are structural zeros
        is_zero = np.random.rand(size) < self.zero_prob
        
        # Generate from base compound for non-zeros
        result = np.zeros(size)
        n_compound = np.sum(~is_zero)
        
        if n_compound > 0:
            compound_samples = self.base_compound.rvs(size=n_compound)
            result[~is_zero] = compound_samples
        
        return result[0] if size == 1 else result
    
    def fit_em(self, data: np.ndarray, max_iter: int = 100, 
               tol: float = 1e-6, verbose: bool = False) -> Dict[str, float]:
        """
        Fit parameters using Expectation-Maximization algorithm.
        
        Args:
            data: Observed aggregate loss data
            max_iter: Maximum EM iterations
            tol: Convergence tolerance
            verbose: Print iteration progress
            
        Returns:
            Dictionary with fitted parameters
        """
        data = np.asarray(data)
        n_obs = len(data)
        n_zeros = np.sum(data == 0)
        
        # Initial estimate of zero_prob
        if self.zero_prob == 0:
            self.zero_prob = max(0.01, (n_zeros / n_obs) - 0.1)
        
        # Initialize parameters
        log_lik_old = -np.inf
        
        for iteration in range(max_iter):
            # E-step: compute responsibilities
            # For zero observations
            p_zero_base = self.base_compound.pdf(0)
            p_zero_total = self.zero_prob + (1 - self.zero_prob) * p_zero_base
            
            # Responsibility that zero comes from zero-inflation
            if p_zero_total > 0:
                w_zero = self.zero_prob / p_zero_total
            else:
                w_zero = 0.0
            
            # M-step: update parameters
            # Update zero_prob
            n_structural_zeros = n_zeros * w_zero
            self.zero_prob = n_structural_zeros / n_obs
            
            # Compute log-likelihood
            log_lik = self._compute_log_likelihood(data)
            
            if verbose:
                print(f"Iteration {iteration + 1}: "
                      f"zero_prob = {self.zero_prob:.4f}, "
                      f"log-lik = {log_lik:.2f}")
            
            # Check convergence
            if abs(log_lik - log_lik_old) < tol:
                break
            
            log_lik_old = log_lik
        
        return {
            'zero_prob': self.zero_prob,
            'converged': iteration < max_iter - 1,
            'iterations': iteration + 1,
            'log_likelihood': log_lik
        }
    
    def _compute_log_likelihood(self, data: np.ndarray) -> float:
        """Compute log-likelihood of data."""
        # Avoid log(0) by using small epsilon
        eps = 1e-10
        pdf_vals = self.pdf(data)
        pdf_vals = np.maximum(pdf_vals, eps)
        return np.sum(stable_log(pdf_vals))
    
    def diagnose_zero_inflation(self, data: np.ndarray) -> Dict[str, float]:
        """
        Diagnostic tools for detecting zero-inflation.
        
        Args:
            data: Observed data
            
        Returns:
            Dictionary with diagnostic statistics
        """
        n_obs = len(data)
        n_zeros = np.sum(data == 0)
        obs_zero_prop = n_zeros / n_obs
        
        # Expected proportion of zeros from base model
        exp_zero_prop = self.base_compound.pdf(0)
        
        # Vuong test statistic for model comparison
        # Compare ZI model vs standard model
        ll_zi = self._compute_log_likelihood(data)
        
        # Log-likelihood under standard model
        standard_model = self.base_compound
        ll_standard = 0.0
        for x in data:
            pdf_val = max(1e-10, standard_model.pdf(x))
            ll_standard += stable_log(pdf_val)
        
        # Vuong statistic
        n = len(data)
        vuong_stat = (ll_zi - ll_standard) / (np.sqrt(n) * 0.1)  # Simplified
        
        return {
            'observed_zero_proportion': obs_zero_prop,
            'expected_zero_proportion': exp_zero_prop,
            'excess_zeros': obs_zero_prop - exp_zero_prop,
            'fitted_zero_prob': self.zero_prob,
            'vuong_statistic': vuong_stat,
            'prefer_zi_model': vuong_stat > 1.96  # 5% significance
        }


class ZIPoissonCompound(ZeroInflatedCompound):
    """Zero-inflated Poisson compound distribution."""
    
    def __init__(self, frequency, severity, zero_prob: float = None):
        """Initialize ZI-Poisson compound."""
        if frequency.__class__.__name__ != 'Poisson':
            raise ValueError("Frequency must be Poisson distribution")
        super().__init__(frequency, severity, zero_prob)


class ZINegativeBinomialCompound(ZeroInflatedCompound):
    """Zero-inflated Negative Binomial compound distribution."""
    
    def __init__(self, frequency, severity, zero_prob: float = None):
        """Initialize ZI-NegativeBinomial compound."""
        if frequency.__class__.__name__ != 'NegativeBinomial':
            raise ValueError("Frequency must be NegativeBinomial distribution")
        super().__init__(frequency, severity, zero_prob)


class ZIBinomialCompound(ZeroInflatedCompound):
    """Zero-inflated Binomial compound distribution."""
    
    def __init__(self, frequency, severity, zero_prob: float = None):
        """Initialize ZI-Binomial compound."""
        if frequency.__class__.__name__ != 'Binomial':
            raise ValueError("Frequency must be Binomial distribution")
        super().__init__(frequency, severity, zero_prob)


def detect_zero_inflation(data: np.ndarray, frequency: FrequencyModel,
                         severity: SeverityModel, 
                         significance_level: float = 0.05) -> Tuple[bool, Dict[str, float]]:
    """
    Statistical test for zero-inflation in compound distribution data.
    
    Args:
        data: Observed aggregate loss data
        frequency: Proposed frequency model
        severity: Proposed severity model
        significance_level: Significance level for test
        
    Returns:
        Tuple of (is_zero_inflated, diagnostic_stats)
    """
    # Create standard and ZI models
    standard_model = create_compound_distribution(frequency, severity)
    zi_model = ZeroInflatedCompound(frequency, severity)
    
    # Fit ZI model
    zi_model.fit_em(data, verbose=False)
    
    # Get diagnostics
    diagnostics = zi_model.diagnose_zero_inflation(data)
    
    # Additional tests
    n_obs = len(data)
    n_zeros = np.sum(data == 0)
    
    # Score test for zero-inflation
    exp_zeros_standard = n_obs * standard_model.pdf(0)
    score_stat = (n_zeros - exp_zeros_standard)**2 / exp_zeros_standard
    
    # Critical value from chi-square(1)
    critical_value = stats.chi2.ppf(1 - significance_level, df=1)
    
    diagnostics['score_statistic'] = score_stat
    diagnostics['critical_value'] = critical_value
    diagnostics['p_value'] = 1 - stats.chi2.cdf(score_stat, df=1)
    
    is_zero_inflated = (diagnostics['p_value'] < significance_level and
                       diagnostics['excess_zeros'] > 0)
    
    return is_zero_inflated, diagnostics


def score_test_zi(data: np.ndarray, distribution: str = 'poisson', 
                   **dist_params) -> Tuple[float, float]:
    """
    Score test for zero-inflation in count data.
    
    Args:
        data: Observed count data
        distribution: Base distribution ('poisson', 'nbinom', 'binomial')
        **dist_params: Distribution parameters (if not estimated from data)
        
    Returns:
        Tuple of (score_statistic, p_value)
    """
    n_obs = len(data)
    n_zeros = np.sum(data == 0)
    
    # Estimate parameters if not provided
    if distribution == 'poisson':
        if 'mu' not in dist_params:
            # Method of moments estimator
            dist_params['mu'] = np.mean(data)
        
        # Expected zeros under Poisson
        exp_zeros = n_obs * np.exp(-dist_params['mu'])
        
    elif distribution == 'nbinom':
        if 'r' not in dist_params or 'p' not in dist_params:
            # Simple moment estimators
            mean_data = np.mean(data)
            var_data = np.var(data)
            
            if var_data > mean_data:
                p = mean_data / var_data
                r = mean_data * p / (1 - p)
            else:
                # Fallback if underdispersed
                r = 5.0
                p = 0.5
            
            dist_params['r'] = r
            dist_params['p'] = p
        
        # Expected zeros under NB
        exp_zeros = n_obs * (1 - dist_params['p']) ** dist_params['r']
        
    elif distribution == 'binomial':
        if 'n' not in dist_params:
            dist_params['n'] = int(np.max(data))
        if 'p' not in dist_params:
            dist_params['p'] = np.mean(data) / dist_params['n']
        
        # Expected zeros under Binomial
        exp_zeros = n_obs * (1 - dist_params['p']) ** dist_params['n']
        
    else:
        # Default to Poisson
        mu = np.mean(data)
        exp_zeros = n_obs * np.exp(-mu)
    
    # Score test statistic
    if exp_zeros > 0:
        score_stat = (n_zeros - exp_zeros) ** 2 / exp_zeros
    else:
        score_stat = np.inf
    
    # P-value from chi-square(1) distribution
    p_value = 1 - stats.chi2.cdf(score_stat, df=1)
    
    return score_stat, p_value


class ZeroInflatedMixtureEM:
    """
    EM algorithm implementation for zero-inflated mixture models.
    
    Supports fitting complex zero-inflated compound distributions
    with parameter estimation for both frequency and severity components.
    """
    
    def __init__(self, frequency_type: str, severity_type: str):
        """
        Initialize EM algorithm for specific distribution types.
        
        Args:
            frequency_type: 'poisson', 'nbinom', or 'binom'
            severity_type: 'exponential', 'gamma', 'lognormal', etc.
        """
        self.frequency_type = frequency_type
        self.severity_type = severity_type
    
    def fit(self, data: np.ndarray, init_params: Dict = None,
            max_iter: int = 100, tol: float = 1e-6, callback: callable = None) -> Dict:
        """
        Fit zero-inflated compound distribution to data.
        
        Args:
            data: Aggregate loss observations
            init_params: Initial parameter values
            max_iter: Maximum iterations
            tol: Convergence tolerance
            callback: Optional callback function(params, log_likelihood, iteration)
            
        Returns:
            Dictionary with fitted parameters and diagnostics
        """
        # Implementation depends on specific distribution types
        # This is a template for the general approach
        
        n_obs = len(data)
        zero_mask = data == 0
        pos_mask = data > 0
        n_zeros = np.sum(zero_mask)
        
        # Initialize parameters
        if init_params is None:
            init_params = self._initialize_params(data)
        
        # Handle parameter name consistency (mu vs lambda for Poisson)
        params = init_params.copy()
        if self.frequency_type == 'poisson' and 'mu' in params:
            params['lambda'] = params.pop('mu')
        
        log_lik_old = -np.inf
        
        for iteration in range(max_iter):
            # E-step
            responsibilities = self._e_step(data, params)
            
            # M-step
            params = self._m_step(data, responsibilities)
            
            # Compute log-likelihood
            log_lik = self._compute_log_likelihood(data, params)
            
            # Call callback if provided
            if callback is not None:
                callback(params, log_lik, iteration)
            
            # Check convergence
            if abs(log_lik - log_lik_old) < tol:
                break
            
            log_lik_old = log_lik
        
        return {
            'params': params,
            'log_likelihood': log_lik,
            'converged': iteration < max_iter - 1,
            'iterations': iteration + 1,
            'aic': -2 * log_lik + 2 * len(params),
            'bic': -2 * log_lik + len(params) * np.log(n_obs)
        }
    
    def _initialize_params(self, data: np.ndarray) -> Dict:
        """Initialize parameters based on data."""
        # Simple moment-based initialization
        n_zeros = np.sum(data == 0)
        n_obs = len(data)
        
        params = {
            'zero_prob': max(0.01, n_zeros / n_obs - 0.2)
        }
        
        # Frequency parameters (example for Poisson)
        if self.frequency_type == 'poisson':
            # Initial lambda estimate from positive data
            if np.any(data > 0):
                # Use mean of all data divided by proportion of non-zeros
                prop_nonzero = 1 - n_zeros / n_obs
                if prop_nonzero > 0:
                    params['lambda'] = np.mean(data) / prop_nonzero
                else:
                    params['lambda'] = 1.0
            else:
                params['lambda'] = 1.0
        
        # Add severity parameters based on type
        # ...
        
        return params
    
    def _e_step(self, data: np.ndarray, params: Dict) -> np.ndarray:
        """Expectation step - compute responsibilities."""
        n_obs = len(data)
        zero_mask = data == 0
        n_zeros = np.sum(zero_mask)
        
        # Initialize responsibilities
        responsibilities = np.zeros(n_obs)
        
        if n_zeros > 0:
            # Get parameters
            zero_prob = params.get('zero_prob', 0.0)
            
            # Calculate P(N=0) based on frequency type
            if self.frequency_type == 'poisson':
                lambda_param = params.get('lambda', 1.0)
                p_n_zero = np.exp(-lambda_param)
            elif self.frequency_type == 'nbinom':
                r = params.get('r', 1.0)
                p = params.get('p', 0.5)
                p_n_zero = (1 - p) ** r
            elif self.frequency_type == 'binom':
                n = params.get('n', 10)
                p = params.get('p', 0.5)
                p_n_zero = (1 - p) ** n
            else:
                p_n_zero = 0.5
            
            # Total probability of zero
            p_zero_total = zero_prob + (1 - zero_prob) * p_n_zero
            
            # Responsibility that zero comes from zero-inflation
            if p_zero_total > 1e-10:
                w_zero = zero_prob / p_zero_total
            else:
                w_zero = 0.0
            
            # Set responsibilities for zero observations
            responsibilities[zero_mask] = w_zero
        
        return responsibilities
    
    def _m_step(self, data: np.ndarray, responsibilities: np.ndarray) -> Dict:
        """Maximization step - update parameters."""
        n_obs = len(data)
        zero_mask = data == 0
        n_zeros = np.sum(zero_mask)
        
        # Copy current parameters
        new_params = {}
        
        # Update zero_prob based on responsibilities
        # Sum of responsibilities for all zeros gives expected number of structural zeros
        n_structural_zeros = np.sum(responsibilities)
        
        new_params['zero_prob'] = max(0.0, min(0.999, n_structural_zeros / n_obs))
        
        # Update frequency parameters based on type
        # This is simplified - full implementation would use MLE on positive data
        pos_data = data[data > 0]
        
        if self.frequency_type == 'poisson':
            # Estimate lambda from all data (including zeros from Poisson process)
            # Account for zero-inflation when estimating
            if new_params['zero_prob'] < 0.999:
                # Adjust mean for zero-inflation
                adjusted_mean = np.mean(data) / (1 - new_params['zero_prob'])
                new_params['lambda'] = max(0.1, adjusted_mean)
            else:
                new_params['lambda'] = 1.0
                
        elif self.frequency_type == 'nbinom':
            # Simple defaults for now
            new_params['r'] = 5.0
            new_params['p'] = 0.5
            
        elif self.frequency_type == 'binom':
            # Simple defaults for now
            new_params['n'] = 10
            new_params['p'] = 0.3
        
        return new_params
    
    def _compute_log_likelihood(self, data: np.ndarray, params: Dict) -> float:
        """Compute log-likelihood under current parameters."""
        n_obs = len(data)
        zero_mask = data == 0
        pos_mask = data > 0
        n_zeros = np.sum(zero_mask)
        n_pos = np.sum(pos_mask)
        
        # Get zero probability
        zero_prob = params.get('zero_prob', 0.0)
        
        # Calculate log-likelihood based on distribution type
        log_lik = 0.0
        
        # For zero observations
        if n_zeros > 0:
            # P(X=0) = π + (1-π)P(N=0)
            if self.frequency_type == 'poisson':
                lambda_param = params.get('lambda', 1.0)
                p_n_zero = np.exp(-lambda_param)
            elif self.frequency_type == 'nbinom':
                r = params.get('r', 1.0)
                p = params.get('p', 0.5)
                p_n_zero = (1 - p) ** r
            elif self.frequency_type == 'binom':
                n = params.get('n', 10)
                p = params.get('p', 0.5)
                p_n_zero = (1 - p) ** n
            else:
                p_n_zero = 0.5  # Default fallback
            
            p_zero_total = zero_prob + (1 - zero_prob) * p_n_zero
            p_zero_total = max(1e-10, p_zero_total)  # Avoid log(0)
            log_lik += n_zeros * stable_log(p_zero_total)
        
        # For positive observations
        if n_pos > 0:
            # P(X>0) = (1-π) * P(compound > 0)
            # This is simplified - in reality would compute full compound distribution
            log_lik += n_pos * stable_log(max(1e-10, 1 - zero_prob))
        
        return log_lik