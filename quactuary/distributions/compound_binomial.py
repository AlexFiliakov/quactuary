"""
Compound binomial distributions for aggregate loss modeling.

This module provides compound distributions with binomial frequency component,
including analytical solutions where available and numerical approximations
for complex cases.
"""

import numpy as np
from scipy import stats, special
from typing import Union, Optional

from quactuary.distributions.compound import CompoundDistribution, SeriesExpansionMixin
from quactuary.utils.numerical import stable_exp, stable_log


class BinomialExponentialCompound(CompoundDistribution, SeriesExpansionMixin):
    """
    Compound Binomial-Exponential distribution.
    
    When N ~ Binomial(n, p) and Xi ~ Exponential(θ), the aggregate loss
    has an atom at 0 with probability (1-p)^n, and for S > 0,
    it follows a mixture of Gamma distributions.
    """
    
    def __init__(self, frequency, severity):
        """Initialize Binomial-Exponential compound distribution.
        
        Args:
            frequency: Binomial frequency distribution
            severity: Exponential severity distribution
        """
        super().__init__(frequency, severity)
        self.n = int(frequency._dist.args[0])
        self.p = frequency._dist.args[1]
        self.theta = severity._dist.kwds.get('scale', 1.0)
        
    def mean(self) -> float:
        """Calculate expected value of aggregate loss.
        
        Returns:
            float: Expected value E[S] = n * p * θ
        """
        return self.n * self.p * self.theta
    
    def var(self) -> float:
        """Calculate variance of aggregate loss.
        
        Returns:
            float: Variance Var[S] = n * p * θ² * (2 - p)
        """
        return self.n * self.p * self.theta**2 * (2 - self.p)
    
    def pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Probability density function.
        
        Args:
            x: Value(s) at which to evaluate PDF
            
        Returns:
            PDF value(s) at x
        """
        x_array = np.atleast_1d(x)
        result = np.zeros_like(x_array, dtype=float)
        
        # Probability of zero claims
        p_zero = (1 - self.p) ** self.n
        result[x_array == 0] = p_zero
        
        # For positive values, use series expansion
        pos_mask = x_array > 0
        if np.any(pos_mask):
            x_pos = x_array[pos_mask]
            
            for k in range(1, self.n + 1):
                # Binomial PMF
                p_k = stats.binom.pmf(k, self.n, self.p)
                # Sum of k exponentials is Gamma(k, θ)
                gamma_pdf = stats.gamma.pdf(x_pos, a=k, scale=self.theta)
                result[pos_mask] += p_k * gamma_pdf
                
                if p_k < 1e-10:
                    break
        
        return result[0] if np.isscalar(x) else result
    
    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """CDF of compound Binomial-Exponential."""
        x_array = np.atleast_1d(x)
        p_zero = (1 - self.p) ** self.n
        
        result = self._series_cdf(
            x_array, p_zero,
            lambda k: stats.binom.pmf(k, self.n, self.p),
            lambda x_pos, k: stats.gamma.cdf(x_pos, a=k, scale=self.theta),
            k_max=self.n
        )
        
        return result[0] if np.isscalar(x) else result
    
    def ppf(self, q: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Quantile function using numerical inversion."""
        q_array = np.atleast_1d(q)
        p_zero = (1 - self.p) ** self.n
        result = self._numerical_ppf(q_array, p_zero, self.mean, self.std)
        return result[0] if np.isscalar(q) else result
    
    def rvs(self, size: int = 1, random_state: Optional[int] = None) -> Union[float, np.ndarray]:
        """Generate random variates."""
        if random_state is not None:
            np.random.seed(random_state)
        
        n_claims = stats.binom.rvs(self.n, self.p, size=size)
        result = np.zeros(size)
        
        for i in range(size):
            if n_claims[i] > 0:
                result[i] = stats.expon.rvs(scale=self.theta, size=n_claims[i]).sum()
        
        return result[0] if size == 1 else result


class BinomialGammaCompound(CompoundDistribution, SeriesExpansionMixin):
    """
    Compound Binomial-Gamma distribution.
    
    When N ~ Binomial(n, p) and Xi ~ Gamma(α, β), the aggregate loss
    follows a mixture of Gamma distributions.
    """
    
    def __init__(self, frequency, severity):
        super().__init__(frequency, severity)
        self.n = int(frequency._dist.args[0])
        self.p = frequency._dist.args[1]
        self.alpha = severity._dist.args[0] if severity._dist.args else 1.0
        self.beta = 1.0 / severity._dist.kwds.get('scale', 1.0)
        
    def mean(self) -> float:
        """E[S] = n * p * α / β"""
        return self.n * self.p * self.alpha / self.beta
    
    def var(self) -> float:
        """Var[S] = n * p * (α/β)² * (1 - p + α)"""
        mean_sev = self.alpha / self.beta
        var_sev = self.alpha / (self.beta**2)
        return self.n * self.p * (var_sev + (1 - self.p) * mean_sev**2)
    
    def pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """PDF using series expansion for Binomial-Gamma compound."""
        x_array = np.atleast_1d(x)
        result = np.zeros_like(x_array, dtype=float)
        
        # Probability of zero claims
        p_zero = (1 - self.p) ** self.n
        result[x_array == 0] = p_zero
        
        # For positive values
        pos_mask = x_array > 0
        if np.any(pos_mask):
            x_pos = x_array[pos_mask]
            
            for k in range(1, self.n + 1):
                p_k = stats.binom.pmf(k, self.n, self.p)
                # Sum of k Gamma(α, β) is Gamma(k*α, β)
                gamma_pdf = stats.gamma.pdf(x_pos, a=k * self.alpha, scale=1/self.beta)
                result[pos_mask] += p_k * gamma_pdf
                
                if p_k < 1e-10:
                    break
        
        return result[0] if np.isscalar(x) else result
    
    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """CDF of compound Binomial-Gamma."""
        x_array = np.atleast_1d(x)
        p_zero = (1 - self.p) ** self.n
        
        result = self._series_cdf(
            x_array, p_zero,
            lambda k: stats.binom.pmf(k, self.n, self.p),
            lambda x_pos, k: stats.gamma.cdf(x_pos, a=k * self.alpha, scale=1/self.beta),
            k_max=self.n
        )
        
        return result[0] if np.isscalar(x) else result
    
    def ppf(self, q: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Quantile function."""
        q_array = np.atleast_1d(q)
        p_zero = (1 - self.p) ** self.n
        result = self._numerical_ppf(q_array, p_zero, self.mean, self.std)
        return result[0] if np.isscalar(q) else result
    
    def rvs(self, size: int = 1, random_state: Optional[int] = None) -> Union[float, np.ndarray]:
        """Generate random variates."""
        if random_state is not None:
            np.random.seed(random_state)
        
        n_claims = stats.binom.rvs(self.n, self.p, size=size)
        result = np.zeros(size)
        
        for i in range(size):
            if n_claims[i] > 0:
                result[i] = stats.gamma.rvs(
                    a=self.alpha, 
                    scale=1/self.beta, 
                    size=n_claims[i]
                ).sum()
        
        return result[0] if size == 1 else result


class BinomialLognormalCompound(CompoundDistribution):
    """
    Compound Binomial-Lognormal distribution.
    
    Uses the method of moments approximation for the sum of lognormals.
    """
    
    def __init__(self, frequency, severity):
        super().__init__(frequency, severity)
        self.n = int(frequency._dist.args[0])
        self.p = frequency._dist.args[1]
        
        # Extract lognormal parameters
        # scipy uses s=sigma, scale=exp(mu)
        self.sigma = severity._dist.args[0] if severity._dist.args else 1.0
        self.scale = severity._dist.kwds.get('scale', 1.0)
        self.mu = np.log(self.scale)
        
    def mean(self) -> float:
        """E[S] = n * p * E[X]"""
        mean_sev = np.exp(self.mu + self.sigma**2 / 2)
        return self.n * self.p * mean_sev
    
    def var(self) -> float:
        """Var[S] = n * p * (Var[X] + (1-p) * E[X]²)"""
        mean_sev = np.exp(self.mu + self.sigma**2 / 2)
        var_sev = (np.exp(self.sigma**2) - 1) * mean_sev**2
        return self.n * self.p * (var_sev + (1 - self.p) * mean_sev**2)
    
    def _fenton_wilkinson_params(self, k):
        """Calculate Fenton-Wilkinson approximation parameters for sum of k lognormals."""
        if k == 0:
            return None, None
        
        # For sum of k i.i.d. lognormals
        mean_sum = k * np.exp(self.mu + self.sigma**2 / 2)
        var_sum = k * (np.exp(self.sigma**2) - 1) * np.exp(2 * self.mu + self.sigma**2)
        
        # Fenton-Wilkinson parameters
        sigma_sum = np.sqrt(np.log(1 + var_sum / mean_sum**2))
        mu_sum = np.log(mean_sum) - sigma_sum**2 / 2
        
        return mu_sum, sigma_sum
    
    def pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """PDF using Fenton-Wilkinson approximation."""
        x_array = np.atleast_1d(x)
        result = np.zeros_like(x_array, dtype=float)
        
        # Probability of zero claims
        p_zero = (1 - self.p) ** self.n
        result[x_array == 0] = p_zero
        
        # For positive values
        pos_mask = x_array > 0
        if np.any(pos_mask):
            x_pos = x_array[pos_mask]
            
            for k in range(1, self.n + 1):
                p_k = stats.binom.pmf(k, self.n, self.p)
                mu_k, sigma_k = self._fenton_wilkinson_params(k)
                
                if mu_k is not None:
                    # Approximate sum as lognormal
                    lognorm_pdf = stats.lognorm.pdf(x_pos, s=sigma_k, scale=np.exp(mu_k))
                    result[pos_mask] += p_k * lognorm_pdf
                
                if p_k < 1e-10:
                    break
        
        return result[0] if np.isscalar(x) else result
    
    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """CDF using Fenton-Wilkinson approximation."""
        x_array = np.atleast_1d(x)
        result = np.zeros_like(x_array, dtype=float)
        
        # Handle negative values
        result[x_array < 0] = 0.0
        
        # Probability of zero claims
        p_zero = (1 - self.p) ** self.n
        result[x_array >= 0] = p_zero
        
        # For positive values
        pos_mask = x_array > 0
        if np.any(pos_mask):
            x_pos = x_array[pos_mask]
            
            for k in range(1, self.n + 1):
                p_k = stats.binom.pmf(k, self.n, self.p)
                mu_k, sigma_k = self._fenton_wilkinson_params(k)
                
                if mu_k is not None:
                    lognorm_cdf = stats.lognorm.cdf(x_pos, s=sigma_k, scale=np.exp(mu_k))
                    result[pos_mask] += p_k * lognorm_cdf
                
                if p_k < 1e-10:
                    break
        
        return result[0] if np.isscalar(x) else result
    
    def ppf(self, q: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Quantile function using numerical inversion."""
        q_array = np.atleast_1d(q)
        p_zero = (1 - self.p) ** self.n
        
        from scipy.optimize import brentq
        result = np.zeros_like(q_array, dtype=float)
        
        for i, qi in enumerate(q_array):
            if qi <= p_zero:
                result[i] = 0.0
            else:
                # Initial guess based on mean and std
                mean = self.mean()
                std = self.std()
                x0 = max(0.01, mean + std * stats.norm.ppf(qi))
                
                try:
                    result[i] = brentq(
                        lambda x: self.cdf(x) - qi,
                        0.01, x0 * 10,
                        xtol=1e-6
                    )
                except:
                    result[i] = brentq(
                        lambda x: self.cdf(x) - qi,
                        0.01, x0 * 100,
                        xtol=1e-6
                    )
        
        return result[0] if np.isscalar(q) else result
    
    def rvs(self, size: int = 1, random_state: Optional[int] = None) -> Union[float, np.ndarray]:
        """Generate random variates."""
        if random_state is not None:
            np.random.seed(random_state)
        
        n_claims = stats.binom.rvs(self.n, self.p, size=size)
        result = np.zeros(size)
        
        for i in range(size):
            if n_claims[i] > 0:
                result[i] = stats.lognorm.rvs(
                    s=self.sigma, 
                    scale=self.scale, 
                    size=n_claims[i]
                ).sum()
        
        return result[0] if size == 1 else result


class PanjerBinomialRecursion:
    """
    Panjer recursion for compound distributions with binomial frequency.
    
    Provides exact computation of the aggregate loss distribution
    when severity distribution has finite support.
    """
    
    def __init__(self, n: int, p: float, severity_pmf: dict):
        """
        Initialize Panjer recursion for binomial frequency.
        
        Args:
            n: Number of trials in binomial distribution
            p: Success probability
            severity_pmf: Dictionary mapping severity values to probabilities
        """
        self.n = n
        self.p = p
        self.severity_pmf = severity_pmf
        self.severity_values = sorted(severity_pmf.keys())
        
    def compute_aggregate_pmf(self, max_value: int = None) -> dict:
        """
        Compute PMF of aggregate loss using Panjer recursion.
        
        Args:
            max_value: Maximum aggregate value to compute
            
        Returns:
            Dictionary mapping aggregate values to probabilities
        """
        if max_value is None:
            max_value = self.n * max(self.severity_values)
        
        # Initialize
        g = {0: (1 - self.p) ** self.n}
        
        # Panjer recursion coefficients for binomial
        a = -self.p / (1 - self.p)
        b = (self.n + 1) * self.p / (1 - self.p)
        
        # Compute recursively
        for s in range(1, max_value + 1):
            g[s] = 0
            for x in self.severity_values:
                if x <= s and (s - x) in g:
                    coeff = (a + b * x / s)
                    g[s] += coeff * self.severity_pmf[x] * g[s - x]
        
        return g