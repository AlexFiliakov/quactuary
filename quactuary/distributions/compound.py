"""
Compound distributions for aggregate loss modeling.

This module provides analytical and simulated compound distributions that combine
frequency and severity distributions for aggregate loss calculations. Where possible,
analytical solutions are used for improved performance and accuracy.

Examples:
    >>> from quactuary.distributions.frequency import Poisson
    >>> from quactuary.distributions.severity import Exponential
    >>> from quactuary.distributions.compound import CompoundDistribution
    >>> 
    >>> freq = Poisson(mu=5.0)
    >>> sev = Exponential(scale=1000.0)
    >>> compound = CompoundDistribution.create(freq, sev)
    >>> 
    >>> # Automatically uses analytical solution if available
    >>> mean_loss = compound.mean()
    >>> samples = compound.rvs(size=1000)
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
from scipy import special, stats
from scipy.integrate import quad

from quactuary.distributions.frequency import FrequencyModel
from quactuary.distributions.severity import SeverityModel


class CompoundDistribution(ABC):
    """
    Base class for compound distributions combining frequency and severity.
    
    Attributes:
        frequency: Frequency distribution model
        severity: Severity distribution model
    """
    
    # Registry of analytical solutions
    _analytical_registry: Dict[Tuple[str, str], Type['CompoundDistribution']] = {}
    
    def __init__(self, frequency: FrequencyModel, severity: SeverityModel):
        """
        Initialize compound distribution.
        
        Args:
            frequency: Distribution for number of claims
            severity: Distribution for claim amounts
        """
        self.frequency = frequency
        self.severity = severity
    
    @classmethod
    def register_analytical(cls, freq_type: str, sev_type: str):
        """Decorator to register analytical compound distributions."""
        def decorator(subclass):
            cls._analytical_registry[(freq_type, sev_type)] = subclass
            return subclass
        return decorator
    
    @classmethod
    def create(cls, frequency: FrequencyModel, severity: SeverityModel) -> 'CompoundDistribution':
        """
        Factory method to create appropriate compound distribution.
        
        Automatically selects analytical solution if available, otherwise
        returns simulated compound distribution.
        
        Args:
            frequency: Frequency distribution
            severity: Severity distribution
            
        Returns:
            CompoundDistribution instance
        """
        freq_type = type(frequency).__name__
        sev_type = type(severity).__name__
        
        # Check for analytical solution
        analytical_class = cls._analytical_registry.get((freq_type, sev_type))
        if analytical_class:
            return analytical_class(frequency, severity)
        
        # Default to simulation
        return SimulatedCompound(frequency, severity)
    
    @abstractmethod
    def mean(self) -> float:
        """Calculate mean of compound distribution."""
        pass
    
    @abstractmethod
    def var(self) -> float:
        """Calculate variance of compound distribution."""
        pass
    
    @abstractmethod
    def std(self) -> float:
        """Calculate standard deviation of compound distribution."""
        pass
    
    @abstractmethod
    def pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Probability density function.
        
        Args:
            x: Values at which to evaluate PDF
            
        Returns:
            PDF values
        """
        pass
    
    @abstractmethod
    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Cumulative distribution function.
        
        Args:
            x: Values at which to evaluate CDF
            
        Returns:
            CDF values
        """
        pass
    
    @abstractmethod
    def ppf(self, q: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Percent point function (inverse CDF).
        
        Args:
            q: Quantiles
            
        Returns:
            Quantile values
        """
        pass
    
    @abstractmethod
    def rvs(self, size: int = 1, random_state: Optional[int] = None) -> Union[float, np.ndarray]:
        """
        Random variate sampling.
        
        Args:
            size: Number of samples
            random_state: Random seed
            
        Returns:
            Random samples
        """
        pass
    
    def has_analytical_solution(self) -> bool:
        """Check if this compound distribution has an analytical solution."""
        return not isinstance(self, SimulatedCompound)


class AnalyticalCompound(CompoundDistribution):
    """Base class for compound distributions with analytical solutions."""
    
    def std(self) -> float:
        """Standard deviation as square root of variance."""
        return np.sqrt(self.var())


class SimulatedCompound(CompoundDistribution):
    """
    Compound distribution using Monte Carlo simulation.
    
    This is the fallback when no analytical solution is available.
    """
    
    def __init__(self, frequency: FrequencyModel, severity: SeverityModel):
        super().__init__(frequency, severity)
        self._cache_size = 10000
        self._cache = None
        self._cache_seed = None
    
    def _generate_cache(self, random_state: Optional[int] = None):
        """Generate cache of simulated values."""
        if random_state is not None:
            np.random.seed(random_state)
        
        samples = []
        for _ in range(self._cache_size):
            n_claims = self.frequency.rvs()
            if n_claims > 0:
                losses = self.severity.rvs(size=n_claims)
                total_loss = np.sum(losses)
            else:
                total_loss = 0.0
            samples.append(total_loss)
        
        self._cache = np.array(samples)
        self._cache_seed = random_state
    
    def mean(self) -> float:
        """Mean = E[N] * E[X]."""
        freq_mean = self.frequency.mean()
        sev_mean = self.severity.mean()
        return freq_mean * sev_mean
    
    def var(self) -> float:
        """Var = E[N] * Var[X] + Var[N] * E[X]²."""
        freq_mean = self.frequency.mean()
        freq_var = self.frequency.var()
        sev_mean = self.severity.mean()
        sev_var = self.severity.var()
        
        return freq_mean * sev_var + freq_var * sev_mean**2
    
    def std(self) -> float:
        """Standard deviation."""
        return np.sqrt(self.var())
    
    def pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Estimate PDF using kernel density estimation on simulated data."""
        if self._cache is None:
            self._generate_cache()
        
        kde = stats.gaussian_kde(self._cache)
        result = kde(x)
        
        # Return scalar if input was scalar
        if np.isscalar(x):
            return float(result[0])
        return result
    
    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Estimate CDF using empirical distribution from simulated data."""
        if self._cache is None:
            self._generate_cache()
        
        x_array = np.atleast_1d(x)
        result = np.zeros_like(x_array, dtype=float)
        
        for i, xi in enumerate(x_array):
            result[i] = np.mean(self._cache <= xi)
        
        return result[0] if np.isscalar(x) else result
    
    def ppf(self, q: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Estimate quantiles using empirical distribution."""
        if self._cache is None:
            self._generate_cache()
        
        result = np.percentile(self._cache, np.array(q) * 100)
        return float(result) if np.isscalar(q) else result
    
    def rvs(self, size: int = 1, random_state: Optional[int] = None) -> Union[float, np.ndarray]:
        """Generate random samples via simulation."""
        if random_state is not None:
            np.random.seed(random_state)
        
        samples = []
        for _ in range(size):
            n_claims = self.frequency.rvs()
            if n_claims > 0:
                losses = self.severity.rvs(size=n_claims)
                total_loss = np.sum(losses)
            else:
                total_loss = 0.0
            samples.append(total_loss)
        
        result = np.array(samples)
        return result[0] if size == 1 else result


# Analytical Compound Distributions

@CompoundDistribution.register_analytical('Poisson', 'Exponential')
class PoissonExponentialCompound(AnalyticalCompound):
    """
    Compound Poisson-Exponential distribution.
    
    When N ~ Poisson(λ) and Xi ~ Exponential(1/θ), the aggregate loss
    S = Σ Xi follows a compound distribution with special properties.
    
    The aggregate loss has an atom at 0 with probability e^(-λ).
    For S > 0, it follows a Gamma distribution.
    """
    
    def __init__(self, frequency, severity):
        super().__init__(frequency, severity)
        self.lam = frequency.mu  # Poisson parameter
        self.theta = severity.scale  # Exponential scale parameter
        
    def mean(self) -> float:
        """E[S] = λ * θ"""
        if not hasattr(self, '_mean_cache'):
            self._mean_cache = self.lam * self.theta
        return self._mean_cache
    
    def var(self) -> float:
        """Var[S] = λ * θ² * 2"""
        if not hasattr(self, '_var_cache'):
            self._var_cache = self.lam * self.theta**2 * 2
        return self._var_cache
    
    def pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        PDF of compound Poisson-Exponential.
        
        P(S = 0) = e^(-λ)
        For x > 0: f(x) = e^(-λ) * Σ(k=1 to ∞) λ^k/k! * Gamma(k, θ).pdf(x)
        """
        x_array = np.atleast_1d(x)
        result = np.zeros_like(x_array, dtype=float)
        
        # Probability of zero claims
        p_zero = np.exp(-self.lam)
    
        # Set atom at zero
        zero_mask = x_array == 0
        result[zero_mask] = p_zero
        
        # For positive values, use series expansion
        positive_mask = x_array > 0
        if np.any(positive_mask):
            x_pos = x_array[positive_mask]
            
            # Truncate series at reasonable k
            k_max = max(50, int(self.lam * 3))
            
            for k in range(1, k_max + 1):
                # Poisson probability of k claims
                p_k = stats.poisson.pmf(k, self.lam)
                
                # Gamma(k, scale=θ) PDF
                gamma_pdf = stats.gamma.pdf(x_pos, a=k, scale=self.theta)
                
                result[positive_mask] += p_k * gamma_pdf
                
                # Early termination if contribution is negligible
                if np.max(p_k * gamma_pdf) < 1e-10:
                    break
        
        return result[0] if np.isscalar(x) else result
    
    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        CDF of compound Poisson-Exponential.
        
        F(x) = e^(-λ) + Σ(k=1 to ∞) P(N=k) * P(Gamma(k,θ) ≤ x)
        """
        x_array = np.atleast_1d(x)
        result = np.zeros_like(x_array, dtype=float)
        
        # Probability of zero claims
        p_zero = np.exp(-self.lam)
        
        # Start with P(S = 0) for all x ≥ 0
        result[x_array >= 0] = p_zero
        
        # Add contributions from positive number of claims
        positive_mask = x_array > 0
        if np.any(positive_mask):
            x_pos = x_array[positive_mask]
            
            k_max = max(50, int(self.lam * 3))
            
            for k in range(1, k_max + 1):
                p_k = stats.poisson.pmf(k, self.lam)
                gamma_cdf = stats.gamma.cdf(x_pos, a=k, scale=self.theta)
                
                result[positive_mask] += p_k * gamma_cdf
                
                if p_k < 1e-10:
                    break
        
        return result[0] if np.isscalar(x) else result
    
    def ppf(self, q: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Quantile function using numerical inversion."""
        q_array = np.atleast_1d(q)
        result = np.zeros_like(q_array, dtype=float)
        
        p_zero = np.exp(-self.lam)
        
        for i, qi in enumerate(q_array):
            if qi <= p_zero:
                result[i] = 0.0
            else:
                # Use numerical root finding
                from scipy.optimize import brentq

                # Adjusted quantile for positive part
                q_adj = (qi - p_zero) / (1 - p_zero)
                
                # Initial guess based on normal approximation
                mean_pos = self.mean()
                std_pos = self.std()
                x0 = mean_pos + std_pos * stats.norm.ppf(q_adj)
                x0 = max(0.01, x0)
                
                # Find quantile - ensure we have a proper bracket
                # Start with a reasonable range
                x_lower = 0.01
                x_upper = max(x0 * 10, 1000 * self.theta)
                
                # Expand bracket if needed
                f_lower = self.cdf(x_lower) - qi
                f_upper = self.cdf(x_upper) - qi
                
                while f_lower * f_upper > 0 and x_upper < 1e6 * self.theta:
                    x_upper *= 10
                    f_upper = self.cdf(x_upper) - qi
                
                try:
                    result[i] = brentq(
                        lambda x: self.cdf(x) - qi,
                        x_lower, x_upper,
                        xtol=1e-6
                    )
                except:
                    # Fallback to larger search range
                    result[i] = brentq(
                        lambda x: self.cdf(x) - qi,
                        0.01, x0 * 100,
                        xtol=1e-6
                    )
        
        return result[0] if np.isscalar(q) else result
    
    def rvs(self, size: int = 1, random_state: Optional[int] = None) -> Union[float, np.ndarray]:
        """
        Generate random variates.
        
        Direct simulation: sample N ~ Poisson(λ), then sum N ~ Exponential(θ).
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        # Sample number of claims
        n_claims = stats.poisson.rvs(self.lam, size=size)
        
        # Generate aggregate losses
        result = np.zeros(size)
        for i in range(size):
            if n_claims[i] > 0:
                result[i] = stats.expon.rvs(scale=self.theta, size=n_claims[i]).sum()
        
        return result[0] if size == 1 else result


@CompoundDistribution.register_analytical('Poisson', 'Gamma')
class PoissonGammaCompound(AnalyticalCompound):
    """
    Compound Poisson-Gamma distribution (Tweedie distribution).
    
    When N ~ Poisson(λ) and Xi ~ Gamma(α, β), the aggregate loss follows
    a Tweedie distribution for specific parameter relationships.
    
    The Tweedie distribution is characterized by power parameter p ∈ (1, 2),
    mean μ, and dispersion φ.
    """
    
    def __init__(self, frequency, severity):
        super().__init__(frequency, severity)
        self.lam = frequency.mu  # Poisson parameter  
        self.alpha = severity.a  # Gamma shape
        self.beta = 1.0 / severity.scale  # Gamma rate = 1/scale
        
        # Tweedie parameters
        self.p = (self.alpha + 2) / (self.alpha + 1)  # Power parameter
        self.mu = self.lam * self.alpha / self.beta  # Mean
        self.phi = (self.beta ** (self.p - 1)) / (self.lam ** (self.p - 2) * (2 - self.p))
    
    def mean(self) -> float:
        """E[S] = λ * α / β"""
        if not hasattr(self, '_mean_cache'):
            self._mean_cache = self.lam * self.alpha / self.beta
        return self._mean_cache
    
    def var(self) -> float:
        """Var[S] = λ * α * (α + 1) / β²"""
        if not hasattr(self, '_var_cache'):
            self._var_cache = self.lam * self.alpha * (self.alpha + 1) / (self.beta ** 2)
        return self._var_cache
    
    def _tweedie_pdf(self, y: np.ndarray) -> np.ndarray:
        """
        Compute Tweedie PDF using series expansion.
        
        For numerical stability, we use a simplified approximation for the PDF.
        """
        result = np.zeros_like(y, dtype=float)
        
        # Handle y = 0 case
        zero_mask = y == 0
        if np.any(zero_mask):
            result[zero_mask] = np.exp(-self.lam)
        
        # For y > 0, use gamma approximation
        pos_mask = y > 0
        if np.any(pos_mask):
            y_pos = y[pos_mask]
            
            # Use the fact that for Poisson-Gamma, we can approximate
            # the PDF using a shifted gamma distribution
            # This is more stable than the full Tweedie series expansion
            
            # Expected number of claims given y > 0
            lambda_cond = self.lam * (1 - np.exp(-self.lam))
            
            # Shape and scale for the conditional distribution
            shape_cond = lambda_cond * self.alpha
            scale_cond = 1 / self.beta
            
            # Approximate PDF using gamma
            if shape_cond > 0:
                result[pos_mask] = (1 - np.exp(-self.lam)) * stats.gamma.pdf(
                    y_pos, a=shape_cond, scale=scale_cond
                )
            else:
                result[pos_mask] = 0.0
        
        return result
    
    def _compute_tweedie_weight(self, k: int, y: np.ndarray) -> np.ndarray:
        """Compute weight for k-th term in Tweedie series."""
        # Simplified weight calculation with numerical stability
        alpha_star = (2 - self.p) / (1 - self.p)
        
        # Use stable computation to avoid overflow
        if k == 0:
            return np.ones_like(y)
        
        # Clip values to prevent overflow
        log_term1 = k * np.log(np.clip(self.mu ** (2 - self.p) / (self.phi * (2 - self.p)), 1e-10, 1e10))
        log_factorial = special.loggamma(k + 1)
        log_term3 = k * alpha_star * np.log(max(k, 1))
        
        log_weight = log_term1 - log_factorial - log_term3
        
        # Clip log_weight to prevent overflow/underflow
        log_weight = np.clip(log_weight, -100, 100)
        
        return np.exp(log_weight)
    
    def pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """PDF of compound Poisson-Gamma (Tweedie)."""
        x_array = np.atleast_1d(x)
        
        # Use series expansion for exact evaluation
        result = self._tweedie_pdf(x_array)
        
        return result[0] if np.isscalar(x) else result
    
    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """CDF of compound Poisson-Gamma."""
        x_array = np.atleast_1d(x)
        result = np.zeros_like(x_array, dtype=float)
        
        # Probability of zero claims
        p_zero = np.exp(-self.lam)
        
        # For x >= 0, start with P(S = 0)
        result[x_array >= 0] = p_zero
        
        # Add contributions from positive claims
        pos_mask = x_array > 0
        if np.any(pos_mask):
            x_pos = x_array[pos_mask]
            
            # Use series expansion
            k_max = max(100, int(self.lam * 4))
            
            for k in range(1, k_max + 1):
                p_k = stats.poisson.pmf(k, self.lam)
                
                # Sum of k Gamma(α, β) random variables is Gamma(kα, β)
                gamma_cdf = stats.gamma.cdf(x_pos, a=k * self.alpha, scale=1/self.beta)
                
                result[pos_mask] += p_k * gamma_cdf
                
                if p_k < 1e-12:
                    break
        
        return result[0] if np.isscalar(x) else result
    
    def ppf(self, q: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Quantile function."""
        q_array = np.atleast_1d(q)
        result = np.zeros_like(q_array, dtype=float)
        
        p_zero = np.exp(-self.lam)
        
        for i, qi in enumerate(q_array):
            if qi <= p_zero:
                result[i] = 0.0
            else:
                # Numerical inversion
                from scipy.optimize import brentq

                # Initial guess
                mean = self.mean()
                std = self.std()
                x0 = mean + std * stats.norm.ppf(qi)
                x0 = max(0.01, x0)
                
                try:
                    # Check bounds to ensure different signs
                    f_low = self.cdf(0.01) - qi
                    f_high = self.cdf(x0 * 10) - qi
                    
                    if f_low * f_high < 0:
                        result[i] = brentq(
                            lambda x: self.cdf(x) - qi,
                            0.01, x0 * 10,
                            xtol=1e-6
                        )
                    else:
                        # Expand search range
                        result[i] = brentq(
                            lambda x: self.cdf(x) - qi,
                            0.01, max(x0 * 100, mean + 5*std),
                            xtol=1e-6
                        )
                except:
                    # Fallback approximation
                    result[i] = x0
        
        return result[0] if np.isscalar(q) else result
    
    def rvs(self, size: int = 1, random_state: Optional[int] = None) -> Union[float, np.ndarray]:
        """Generate random variates."""
        if random_state is not None:
            np.random.seed(random_state)
        
        # Direct simulation
        n_claims = stats.poisson.rvs(self.lam, size=size)
        
        result = np.zeros(size)
        for i in range(size):
            if n_claims[i] > 0:
                # Sum of Gamma variables
                result[i] = stats.gamma.rvs(
                    a=self.alpha * n_claims[i], 
                    scale=1/self.beta
                )
        
        return result[0] if size == 1 else result


@CompoundDistribution.register_analytical('Geometric', 'Exponential')
class GeometricExponentialCompound(AnalyticalCompound):
    """
    Compound Geometric-Exponential distribution.
    
    When N ~ Geometric(p) and Xi ~ Exponential(θ), the aggregate loss
    S = Σ Xi also follows an Exponential distribution with parameter (1-p)/θ.
    
    Note: This uses the "number of failures" parameterization where
    P(N = k) = (1-p)^k * p for k = 0, 1, 2, ...
    """
    
    def __init__(self, frequency, severity):
        super().__init__(frequency, severity)
        self.p = frequency.p  # Geometric success probability
        self.theta = severity.scale  # Exponential scale
        
        # Aggregate exponential parameter
        self.aggregate_scale = self.theta / (1 - self.p) if self.p < 1 else float('inf')
        
        # Cache moments
        self._mean_cache = None
        self._var_cache = None
    
    def mean(self) -> float:
        """E[S] = θ / p"""
        if self._mean_cache is None:
            if self.p > 0:
                self._mean_cache = self.theta / self.p
            else:
                self._mean_cache = float('inf')
        return self._mean_cache
    
    def var(self) -> float:
        """Var[S] = θ² * (2 - p) / p²"""
        if self._var_cache is None:
            if self.p > 0:
                self._var_cache = self.theta**2 * (2 - self.p) / (self.p**2)
            else:
                self._var_cache = float('inf')
        return self._var_cache
    
    def pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        PDF of compound Geometric-Exponential.
        
        The aggregate follows Exponential(scale = θ/(1-p)) for p < 1.
        """
        x_array = np.atleast_1d(x)
        
        if self.p >= 1:
            # Degenerate case - no claims
            result = np.zeros_like(x_array)
            result[x_array == 0] = 1.0
        else:
            # Use exponential PDF
            result = stats.expon.pdf(x_array, scale=self.aggregate_scale)
        
        return result[0] if np.isscalar(x) else result
    
    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """CDF of compound Geometric-Exponential."""
        x_array = np.atleast_1d(x)
        
        if self.p >= 1:
            # Degenerate case
            result = np.ones_like(x_array, dtype=float)
            result[x_array < 0] = 0.0
        else:
            # Use exponential CDF
            result = stats.expon.cdf(x_array, scale=self.aggregate_scale)
        
        return result[0] if np.isscalar(x) else result
    
    def ppf(self, q: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Quantile function."""
        q_array = np.atleast_1d(q)
        
        if self.p >= 1:
            # Degenerate case
            result = np.zeros_like(q_array)
        else:
            # Use exponential PPF
            result = stats.expon.ppf(q_array, scale=self.aggregate_scale)
        
        return result[0] if np.isscalar(q) else result
    
    def rvs(self, size: int = 1, random_state: Optional[int] = None) -> Union[float, np.ndarray]:
        """Generate random variates."""
        if random_state is not None:
            np.random.seed(random_state)
        
        if self.p >= 1:
            # Degenerate case
            return np.zeros(size) if size > 1 else 0.0
        else:
            # Direct sampling from exponential
            return stats.expon.rvs(scale=self.aggregate_scale, size=size)


@CompoundDistribution.register_analytical('NegativeBinomial', 'Gamma')
class NegativeBinomialGammaCompound(AnalyticalCompound):
    """
    Compound NegativeBinomial-Gamma distribution.
    
    When N ~ NegativeBinomial(r, p) and Xi ~ Gamma(α, β), the aggregate loss
    S = Σ Xi follows a Generalized Beta Prime distribution (Beta of the second kind).
    
    For computational efficiency, we use moment-based approximations and
    series expansions.
    """
    
    def __init__(self, frequency, severity):
        super().__init__(frequency, severity)
        self.r = frequency.n  # NegativeBinomial number of failures
        self.p = frequency.p  # NegativeBinomial success probability
        self.alpha = severity.a  # Gamma shape
        self.beta = 1.0 / severity.scale  # Gamma rate
        
        # Cache moments
        self._mean_cache = None
        self._var_cache = None
    
    def mean(self) -> float:
        """E[S] = r * (1-p) / p * α / β"""
        if self._mean_cache is None:
            freq_mean = self.r * (1 - self.p) / self.p
            sev_mean = self.alpha / self.beta
            self._mean_cache = freq_mean * sev_mean
        return self._mean_cache
    
    def var(self) -> float:
        """Var[S] = E[N] * Var[X] + Var[N] * E[X]²"""
        if self._var_cache is None:
            freq_mean = self.r * (1 - self.p) / self.p
            freq_var = self.r * (1 - self.p) / (self.p ** 2)
            sev_mean = self.alpha / self.beta
            sev_var = self.alpha / (self.beta ** 2)
            
            self._var_cache = freq_mean * sev_var + freq_var * sev_mean**2
        return self._var_cache
    
    def pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        PDF using series expansion.
        
        The exact distribution is complex, so we use a series representation
        based on the negative binomial expansion.
        """
        x_array = np.atleast_1d(x)
        result = np.zeros_like(x_array, dtype=float)
        
        # Handle x = 0
        zero_mask = x_array == 0
        if np.any(zero_mask):
            # P(S = 0) = P(N = 0) = p^r
            result[zero_mask] = self.p ** self.r
        
        # For x > 0, use series expansion
        pos_mask = x_array > 0
        if np.any(pos_mask):
            x_pos = x_array[pos_mask]
            
            # Truncate series
            k_max = max(100, int(self.mean() / self.alpha * 3))
            
            for k in range(1, k_max + 1):
                # NegativeBinomial probability
                p_k = stats.nbinom.pmf(k, self.r, self.p)
                
                # Sum of k Gamma(α, β) is Gamma(kα, β)
                gamma_pdf = stats.gamma.pdf(x_pos, a=k * self.alpha, scale=1/self.beta)
                
                result[pos_mask] += p_k * gamma_pdf
                
                if np.max(p_k * gamma_pdf) < 1e-12:
                    break
        
        return result[0] if np.isscalar(x) else result
    
    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """CDF using series expansion."""
        x_array = np.atleast_1d(x)
        result = np.zeros_like(x_array, dtype=float)
        
        # P(S = 0) = p^r
        p_zero = self.p ** self.r
        
        # For x >= 0, start with P(S = 0)
        result[x_array >= 0] = p_zero
        
        # Add positive contributions
        pos_mask = x_array > 0
        if np.any(pos_mask):
            x_pos = x_array[pos_mask]
            
            k_max = max(100, int(self.mean() / self.alpha * 3))
            
            for k in range(1, k_max + 1):
                p_k = stats.nbinom.pmf(k, self.r, self.p)
                gamma_cdf = stats.gamma.cdf(x_pos, a=k * self.alpha, scale=1/self.beta)
                
                result[pos_mask] += p_k * gamma_cdf
                
                if p_k < 1e-12:
                    break
        
        return result[0] if np.isscalar(x) else result
    
    def ppf(self, q: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Quantile function using numerical inversion."""
        q_array = np.atleast_1d(q)
        result = np.zeros_like(q_array, dtype=float)
        
        p_zero = self.p ** self.r
        
        for i, qi in enumerate(q_array):
            if qi <= p_zero:
                result[i] = 0.0
            else:
                from scipy.optimize import brentq

                # Initial guess using normal approximation
                mean = self.mean()
                std = self.std()
                x0 = mean + std * stats.norm.ppf(qi)
                x0 = max(0.01, x0)
                
                try:
                    result[i] = brentq(
                        lambda x: self.cdf(x) - qi,
                        0, x0 * 10,
                        xtol=1e-6
                    )
                except:
                    result[i] = brentq(
                        lambda x: self.cdf(x) - qi,
                        0, x0 * 100,
                        xtol=1e-6
                    )
        
        return result[0] if np.isscalar(q) else result
    
    def rvs(self, size: int = 1, random_state: Optional[int] = None) -> Union[float, np.ndarray]:
        """Generate random variates via direct simulation."""
        if random_state is not None:
            np.random.seed(random_state)
        
        # Sample number of claims
        n_claims = stats.nbinom.rvs(self.r, self.p, size=size)
        
        result = np.zeros(size)
        for i in range(size):
            if n_claims[i] > 0:
                # Sum of Gamma variables
                result[i] = stats.gamma.rvs(
                    a=self.alpha * n_claims[i],
                    scale=1/self.beta
                )
        
        return result[0] if size == 1 else result


# Approximation Methods

class PanjerRecursion:
    """
    Panjer recursion for exact calculation of compound distributions.
    
    Works for frequency distributions in the (a, b, 0) class:
    - Poisson: a = 0, b = λ
    - Binomial: a = -p/(1-p), b = (n+1)p/(1-p)
    - Negative Binomial: a = (1-p), b = (r-1)(1-p)
    
    The severity distribution must be discrete or discretized.
    """
    
    def __init__(self, frequency: FrequencyModel, severity: SeverityModel, 
                 discretization_step: float = 1.0, max_value: float = None):
        """
        Initialize Panjer recursion.
        
        Args:
            frequency: Frequency distribution (must be in (a,b,0) class)
            severity: Severity distribution
            discretization_step: Step size for discretization
            max_value: Maximum value to consider (default: 99.9th percentile)
        """
        self.frequency = frequency
        self.severity = severity
        self.h = discretization_step
        
        # Determine maximum value
        if max_value is None:
            try:
                max_value = severity.ppf(0.999) * frequency.ppf(0.999)
            except (AttributeError, TypeError):
                max_value = 1000.0  # Default fallback
        self.max_value = max_value
        self.n_steps = int(max_value / self.h) + 1
        
        # Identify frequency distribution parameters
        self._identify_panjer_params()
        
        # Discretize severity distribution
        self._discretize_severity()
    
    def _identify_panjer_params(self):
        """Identify a, b parameters for Panjer recursion."""
        freq_name = type(self.frequency).__name__
        
        if freq_name == 'Poisson':
            self.a = 0
            self.b = self.frequency.mu
            self.p0 = np.exp(-self.frequency.mu)
        elif freq_name == 'Binomial':
            p = self.frequency.p
            n = self.frequency.n
            self.a = -p / (1 - p)
            self.b = (n + 1) * p / (1 - p)
            self.p0 = (1 - p) ** n
        elif freq_name == 'NegativeBinomial':
            p = self.frequency.p
            r = self.frequency.n
            self.a = 1 - p
            self.b = (r - 1) * (1 - p)
            self.p0 = p ** r
        else:
            raise ValueError(f"Frequency distribution {freq_name} not supported by Panjer recursion")
    
    def _discretize_severity(self):
        """Discretize severity distribution."""
        # Discretization points
        x_points = np.arange(0, self.max_value + self.h, self.h)
        
        # Calculate probabilities using midpoint rule
        self.f = np.zeros(len(x_points))
        
        # Special handling for point mass at 0
        self.f[0] = self.severity.cdf(self.h / 2)
        
        # Other points
        for i in range(1, len(x_points) - 1):
            lower = x_points[i] - self.h / 2
            upper = x_points[i] + self.h / 2
            self.f[i] = self.severity.cdf(upper) - self.severity.cdf(lower)
        
        # Last point captures all remaining probability
        if len(x_points) > 1:
            self.f[-1] = 1.0 - self.severity.cdf(x_points[-2] + self.h / 2)
        
        # Normalize (ensure sum is 1)
        total = self.f.sum()
        if total > 0 and abs(total - 1.0) > 1e-10:
            self.f = self.f / total
    
    def calculate_aggregate_pmf(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate aggregate loss PMF using Panjer recursion.
        
        Returns:
            Tuple of (loss_values, probabilities)
        """
        # Initialize aggregate distribution
        g = np.zeros(self.n_steps)
        g[0] = self.p0
        
        # Apply Panjer recursion
        for x in range(1, self.n_steps):
            sum_term = 0
            # Adjust the range to properly handle the discretization
            # y represents the number of discretization units
            for y in range(1, min(x + 1, len(self.f))):
                if x - y >= 0 and x - y < self.n_steps and y < len(self.f):
                    sum_term += (self.a + self.b * y / x) * self.f[y] * g[x - y]
            
            # Avoid division by zero or negative denominator
            denominator = 1 - self.a * self.f[0]
            if abs(denominator) > 1e-10:
                g[x] = sum_term / denominator
            else:
                g[x] = 0
        
        # Return loss values and probabilities
        loss_values = np.arange(self.n_steps) * self.h
        
        return loss_values, g
    
    def mean(self) -> float:
        """Calculate mean from PMF."""
        x, p = self.calculate_aggregate_pmf()
        return np.sum(x * p)
    
    def var(self) -> float:
        """Calculate variance from PMF."""
        x, p = self.calculate_aggregate_pmf()
        mean = np.sum(x * p)
        return np.sum(x**2 * p) - mean**2
    
    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Calculate CDF from PMF."""
        loss_values, pmf = self.calculate_aggregate_pmf()
        cdf_values = np.cumsum(pmf)
        
        # Interpolate for requested x values
        x_array = np.atleast_1d(x)
        result = np.interp(x_array, loss_values, cdf_values)
        
        return result[0] if np.isscalar(x) else result


@CompoundDistribution.register_analytical('Binomial', 'Lognormal')
class BinomialLognormalApproximation(AnalyticalCompound):
    """
    Approximation for Binomial-Lognormal compound distribution.
    
    Uses moment matching and the fact that sum of lognormals can be
    approximated by another lognormal (Fenton-Wilkinson approximation).
    """
    
    def __init__(self, frequency, severity):
        super().__init__(frequency, severity)
        self.n = frequency.n  # Binomial trials
        self.p = frequency.p  # Binomial probability
        self.mu_ln = severity.mu  # Lognormal log-mean
        self.sigma_ln = severity.sigma  # Lognormal log-std
        
        # Cache for approximate parameters
        self._approx_params = None
    
    def _get_approx_params(self) -> Tuple[float, float]:
        """
        Calculate approximate lognormal parameters for aggregate loss.
        
        Uses Fenton-Wilkinson approximation for sum of lognormals.
        """
        if self._approx_params is None:
            # Moments of individual lognormal
            m1 = np.exp(self.mu_ln + self.sigma_ln**2 / 2)
            m2 = np.exp(2 * self.mu_ln + 2 * self.sigma_ln**2)
            
            # Expected number of claims
            E_N = self.n * self.p
            Var_N = self.n * self.p * (1 - self.p)
            
            # Approximate aggregate moments
            E_S = E_N * m1
            Var_S = E_N * (m2 - m1**2) + Var_N * m1**2
            
            # Match lognormal parameters
            if E_S > 0:
                cv_S = np.sqrt(Var_S) / E_S  # Coefficient of variation
                sigma_approx = np.sqrt(np.log(1 + cv_S**2))
                mu_approx = np.log(E_S) - sigma_approx**2 / 2
            else:
                # Degenerate case
                mu_approx = -np.inf
                sigma_approx = 0
            
            self._approx_params = (mu_approx, sigma_approx)
        
        return self._approx_params
    
    def mean(self) -> float:
        """E[S] = n * p * E[X]"""
        return self.n * self.p * np.exp(self.mu_ln + self.sigma_ln**2 / 2)
    
    def var(self) -> float:
        """Var[S] using compound distribution formula"""
        m1 = np.exp(self.mu_ln + self.sigma_ln**2 / 2)
        m2 = np.exp(2 * self.mu_ln + 2 * self.sigma_ln**2)
        
        E_N = self.n * self.p
        Var_N = self.n * self.p * (1 - self.p)
        
        return E_N * (m2 - m1**2) + Var_N * m1**2
    
    def pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Approximate PDF using lognormal approximation."""
        x_array = np.atleast_1d(x)
        result = np.zeros_like(x_array, dtype=float)
        
        # Probability of zero claims
        p_zero = (1 - self.p) ** self.n
        
        # For x = 0
        zero_mask = x_array == 0
        result[zero_mask] = p_zero
        
        # For x > 0, use lognormal approximation
        pos_mask = x_array > 0
        if np.any(pos_mask):
            mu_approx, sigma_approx = self._get_approx_params()
            
            if sigma_approx > 0:
                # Weight by probability of positive claims
                p_positive = 1 - p_zero
                result[pos_mask] = p_positive * stats.lognorm.pdf(
                    x_array[pos_mask], 
                    s=sigma_approx, 
                    scale=np.exp(mu_approx)
                )
        
        return result[0] if np.isscalar(x) else result
    
    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Approximate CDF using lognormal approximation."""
        x_array = np.atleast_1d(x)
        result = np.zeros_like(x_array, dtype=float)
        
        # Probability of zero claims
        p_zero = (1 - self.p) ** self.n
        
        # For x >= 0, start with P(S = 0)
        result[x_array >= 0] = p_zero
        
        # For x > 0, add lognormal CDF
        pos_mask = x_array > 0
        if np.any(pos_mask):
            mu_approx, sigma_approx = self._get_approx_params()
            
            if sigma_approx > 0:
                p_positive = 1 - p_zero
                result[pos_mask] += p_positive * stats.lognorm.cdf(
                    x_array[pos_mask],
                    s=sigma_approx,
                    scale=np.exp(mu_approx)
                )
        
        return result[0] if np.isscalar(x) else result
    
    def ppf(self, q: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Approximate quantiles using lognormal approximation."""
        q_array = np.atleast_1d(q)
        result = np.zeros_like(q_array, dtype=float)
        
        p_zero = (1 - self.p) ** self.n
        
        for i, qi in enumerate(q_array):
            if qi <= p_zero:
                result[i] = 0.0
            else:
                mu_approx, sigma_approx = self._get_approx_params()
                
                if sigma_approx > 0:
                    # Adjust quantile for positive part
                    q_adj = (qi - p_zero) / (1 - p_zero)
                    result[i] = stats.lognorm.ppf(
                        q_adj,
                        s=sigma_approx,
                        scale=np.exp(mu_approx)
                    )
        
        return result[0] if np.isscalar(q) else result
    
    def rvs(self, size: int = 1, random_state: Optional[int] = None) -> Union[float, np.ndarray]:
        """Generate samples using approximation."""
        if random_state is not None:
            np.random.seed(random_state)
        
        # Sample number of claims
        n_claims = stats.binom.rvs(self.n, self.p, size=size)
        
        # For efficiency, use lognormal approximation for large n_claims
        result = np.zeros(size)
        
        for i in range(size):
            if n_claims[i] == 0:
                result[i] = 0
            elif n_claims[i] > 30:  # Use approximation for large counts
                # Approximate sum of lognormals
                sum_mu = self.mu_ln + np.log(n_claims[i])
                sum_sigma = self.sigma_ln / np.sqrt(n_claims[i])
                result[i] = stats.lognorm.rvs(s=sum_sigma, scale=np.exp(sum_mu))
            else:
                # Direct simulation for small counts
                result[i] = stats.lognorm.rvs(
                    s=self.sigma_ln,
                    scale=np.exp(self.mu_ln),
                    size=n_claims[i]
                ).sum()
        
        return result[0] if size == 1 else result