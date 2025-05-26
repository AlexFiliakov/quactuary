"""
Extensions and integration for compound distributions.

This module provides factory updates, performance optimizations,
and integration utilities for extended distribution support.
"""

import numpy as np
from typing import Union, Optional, Dict, Callable
from functools import lru_cache
import multiprocessing as mp

from quactuary.distributions.compound import CompoundDistribution, create_compound_distribution
from quactuary.distributions.frequency import FrequencyModel
from quactuary.distributions.severity import SeverityModel
from quactuary.distributions.compound_binomial import (
    BinomialExponentialCompound,
    BinomialGammaCompound, 
    BinomialLognormalCompound
)
from quactuary.distributions.mixed_poisson import (
    PoissonGammaMixture,
    PoissonInverseGaussianMixture,
    HierarchicalPoissonMixture
)
from quactuary.distributions.zero_inflated import (
    ZeroInflatedCompound,
    ZIPoissonCompound,
    ZINegativeBinomialCompound,
    ZIBinomialCompound
)
from quactuary.distributions.edgeworth import (
    EdgeworthExpansion,
    CompoundDistributionEdgeworth,
    automatic_order_selection
)


def create_extended_compound_distribution(
    frequency: Union[FrequencyModel, str],
    severity: Union[SeverityModel, str],
    zero_inflated: bool = False,
    zero_prob: float = None,
    use_edgeworth: bool = False,
    cache_size: int = 10000,
    parallel: bool = False,
    **kwargs
) -> CompoundDistribution:
    """
    Enhanced factory function for creating compound distributions.
    
    Supports all standard combinations plus:
    - Zero-inflated variants
    - Mixed Poisson processes
    - Edgeworth approximations
    - Performance optimizations
    
    Args:
        frequency: Frequency model or string identifier
        severity: Severity model or string identifier
        zero_inflated: Whether to use zero-inflated variant
        zero_prob: Zero-inflation probability (if None, estimate from data)
        use_edgeworth: Use Edgeworth expansion for approximation
        cache_size: Size of cache for simulated distributions
        parallel: Enable parallel computation where applicable
        **kwargs: Additional parameters for specific distributions
        
    Returns:
        Appropriate compound distribution instance
    """
    # Convert string identifiers to models if needed
    if isinstance(frequency, str):
        frequency = _create_frequency_from_string(frequency, **kwargs)
    if isinstance(severity, str):
        severity = _create_severity_from_string(severity, **kwargs)
    
    # Check for mixed Poisson
    if hasattr(frequency, 'mixing_dist_name'):
        # Already a mixed Poisson distribution
        base_compound = SimulatedCompoundOptimized(frequency, severity, cache_size, parallel)
    else:
        # Standard compound distribution
        base_compound = create_compound_distribution(frequency, severity)
    
    # Apply zero-inflation if requested
    if zero_inflated:
        freq_name = type(frequency).__name__
        
        if freq_name == 'Poisson':
            compound = ZIPoissonCompound(frequency, severity, zero_prob)
        elif freq_name == 'NegativeBinomial':
            compound = ZINegativeBinomialCompound(frequency, severity, zero_prob)
        elif freq_name == 'Binomial':
            compound = ZIBinomialCompound(frequency, severity, zero_prob)
        else:
            compound = ZeroInflatedCompound(frequency, severity, zero_prob)
    else:
        compound = base_compound
    
    # Apply Edgeworth approximation if requested
    if use_edgeworth and hasattr(compound, 'mean') and hasattr(compound, 'var'):
        compound = EdgeworthCompoundWrapper(compound)
    
    return compound


class SimulatedCompoundOptimized(CompoundDistribution):
    """
    Optimized version of SimulatedCompound with caching and parallel support.
    """
    
    def __init__(self, frequency: FrequencyModel, severity: SeverityModel,
                 cache_size: int = 10000, parallel: bool = False):
        super().__init__(frequency, severity)
        self.cache_size = cache_size
        self.parallel = parallel
        self._cache = None
        self._cache_stats = None
    
    @lru_cache(maxsize=128)
    def mean(self) -> float:
        """Cached mean calculation."""
        freq_mean = self.frequency.mean() if hasattr(self.frequency, 'mean') else self.frequency._dist.mean()
        sev_mean = self.severity._dist.mean()
        return freq_mean * sev_mean
    
    @lru_cache(maxsize=128)
    def var(self) -> float:
        """Cached variance calculation."""
        freq_mean = self.frequency.mean() if hasattr(self.frequency, 'mean') else self.frequency._dist.mean()
        freq_var = self.frequency.var() if hasattr(self.frequency, 'var') else self.frequency._dist.var()
        sev_mean = self.severity._dist.mean()
        sev_var = self.severity._dist.var()
        
        return freq_mean * sev_var + freq_var * sev_mean**2
    
    def _ensure_cache(self):
        """Generate cache with optional parallel processing."""
        if self._cache is None:
            if self.parallel and self.cache_size >= 1000:
                self._generate_cache_parallel()
            else:
                self._generate_cache_serial()
            
            # Compute cache statistics
            self._cache_stats = {
                'mean': np.mean(self._cache),
                'std': np.std(self._cache),
                'quantiles': np.percentile(self._cache, [1, 5, 10, 25, 50, 75, 90, 95, 99])
            }
    
    def _generate_cache_serial(self):
        """Serial cache generation."""
        samples = []
        for _ in range(self.cache_size):
            n_claims = self.frequency.rvs()
            if n_claims > 0:
                losses = self.severity.rvs(size=n_claims)
                total_loss = np.sum(losses)
            else:
                total_loss = 0.0
            samples.append(total_loss)
        self._cache = np.array(samples)
    
    def _generate_cache_parallel(self):
        """Parallel cache generation using multiprocessing."""
        n_cores = min(mp.cpu_count(), 4)
        chunk_size = self.cache_size // n_cores
        
        with mp.Pool(n_cores) as pool:
            chunks = pool.map(
                self._generate_chunk,
                [chunk_size] * n_cores
            )
        
        self._cache = np.concatenate(chunks)[:self.cache_size]
    
    def _generate_chunk(self, size: int) -> np.ndarray:
        """Generate a chunk of samples for parallel processing."""
        samples = []
        for _ in range(size):
            n_claims = self.frequency.rvs()
            if n_claims > 0:
                losses = self.severity.rvs(size=n_claims)
                total_loss = np.sum(losses)
            else:
                total_loss = 0.0
            samples.append(total_loss)
        return np.array(samples)
    
    def pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """PDF using kernel density estimation on cache."""
        self._ensure_cache()
        from scipy.stats import gaussian_kde
        
        kde = gaussian_kde(self._cache, bw_method='scott')
        return kde(x)
    
    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """CDF using empirical distribution from cache."""
        self._ensure_cache()
        x_array = np.atleast_1d(x)
        
        result = np.zeros_like(x_array, dtype=float)
        for i, xi in enumerate(x_array):
            result[i] = np.mean(self._cache <= xi)
        
        return result[0] if np.isscalar(x) else result
    
    def ppf(self, q: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Quantile function using cache."""
        self._ensure_cache()
        return np.percentile(self._cache, np.array(q) * 100)
    
    def rvs(self, size: int = 1, random_state: Optional[int] = None) -> Union[float, np.ndarray]:
        """Random sampling with optional resampling from cache."""
        if size <= self.cache_size // 10:
            # For small samples, resample from cache
            self._ensure_cache()
            if random_state is not None:
                np.random.seed(random_state)
            indices = np.random.choice(self.cache_size, size=size, replace=True)
            samples = self._cache[indices]
        else:
            # For large samples, generate new
            if random_state is not None:
                np.random.seed(random_state)
            
            samples = np.zeros(size)
            for i in range(size):
                n_claims = self.frequency.rvs()
                if n_claims > 0:
                    losses = self.severity.rvs(size=n_claims)
                    samples[i] = np.sum(losses)
        
        return samples[0] if size == 1 else samples


class EdgeworthCompoundWrapper:
    """
    Wrapper to add Edgeworth approximation to any compound distribution.
    """
    
    def __init__(self, base_compound: CompoundDistribution):
        self.base_compound = base_compound
        self._setup_edgeworth()
    
    def _setup_edgeworth(self):
        """Setup Edgeworth expansion from base distribution moments."""
        # Get moments from base distribution
        mean = self.base_compound.mean()
        var = self.base_compound.var()
        
        # Estimate higher moments using simulation if not available
        samples = self.base_compound.rvs(size=10000)
        
        # Standardized moments
        std = np.sqrt(var)
        standardized = (samples - mean) / std
        
        skewness = np.mean(standardized**3)
        excess_kurtosis = np.mean(standardized**4) - 3
        
        # Create Edgeworth expansion
        self.edgeworth = EdgeworthExpansion(
            mean=mean,
            variance=var,
            skewness=skewness,
            excess_kurtosis=excess_kurtosis
        )
        
        # Determine optimal order
        self.order = automatic_order_selection(skewness, excess_kurtosis, 1000)
    
    def mean(self) -> float:
        return self.base_compound.mean()
    
    def var(self) -> float:
        return self.base_compound.var()
    
    def std(self) -> float:
        return self.base_compound.std()
    
    def pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self.edgeworth.pdf(x, order=self.order)
    
    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self.edgeworth.cdf(x, order=self.order)
    
    def ppf(self, q: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self.edgeworth.ppf(q, order=self.order)
    
    def rvs(self, size: int = 1, random_state: Optional[int] = None) -> Union[float, np.ndarray]:
        # Use base distribution for sampling
        return self.base_compound.rvs(size=size, random_state=random_state)


def distribution_selection_guide(
    mean_frequency: float,
    cv_frequency: float,
    mean_severity: float,
    cv_severity: float,
    has_zeros: bool = False,
    sample_size: int = None
) -> str:
    """
    Guide for selecting appropriate compound distribution.
    
    Args:
        mean_frequency: Average number of claims
        cv_frequency: Coefficient of variation for frequency
        mean_severity: Average claim size
        cv_severity: Coefficient of variation for severity
        has_zeros: Whether data has excess zeros
        sample_size: Available sample size
        
    Returns:
        Recommended distribution type
    """
    recommendations = []
    
    # Frequency distribution selection
    if cv_frequency < 0.9:
        freq_dist = "Poisson"
    elif cv_frequency < 1.5:
        freq_dist = "Negative Binomial"
    else:
        freq_dist = "Mixed Poisson"
    
    # Severity distribution selection
    if cv_severity < 0.5:
        sev_dist = "Exponential"
    elif cv_severity < 1.5:
        sev_dist = "Gamma"
    else:
        sev_dist = "Lognormal"
    
    # Check for analytical solution
    analytical_pairs = [
        ("Poisson", "Exponential"),
        ("Poisson", "Gamma"),
        ("Binomial", "Exponential"),
        ("Binomial", "Gamma"),
        ("Binomial", "Lognormal")
    ]
    
    base_recommendation = f"{freq_dist}-{sev_dist}"
    
    if (freq_dist, sev_dist) in analytical_pairs:
        recommendations.append(f"Use {base_recommendation} (analytical solution available)")
    else:
        recommendations.append(f"Use {base_recommendation} (simulation required)")
    
    # Zero-inflation check
    if has_zeros:
        recommendations.append("Consider zero-inflated variant")
    
    # Sample size considerations
    if sample_size and sample_size < 100:
        recommendations.append("Small sample: consider Edgeworth approximation with order 2")
    elif sample_size and sample_size >= 1000:
        recommendations.append("Large sample: Edgeworth order 4 or simulation are both viable")
    
    return "\n".join(recommendations)


def _create_frequency_from_string(name: str, **params) -> FrequencyModel:
    """Helper to create frequency distribution from string."""
    from quactuary.distributions.frequency import Poisson, Binomial, NegativeBinomial
    
    if name.lower() == 'poisson':
        return Poisson(mu=params.get('mu', 10))
    elif name.lower() == 'binomial':
        return Binomial(n=params.get('n', 20), p=params.get('p', 0.5))
    elif name.lower() == 'negativebinomial':
        return NegativeBinomial(r=params.get('r', 5), p=params.get('p', 0.5))
    else:
        raise ValueError(f"Unknown frequency distribution: {name}")


def _create_severity_from_string(name: str, **params) -> SeverityModel:
    """Helper to create severity distribution from string."""
    from quactuary.distributions.severity import Exponential, Gamma, LogNormal
    
    if name.lower() == 'exponential':
        return Exponential(scale=params.get('scale', 1000))
    elif name.lower() == 'gamma':
        return Gamma(a=params.get('a', 2), scale=params.get('scale', 500))
    elif name.lower() == 'lognormal':
        return LogNormal(mu=params.get('mu', 6), sigma=params.get('sigma', 1))
    else:
        raise ValueError(f"Unknown severity distribution: {name}")