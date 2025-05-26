"""
QMC wrapper for frequency and severity distributions.

This module provides wrapper classes that intercept rvs() calls and use
Sobol sequences when QMC is enabled.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional
from scipy import stats

from quactuary.sobol import get_qmc_simulator
from quactuary.distributions.frequency import FrequencyModel
from quactuary.distributions.severity import SeverityModel


class QMCFrequencyWrapper:
    """
    Wrapper that adds QMC sampling to any frequency distribution.
    
    This wrapper intercepts rvs() calls and uses Sobol sequences
    via inverse transform sampling when QMC is enabled.
    """
    
    def __init__(self, distribution: FrequencyModel):
        """
        Initialize wrapper with a frequency distribution.
        
        Args:
            distribution: The frequency distribution to wrap
        """
        self._dist = distribution
        
    def __getattr__(self, name):
        """Delegate all other attributes to the wrapped distribution."""
        return getattr(self._dist, name)
    
    def rvs(self, size: int = 1) -> Union[pd.Series, np.integer]:
        """
        Generate samples using QMC if enabled, otherwise use standard sampling.
        
        Args:
            size: Number of samples
            
        Returns:
            Samples as Series (if size > 1) or scalar
        """
        simulator = get_qmc_simulator()
        
        if simulator is None:
            # No QMC configured, use standard sampling
            return self._dist.rvs(size)
        
        # Get uniform QMC samples
        uniform_samples = simulator.uniform(n_samples=size, dimension=1)
        
        # Transform to discrete distribution using inverse CDF
        samples = self._inverse_transform(uniform_samples)
        
        if size == 1:
            return samples[0]
        else:
            return pd.Series(samples)
    
    def _inverse_transform(self, uniform_samples: np.ndarray) -> np.ndarray:
        """
        Transform uniform [0,1] samples to the target distribution.
        
        Uses inverse transform method with numerical search for discrete distributions.
        
        Args:
            uniform_samples: Uniform samples from QMC
            
        Returns:
            Integer samples from the frequency distribution
        """
        samples = np.zeros_like(uniform_samples, dtype=int)
        
        for i, u in enumerate(uniform_samples):
            # Find k such that CDF(k-1) < u <= CDF(k)
            # Use binary search for efficiency
            k = self._binary_search_ppf(u)
            samples[i] = k
            
        return samples
    
    def _binary_search_ppf(self, u: float, max_k: int = 1000) -> int:
        """
        Find quantile using binary search on CDF.
        
        Args:
            u: Uniform value in [0,1]
            max_k: Maximum value to search
            
        Returns:
            Integer k such that CDF(k-1) < u <= CDF(k)
        """
        if u <= 0:
            return 0
        if u >= 1:
            return max_k
            
        low, high = 0, max_k
        
        # Handle edge case where CDF(0) >= u
        if self._dist.cdf(0) >= u:
            return 0
        
        while low < high:
            mid = (low + high) // 2
            cdf_mid = self._dist.cdf(mid)
            
            if cdf_mid < u:
                low = mid + 1
            else:
                high = mid
                
        return low


class QMCSeverityWrapper:
    """
    Wrapper that adds QMC sampling to any severity distribution.
    
    This wrapper intercepts rvs() calls and uses Sobol sequences
    via inverse transform sampling when QMC is enabled.
    """
    
    def __init__(self, distribution: SeverityModel):
        """
        Initialize wrapper with a severity distribution.
        
        Args:
            distribution: The severity distribution to wrap
        """
        self._dist = distribution
        
    def __getattr__(self, name):
        """Delegate all other attributes to the wrapped distribution."""
        return getattr(self._dist, name)
    
    def rvs(self, size: int = 1) -> Union[float, np.ndarray, pd.Series]:
        """
        Generate samples using QMC if enabled, otherwise use standard sampling.
        
        Args:
            size: Number of samples
            
        Returns:
            Samples as array/Series (if size > 1) or scalar
        """
        simulator = get_qmc_simulator()
        
        if simulator is None or size == 0:
            # No QMC configured or zero samples, use standard sampling
            return self._dist.rvs(size)
        
        # Get uniform QMC samples
        uniform_samples = simulator.uniform(n_samples=size, dimension=1)
        
        # Transform to continuous distribution using inverse CDF
        samples = self._inverse_transform(uniform_samples)
        
        # Match the return type of the original distribution
        original_sample = self._dist.rvs(size=1)
        
        if size == 1:
            return float(samples[0])
        elif isinstance(original_sample, pd.Series) or (size > 1 and isinstance(self._dist.rvs(size=2), pd.Series)):
            return pd.Series(samples)
        else:
            return samples
    
    def _inverse_transform(self, uniform_samples: np.ndarray) -> np.ndarray:
        """
        Transform uniform [0,1] samples to the target distribution.
        
        Args:
            uniform_samples: Uniform samples from QMC
            
        Returns:
            Samples from the severity distribution
        """
        # Try to use ppf if available
        if hasattr(self._dist, 'ppf'):
            return self._dist.ppf(uniform_samples)
        elif hasattr(self._dist, '_dist') and hasattr(self._dist._dist, 'ppf'):
            # For scipy adapter classes
            return self._dist._dist.ppf(uniform_samples)
        else:
            # Fallback: use numerical inversion
            # This is slower but works for any distribution with CDF
            samples = np.zeros_like(uniform_samples)
            for i, u in enumerate(uniform_samples):
                samples[i] = self._numerical_ppf(u)
            return samples
    
    def _numerical_ppf(self, u: float, tol: float = 1e-8, max_iter: int = 50) -> float:
        """
        Numerically compute quantile using root finding on CDF.
        
        Args:
            u: Uniform value in [0,1]
            tol: Tolerance for root finding
            max_iter: Maximum iterations
            
        Returns:
            Value x such that CDF(x) ≈ u
        """
        # Initial bounds - adapt based on distribution characteristics
        if hasattr(self._dist, 'mean') and hasattr(self._dist, 'std'):
            mean = self._dist.mean()
            std = self._dist.std()
            # Start with ±5 standard deviations
            low = mean - 5 * std
            high = mean + 5 * std
        else:
            # Generic bounds
            low = 0.0
            high = 1000.0
        
        # Ensure bounds contain the target
        while self._dist.cdf(low) > u:
            low = low * 2 - high
        while self._dist.cdf(high) < u:
            high = high * 2 - low
        
        # Binary search
        for _ in range(max_iter):
            mid = (low + high) / 2
            cdf_mid = self._dist.cdf(mid)
            
            if abs(cdf_mid - u) < tol:
                return mid
            elif cdf_mid < u:
                low = mid
            else:
                high = mid
        
        return (low + high) / 2


def wrap_for_qmc(distribution: Union[FrequencyModel, SeverityModel]) -> Union[FrequencyModel, SeverityModel]:
    """
    Wrap a distribution to enable QMC sampling.
    
    Args:
        distribution: Frequency or severity distribution
        
    Returns:
        Wrapped distribution with QMC support
    """
    if isinstance(distribution, FrequencyModel):
        return QMCFrequencyWrapper(distribution)
    elif isinstance(distribution, SeverityModel):
        return QMCSeverityWrapper(distribution)
    else:
        return distribution