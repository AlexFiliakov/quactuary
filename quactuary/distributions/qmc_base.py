"""
Base classes for quasi-Monte Carlo integration in distributions.

This module provides base functionality to integrate Sobol sequences and other
QMC methods with frequency and severity distributions.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union
from scipy import stats

from quactuary.sobol import get_qmc_simulator


class QMCDistributionMixin:
    """
    Mixin class that adds QMC sampling capabilities to distributions.
    
    This mixin modifies the rvs() method to use Sobol sequences or other
    QMC methods instead of pseudo-random sampling when a QMC simulator
    is configured.
    """
    
    def _qmc_enabled(self) -> bool:
        """Check if QMC sampling is enabled."""
        return get_qmc_simulator() is not None
    
    def _get_uniform_qmc(self, size: int) -> np.ndarray:
        """
        Get uniform QMC samples in [0, 1].
        
        Args:
            size: Number of samples needed
            
        Returns:
            Array of uniform samples
        """
        simulator = get_qmc_simulator()
        if simulator is None:
            # Fallback to standard random
            return np.random.rand(size)
        return simulator.uniform(n_samples=size, dimension=1)
    
    def _transform_uniform_to_distribution(
        self, 
        uniform_samples: np.ndarray,
        ppf_func=None
    ) -> np.ndarray:
        """
        Transform uniform [0,1] samples to target distribution.
        
        Args:
            uniform_samples: Uniform samples from QMC
            ppf_func: Percent point function (inverse CDF)
            
        Returns:
            Samples from target distribution
        """
        if ppf_func is None:
            # Use self.ppf if available
            if hasattr(self, 'ppf'):
                ppf_func = self.ppf
            else:
                raise NotImplementedError(
                    "Distribution must provide ppf (inverse CDF) for QMC sampling"
                )
        
        # Apply inverse transform sampling
        return ppf_func(uniform_samples)


class QMCFrequencyMixin(QMCDistributionMixin):
    """
    Mixin for frequency distributions to support QMC sampling.
    
    Overrides rvs() to use QMC when enabled.
    """
    
    def rvs_qmc(self, size: int = 1) -> Union[pd.Series, np.integer]:
        """
        Generate samples using QMC if enabled, otherwise fallback to standard.
        
        Args:
            size: Number of samples
            
        Returns:
            Samples as Series (if size > 1) or scalar
        """
        if not self._qmc_enabled():
            # Fallback to original rvs method
            return self.rvs_standard(size)
        
        # Get uniform QMC samples
        uniform_samples = self._get_uniform_qmc(size)
        
        # Transform to discrete distribution
        # For discrete distributions, we need the quantile function
        if hasattr(self, 'ppf'):
            samples = self.ppf(uniform_samples)
        elif hasattr(self, '_dist') and hasattr(self._dist, 'ppf'):
            # For scipy adapter classes
            samples = self._dist.ppf(uniform_samples)
        else:
            # Fallback for distributions without ppf
            # Use numerical inversion of CDF
            samples = np.array([self._numerical_ppf(u) for u in uniform_samples])
        
        # Ensure integer type for frequency
        samples = np.round(samples).astype(int)
        
        if size == 1:
            return samples[0]
        else:
            return pd.Series(samples)
    
    def _numerical_ppf(self, q: float, max_k: int = 1000) -> int:
        """
        Numerical computation of quantile function for discrete distributions.
        
        Args:
            q: Quantile in [0, 1]
            max_k: Maximum value to search
            
        Returns:
            Integer k such that CDF(k-1) < q <= CDF(k)
        """
        if q <= 0:
            return 0
        if q >= 1:
            return max_k
        
        # Binary search for efficiency
        low, high = 0, max_k
        while low < high:
            mid = (low + high) // 2
            if self.cdf(mid) < q:
                low = mid + 1
            else:
                high = mid
        return low
    
    def rvs_standard(self, size: int = 1):
        """Standard rvs method (to be overridden by actual implementation)."""
        raise NotImplementedError("Must be implemented by distribution class")


class QMCSeverityMixin(QMCDistributionMixin):
    """
    Mixin for severity distributions to support QMC sampling.
    
    Overrides rvs() to use QMC when enabled.
    """
    
    def rvs_qmc(self, size: int = 1) -> Union[pd.Series, float, np.ndarray]:
        """
        Generate samples using QMC if enabled, otherwise fallback to standard.
        
        Args:
            size: Number of samples
            
        Returns:
            Samples as Series/array (if size > 1) or scalar
        """
        if not self._qmc_enabled() or size == 0:
            # Fallback to original rvs method
            return self.rvs_standard(size)
        
        # Get uniform QMC samples
        uniform_samples = self._get_uniform_qmc(size)
        
        # Transform to continuous distribution
        if hasattr(self, 'ppf'):
            samples = self.ppf(uniform_samples)
        elif hasattr(self, '_dist') and hasattr(self._dist, 'ppf'):
            # For scipy adapter classes
            samples = self._dist.ppf(uniform_samples)
        else:
            # Fallback - use standard sampling
            return self.rvs_standard(size)
        
        if size == 1:
            return float(samples[0])
        else:
            return samples
    
    def rvs_standard(self, size: int = 1):
        """Standard rvs method (to be overridden by actual implementation)."""
        raise NotImplementedError("Must be implemented by distribution class")