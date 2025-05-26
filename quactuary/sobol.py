"""
Sobol sequence implementation for quasi-Monte Carlo simulations.

This module provides low-discrepancy Sobol sequences to replace standard pseudo-random
number generation in Monte Carlo simulations, offering faster convergence rates.

Examples:
    >>> from quactuary.sobol import SobolEngine
    >>> engine = SobolEngine(dimension=2)
    >>> points = engine.generate(n_points=1000)
    >>> # Points are uniformly distributed in [0,1]^2 with low discrepancy
"""

from typing import Optional, Union, Literal
import numpy as np
from scipy.stats import qmc


class SobolEngine:
    """
    Sobol sequence generator for quasi-Monte Carlo sampling.
    
    This class wraps scipy's Sobol implementation and provides enhanced features
    for actuarial simulations including scrambling, dimension management, and
    integration with distribution sampling.
    
    Attributes:
        dimension: Number of dimensions for the Sobol sequence
        scramble: Whether to use scrambling for randomized QMC
        skip: Number of initial points to skip (default: 1024 for better uniformity)
        seed: Random seed for scrambling
    """
    
    def __init__(
        self,
        dimension: int,
        scramble: bool = True,
        skip: int = 1024,
        seed: Optional[int] = None
    ):
        """
        Initialize Sobol sequence generator.
        
        Args:
            dimension: Number of dimensions needed
            scramble: Whether to apply Owen scrambling
            skip: Number of initial points to skip (burn-in)
            seed: Random seed for reproducible scrambling
        """
        if dimension < 1:
            raise ValueError("Dimension must be at least 1")
        if dimension > 21201:
            raise ValueError("Maximum supported dimension is 21201")
        
        self.dimension = dimension
        self.scramble = scramble
        self.skip = skip
        self.seed = seed
        
        # Initialize scipy's Sobol generator
        self._engine = qmc.Sobol(d=dimension, scramble=scramble, seed=seed)
        
        # Skip initial points for better uniformity
        if skip > 0:
            _ = self._engine.random(skip)
        
        # Track number of points generated
        self._n_generated = 0
        
    def generate(self, n_points: int) -> np.ndarray:
        """
        Generate next batch of Sobol points.
        
        Args:
            n_points: Number of points to generate
            
        Returns:
            Array of shape (n_points, dimension) with values in [0, 1]
        """
        points = self._engine.random(n_points)
        self._n_generated += n_points
        return points
    
    def reset(self):
        """Reset the generator to initial state."""
        self._engine = qmc.Sobol(
            d=self.dimension, 
            scramble=self.scramble, 
            seed=self.seed
        )
        if self.skip > 0:
            _ = self._engine.random(self.skip)
        self._n_generated = 0
    
    @property
    def total_generated(self) -> int:
        """Total number of points generated since initialization or reset."""
        return self._n_generated


class DimensionAllocator:
    """
    Manages dimension allocation for portfolio simulations.
    
    Allocates Sobol dimensions efficiently across frequency and severity
    sampling for multiple policies to maintain low-discrepancy properties.
    """
    
    def __init__(self, n_policies: int, max_claims_per_sim: int = 1000):
        """
        Initialize dimension allocator.
        
        Args:
            n_policies: Number of policies in portfolio
            max_claims_per_sim: Maximum expected claims in a single simulation
        """
        self.n_policies = n_policies
        self.max_claims_per_sim = max_claims_per_sim
        
        # Dimension allocation strategy:
        # - First n_policies dimensions: frequency sampling
        # - Next n_policies dimensions: primary severity sampling
        # - Remaining: additional severity sampling for high claim counts
        self.freq_start = 0
        self.freq_end = n_policies
        self.sev_primary_start = n_policies
        self.sev_primary_end = 2 * n_policies
        self.sev_additional_start = 2 * n_policies
        
        # Total dimensions needed
        self.total_dimensions = 2 * n_policies + max_claims_per_sim
        
    def get_frequency_dims(self, policy_idx: int) -> int:
        """Get dimension index for frequency sampling of a specific policy."""
        return self.freq_start + policy_idx
    
    def get_severity_dims(self, policy_idx: int, claim_idx: int) -> int:
        """Get dimension index for severity sampling of a specific claim."""
        if claim_idx == 0:
            # Use primary severity dimension for first claim
            return self.sev_primary_start + policy_idx
        else:
            # Use additional dimensions for subsequent claims
            dim_idx = self.sev_additional_start + claim_idx - 1
            if dim_idx >= self.total_dimensions:
                # Wrap around using modulo for dimension reuse
                dim_idx = self.sev_additional_start + (
                    (claim_idx - 1) % (self.total_dimensions - self.sev_additional_start)
                )
            return dim_idx


class QMCSimulator:
    """
    Quasi-Monte Carlo simulator using Sobol sequences.
    
    Replaces standard Monte Carlo sampling with low-discrepancy sequences
    for improved convergence in actuarial simulations.
    """
    
    def __init__(
        self,
        method: Literal["sobol", "halton", "random"] = "sobol",
        scramble: bool = True,
        skip: int = 1024,
        seed: Optional[int] = None
    ):
        """
        Initialize QMC simulator.
        
        Args:
            method: QMC method to use ("sobol", "halton", or "random")
            scramble: Whether to apply scrambling
            skip: Number of initial points to skip
            seed: Random seed for reproducibility
        """
        self.method = method
        self.scramble = scramble
        self.skip = skip
        self.seed = seed
        self._engines = {}  # Cache engines by dimension
        
    def get_engine(self, dimension: int) -> Union[SobolEngine, qmc.QMCEngine]:
        """Get or create QMC engine for specified dimension."""
        if dimension not in self._engines:
            if self.method == "sobol":
                self._engines[dimension] = SobolEngine(
                    dimension=dimension,
                    scramble=self.scramble,
                    skip=self.skip,
                    seed=self.seed
                )
            elif self.method == "halton":
                engine = qmc.Halton(d=dimension, scramble=self.scramble, seed=self.seed)
                if self.skip > 0:
                    _ = engine.random(self.skip)
                self._engines[dimension] = engine
            else:  # random
                self._engines[dimension] = np.random.RandomState(self.seed)
                
        return self._engines[dimension]
    
    def uniform(self, n_samples: int, dimension: int = 1) -> np.ndarray:
        """
        Generate uniform samples in [0, 1].
        
        Args:
            n_samples: Number of samples
            dimension: Number of dimensions
            
        Returns:
            Array of shape (n_samples, dimension) if dimension > 1,
            or (n_samples,) if dimension == 1
        """
        engine = self.get_engine(dimension)
        
        if self.method == "random":
            samples = engine.rand(n_samples, dimension)
        else:
            samples = engine.generate(n_samples) if hasattr(engine, 'generate') else engine.random(n_samples)
        
        return samples.squeeze() if dimension == 1 else samples
    
    def reset(self):
        """Reset all cached engines."""
        for engine in self._engines.values():
            if hasattr(engine, 'reset'):
                engine.reset()
        self._engines.clear()


# Global QMC simulator instance (can be configured by users)
_qmc_simulator = None


def get_qmc_simulator() -> Optional[QMCSimulator]:
    """Get the global QMC simulator instance."""
    return _qmc_simulator


def set_qmc_simulator(
    method: Literal["sobol", "halton", "random"] = "sobol",
    scramble: bool = True,
    skip: int = 1024,
    seed: Optional[int] = None
) -> QMCSimulator:
    """
    Configure and set the global QMC simulator.
    
    Args:
        method: QMC method to use
        scramble: Whether to apply scrambling
        skip: Number of initial points to skip
        seed: Random seed
        
    Returns:
        The configured QMC simulator
    """
    global _qmc_simulator
    _qmc_simulator = QMCSimulator(
        method=method,
        scramble=scramble,
        skip=skip,
        seed=seed
    )
    return _qmc_simulator


def reset_qmc_simulator():
    """Reset the global QMC simulator to None (use standard random)."""
    global _qmc_simulator
    _qmc_simulator = None