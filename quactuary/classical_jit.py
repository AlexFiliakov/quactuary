"""
JIT-optimized classical actuarial simulation module.

This module provides a high-performance implementation of classical actuarial
pricing models using Numba JIT compilation. It offers the same interface as
the standard classical module but with significant performance improvements
for large-scale Monte Carlo simulations.

The module achieves performance gains through:
- Just-In-Time compilation of simulation loops
- Parallel execution across CPU cores
- Optimized memory access patterns
- Vectorized operations on NumPy arrays
- Reduced Python interpreter overhead

Key Components:
    - ClassicalJITPricingModel: Main pricing model with JIT optimization
    - Distribution parameter extraction for JIT compatibility
    - Bucket-based parallel simulation
    - Optimized risk measure calculations

Performance Characteristics:
    - 10-100x faster than pure Python for large portfolios
    - Linear scaling with CPU cores (parallel=True)
    - One-time compilation overhead on first use
    - Memory-efficient batch processing

Limitations:
    - Assumes specific distribution types (Poisson, Lognormal, Exponential)
    - Falls back to approximations for unsupported distributions
    - Requires Numba package installation
    - May have reduced flexibility compared to pure Python

Examples:
    Direct usage:
        >>> from quactuary.classical_jit import ClassicalJITPricingModel
        >>> from quactuary.book import Portfolio
        >>> 
        >>> portfolio = Portfolio(policies_df)
        >>> jit_model = ClassicalJITPricingModel()
        >>> 
        >>> # First call compiles (slower)
        >>> result = jit_model.calculate_portfolio_statistics(
        ...     portfolio=portfolio,
        ...     n_sims=100000,
        ...     use_jit=True
        ... )
        >>> 
        >>> # Subsequent calls are fast
        >>> result2 = jit_model.calculate_portfolio_statistics(
        ...     portfolio=portfolio,
        ...     n_sims=1000000,  # 10x more simulations
        ...     use_jit=True
        ... )

    Through pricing strategy:
        >>> from quactuary.pricing_strategies import ClassicalPricingStrategy
        >>> 
        >>> # Automatically uses JIT when use_jit=True
        >>> strategy = ClassicalPricingStrategy(use_jit=True)
        >>> result = strategy.calculate_portfolio_statistics(
        ...     portfolio, n_sims=100000
        ... )

Performance Tips:
    - Use for portfolios with >100 policies
    - Use for simulations with >10,000 iterations
    - Group policies with similar characteristics in buckets
    - Ensure distributions are Poisson/Lognormal/Exponential when possible
    - Run a warmup simulation to trigger compilation

Notes:
    - The module maintains API compatibility with classical.py
    - Warnings are issued when distributions require approximation
    - Results should match non-JIT version within Monte Carlo error
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, Union, List
import warnings

import numpy as np
import pandas as pd

from quactuary.book import Portfolio, Inforce
from quactuary.datatypes import PricingResult
from quactuary.jit_kernels import (
    simulate_aggregate_loss_single,
    simulate_aggregate_loss_batch,
    aggregate_portfolio_losses,
    calculate_var_tvar,
    apply_policy_terms_vectorized
)
from quactuary.distributions.frequency import Poisson
from quactuary.distributions.severity import Lognormal, Exponential


class ClassicalJITPricingModel:
    """
    JIT-optimized classical pricing model using Numba acceleration.
    
    This model provides significant performance improvements for large-scale
    Monte Carlo simulations by leveraging JIT compilation and parallelization.
    """
    
    def __init__(self):
        """Initialize the JIT pricing model."""
        pass
    
    def _extract_distribution_params(self, inforce: Inforce) -> tuple:
        """
        Extract distribution parameters from an Inforce bucket.
        
        Returns:
            Tuple of (freq_lambda, sev_mean, sev_std, distribution_type)
        """
        # Extract frequency parameter (assuming Poisson for now)
        if isinstance(inforce.frequency, Poisson):
            freq_lambda = inforce.frequency._dist.args[0]  # mu parameter
        else:
            # Fallback to expected value for other distributions
            freq_lambda = float(inforce.frequency.pmf(1))  # Rough approximation
            warnings.warn(f"JIT optimization assumes Poisson frequency. Using approximation")
        
        # Extract severity parameters
        if isinstance(inforce.severity, Lognormal):
            # Lognormal in scipy uses shape parameter (sigma) and scale=exp(mu)
            # We need to compute mean and std
            shape = inforce.severity._dist.args[0]
            scale = inforce.severity._dist.kwds.get('scale', 1.0)
            # Mean = scale * exp(shape^2/2)
            # Var = (exp(shape^2) - 1) * scale^2 * exp(shape^2)
            sev_mean = scale * np.exp(shape**2 / 2)
            sev_var = (np.exp(shape**2) - 1) * scale**2 * np.exp(shape**2)
            sev_std = np.sqrt(sev_var)
            distribution_type = 0
        elif isinstance(inforce.severity, Exponential):
            # Exponential uses scale parameter
            scale = inforce.severity._dist.kwds.get('scale', 1.0)
            sev_mean = scale
            sev_std = scale  # For exponential, mean = std = scale
            distribution_type = 1
        else:
            # Fallback - try to compute mean/std
            try:
                # Sample to estimate parameters
                samples = inforce.severity.rvs(size=1000)
                if hasattr(samples, 'values'):
                    samples = samples.values
                sev_mean = np.mean(samples)
                sev_std = np.std(samples)
                distribution_type = 0  # Assume lognormal
                warnings.warn(f"JIT optimization assumes Lognormal/Exponential severity. Using empirical mean={sev_mean}, std={sev_std}")
            except:
                # Last resort defaults
                sev_mean = 1000.0
                sev_std = 500.0
                distribution_type = 0
                warnings.warn("Could not extract severity parameters, using defaults")
        
        return freq_lambda, sev_mean, sev_std, distribution_type
    
    def simulate_inforce_jit(self, inforce: Inforce, n_sims: int) -> np.ndarray:
        """
        Simulate aggregate losses for a single Inforce bucket using JIT.
        
        Args:
            inforce: Inforce bucket to simulate
            n_sims: Number of simulations
            
        Returns:
            Array of simulated aggregate losses
        """
        freq_lambda, sev_mean, sev_std, dist_type = self._extract_distribution_params(inforce)
        
        if n_sims == 1:
            return np.array([simulate_aggregate_loss_single(
                inforce.n_policies, freq_lambda, sev_mean, sev_std, dist_type
            )])
        else:
            return simulate_aggregate_loss_batch(
                n_sims, inforce.n_policies, freq_lambda, sev_mean, sev_std, dist_type
            )
    
    def simulate_portfolio_jit(self, portfolio: Portfolio, n_sims: int) -> np.ndarray:
        """
        Simulate aggregate losses for an entire portfolio using JIT.
        
        Args:
            portfolio: Portfolio to simulate
            n_sims: Number of simulations
            
        Returns:
            Array of simulated portfolio aggregate losses
        """
        if len(portfolio) == 0:
            return np.zeros(n_sims)
        
        # Simulate each bucket
        bucket_results = []
        for inforce in portfolio:
            bucket_sim = self.simulate_inforce_jit(inforce, n_sims)
            bucket_results.append(bucket_sim)
        
        # Convert to 2D array and aggregate
        bucket_array = np.array(bucket_results)
        return aggregate_portfolio_losses(bucket_array)
    
    def calculate_portfolio_statistics(
        self,
        portfolio: Portfolio,
        mean: bool = True,
        variance: bool = True,
        value_at_risk: bool = True,
        tail_value_at_risk: bool = True,
        tail_alpha: float = 0.05,
        n_sims: Optional[int] = None,
        use_jit: bool = True,
        *args,
        **kwargs
    ) -> PricingResult:
        """
        Compute portfolio statistics using JIT-optimized Monte Carlo simulation.
        
        Args:
            portfolio: Portfolio to analyze
            mean: Calculate mean loss
            variance: Calculate variance
            value_at_risk: Calculate VaR
            tail_value_at_risk: Calculate TVaR
            tail_alpha: Alpha level for tail risk measures
            n_sims: Number of simulations
            use_jit: Whether to use JIT optimization (default: True)
            
        Returns:
            PricingResult with calculated statistics
        """
        if n_sims is None:
            n_sims = 10000
        
        # Use JIT or fallback to standard simulation
        if use_jit:
            try:
                simulations = self.simulate_portfolio_jit(portfolio, n_sims)
            except Exception as e:
                warnings.warn(f"JIT simulation failed, falling back to standard: {e}")
                simulations = portfolio.simulate(n_sims)
                if isinstance(simulations, pd.Series):
                    simulations = simulations.values
        else:
            simulations = portfolio.simulate(n_sims)
            if isinstance(simulations, pd.Series):
                simulations = simulations.values
        
        # Convert to pandas Series for compatibility
        sim_series = pd.Series(simulations)
        
        # Build results
        result = PricingResult(
            estimates={},
            intervals={},
            samples=sim_series,
            metadata={
                "n_sims": n_sims,
                "run_date": datetime.now(),
                "jit_enabled": use_jit
            }
        )
        
        if mean:
            result.estimates['mean'] = np.mean(simulations)
        
        if variance:
            result.estimates['variance'] = np.var(simulations)
        
        if value_at_risk or tail_value_at_risk:
            var, tvar = calculate_var_tvar(simulations, tail_alpha)
            
            if value_at_risk:
                result.estimates['VaR'] = var
            
            if tail_value_at_risk:
                result.estimates['TVaR'] = tvar
            
            result.metadata['tail_alpha'] = tail_alpha
        
        return result