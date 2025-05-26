"""
Vectorized simulation improvements for better performance.

This module implements vectorized operations to replace scalar loops
in the simulation process, particularly for distribution sampling and
loss aggregation.
"""

import numpy as np
from typing import Optional, Tuple
import warnings

from quactuary.book import Inforce
from quactuary.distributions.frequency import FrequencyModel
from quactuary.distributions.severity import SeverityModel
from quactuary.parallel_processing import ParallelSimulator, ParallelConfig


class VectorizedSimulator:
    """
    Implements vectorized simulation methods for improved performance.
    
    Key optimizations:
    - Batch sampling from distributions
    - Vectorized loss aggregation
    - Pre-allocation of arrays
    - Minimal Python loops
    """
    
    def __init__(self):
        """
        Initialize the vectorized simulator.
        
        The simulator is stateless and doesn't require configuration.
        All parameters are passed to individual simulation methods.
        
        Examples:
            >>> simulator = VectorizedSimulator()
            >>> # Ready to use for any portfolio simulation
        """
        pass
    
    @staticmethod
    def simulate_inforce_vectorized(
        inforce: Inforce,
        n_sims: int,
        batch_size: Optional[int] = None,
        parallel: bool = False,
        n_workers: Optional[int] = None
    ) -> np.ndarray:
        """
        Simulate aggregate losses using vectorized operations.
        
        Args:
            inforce: Inforce bucket to simulate
            n_sims: Number of simulations
            batch_size: Size of batches for memory efficiency
            parallel: Whether to use parallel processing
            n_workers: Number of parallel workers (None for auto)
            
        Returns:
            Array of aggregate losses
        """
        if parallel:
            # Use parallel processing for large simulations
            config = ParallelConfig(
                n_workers=n_workers,
                show_progress=True,
                fallback_to_serial=True
            )
            simulator = ParallelSimulator(config)
            
            # Create simulation function for parallel execution
            def simulate_batch(n_batch, n_policies):
                return VectorizedSimulator._simulate_batch(
                    inforce.frequency,
                    inforce.severity,
                    n_policies,
                    n_batch
                )
            
            return simulator.simulate_parallel_multiprocessing(
                simulate_batch,
                n_sims,
                inforce.n_policies
            )
        
        if batch_size is None:
            batch_size = min(n_sims, 10000)  # Default batch size
        
        # Pre-allocate result array
        results = np.zeros(n_sims)
        
        # Process in batches for memory efficiency
        for batch_start in range(0, n_sims, batch_size):
            batch_end = min(batch_start + batch_size, n_sims)
            batch_n = batch_end - batch_start
            
            # Vectorized frequency sampling for all simulations and policies
            # Shape: (n_policies, batch_n)
            freq_samples = np.zeros((inforce.n_policies, batch_n))
            for i in range(inforce.n_policies):
                freq_samples[i, :] = inforce.frequency.rvs(size=batch_n)
            
            # Total frequencies per simulation
            total_freqs = freq_samples.sum(axis=0).astype(int)
            
            # Vectorized severity sampling
            # Get maximum frequency to pre-allocate
            max_freq = total_freqs.max()
            if max_freq > 0:
                # Sample all severities at once
                # This is more efficient than sampling per simulation
                all_severities = inforce.severity.rvs(size=total_freqs.sum())
                
                # Distribute severities to simulations
                severity_idx = 0
                for sim_idx, freq in enumerate(total_freqs):
                    if freq > 0:
                        sim_severities = all_severities[severity_idx:severity_idx + freq]
                        results[batch_start + sim_idx] = sim_severities.sum()
                        severity_idx += freq
            
        return results
    
    @staticmethod
    def _simulate_batch(
        frequency: FrequencyModel,
        severity: SeverityModel,
        n_policies: int,
        n_sims: int
    ) -> np.ndarray:
        """Helper method for parallel batch simulation."""
        # Vectorized frequency sampling
        freq_samples = np.zeros((n_policies, n_sims))
        for i in range(n_policies):
            freq_samples[i, :] = frequency.rvs(size=n_sims)
        
        # Total frequencies per simulation
        total_freqs = freq_samples.sum(axis=0).astype(int)
        
        # Pre-allocate results
        results = np.zeros(n_sims)
        
        # Vectorized severity sampling
        max_freq = total_freqs.max()
        if max_freq > 0:
            # Sample all severities at once
            all_severities = severity.rvs(size=total_freqs.sum())
            
            # Distribute severities to simulations
            severity_idx = 0
            for sim_idx, freq in enumerate(total_freqs):
                if freq > 0:
                    sim_severities = all_severities[severity_idx:severity_idx + freq]
                    results[sim_idx] = sim_severities.sum()
                    severity_idx += freq
        
        return results
    
    @staticmethod
    def simulate_inforce_vectorized_v2(
        inforce: Inforce,
        n_sims: int
    ) -> np.ndarray:
        """
        Alternative vectorized implementation with different strategy.
        
        This version uses a different approach that may be faster
        for certain distribution combinations.
        """
        # Sample all frequencies at once
        # Shape: (n_sims, n_policies)
        freq_matrix = np.zeros((n_sims, inforce.n_policies))
        
        # Vectorized sampling per policy
        for i in range(inforce.n_policies):
            freq_matrix[:, i] = inforce.frequency.rvs(size=n_sims)
        
        # Total frequencies per simulation
        total_freqs = freq_matrix.sum(axis=1).astype(int)
        
        # Pre-allocate results
        results = np.zeros(n_sims)
        
        # Group simulations by frequency for efficient severity sampling
        unique_freqs = np.unique(total_freqs)
        
        for freq in unique_freqs:
            if freq == 0:
                continue
                
            # Find simulations with this frequency
            mask = total_freqs == freq
            n_with_freq = mask.sum()
            
            # Sample severities for all simulations with this frequency
            if n_with_freq > 0:
                # Sample as 1D array and reshape
                severities = inforce.severity.rvs(size=n_with_freq * freq)
                if hasattr(severities, 'values'):
                    severities = severities.values
                severities = severities.reshape(n_with_freq, freq)
                results[mask] = severities.sum(axis=1)
        
        return results
    
    @staticmethod
    def apply_policy_terms_vectorized(
        ground_up_losses: np.ndarray,
        deductible: float = 0.0,
        limit: Optional[float] = None,
        attachment: float = 0.0,
        coinsurance: float = 1.0
    ) -> np.ndarray:
        """
        Apply policy terms using vectorized operations.
        
        Args:
            ground_up_losses: Array of ground-up losses
            deductible: Policy deductible
            limit: Policy limit (None for unlimited)
            attachment: Attachment point for excess layers
            coinsurance: Coinsurance factor
            
        Returns:
            Array of losses after policy terms
        """
        # Apply deductible
        losses = np.maximum(ground_up_losses - deductible, 0.0)
        
        # Apply attachment if specified
        if attachment > 0:
            losses = np.maximum(losses - attachment, 0.0)
        
        # Apply limit
        if limit is not None and limit > 0:
            losses = np.minimum(losses, limit)
        
        # Apply coinsurance
        if coinsurance != 1.0:
            losses = losses * coinsurance
        
        return losses
    
    @staticmethod
    def calculate_statistics_vectorized(
        losses: np.ndarray,
        confidence_levels: Optional[list] = None
    ) -> dict:
        """
        Calculate statistics using vectorized operations.
        
        Args:
            losses: Array of loss values
            confidence_levels: List of confidence levels for VaR/TVaR
            
        Returns:
            Dictionary of statistics
        """
        if confidence_levels is None:
            confidence_levels = [0.90, 0.95, 0.99]
        
        # Basic statistics (all vectorized)
        results = {
            'mean': np.mean(losses),
            'std': np.std(losses),
            'variance': np.var(losses),
            'min': np.min(losses),
            'max': np.max(losses),
            'count': len(losses)
        }
        
        # Quantiles (VaR) - vectorized
        quantile_values = np.percentile(losses, [cl * 100 for cl in confidence_levels])
        for cl, var in zip(confidence_levels, quantile_values):
            results[f'var_{cl:.0%}'] = var
        
        # TVaR (Tail Value at Risk) - vectorized
        sorted_losses = np.sort(losses)
        n = len(losses)
        
        for cl in confidence_levels:
            var_idx = int(np.ceil(cl * n)) - 1
            var_idx = max(0, min(var_idx, n - 1))
            tail_losses = sorted_losses[var_idx:]
            results[f'tvar_{cl:.0%}'] = np.mean(tail_losses) if len(tail_losses) > 0 else sorted_losses[var_idx]
        
        return results


def benchmark_vectorization():
    """Benchmark vectorized vs standard simulation."""
    import time
    from quactuary.book import Portfolio, Inforce, PolicyTerms
    from quactuary.distributions.frequency import Poisson
    from quactuary.distributions.severity import Lognormal
    import pandas as pd
    
    print("VECTORIZATION BENCHMARK")
    print("=" * 60)
    
    # Create test inforce
    terms = PolicyTerms(
        effective_date=pd.Timestamp('2024-01-01'),
        expiration_date=pd.Timestamp('2024-12-31')
    )
    
    inforce = Inforce(
        n_policies=100,
        terms=terms,
        frequency=Poisson(mu=1.5),
        severity=Lognormal(shape=1.0, scale=np.exp(8.0)),
        name="Test"
    )
    
    n_sims = 10000
    
    # Standard simulation
    start = time.perf_counter()
    standard_results = inforce.simulate(n_sims)
    standard_time = time.perf_counter() - start
    
    # Vectorized simulation v1
    start = time.perf_counter()
    vectorized_results_v1 = VectorizedSimulator.simulate_inforce_vectorized(inforce, n_sims)
    vectorized_time_v1 = time.perf_counter() - start
    
    # Vectorized simulation v2
    start = time.perf_counter()
    vectorized_results_v2 = VectorizedSimulator.simulate_inforce_vectorized_v2(inforce, n_sims)
    vectorized_time_v2 = time.perf_counter() - start
    
    print(f"Standard:      {standard_time:.3f}s")
    print(f"Vectorized v1: {vectorized_time_v1:.3f}s (speedup: {standard_time/vectorized_time_v1:.1f}x)")
    print(f"Vectorized v2: {vectorized_time_v2:.3f}s (speedup: {standard_time/vectorized_time_v2:.1f}x)")
    
    # Verify results are similar
    standard_mean = np.mean(standard_results)
    vec_mean_v1 = np.mean(vectorized_results_v1)
    vec_mean_v2 = np.mean(vectorized_results_v2)
    
    print(f"\nMean comparison:")
    print(f"Standard:      {standard_mean:,.2f}")
    print(f"Vectorized v1: {vec_mean_v1:,.2f} (diff: {abs(vec_mean_v1-standard_mean)/standard_mean:.1%})")
    print(f"Vectorized v2: {vec_mean_v2:,.2f} (diff: {abs(vec_mean_v2-standard_mean)/standard_mean:.1%})")


if __name__ == "__main__":
    benchmark_vectorization()