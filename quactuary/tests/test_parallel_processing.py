"""Test parallel processing functionality."""

import numpy as np
import pandas as pd
import pytest
from quactuary.book import PolicyTerms, Inforce, Portfolio
from quactuary.distributions.frequency import Poisson
from quactuary.distributions.severity import Lognormal
from quactuary.parallel_processing import ParallelSimulator, ParallelConfig
from quactuary.vectorized_simulation import VectorizedSimulator


class TestParallelProcessing:
    """Test parallel processing capabilities."""
    
    def test_parallel_config(self):
        """Test ParallelConfig initialization."""
        config = ParallelConfig(n_workers=4, show_progress=False)
        assert config.n_workers == 4
        assert config.show_progress == False
        assert config.fallback_to_serial == True
    
    def test_parallel_simulator_basic(self):
        """Test basic parallel simulation."""
        config = ParallelConfig(n_workers=2, show_progress=False)
        simulator = ParallelSimulator(config)
        
        # Simple simulation function
        def simulate_func(n_sims, n_policies):
            return np.random.normal(100, 10, size=n_sims)
        
        results = simulator.simulate_parallel_multiprocessing(
            simulate_func, 1000, 10
        )
        
        assert len(results) == 1000
        assert 80 < np.mean(results) < 120  # Should be around 100
    
    def test_vectorized_parallel(self):
        """Test vectorized simulation with parallel option."""
        # Create test inforce
        terms = PolicyTerms(
            effective_date=pd.Timestamp('2024-01-01'),
            expiration_date=pd.Timestamp('2024-12-31')
        )
        
        inforce = Inforce(
            n_policies=10,
            frequency=Poisson(mu=1.5),
            severity=Lognormal(shape=1.0, scale=np.exp(8.0)),
            terms=terms
        )
        
        # Test serial
        results_serial = VectorizedSimulator.simulate_inforce_vectorized(
            inforce, 100, parallel=False
        )
        
        # Test parallel
        results_parallel = VectorizedSimulator.simulate_inforce_vectorized(
            inforce, 100, parallel=True, n_workers=2
        )
        
        assert len(results_serial) == 100
        assert len(results_parallel) == 100
        
        # Results should be similar in distribution
        assert abs(np.mean(results_serial) - np.mean(results_parallel)) / np.mean(results_serial) < 0.5
    
    def test_portfolio_parallel(self):
        """Test portfolio simulation with parallel option."""
        terms = PolicyTerms(
            effective_date=pd.Timestamp('2024-01-01'),
            expiration_date=pd.Timestamp('2024-12-31')
        )
        
        # Create small portfolio
        buckets = []
        for i in range(3):
            bucket = Inforce(
                n_policies=10,
                frequency=Poisson(mu=1.0),
                severity=Lognormal(shape=1.0, scale=np.exp(8.0)),
                terms=terms
            )
            buckets.append(bucket)
        
        portfolio = Portfolio(buckets)
        
        # Test small simulation (should use serial)
        results_small = portfolio.simulate(100, parallel=True)
        assert len(results_small) == 100
        
        # Test larger simulation (should use parallel)
        results_large = portfolio.simulate(20000, parallel=True, n_workers=2)
        assert len(results_large) == 20000
    
    def test_error_handling(self):
        """Test error handling in parallel processing."""
        config = ParallelConfig(
            n_workers=2,
            show_progress=False,
            fallback_to_serial=True,
            max_retries=1
        )
        simulator = ParallelSimulator(config)
        
        # Function that sometimes fails
        def flaky_func(n_sims, n_policies):
            if np.random.rand() < 0.3:
                raise RuntimeError("Random failure")
            return np.ones(n_sims)
        
        # Should still complete with fallback
        results = simulator.simulate_parallel_multiprocessing(
            flaky_func, 100, 10
        )
        assert len(results) == 100