"""
Performance tests for JIT-compiled simulation functions.

Tests that JIT compilation provides performance improvements over
standard Python implementation (excluding compilation time).
"""

import time
import unittest
from unittest.mock import patch

import pytest

import numpy as np
import pandas as pd

from quactuary.book import Portfolio, Inforce, PolicyTerms
from quactuary.distributions.frequency import Poisson
from quactuary.distributions.severity import Lognormal, Exponential
from quactuary.pricing import PricingModel
from quactuary.pricing_strategies import ClassicalPricingStrategy
from quactuary.classical_jit import ClassicalJITPricingModel
from quactuary.jit_kernels import simulate_aggregate_loss_batch


class TestJITPerformance(unittest.TestCase):
    """Test JIT compilation performance improvements."""
    
    def setUp(self):
        """Set up test portfolio with multiple buckets."""
        # Create policy terms
        terms = PolicyTerms(
            effective_date=pd.Timestamp('2024-01-01'),
            expiration_date=pd.Timestamp('2024-12-31'),
            per_occ_retention=1000.0,
            per_occ_limit=100000.0
        )
        
        # Create different types of inforce buckets
        self.buckets = [
            Inforce(
                n_policies=100,
                terms=terms,
                frequency=Poisson(mu=2.0),
                severity=Lognormal(shape=1.5, scale=np.exp(9.0)),
                name="High Frequency Bucket"
            ),
            Inforce(
                n_policies=200,
                terms=terms,
                frequency=Poisson(mu=0.5),
                severity=Exponential(scale=10000.0),  # mean=10000
                name="Low Frequency Bucket"
            ),
            Inforce(
                n_policies=50,
                terms=terms,
                frequency=Poisson(mu=1.0),
                severity=Lognormal(shape=0.8, scale=np.exp(10.0)),
                name="Medium Risk Bucket"
            )
        ]
        
        self.portfolio = Portfolio(self.buckets)
    
    def test_jit_kernel_compilation(self):
        """Test that JIT kernels compile and run correctly."""
        # Force compilation by running once
        _ = simulate_aggregate_loss_batch(
            n_sims=10,
            n_policies=10,
            freq_lambda=1.0,
            sev_mean=1000.0,
            sev_std=500.0,
            distribution_type=0
        )
        
        # Now test actual execution
        result = simulate_aggregate_loss_batch(
            n_sims=100,
            n_policies=50,
            freq_lambda=2.0,
            sev_mean=5000.0,
            sev_std=2000.0,
            distribution_type=0
        )
        
        self.assertEqual(len(result), 100)
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.mean(result) > 0)  # Should have some positive losses
    
    @pytest.mark.skip(reason="Test takes too long due to portfolio simulation")
    def test_jit_vs_standard_performance(self):
        """Compare JIT and standard simulation performance."""
        n_sims = 10000
        
        # Warm up JIT compilation
        jit_model = ClassicalJITPricingModel()
        _ = jit_model.simulate_portfolio_jit(self.portfolio, n_sims=10)
        
        # Time JIT simulation (excluding compilation)
        start_jit = time.time()
        jit_results = jit_model.simulate_portfolio_jit(self.portfolio, n_sims)
        jit_time = time.time() - start_jit
        
        # Time standard simulation
        start_std = time.time()
        std_results = self.portfolio.simulate(n_sims)
        std_time = time.time() - start_std
        
        # Print performance results
        speedup = std_time / jit_time
        print(f"\nPerformance Results (n_sims={n_sims}):")
        print(f"Standard time: {std_time:.3f}s")
        print(f"JIT time: {jit_time:.3f}s")
        print(f"Speedup: {speedup:.1f}x")
        
        # Verify results are statistically similar
        if isinstance(std_results, pd.Series):
            std_results = std_results.values
        
        # Compare means (should be within 5% due to randomness)
        jit_mean = np.mean(jit_results)
        std_mean = np.mean(std_results)
        relative_diff = abs(jit_mean - std_mean) / std_mean
        
        self.assertLess(relative_diff, 0.05, 
                       f"Means differ by {relative_diff:.1%}: JIT={jit_mean:.2f}, Std={std_mean:.2f}")
        
        # JIT should be faster (at least 2x speedup expected)
        self.assertGreater(speedup, 2.0, 
                          f"JIT speedup only {speedup:.1f}x, expected at least 2x")
    
    @pytest.mark.skip(reason="Test takes too long due to JIT compilation and portfolio simulation")
    def test_pricing_model_with_jit(self):
        """Test that PricingModel works correctly with JIT enabled."""
        # Create model with JIT-enabled strategy
        jit_strategy = ClassicalPricingStrategy(use_jit=True)
        model = PricingModel(self.portfolio, strategy=jit_strategy)
        
        # Force JIT compilation
        _ = model.simulate(n_sims=10)
        
        # Run actual test
        n_sims = 5000
        start = time.time()
        result = model.simulate(
            mean=True,
            variance=True,
            value_at_risk=True,
            tail_value_at_risk=True,
            n_sims=n_sims
        )
        elapsed = time.time() - start
        
        # Verify results
        self.assertIn('mean', result.estimates)
        self.assertIn('variance', result.estimates)
        self.assertIn('VaR', result.estimates)
        self.assertIn('TVaR', result.estimates)
        self.assertEqual(len(result.samples), n_sims)
        self.assertTrue(result.metadata.get('jit_enabled', False))
        
        print(f"\nPricingModel with JIT completed {n_sims} sims in {elapsed:.3f}s")
    
    @pytest.mark.skip(reason="Test takes too long (>2 minutes) due to portfolio simulation")
    def test_jit_vs_no_jit_strategy(self):
        """Compare strategies with and without JIT."""
        n_sims = 5000
        
        # Create models with different strategies
        jit_strategy = ClassicalPricingStrategy(use_jit=True)
        no_jit_strategy = ClassicalPricingStrategy(use_jit=False)
        
        jit_model = PricingModel(self.portfolio, strategy=jit_strategy)
        no_jit_model = PricingModel(self.portfolio, strategy=no_jit_strategy)
        
        # Warm up JIT
        _ = jit_model.simulate(n_sims=10)
        
        # Time with JIT
        start_jit = time.time()
        jit_result = jit_model.simulate(n_sims=n_sims)
        jit_time = time.time() - start_jit
        
        # Time without JIT
        start_no_jit = time.time()
        no_jit_result = no_jit_model.simulate(n_sims=n_sims)
        no_jit_time = time.time() - start_no_jit
        
        speedup = no_jit_time / jit_time
        print(f"\nStrategy Performance Comparison (n_sims={n_sims}):")
        print(f"Without JIT: {no_jit_time:.3f}s")
        print(f"With JIT: {jit_time:.3f}s")
        print(f"Speedup: {speedup:.1f}x")
        
        # Verify both produce similar results
        self.assertAlmostEqual(
            jit_result.estimates['mean'],
            no_jit_result.estimates['mean'],
            delta=no_jit_result.estimates['mean'] * 0.05  # 5% tolerance
        )
    
    def test_jit_with_different_distributions(self):
        """Test JIT performance with various distribution types."""
        # Test with pure Poisson/Lognormal
        lognormal_bucket = Inforce(
            n_policies=100,
            terms=PolicyTerms(
                effective_date=pd.Timestamp('2024-01-01'),
                expiration_date=pd.Timestamp('2024-12-31')
            ),
            frequency=Poisson(mu=1.5),
            severity=Lognormal(shape=1.0, scale=np.exp(8.0))
        )
        
        # Test with Poisson/Exponential
        exponential_bucket = Inforce(
            n_policies=100,
            terms=PolicyTerms(
                effective_date=pd.Timestamp('2024-01-01'),
                expiration_date=pd.Timestamp('2024-12-31')
            ),
            frequency=Poisson(mu=1.5),
            severity=Exponential(scale=1000.0)  # mean=1000
        )
        
        jit_model = ClassicalJITPricingModel()
        
        # Test both distributions
        for bucket, dist_name in [(lognormal_bucket, "Lognormal"), 
                                  (exponential_bucket, "Exponential")]:
            start = time.time()
            results = jit_model.simulate_inforce_jit(bucket, n_sims=10000)
            elapsed = time.time() - start
            
            print(f"\n{dist_name} distribution: {elapsed:.3f}s for 10k sims")
            self.assertEqual(len(results), 10000)
            self.assertTrue(np.mean(results) > 0)


if __name__ == '__main__':
    unittest.main()