"""
Consolidated performance tests for JIT-compiled simulation functions.

This file combines tests from test_jit_speedup.py and test_jit_performance.py
to provide comprehensive JIT performance testing in one place.

Tests include:
- JIT speedup vs baseline measurements
- JIT compilation overhead analysis
- Performance with different portfolio sizes
- Performance with different distribution types
- Integration with PricingModel and strategies
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


class TestJITSpeedup(unittest.TestCase):
    """Test JIT speedup compared to baseline performance."""
    
    def create_test_portfolios(self):
        """Create test portfolios of different sizes."""
        terms = PolicyTerms(
            effective_date=pd.Timestamp('2024-01-01'),
            expiration_date=pd.Timestamp('2024-12-31'),
            per_occ_retention=1000.0,
            per_occ_limit=100000.0
        )
        
        return {
            'small': Portfolio([
                Inforce(
                    n_policies=10,
                    terms=terms,
                    frequency=Poisson(mu=2.0),
                    severity=Lognormal(shape=1.0, scale=np.exp(8.0)),
                    name="Small"
                )
            ]),
            'medium': Portfolio([
                Inforce(
                    n_policies=50,
                    terms=terms,
                    frequency=Poisson(mu=1.5),
                    severity=Lognormal(shape=1.2, scale=np.exp(8.5)),
                    name="Med1"
                ),
                Inforce(
                    n_policies=50,
                    terms=terms,
                    frequency=Poisson(mu=0.8),
                    severity=Exponential(scale=5000.0),
                    name="Med2"
                )
            ]),
            'large': Portfolio([
                Inforce(
                    n_policies=100,
                    terms=terms,
                    frequency=Poisson(mu=1.0 + i*0.1),
                    severity=Lognormal(shape=0.8 + i*0.05, scale=np.exp(8.0 + i*0.1)),
                    name=f"Large{i}"
                ) for i in range(5)
            ])
        }
    
    def test_jit_speedup_small_portfolio(self):
        """Test JIT speedup with small portfolio."""
        portfolio = self.create_test_portfolios()['small']
        n_sims = 1000
        
        # Baseline (no JIT)
        model_baseline = PricingModel(portfolio, strategy=ClassicalPricingStrategy(use_jit=False))
        start = time.perf_counter()
        result_baseline = model_baseline.simulate(n_sims=n_sims)
        baseline_time = time.perf_counter() - start
        
        # JIT enabled
        model_jit = PricingModel(portfolio, strategy=ClassicalPricingStrategy(use_jit=True))
        # Warm up JIT
        _ = model_jit.simulate(n_sims=10)
        
        start = time.perf_counter()
        result_jit = model_jit.simulate(n_sims=n_sims)
        jit_time = time.perf_counter() - start
        
        # Calculate speedup
        speedup = baseline_time / jit_time
        
        # Verify results are similar
        baseline_mean = result_baseline.estimates.get('mean', 0)
        jit_mean = result_jit.estimates.get('mean', 0)
        relative_diff = abs(jit_mean - baseline_mean) / baseline_mean if baseline_mean > 0 else 0
        
        # Assert expectations
        self.assertLess(relative_diff, 0.05, 
                       f"Results differ by {relative_diff:.1%}")
        # For small portfolios, JIT speedup might be modest
        self.assertGreater(speedup, 1.0, 
                          f"JIT should be at least as fast as baseline, got {speedup:.1f}x")
    
    @pytest.mark.slow
    def test_jit_speedup_large_portfolio(self):
        """Test JIT speedup with large portfolio."""
        portfolio = self.create_test_portfolios()['large']
        n_sims = 1000
        
        # Baseline (no JIT)
        model_baseline = PricingModel(portfolio, strategy=ClassicalPricingStrategy(use_jit=False))
        start = time.perf_counter()
        result_baseline = model_baseline.simulate(n_sims=n_sims)
        baseline_time = time.perf_counter() - start
        
        # JIT enabled
        model_jit = PricingModel(portfolio, strategy=ClassicalPricingStrategy(use_jit=True))
        # Warm up JIT
        _ = model_jit.simulate(n_sims=10)
        
        start = time.perf_counter()
        result_jit = model_jit.simulate(n_sims=n_sims)
        jit_time = time.perf_counter() - start
        
        # Calculate speedup
        speedup = baseline_time / jit_time
        
        # For larger portfolios, JIT should show more benefit
        self.assertGreater(speedup, 1.5, 
                          f"JIT speedup only {speedup:.1f}x, expected at least 1.5x for large portfolio")
    
    def test_jit_compilation_overhead(self):
        """Test JIT compilation overhead."""
        terms = PolicyTerms(
            effective_date=pd.Timestamp('2024-01-01'),
            expiration_date=pd.Timestamp('2024-12-31')
        )
        
        portfolio = Portfolio([
            Inforce(
                n_policies=100,
                terms=terms,
                frequency=Poisson(mu=1.5),
                severity=Lognormal(shape=1.0, scale=np.exp(8.0))
            )
        ])
        
        model = PricingModel(portfolio, strategy=ClassicalPricingStrategy(use_jit=True))
        
        # First run (includes compilation)
        start = time.perf_counter()
        _ = model.simulate(n_sims=100)
        first_run_time = time.perf_counter() - start
        
        # Second run (already compiled)
        start = time.perf_counter()
        _ = model.simulate(n_sims=100)
        second_run_time = time.perf_counter() - start
        
        # Compilation overhead should exist
        compilation_overhead = first_run_time - second_run_time
        self.assertGreater(compilation_overhead, 0, 
                          "First run should be slower due to JIT compilation")
        
        # But overhead shouldn't be excessive (< 2 seconds)
        self.assertLess(compilation_overhead, 2.0, 
                       f"Compilation overhead too high: {compilation_overhead:.3f}s")


class TestJITKernels(unittest.TestCase):
    """Test low-level JIT kernel functionality."""
    
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
    
    def test_jit_kernel_distributions(self):
        """Test JIT kernels with different distribution types."""
        n_sims = 1000
        
        # Test Lognormal (type 0)
        lognormal_results = simulate_aggregate_loss_batch(
            n_sims=n_sims,
            n_policies=100,
            freq_lambda=1.5,
            sev_mean=10000.0,
            sev_std=5000.0,
            distribution_type=0
        )
        
        # Test Exponential (type 1)
        exponential_results = simulate_aggregate_loss_batch(
            n_sims=n_sims,
            n_policies=100,
            freq_lambda=1.5,
            sev_mean=10000.0,
            sev_std=10000.0,  # For exponential, std = mean
            distribution_type=1
        )
        
        # Both should produce valid results
        self.assertEqual(len(lognormal_results), n_sims)
        self.assertEqual(len(exponential_results), n_sims)
        
        # Means should be roughly similar (within order of magnitude)
        ln_mean = np.mean(lognormal_results)
        exp_mean = np.mean(exponential_results)
        self.assertGreater(ln_mean, exp_mean * 0.1)
        self.assertLess(ln_mean, exp_mean * 10.0)


class TestJITIntegration(unittest.TestCase):
    """Test JIT integration with pricing models and strategies."""
    
    def setUp(self):
        """Set up test portfolio."""
        terms = PolicyTerms(
            effective_date=pd.Timestamp('2024-01-01'),
            expiration_date=pd.Timestamp('2024-12-31'),
            per_occ_retention=1000.0,
            per_occ_limit=100000.0
        )
        
        self.portfolio = Portfolio([
            Inforce(
                n_policies=100,
                terms=terms,
                frequency=Poisson(mu=2.0),
                severity=Lognormal(shape=1.5, scale=np.exp(9.0)),
                name="High Frequency"
            ),
            Inforce(
                n_policies=200,
                terms=terms,
                frequency=Poisson(mu=0.5),
                severity=Exponential(scale=10000.0),
                name="Low Frequency"
            )
        ])
    
    def test_pricing_model_jit_strategy(self):
        """Test that PricingModel works correctly with JIT strategy."""
        # Create model with JIT-enabled strategy
        jit_strategy = ClassicalPricingStrategy(use_jit=True)
        model = PricingModel(self.portfolio, strategy=jit_strategy)
        
        # Force JIT compilation
        _ = model.simulate(n_sims=10)
        
        # Run actual test
        n_sims = 1000
        result = model.simulate(
            mean=True,
            variance=True,
            value_at_risk=True,
            tail_value_at_risk=True,
            n_sims=n_sims
        )
        
        # Verify results
        self.assertIn('mean', result.estimates)
        self.assertIn('variance', result.estimates)
        self.assertIn('VaR', result.estimates)
        self.assertIn('TVaR', result.estimates)
        self.assertEqual(len(result.samples), n_sims)
        self.assertTrue(result.metadata.get('jit_enabled', False))
    
    def test_jit_vs_no_jit_consistency(self):
        """Test that JIT and non-JIT produce consistent results."""
        n_sims = 1000
        
        # Create models with different strategies
        jit_model = PricingModel(self.portfolio, 
                                strategy=ClassicalPricingStrategy(use_jit=True))
        no_jit_model = PricingModel(self.portfolio, 
                                   strategy=ClassicalPricingStrategy(use_jit=False))
        
        # Set random seed for reproducibility
        np.random.seed(42)
        jit_result = jit_model.simulate(n_sims=n_sims, mean=True, variance=True)
        
        np.random.seed(42)
        no_jit_result = no_jit_model.simulate(n_sims=n_sims, mean=True, variance=True)
        
        # Results should be statistically similar (not identical due to different RNG)
        # Allow 10% relative difference due to randomness
        self.assertAlmostEqual(
            jit_result.estimates['mean'],
            no_jit_result.estimates['mean'],
            delta=no_jit_result.estimates['mean'] * 0.10
        )
        
        self.assertAlmostEqual(
            jit_result.estimates['variance'],
            no_jit_result.estimates['variance'],
            delta=no_jit_result.estimates['variance'] * 0.20  # Variance has more variation
        )


class TestJITSpecificDistributions(unittest.TestCase):
    """Test JIT with specific distribution combinations."""
    
    def test_classical_jit_model_direct(self):
        """Test ClassicalJITPricingModel directly."""
        # Create simple inforce
        inforce = Inforce(
            n_policies=100,
            terms=PolicyTerms(
                effective_date=pd.Timestamp('2024-01-01'),
                expiration_date=pd.Timestamp('2024-12-31')
            ),
            frequency=Poisson(mu=1.5),
            severity=Lognormal(shape=1.0, scale=np.exp(8.0))
        )
        
        jit_model = ClassicalJITPricingModel()
        
        # Warm up
        _ = jit_model.simulate_inforce_jit(inforce, n_sims=10)
        
        # Test performance
        start = time.time()
        results = jit_model.simulate_inforce_jit(inforce, n_sims=10000)
        elapsed = time.time() - start
        
        self.assertEqual(len(results), 10000)
        self.assertTrue(np.mean(results) > 0)
        # Should be fast after compilation
        self.assertLess(elapsed, 1.0, f"10k sims took {elapsed:.3f}s, expected < 1s")
    
    def test_mixed_severity_distributions(self):
        """Test JIT with mixed severity distribution types."""
        terms = PolicyTerms(
            effective_date=pd.Timestamp('2024-01-01'),
            expiration_date=pd.Timestamp('2024-12-31')
        )
        
        # Portfolio with mixed distributions
        portfolio = Portfolio([
            Inforce(
                n_policies=50,
                terms=terms,
                frequency=Poisson(mu=2.0),
                severity=Lognormal(shape=1.0, scale=np.exp(8.0)),
                name="Lognormal"
            ),
            Inforce(
                n_policies=50,
                terms=terms,
                frequency=Poisson(mu=2.0),
                severity=Exponential(scale=5000.0),
                name="Exponential"
            )
        ])
        
        model = PricingModel(portfolio, strategy=ClassicalPricingStrategy(use_jit=True))
        result = model.simulate(n_sims=1000)
        
        # Should handle mixed distributions
        self.assertEqual(len(result.samples), 1000)
        self.assertGreater(result.estimates['mean'], 0)