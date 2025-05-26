"""
Example performance tests demonstrating the adaptive baseline system.

This file shows how to write performance tests using the quactuary
performance testing framework.
"""

import numpy as np
import pandas as pd

from quactuary.book import Inforce, PolicyTerms, Portfolio
from quactuary.distributions.frequency import Poisson
from quactuary.distributions.severity import Lognormal
from quactuary.performance_testing import PerformanceTestCase, performance_test
from quactuary.pricing import PricingModel


class TestPerformanceExample(PerformanceTestCase):
    """Example performance test cases."""

    def setUp(self):
        """Set up test data."""
        # Create a simple test portfolio
        terms = PolicyTerms(
            effective_date=pd.Timestamp('2024-01-01'),
            expiration_date=pd.Timestamp('2024-12-31'),
            per_occ_retention=1000.0,
            per_occ_limit=100000.0
        )

        self.small_portfolio = Portfolio([
            Inforce(
                n_policies=10,
                terms=terms,
                frequency=Poisson(mu=2.0),
                severity=Lognormal(shape=1.0, scale=np.exp(8.0)),
                name="Small Test Portfolio"
            )
        ])

        self.medium_portfolio = Portfolio([
            Inforce(
                n_policies=100,
                terms=terms,
                frequency=Poisson(mu=1.5),
                severity=Lognormal(shape=1.2, scale=np.exp(8.5)),
                name="Medium Test Portfolio"
            )
        ])

    # TODO: Uncomment and implement `test_small_portfolio_performance` when ready
    # @performance_test("small_portfolio_pricing", sample_size=1000)
    # def test_small_portfolio_performance(self):
    #     """Test performance of pricing small portfolio."""
    #     model = PricingModel(self.small_portfolio)
    #     result = model.simulate(n_sims=1000)

    #     # Verify result is valid
    #     self.assertIn('mean', result.estimates)
    #     self.assertGreater(result.estimates['mean'], 0)

    # def test_medium_portfolio_with_requirements(self):
    #     """Test that medium portfolio meets specific performance requirements."""
    #     model = PricingModel(self.medium_portfolio)

    #     # Define the operation to benchmark
    #     def run_pricing():
    #         return model.simulate(
    #             mean=True,
    #             variance=True,
    #             value_at_risk=True,
    #             tail_value_at_risk=True,
    #             n_sims=10000
    #         )

    #     # Assert performance with maximum time requirement
    #     self.assertPerformance(
    #         test_func=run_pricing,
    #         test_name="medium_portfolio_full_metrics",
    #         sample_size=10000 * 100,  # n_sims * n_policies
    #         max_time=5.0,  # Must complete within 5 seconds
    #         check_regression=True,
    #         metadata={
    #             'portfolio_size': 100,
    #             'n_simulations': 10000,
    #             'metrics': ['mean', 'variance', 'VaR', 'TVaR']
    #         }
    #     )

    @performance_test("numpy_operations", sample_size=1000000)
    def test_numpy_performance(self):
        """Test performance of numpy operations (baseline comparison)."""
        # Generate random data
        data = np.random.randn(1000, 1000)

        # Perform matrix operations
        result = np.dot(data, data.T)
        eigenvalues = np.linalg.eigvals(result[:100, :100])  # Subset for speed

        # Verify results
        self.assertEqual(result.shape, (1000, 1000))
        self.assertEqual(len(eigenvalues), 100)

    def test_jit_warmup_performance(self):
        """Test JIT performance with proper warm-up."""
        from quactuary.pricing_strategies import ClassicalPricingStrategy

        # Create JIT-enabled model
        model = PricingModel(
            self.small_portfolio,
            strategy=ClassicalPricingStrategy(use_jit=True)
        )

        # Warm up JIT compiler
        for _ in range(5):
            model.simulate(n_sims=10)

        # Actual performance test
        self.assertPerformance(
            test_func=lambda: model.simulate(n_sims=1000),
            test_name="jit_pricing_after_warmup",
            sample_size=1000 * 10,  # n_sims * n_policies
            check_regression=True,
            metadata={'jit_enabled': True, 'warmup_runs': 5}
        )

    def test_performance_comparison(self):
        """Compare performance of different methods."""
        model = PricingModel(self.small_portfolio)

        # Test baseline method
        self.assertPerformance(
            test_func=lambda: model.simulate(n_sims=1000),
            test_name="comparison_baseline",
            sample_size=1000 * 10,
            metadata={'method': 'baseline'}
        )

        # Test with QMC
        self.assertPerformance(
            test_func=lambda: model.simulate(n_sims=1000, qmc_method="sobol"),
            test_name="comparison_qmc",
            sample_size=1000 * 10,
            metadata={'method': 'qmc', 'qmc_type': 'sobol'}
        )


class TestRegressionScenarios(PerformanceTestCase):
    """Test cases that demonstrate regression detection."""

    # Set to True to allow regressions (useful for CI)
    allow_regressions = False

    # TODO: Uncomment and implement `test_intentional_slow_operation` when ready
    # def test_intentional_slow_operation(self):
    #     """
    #     Example of a test that might trigger regression detection.

    #     This test simulates a performance regression by adding
    #     artificial delay. In real tests, this would be actual
    #     algorithm performance.
    #     """
    #     import time

    #     def slow_operation():
    #         # Simulate work with intentional slowdown
    #         data = np.random.randn(100, 100)
    #         for _ in range(10):
    #             np.linalg.svd(data)
    #         # Uncomment to simulate regression:
    #         # time.sleep(0.1)
    #         return data

    #     # This will track performance and detect regressions
    #     self.assertPerformance(
    #         test_func=slow_operation,
    #         test_name="matrix_svd_operations",
    #         sample_size=1000,  # 100x100 matrix = 10000 elements, but we do 10 iterations
    #         check_regression=True
    #     )
