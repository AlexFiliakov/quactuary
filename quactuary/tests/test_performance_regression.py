import pytest
import numpy as np
import time
import json
import os
from datetime import date
from pathlib import Path

from quactuary.book import PolicyTerms, Inforce, Portfolio
from quactuary.distributions.frequency import Poisson
from quactuary.distributions.severity import Lognormal
from quactuary.pricing import PricingModel
from quactuary.pricing_strategies import ClassicalPricingStrategy
from quactuary.benchmarks import PerformanceBenchmark


class TestPerformanceRegression:
    """Test suite for detecting performance regressions."""
    
    # Performance thresholds - adjust based on your CI environment
    BASELINE_THRESHOLD_MULTIPLIER = 1.5  # Allow 50% slower than baseline
    MINIMUM_SPEEDUP_JIT = 1.2  # JIT should be at least 20% faster
    MINIMUM_SPEEDUP_QMC = 0.95  # QMC might be slightly slower but more accurate
    
    @pytest.fixture
    def performance_baseline_path(self):
        """Path to store performance baseline data."""
        return Path("quactuary/tests/performance_baseline.json")
    
    @pytest.fixture
    def standard_portfolio(self):
        """Create a standard portfolio for performance testing."""
        portfolio = []
        for i in range(50):  # 50 policies for meaningful performance test
            policy = Inforce(
                n_policies=1,
                name=f"P{i:03d}",
                frequency=Poisson(mu=5.0),
                severity=Lognormal(shape=1.0, scale=np.exp(8.0)),
                terms=PolicyTerms(
                    effective_date=date(2024, 1, 1),
                    expiration_date=date(2024, 12, 31),
                    per_occ_retention=1000.0,
                    agg_retention=5000.0,
                    per_occ_limit=100000.0,
                    agg_limit=500000.0
                )
            )
            portfolio.append(policy)
        return Portfolio(portfolio)
    
    def measure_performance(self, portfolio, strategy, n_sims=5000, **kwargs):
        """Measure performance of a pricing model."""
        model = PricingModel(portfolio, strategy=strategy)
        
        # Warm up for JIT
        if hasattr(strategy, 'use_jit') and strategy.use_jit:
            model.simulate(n_sims=100)
        
        # Measure time
        start_time = time.perf_counter()
        result = model.simulate(n_sims=n_sims, **kwargs)
        end_time = time.perf_counter()
        
        return {
            'time_seconds': end_time - start_time,
            'mean_loss': result.estimates['mean'],
            'std_loss': np.sqrt(result.estimates['variance'])
        }
    
    def load_baseline(self, baseline_path):
        """Load baseline performance data."""
        if baseline_path.exists():
            with open(baseline_path, 'r') as f:
                return json.load(f)
        return None
    
    def save_baseline(self, baseline_path, data):
        """Save baseline performance data."""
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        with open(baseline_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @pytest.mark.benchmark
    def test_establish_baseline(self, standard_portfolio, performance_baseline_path):
        """Establish or update performance baseline."""
        n_sims = 5000
        
        # Measure baseline (no optimization)
        strategy_baseline = ClassicalPricingStrategy(use_jit=False)
        baseline_perf = self.measure_performance(
            standard_portfolio, 
            strategy_baseline, 
            n_sims=n_sims,
            qmc_seed=42
        )
        
        # Measure JIT performance
        strategy_jit = ClassicalPricingStrategy(use_jit=True)
        jit_perf = self.measure_performance(
            standard_portfolio,
            strategy_jit,
            n_sims=n_sims,
            qmc_seed=42
        )
        
        # Measure QMC performance
        qmc_perf = self.measure_performance(
            standard_portfolio,
            strategy_baseline,
            n_sims=4096,  # Power of 2 for Sobol
            qmc_method="sobol",
            qmc_scramble=True,
            qmc_seed=42
        )
        
        # Measure combined optimization
        combined_perf = self.measure_performance(
            standard_portfolio,
            strategy_jit,
            n_sims=4096,
            qmc_method="sobol",
            qmc_scramble=True,
            qmc_seed=42
        )
        
        # Store results
        baseline_data = {
            'baseline': baseline_perf,
            'jit': jit_perf,
            'qmc': qmc_perf,
            'combined': combined_perf,
            'portfolio_size': len(standard_portfolio),
            'n_sims': n_sims
        }
        
        # Print performance summary
        print("\nPerformance Summary:")
        print(f"Baseline: {baseline_perf['time_seconds']:.3f}s")
        print(f"JIT: {jit_perf['time_seconds']:.3f}s (speedup: {baseline_perf['time_seconds']/jit_perf['time_seconds']:.2f}x)")
        print(f"QMC: {qmc_perf['time_seconds']:.3f}s (speedup: {baseline_perf['time_seconds']/qmc_perf['time_seconds']:.2f}x)")
        print(f"Combined: {combined_perf['time_seconds']:.3f}s (speedup: {baseline_perf['time_seconds']/combined_perf['time_seconds']:.2f}x)")
        
        # Save baseline for future comparisons
        # Uncomment to update baseline:
        # self.save_baseline(performance_baseline_path, baseline_data)
    
    @pytest.mark.benchmark
    def test_performance_regression_detection(self, standard_portfolio, performance_baseline_path):
        """Test that current performance doesn't regress from baseline."""
        # Load baseline
        baseline_data = self.load_baseline(performance_baseline_path)
        if baseline_data is None:
            pytest.skip("No baseline data available. Run test_establish_baseline first.")
        
        # Measure current performance
        strategy_current = ClassicalPricingStrategy(use_jit=False)
        current_perf = self.measure_performance(
            standard_portfolio,
            strategy_current,
            n_sims=baseline_data['n_sims'],
            qmc_seed=42
        )
        
        # Check regression
        baseline_time = baseline_data['baseline']['time_seconds']
        current_time = current_perf['time_seconds']
        
        assert current_time <= baseline_time * self.BASELINE_THRESHOLD_MULTIPLIER, \
            f"Performance regression detected: {current_time:.3f}s vs baseline {baseline_time:.3f}s"
    
    @pytest.mark.benchmark
    def test_jit_optimization_effectiveness(self, standard_portfolio):
        """Test that JIT optimization provides expected speedup."""
        n_sims = 5000
        
        # Measure without JIT
        strategy_no_jit = ClassicalPricingStrategy(use_jit=False)
        no_jit_perf = self.measure_performance(
            standard_portfolio,
            strategy_no_jit,
            n_sims=n_sims,
            qmc_seed=42
        )
        
        # Measure with JIT
        strategy_jit = ClassicalPricingStrategy(use_jit=True)
        jit_perf = self.measure_performance(
            standard_portfolio,
            strategy_jit,
            n_sims=n_sims,
            qmc_seed=42
        )
        
        # Calculate speedup
        speedup = no_jit_perf['time_seconds'] / jit_perf['time_seconds']
        
        print(f"\nJIT Speedup: {speedup:.2f}x")
        print(f"No JIT: {no_jit_perf['time_seconds']:.3f}s")
        print(f"With JIT: {jit_perf['time_seconds']:.3f}s")
        
        # JIT should provide meaningful speedup
        assert speedup >= self.MINIMUM_SPEEDUP_JIT, \
            f"JIT speedup {speedup:.2f}x is below expected {self.MINIMUM_SPEEDUP_JIT}x"
    
    @pytest.mark.benchmark
    def test_scalability_performance(self):
        """Test performance scalability with portfolio size."""
        sizes = [10, 50, 100]
        times = []
        
        strategy = ClassicalPricingStrategy(use_jit=True)
        
        for size in sizes:
            # Create portfolio of given size
            portfolio = []
            for i in range(size):
                policy = Inforce(
                    n_policies=1,
                    name=f"P{i:03d}",
                    frequency=Poisson(mu=5.0),
                    severity=Lognormal(shape=1.0, scale=np.exp(8.0)),
                    terms=PolicyTerms(
                        effective_date=date(2024, 1, 1),
                        expiration_date=date(2024, 12, 31)
                    )
                )
                portfolio.append(policy)
            
            # Measure performance
            perf = self.measure_performance(portfolio, strategy, n_sims=1000)
            times.append(perf['time_seconds'])
        
        # Check that time scales roughly linearly
        print("\nScalability Test:")
        for size, time_taken in zip(sizes, times):
            print(f"Portfolio size {size}: {time_taken:.3f}s")
        
        # Time should scale sub-linearly with size (due to vectorization)
        time_ratio_50_10 = times[1] / times[0]
        size_ratio_50_10 = sizes[1] / sizes[0]
        
        time_ratio_100_50 = times[2] / times[1]
        size_ratio_100_50 = sizes[2] / sizes[1]
        
        # Allow for some overhead, but should be better than linear
        assert time_ratio_50_10 < size_ratio_50_10 * 1.2
        assert time_ratio_100_50 < size_ratio_100_50 * 1.2
    
    @pytest.mark.benchmark
    def test_memory_efficiency(self, standard_portfolio):
        """Test that optimizations don't significantly increase memory usage."""
        # This is a placeholder - actual memory testing would require
        # memory profiling tools like memory_profiler
        
        # For now, just ensure the optimized versions run without memory errors
        strategy_jit = ClassicalPricingStrategy(use_jit=True)
        model = PricingModel(standard_portfolio, strategy=strategy_jit)
        
        # Run with large number of simulations
        result = model.simulate(n_sims=10000, qmc_seed=42)
        
        # Basic sanity checks
        assert result is not None
        assert result.samples.shape[0] == 10000
        assert np.all(np.isfinite(result.samples))
    
    @pytest.mark.benchmark
    @pytest.mark.parametrize("backend,n_sims", [
        ("classical", 5000),
        ("classical_jit", 5000),
        ("classical_qmc", 4096),
        ("classical_jit_qmc", 4096),
    ])
    def test_optimization_combinations_performance(self, standard_portfolio, backend, n_sims):
        """Test performance of different optimization combinations."""
        use_jit = "jit" in backend
        use_qmc = "qmc" in backend
        
        strategy = ClassicalPricingStrategy(use_jit=use_jit)
        
        kwargs = {'n_sims': n_sims, 'qmc_seed': 42}
        if use_qmc:
            kwargs.update({
                'qmc_method': 'sobol',
                'qmc_scramble': True,
                'qmc_seed': 42
            })
        
        # Measure performance
        perf = self.measure_performance(standard_portfolio, strategy, **kwargs)
        
        print(f"\n{backend} performance: {perf['time_seconds']:.3f}s")
        
        # All methods should complete in reasonable time
        assert perf['time_seconds'] < 12.0  # 12 seconds max for 50 policies
        
        # Results should be reasonable
        assert perf['mean_loss'] > 0
        assert perf['std_loss'] > 0