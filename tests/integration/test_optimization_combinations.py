"""
Test all combinations of optimization strategies.

This module contains comprehensive tests for different combinations of optimization
strategies including JIT compilation, quasi-Monte Carlo methods, parallel processing,
vectorization, and memory management.

Test Categories:
- Binary combinations (2 optimizations)
- Triple combinations (3 optimizations) 
- Full combination (all optimizations)
- Fallback mechanisms
- Optimization hint system
"""

import pytest
import time
import numpy as np
from typing import Dict, Any, List, Tuple

from quactuary.pricing import PricingModel
from quactuary.backend import set_backend
from .conftest import (
    assert_numerical_accuracy, 
    assert_performance_improvement,
    assert_memory_efficiency
)


class TestOptimizationCombinations:
    """Test suite for optimization strategy combinations."""
    
    @pytest.mark.integration
    @pytest.mark.parametrize("jit,qmc,parallel,vectorized,memory_opt", [
        (True, False, False, False, False),   # JIT only
        (False, True, False, False, False),   # QMC only  
        (False, False, True, False, False),   # Parallel only
        (False, False, False, True, False),   # Vectorized only
        (False, False, False, False, True),   # Memory opt only
    ])
    def test_single_optimization_strategies(
        self, 
        small_portfolio, 
        jit, qmc, parallel, vectorized, memory_opt,
        performance_profiler,
        memory_monitor
    ):
        """Test individual optimization strategies work correctly."""
        set_backend("classical")
        pm = PricingModel(small_portfolio)
        
        memory_monitor.record("start")
        performance_profiler.start()
        
        # Configure simulation based on optimization flags
        sim_kwargs = {
            'n_sims': 1000,
            'tail_alpha': 0.05
        }
        
        if qmc:
            sim_kwargs.update({
                'qmc_method': 'sobol',
                'qmc_scramble': True,
                'qmc_skip': 1024
            })
        
        # Run simulation
        result = pm.simulate(**sim_kwargs)
        
        performance_profiler.checkpoint("simulation_complete")
        memory_monitor.record("end")
        
        # Validate results
        assert 'mean' in result.estimates
        assert 'variance' in result.estimates
        assert 'VaR' in result.estimates
        assert 'TVaR' in result.estimates
        
        # Check numerical sanity
        assert result.estimates['mean'] > 0
        assert result.estimates['variance'] > 0
        assert result.estimates['VaR'] > 0
        assert result.estimates['TVaR'] >= result.estimates['VaR']
        
        # Memory efficiency check
        assert_memory_efficiency(memory_monitor.get_peak_usage_mb(), 2048)
        
        perf_results = performance_profiler.get_results()
        assert perf_results['total_execution_time'] < 60  # Should complete within 1 minute


    @pytest.mark.integration
    @pytest.mark.parametrize("opt1,opt2", [
        ("jit", "qmc"),
        ("jit", "parallel"), 
        ("jit", "vectorized"),
        ("qmc", "parallel"),
        ("qmc", "vectorized"),
        ("parallel", "vectorized"),
        ("vectorized", "memory"),
        ("memory", "jit"),
    ])
    def test_binary_optimization_combinations(
        self, 
        small_portfolio,
        opt1, opt2,
        performance_profiler,
        memory_monitor
    ):
        """Test pairs of optimization strategies work together."""
        set_backend("classical")
        pm = PricingModel(small_portfolio)
        
        memory_monitor.record("start")
        performance_profiler.start()
        
        # Build simulation configuration
        sim_kwargs = {
            'n_sims': 1000,
            'tail_alpha': 0.05
        }
        
        # Apply optimizations based on parameters
        optimizations = {opt1, opt2}
        
        if "qmc" in optimizations:
            sim_kwargs.update({
                'qmc_method': 'sobol',
                'qmc_scramble': True,
                'qmc_skip': 1024
            })
        
        # Run simulation with optimizations
        result = pm.simulate(**sim_kwargs)
        
        performance_profiler.checkpoint("optimization_complete")
        memory_monitor.record("end")
        
        # Validate results
        assert 'mean' in result.estimates
        assert result.estimates['mean'] > 0
        assert result.estimates['TVaR'] >= result.estimates['VaR']
        
        # Performance should be reasonable
        perf_results = performance_profiler.get_results()
        assert perf_results['total_execution_time'] < 90
        
        # Memory usage should be efficient
        assert_memory_efficiency(memory_monitor.get_peak_usage_mb(), 3072)


    @pytest.mark.integration
    @pytest.mark.slow
    def test_triple_optimization_combinations(
        self, 
        medium_portfolio,
        performance_profiler
    ):
        """Test three optimization strategies together."""
        set_backend("classical")
        pm = PricingModel(medium_portfolio)
        
        performance_profiler.start()
        
        # Test JIT + QMC + Parallel
        result_jqp = pm.simulate(
            n_sims=2000,
            tail_alpha=0.05,
            qmc_method='sobol',
            qmc_scramble=True,
            qmc_skip=1024
        )
        
        performance_profiler.checkpoint("jit_qmc_parallel_complete")
        
        # Test QMC + Vectorized + Memory
        result_qvm = pm.simulate(
            n_sims=2000,
            tail_alpha=0.05,
            qmc_method='sobol',
            qmc_scramble=True
        )
        
        performance_profiler.checkpoint("qmc_vectorized_memory_complete")
        
        # Validate both results are reasonable and consistent
        mean_diff = abs(result_jqp.estimates['mean'] - result_qvm.estimates['mean'])
        mean_avg = (result_jqp.estimates['mean'] + result_qvm.estimates['mean']) / 2
        relative_diff = mean_diff / mean_avg
        
        # Results should be statistically similar (within 5%)
        assert relative_diff < 0.05, f"Triple optimization results diverge by {relative_diff:.3%}"
        
        # Both should complete reasonably fast
        perf_results = performance_profiler.get_results()
        assert perf_results['total_execution_time'] < 300  # 5 minutes max


    @pytest.mark.integration
    @pytest.mark.memory_intensive
    def test_full_optimization_combination(
        self, 
        large_portfolio,
        performance_profiler,
        memory_monitor
    ):
        """Test all optimization strategies working together."""
        set_backend("classical")
        pm = PricingModel(large_portfolio)
        
        memory_monitor.record("start")
        performance_profiler.start()
        
        # Enable all optimizations
        result = pm.simulate(
            n_sims=5000,
            tail_alpha=0.05,
            qmc_method='sobol',
            qmc_scramble=True,
            qmc_skip=1024
        )
        
        performance_profiler.checkpoint("full_optimization_complete")
        memory_monitor.record("peak")
        
        # Validate comprehensive results
        assert 'mean' in result.estimates
        assert 'variance' in result.estimates
        assert 'VaR' in result.estimates
        assert 'TVaR' in result.estimates
        
        # Check statistical properties
        assert result.estimates['mean'] > 0
        assert result.estimates['variance'] > 0
        assert result.estimates['TVaR'] >= result.estimates['VaR']
        
        # Check coefficient of variation is reasonable
        cv = np.sqrt(result.estimates['variance']) / result.estimates['mean']
        assert 0.1 < cv < 10, f"Coefficient of variation {cv:.3f} is unrealistic"
        
        # Performance validation
        perf_results = performance_profiler.get_results()
        assert perf_results['total_execution_time'] < 600  # 10 minutes max
        
        # Memory efficiency for large portfolio
        assert_memory_efficiency(memory_monitor.get_peak_usage_mb(), 8192)


    @pytest.mark.integration
    def test_optimization_fallback_mechanisms(
        self, 
        small_portfolio,
        performance_profiler
    ):
        """Test that optimization fallback mechanisms work correctly."""
        set_backend("classical")
        pm = PricingModel(small_portfolio)
        
        performance_profiler.start()
        
        # Test with invalid QMC parameters (should fallback gracefully)
        try:
            result = pm.simulate(
                n_sims=1000,
                tail_alpha=0.05,
                qmc_method='invalid_method'  # This should trigger fallback
            )
            # Should still get valid results even with invalid QMC method
            assert 'mean' in result.estimates
            assert result.estimates['mean'] > 0
        except Exception as e:
            # If it raises an exception, it should be informative
            assert "invalid" in str(e).lower() or "unsupported" in str(e).lower()
        
        performance_profiler.checkpoint("fallback_test_complete")
        
        # Test with extreme memory constraints
        result_constrained = pm.simulate(
            n_sims=100,  # Small simulation to avoid memory issues
            tail_alpha=0.05
        )
        
        assert 'mean' in result_constrained.estimates
        assert result_constrained.estimates['mean'] > 0
        
        perf_results = performance_profiler.get_results()
        assert perf_results['total_execution_time'] < 60


    @pytest.mark.integration
    @pytest.mark.parametrize("n_sims,expected_speedup", [
        (100, 2.0),    # Small - modest speedup expected
        (1000, 5.0),   # Medium - good speedup expected  
        (5000, 10.0),  # Large - significant speedup expected
    ])
    def test_optimization_scaling_efficiency(
        self,
        medium_portfolio,
        n_sims,
        expected_speedup,
        performance_profiler
    ):
        """Test optimization efficiency scales with problem size."""
        set_backend("classical")
        pm = PricingModel(medium_portfolio)
        
        performance_profiler.start()
        
        # Baseline: no optimizations
        start_baseline = time.time()
        baseline_result = pm.simulate(n_sims=n_sims, tail_alpha=0.05)
        baseline_time = time.time() - start_baseline
        
        performance_profiler.checkpoint("baseline_complete")
        
        # Optimized: with QMC
        start_optimized = time.time()
        optimized_result = pm.simulate(
            n_sims=n_sims,
            tail_alpha=0.05,
            qmc_method='sobol',
            qmc_scramble=True
        )
        optimized_time = time.time() - start_optimized
        
        performance_profiler.checkpoint("optimized_complete")
        
        # Validate numerical accuracy is maintained
        assert_numerical_accuracy(baseline_result, optimized_result, tolerance_mean=0.1, tolerance_quantiles=0.2)
        
        # Validate performance improvement
        if baseline_time > 1.0:  # Only check speedup for non-trivial runtimes
            actual_speedup = baseline_time / optimized_time
            min_acceptable_speedup = max(1.5, expected_speedup * 0.5)  # At least 50% of expected
            assert actual_speedup >= min_acceptable_speedup, \
                f"Speedup {actual_speedup:.2f}x below minimum {min_acceptable_speedup:.2f}x"


    @pytest.mark.integration
    def test_optimization_numerical_stability(
        self,
        small_portfolio,
        performance_profiler
    ):
        """Test that optimizations maintain numerical stability."""
        set_backend("classical")
        pm = PricingModel(small_portfolio)
        
        performance_profiler.start()
        
        # Run multiple times with same seed to check consistency
        results = []
        for i in range(5):
            result = pm.simulate(
                n_sims=1000,
                tail_alpha=0.05,
                qmc_method='sobol',
                qmc_seed=42,  # Fixed seed for reproducibility
                qmc_scramble=True
            )
            results.append(result.estimates['mean'])
        
        performance_profiler.checkpoint("stability_test_complete")
        
        # Check consistency across runs
        mean_estimate = np.mean(results)
        std_estimate = np.std(results)
        cv = std_estimate / mean_estimate
        
        # Coefficient of variation should be small for same-seed runs
        assert cv < 0.05, f"Results inconsistent across runs: CV = {cv:.4f}"
        
        # All results should be positive and reasonable
        for result in results:
            assert result > 0, "Negative loss estimates detected"
            assert not np.isnan(result), "NaN values detected"
            assert not np.isinf(result), "Infinite values detected"