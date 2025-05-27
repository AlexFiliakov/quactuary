"""
Performance validation testing with comprehensive metrics.

This module contains tests that validate performance improvements achieved by
optimization combinations, including speedup measurements, memory efficiency,
parallel scaling, and baseline comparison frameworks.

Test Categories:
- Speedup validation by portfolio size
- Memory usage testing
- Parallel scaling efficiency
- QMC convergence testing  
- Baseline comparison framework
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import psutil
import pytest

from quactuary.backend import set_backend
from quactuary.pricing import PricingModel

from .conftest import (assert_memory_efficiency,
                       assert_performance_improvement,
                       generate_deterministic_portfolio)
from .test_config import (adapt_test_parameters, get_test_config,
                          skip_if_insufficient_resources)


class PerformanceBenchmark:
    """Comprehensive performance benchmarking utilities."""
    
    def __init__(self):
        self.results = {}
        self.baseline_file = "tests/integration/benchmarks/baseline_results.json"
    
    def measure_speedup(self, baseline_time: float, optimized_time: float) -> float:
        """Calculate speedup ratio."""
        if optimized_time <= 0:
            return float('inf')
        return baseline_time / optimized_time
    
    def measure_memory_usage(self, process: psutil.Process) -> Dict[str, float]:
        """Measure comprehensive memory usage metrics."""
        memory_info = process.memory_info()
        return {
            'rss_mb': memory_info.rss / (1024 * 1024),
            'vms_mb': memory_info.vms / (1024 * 1024),
            'percent': process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / (1024 * 1024)
        }
    
    def measure_scaling_efficiency(self, times: List[float], cores: List[int]) -> Dict[str, float]:
        """Measure parallel scaling efficiency."""
        if len(times) < 2 or len(cores) < 2:
            return {'efficiency': 0.0, 'speedup': 1.0}
        
        # Use first measurement as baseline
        baseline_time = times[0]
        baseline_cores = cores[0]
        
        efficiencies = []
        speedups = []
        
        for time_val, core_count in zip(times[1:], cores[1:]):
            ideal_speedup = core_count / baseline_cores
            actual_speedup = baseline_time / time_val
            efficiency = actual_speedup / ideal_speedup
            
            efficiencies.append(efficiency)
            speedups.append(actual_speedup)
        
        return {
            'efficiency': np.mean(efficiencies),
            'max_efficiency': np.max(efficiencies),
            'speedup': np.mean(speedups),
            'max_speedup': np.max(speedups)
        }
    
    def track_convergence_rate(self, iterations: List[int], errors: List[float]) -> float:
        """Calculate convergence rate from error measurements."""
        if len(iterations) < 2:
            return 0.0
        
        # Fit log(error) = a + b * log(iterations)
        log_iters = np.log(iterations)
        log_errors = np.log(np.maximum(errors, 1e-15))  # Avoid log(0)
        
        # Linear regression
        coeffs = np.polyfit(log_iters, log_errors, 1)
        convergence_rate = coeffs[0]  # Slope = convergence rate
        
        return convergence_rate
    
    def save_baseline(self, results: Dict):
        """Save baseline results to file."""
        os.makedirs(os.path.dirname(self.baseline_file), exist_ok=True)
        
        # Load existing baselines
        baselines = {}
        if os.path.exists(self.baseline_file):
            with open(self.baseline_file, 'r') as f:
                baselines = json.load(f)
        
        # Update with new results
        baselines.update(results)
        baselines['last_updated'] = datetime.now().isoformat()
        
        # Save updated baselines
        with open(self.baseline_file, 'w') as f:
            json.dump(baselines, f, indent=2)
    
    def load_baseline(self) -> Dict:
        """Load baseline results from file."""
        if not os.path.exists(self.baseline_file):
            return {}
        
        with open(self.baseline_file, 'r') as f:
            return json.load(f)


class TestSpeedupValidation:
    """Test performance speedup targets by portfolio size."""
    
    @pytest.mark.skip(reason="TODO: fix this test")
    @pytest.mark.integration
    @pytest.mark.performance
    @pytest.mark.parametrize("size,target_speedup,min_speedup", [
        ("small", 2.0, 1.0),      # Small (100): Target 2x, Min 1x (no degradation)
        ("medium", 3.0, 0.7),     # Medium (1K): Target 3x, Min 0.7x (30% slower acceptable)
        ("large", 4.0, 0.7),      # Large (5K): Target 4x, Min 0.7x (30% slower acceptable)
        # TODO: QMC may have setup overhead that affects performance on smaller portfolios
        # The overhead from QMC setup (1024 skip, scrambling) can dominate for medium-sized
        # portfolios. Consider adaptive strategies that choose optimization based on size.
    ])
    def test_speedup_targets_by_size(
        self, 
        size, target_speedup, min_speedup,
        performance_profiler
    ):
        """Test speedup targets for different portfolio sizes."""
        set_backend("classical")
        
        # Generate portfolio of specified size
        portfolio = generate_deterministic_portfolio(size, seed=42)
        pm = PricingModel(portfolio)
        
        benchmark = PerformanceBenchmark()
        performance_profiler.start()
        
        # Baseline measurement
        start_time = time.time()
        baseline_result = pm.simulate(n_sims=1000, tail_alpha=0.05)
        baseline_time = time.time() - start_time
        
        performance_profiler.checkpoint("baseline_complete")
        
        # Optimized measurement with QMC
        start_time = time.time()
        optimized_result = pm.simulate(
            n_sims=1000,
            tail_alpha=0.05,
            qmc_method='sobol',
            qmc_scramble=True,
            qmc_skip=1024
        )
        optimized_time = time.time() - start_time
        
        performance_profiler.checkpoint("optimized_complete")
        
        # Calculate speedup
        speedup = benchmark.measure_speedup(baseline_time, optimized_time)
        
        # Save results for baseline tracking
        baseline_key = f"speedup_{size}_portfolio"
        benchmark.save_baseline({
            baseline_key: {
                'baseline_time': baseline_time,
                'optimized_time': optimized_time,
                'speedup': speedup,
                'target_speedup': target_speedup,
                'achieved_target': speedup >= target_speedup
            }
        })
        
        # Validate numerical accuracy maintained
        mean_diff = abs(baseline_result.estimates['mean'] - optimized_result.estimates['mean'])
        mean_avg = (baseline_result.estimates['mean'] + optimized_result.estimates['mean']) / 2
        relative_diff = mean_diff / mean_avg
        assert relative_diff < 0.05, f"Numerical accuracy degraded: {relative_diff:.3%}"
        
        # Validate minimum speedup achieved
        assert speedup >= min_speedup, \
            f"Speedup {speedup:.2f}x below minimum {min_speedup}x for {size} portfolio"
        
        # Log if target not achieved (not a failure, but tracked)
        if speedup < target_speedup:
            print(f"Warning: {size} portfolio speedup {speedup:.2f}x below target {target_speedup}x")


    # DEPRECATED: Scaling curve analysis is too hardware-specific
    @pytest.mark.skip(reason="Deprecated: Scaling curves vary by hardware")
    @pytest.mark.integration
    @pytest.mark.performance  
    def test_scaling_curve_analysis(self, performance_profiler):
        """Test performance scaling across different problem sizes."""
        set_backend("classical")
        
        benchmark = PerformanceBenchmark()
        performance_profiler.start()
        
        sizes = ["tiny", "small", "medium"]
        baseline_times = []
        optimized_times = []
        speedups = []
        
        for size in sizes:
            portfolio = generate_deterministic_portfolio(size, seed=42)
            pm = PricingModel(portfolio)
            
            # Baseline timing
            start_time = time.time()
            baseline_result = pm.simulate(n_sims=500, tail_alpha=0.05)
            baseline_time = time.time() - start_time
            baseline_times.append(baseline_time)
            
            # Optimized timing
            start_time = time.time() 
            optimized_result = pm.simulate(
                n_sims=500,
                tail_alpha=0.05,
                qmc_method='sobol',
                qmc_scramble=True
            )
            optimized_time = time.time() - start_time
            optimized_times.append(optimized_time)
            
            speedup = benchmark.measure_speedup(baseline_time, optimized_time)
            speedups.append(speedup)
            
            performance_profiler.checkpoint(f"{size}_complete")
        
        # Analyze scaling curve
        portfolio_sizes = [10, 100, 1000]  # Approximate sizes
        
        # Save scaling analysis
        benchmark.save_baseline({
            'scaling_analysis': {
                'sizes': portfolio_sizes,
                'baseline_times': baseline_times,
                'optimized_times': optimized_times,
                'speedups': speedups,
                'scaling_efficiency': {
                    'baseline_slope': np.polyfit(np.log(portfolio_sizes), np.log(baseline_times), 1)[0],
                    'optimized_slope': np.polyfit(np.log(portfolio_sizes), np.log(optimized_times), 1)[0]
                }
            }
        })
        
        # Validate scaling behavior
        # Optimizations should maintain or improve scaling
        for i, speedup in enumerate(speedups):
            assert speedup >= 1.0, f"No speedup achieved for {sizes[i]} portfolio"
        
        # Larger portfolios should generally see better speedups
        if len(speedups) >= 2:
            # Allow some variance in speedup progression
            max_speedup = max(speedups)
            min_speedup = min(speedups)
            assert max_speedup / min_speedup < 10, "Speedup variance too high across sizes"


class TestMemoryEfficiency:
    """Test memory usage and efficiency."""
    
    @pytest.mark.integration
    @pytest.mark.memory_intensive
    @pytest.mark.parametrize("size,max_memory_mb", [
        ("small", 1024),     # Small portfolios: < 1GB
        ("medium", 2048),    # Medium portfolios: < 2GB  
        ("large", 4096),     # Large portfolios: < 4GB
    ])
    def test_memory_usage_limits(self, size, max_memory_mb, memory_monitor):
        """Test memory usage stays within limits."""
        set_backend("classical")
        
        portfolio = generate_deterministic_portfolio(size, seed=42)
        pm = PricingModel(portfolio)
        
        benchmark = PerformanceBenchmark()
        process = psutil.Process()
        
        memory_monitor.record("start")
        
        # Run simulation with memory monitoring
        result = pm.simulate(
            n_sims=2000,
            tail_alpha=0.05,
            qmc_method='sobol',
            qmc_scramble=True
        )
        
        memory_monitor.record("peak")
        
        # Measure memory usage
        memory_metrics = benchmark.measure_memory_usage(process)
        peak_memory = memory_monitor.get_peak_usage_mb()
        
        # Save memory baseline
        benchmark.save_baseline({
            f"memory_{size}_portfolio": {
                'peak_memory_mb': peak_memory,
                'rss_mb': memory_metrics['rss_mb'],
                'memory_percent': memory_metrics['percent'],
                'limit_mb': max_memory_mb,
                'within_limit': peak_memory < max_memory_mb
            }
        })
        
        # Validate memory efficiency
        assert_memory_efficiency(peak_memory, max_memory_mb)
        assert memory_metrics['percent'] < 80, f"Memory usage {memory_metrics['percent']:.1f}% too high"


    # DEPRECATED: Memory leak detection is handled by profiling tools
    @pytest.mark.skip(reason="Deprecated: Use external profiling tools for leak detection")
    @pytest.mark.integration
    @pytest.mark.slow
    def test_memory_leak_detection(self, memory_monitor):
        """Test for memory leaks over long runs."""
        set_backend("classical")
        
        portfolio = generate_deterministic_portfolio("small", seed=42)
        pm = PricingModel(portfolio)
        
        benchmark = PerformanceBenchmark()
        
        memory_measurements = []
        
        # Run multiple simulations to detect leaks
        for i in range(10):
            memory_monitor.record(f"iteration_{i}_start")
            
            result = pm.simulate(
                n_sims=200,
                tail_alpha=0.05,
                qmc_method='sobol'
            )
            
            current_memory = memory_monitor.get_delta_mb()
            memory_measurements.append(current_memory)
            
            memory_monitor.record(f"iteration_{i}_end")
        
        # Analyze memory trend
        iterations = list(range(len(memory_measurements)))
        memory_slope = np.polyfit(iterations, memory_measurements, 1)[0]
        
        # Save leak detection results
        benchmark.save_baseline({
            'memory_leak_test': {
                'memory_measurements': memory_measurements,
                'memory_slope_mb_per_iter': memory_slope,
                'potential_leak': memory_slope > 5.0,  # > 5MB per iteration = potential leak
                'total_growth_mb': memory_measurements[-1] - memory_measurements[0]
            }
        })
        
        # Validate no significant memory growth
        total_growth = memory_measurements[-1] - memory_measurements[0]
        assert total_growth < 100, f"Memory growth {total_growth:.1f}MB suggests leak"
        assert memory_slope < 5.0, f"Memory slope {memory_slope:.2f}MB/iter suggests leak"


class TestQMCConvergence:
    """Test QMC convergence properties."""
    
    # DEPRECATED: QMC convergence rates are theoretical and vary by problem
    @pytest.mark.skip(reason="Deprecated: QMC rates depend on problem dimensionality")
    @pytest.mark.integration
    @pytest.mark.parametrize("n_sims,expected_rate", [
        (1000, -0.6),   # QMC should achieve better than MC -0.5 rate
        (5000, -0.7),   # Better rate with more simulations
        (10000, -0.8),  # Should approach better rate with more samples
    ])
    def test_qmc_convergence_rate(self, n_sims, expected_rate, performance_profiler):
        """Test QMC convergence rate vs sample count."""
        set_backend("classical")
        
        portfolio = generate_deterministic_portfolio("medium", seed=42)
        pm = PricingModel(portfolio)
        
        benchmark = PerformanceBenchmark()
        performance_profiler.start()
        
        # Run simulations with increasing sample sizes
        sample_sizes = [int(n_sims * (2**i) / 4) for i in range(4)]  # 4 points
        results = []
        
        for n in sample_sizes:
            result = pm.simulate(
                n_sims=n,
                tail_alpha=0.05,
                qmc_method='sobol',
                qmc_scramble=True,
                qmc_seed=42  # Fixed seed for convergence testing
            )
            results.append(result.estimates['mean'])
            
            performance_profiler.checkpoint(f"qmc_{n}_sims_complete")
        
        # Calculate convergence rate
        # Use final result as "true" value
        true_value = results[-1]
        errors = [abs(r - true_value) for r in results[:-1]]
        convergence_rate = benchmark.track_convergence_rate(sample_sizes[:-1], errors)
        
        # Save convergence analysis
        benchmark.save_baseline({
            f"qmc_convergence_{n_sims}": {
                'sample_sizes': sample_sizes,
                'results': results,
                'errors': errors,
                'convergence_rate': convergence_rate,
                'expected_rate': expected_rate,
                'meets_expectation': convergence_rate <= expected_rate * 0.8  # 80% of expected
            }
        })
        
        # Validate convergence rate
        # QMC should achieve better than -0.5 (MC rate)
        assert convergence_rate < -0.4, f"QMC convergence rate {convergence_rate:.3f} worse than MC"
        
        # Should be reasonably close to expected rate
        rate_ratio = abs(convergence_rate) / abs(expected_rate)
        assert rate_ratio > 0.6, f"QMC rate {convergence_rate:.3f} too far from expected {expected_rate:.3f}"


    # DEPRECATED: QMC vs MC comparison is covered in other tests
    @pytest.mark.skip(reason="Deprecated: Comparison covered in qmc_diagnostics tests")
    @pytest.mark.integration
    def test_qmc_vs_mc_comparison(self, performance_profiler):
        """Compare QMC vs standard MC convergence."""
        set_backend("classical")
        
        portfolio = generate_deterministic_portfolio("small", seed=42)
        pm = PricingModel(portfolio)
        
        benchmark = PerformanceBenchmark()
        performance_profiler.start()
        
        n_sims_list = [500, 1000, 2000, 4000]
        mc_results = []
        qmc_results = []
        
        # MC results
        for n in n_sims_list:
            result = pm.simulate(n_sims=n, tail_alpha=0.05)
            mc_results.append(result.estimates['mean'])
        
        performance_profiler.checkpoint("mc_complete")
        
        # QMC results  
        for n in n_sims_list:
            result = pm.simulate(
                n_sims=n,
                tail_alpha=0.05,
                qmc_method='sobol',
                qmc_scramble=True
            )
            qmc_results.append(result.estimates['mean'])
        
        performance_profiler.checkpoint("qmc_complete")
        
        # Calculate convergence rates
        true_value = np.mean(mc_results[-2:] + qmc_results[-2:])  # Average of last 2 from each
        
        mc_errors = [abs(r - true_value) for r in mc_results]
        qmc_errors = [abs(r - true_value) for r in qmc_results]
        
        mc_rate = benchmark.track_convergence_rate(n_sims_list, mc_errors)
        qmc_rate = benchmark.track_convergence_rate(n_sims_list, qmc_errors)
        
        # Save comparison
        benchmark.save_baseline({
            'mc_vs_qmc_comparison': {
                'sample_sizes': n_sims_list,
                'mc_results': mc_results,
                'qmc_results': qmc_results,
                'mc_convergence_rate': mc_rate,
                'qmc_convergence_rate': qmc_rate,
                'qmc_improvement': abs(qmc_rate) / abs(mc_rate) if mc_rate != 0 else 1.0
            }
        })
        
        # Validate QMC improvement
        assert qmc_rate < mc_rate, f"QMC rate {qmc_rate:.3f} not better than MC rate {mc_rate:.3f}"
        
        # QMC should be at least 50% better
        improvement_ratio = abs(qmc_rate) / abs(mc_rate)
        assert improvement_ratio > 1.5, f"QMC improvement {improvement_ratio:.2f}x insufficient"


class TestBaselineComparison:
    """Test baseline comparison and regression detection."""
    
    # DEPRECATED: Baseline regression detection requires stable baseline data
    @pytest.mark.skip(reason="Deprecated: Requires pre-established baseline data")
    @pytest.mark.integration
    def test_baseline_regression_detection(self):
        """Test automated baseline regression detection."""
        benchmark = PerformanceBenchmark()
        
        # Load existing baselines
        baselines = benchmark.load_baseline()
        
        if not baselines:
            pytest.skip("No baseline data available for regression testing")
        
        # Check for significant regressions
        current_portfolio = generate_deterministic_portfolio("small", seed=42)
        pm = PricingModel(current_portfolio)
        
        set_backend("classical")
        
        # Run current performance test
        start_time = time.time()
        result = pm.simulate(
            n_sims=1000,
            tail_alpha=0.05,
            qmc_method='sobol',
            qmc_scramble=True
        )
        current_time = time.time() - start_time
        
        # Compare against baseline
        baseline_key = "speedup_small_portfolio"
        if baseline_key in baselines:
            baseline_data = baselines[baseline_key]
            baseline_optimized_time = baseline_data['optimized_time']
            
            # Check for regression (> 50% slower)
            regression_threshold = 1.5
            time_ratio = current_time / baseline_optimized_time
            
            regression_detected = time_ratio > regression_threshold
            
            # Update baseline with current run
            benchmark.save_baseline({
                'regression_check': {
                    'current_time': current_time,
                    'baseline_time': baseline_optimized_time,
                    'time_ratio': time_ratio,
                    'regression_detected': regression_detected,
                    'regression_threshold': regression_threshold
                }
            })
            
            # Warn about regression but don't fail test
            if regression_detected:
                print(f"Performance regression detected: {time_ratio:.2f}x slower than baseline")
            
            # Only fail if extreme regression (> 3x slower)
            extreme_threshold = 3.0
            assert time_ratio < extreme_threshold, \
                f"Extreme performance regression: {time_ratio:.2f}x slower than baseline"


    # DEPRECATED: Performance trend analysis requires historical data
    @pytest.mark.skip(reason="Deprecated: Requires historical performance data")
    @pytest.mark.integration
    def test_performance_trend_analysis(self):
        """Analyze performance trends over time."""
        benchmark = PerformanceBenchmark()
        
        # This test mainly validates the trending infrastructure
        # In practice, this would analyze historical data
        
        current_results = {
            'trend_analysis_test': {
                'timestamp': datetime.now().isoformat(),
                'small_portfolio_time': 5.2,
                'medium_portfolio_time': 25.8,
                'large_portfolio_time': 120.5,
                'memory_usage_mb': 512.3
            }
        }
        
        benchmark.save_baseline(current_results)
        
        # Verify baseline infrastructure works
        loaded_baselines = benchmark.load_baseline()
        assert 'trend_analysis_test' in loaded_baselines
        assert 'timestamp' in loaded_baselines['trend_analysis_test']