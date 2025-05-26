"""
Enhanced performance testing framework with adaptive baselines.

This module extends the existing PerformanceBenchmark class to integrate with
the adaptive baseline system for environment-aware performance testing.
"""

import time
import functools
import warnings
from typing import Callable, Dict, Any, Optional, Tuple
from contextlib import contextmanager
import unittest

from quactuary.benchmarks import PerformanceBenchmark, BenchmarkResult
from quactuary.performance_baseline import AdaptiveBaselineManager, HardwareProfile


class AdaptivePerformanceBenchmark(PerformanceBenchmark):
    """
    Extended performance benchmark with adaptive baseline support.
    
    This class enhances the existing PerformanceBenchmark with:
    - Automatic baseline management
    - Hardware-aware performance comparison
    - Regression detection
    - CI/CD integration support
    """
    
    def __init__(self, output_dir: str = "./benchmark_results", 
                 baseline_dir: str = "./performance_baselines"):
        """Initialize adaptive benchmark framework."""
        super().__init__(output_dir)
        self.baseline_manager = AdaptiveBaselineManager(baseline_dir)
        self.regression_failures = []
        
    def benchmark_with_baseline(
        self,
        test_func: Callable,
        test_name: str,
        sample_size: int,
        warm_up_runs: int = 1,
        check_regression: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[BenchmarkResult, bool, str]:
        """
        Run benchmark and compare with baseline.
        
        Args:
            test_func: Function to benchmark
            test_name: Name of the test
            sample_size: Number of samples/iterations
            warm_up_runs: Number of warm-up runs before measurement
            check_regression: Whether to check for regression
            metadata: Additional metadata to store
            
        Returns:
            Tuple of (BenchmarkResult, is_regression, message)
        """
        # Warm up
        for _ in range(warm_up_runs):
            test_func()
        
        # Measure performance
        with self.measure_performance(test_name):
            result = test_func()
        
        # Create benchmark result
        execution_time = self.last_measurement['execution_time']
        
        # Record in baseline system
        baseline = self.baseline_manager.record_performance(
            test_name=test_name,
            execution_time=execution_time,
            sample_size=sample_size,
            metadata=metadata
        )
        
        # Check for regression if requested
        is_regression = False
        message = "Performance recorded"
        
        if check_regression:
            is_regression, expected_time, message = self.baseline_manager.check_regression(
                test_name=test_name,
                current_time=execution_time,
                sample_size=sample_size
            )
            
            if is_regression:
                self.regression_failures.append({
                    'test_name': test_name,
                    'execution_time': execution_time,
                    'expected_time': expected_time,
                    'message': message
                })
        
        # Create standard benchmark result
        benchmark_result = BenchmarkResult(
            name=test_name,
            portfolio_size=sample_size,
            n_simulations=1,
            execution_time=execution_time,
            memory_used=self.last_measurement.get('memory_used', 0),
            memory_peak=self.last_measurement.get('memory_peak', 0),
            samples_per_second=sample_size / execution_time if execution_time > 0 else 0,
            metadata={
                'baseline': baseline.to_dict(),
                'regression_check': {
                    'is_regression': is_regression,
                    'message': message
                },
                **(metadata or {})
            }
        )
        
        self.results.append(benchmark_result)
        
        return benchmark_result, is_regression, message
    
    def get_regression_report(self) -> Dict[str, Any]:
        """Get summary of all regression failures."""
        return {
            'total_tests': len(self.results),
            'regression_count': len(self.regression_failures),
            'regression_rate': len(self.regression_failures) / len(self.results) if self.results else 0,
            'failures': self.regression_failures,
            'hardware_profile': self.baseline_manager.current_profile.to_dict()
        }


class PerformanceTestCase(unittest.TestCase):
    """
    Base test case for performance tests with adaptive baseline support.
    
    Inherit from this class to create performance tests that automatically
    track baselines and detect regressions.
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up performance testing infrastructure."""
        cls.benchmark = AdaptivePerformanceBenchmark()
        cls.performance_results = []
        cls.allow_regressions = False  # Set to True in CI to allow but warn
    
    def assertPerformance(
        self,
        test_func: Callable,
        test_name: str,
        sample_size: int,
        max_time: Optional[float] = None,
        check_regression: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Assert that performance meets expectations.
        
        Args:
            test_func: Function to benchmark
            test_name: Name of the test
            sample_size: Number of samples/iterations
            max_time: Maximum allowed time (optional)
            check_regression: Whether to check against baseline
            metadata: Additional metadata
        """
        result, is_regression, message = self.benchmark.benchmark_with_baseline(
            test_func=test_func,
            test_name=test_name,
            sample_size=sample_size,
            check_regression=check_regression,
            metadata=metadata
        )
        
        self.performance_results.append({
            'test': test_name,
            'result': result,
            'regression': is_regression,
            'message': message
        })
        
        # Check absolute time limit if specified
        if max_time is not None:
            self.assertLessEqual(
                result.execution_time,
                max_time,
                f"Performance test '{test_name}' exceeded time limit: "
                f"{result.execution_time:.3f}s > {max_time:.3f}s"
            )
        
        # Check for regression
        if is_regression:
            if self.allow_regressions:
                warnings.warn(f"Performance regression in '{test_name}': {message}")
            else:
                self.fail(f"Performance regression detected: {message}")
    
    @classmethod
    def tearDownClass(cls):
        """Generate performance report after all tests."""
        if cls.performance_results:
            print("\n" + "=" * 60)
            print("PERFORMANCE TEST SUMMARY")
            print("=" * 60)
            
            for result in cls.performance_results:
                status = "REGRESSION" if result['regression'] else "OK"
                print(f"{result['test']}: {result['result'].execution_time:.3f}s [{status}]")
                if result['regression']:
                    print(f"  -> {result['message']}")
            
            # Save detailed report
            cls.benchmark.save_results()
            cls.benchmark.generate_report()


def performance_test(
    test_name: Optional[str] = None,
    sample_size: int = 1,
    check_regression: bool = True,
    warm_up: int = 1
):
    """
    Decorator for performance tests.
    
    Usage:
        @performance_test("my_algorithm", sample_size=1000)
        def test_my_algorithm():
            # Test code here
            pass
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self):
            # Use function name as test name if not provided
            name = test_name or func.__name__
            
            # Create test function
            def test_func():
                return func(self)
            
            # Run performance test
            self.assertPerformance(
                test_func=test_func,
                test_name=name,
                sample_size=sample_size,
                check_regression=check_regression
            )
        
        return wrapper
    
    return decorator


@contextmanager
def performance_context(test_name: str, baseline_manager: AdaptiveBaselineManager):
    """
    Context manager for ad-hoc performance measurements.
    
    Usage:
        baseline_manager = AdaptiveBaselineManager()
        with performance_context("my_operation", baseline_manager) as timer:
            # Code to measure
            expensive_operation()
            timer.sample_size = 1000  # Set sample size
    """
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.sample_size = 1
            self.metadata = {}
    
    timer = Timer()
    timer.start_time = time.perf_counter()
    
    try:
        yield timer
    finally:
        timer.end_time = time.perf_counter()
        execution_time = timer.end_time - timer.start_time
        
        # Record performance
        baseline_manager.record_performance(
            test_name=test_name,
            execution_time=execution_time,
            sample_size=timer.sample_size,
            metadata=timer.metadata
        )
        
        # Check for regression
        is_regression, expected_time, message = baseline_manager.check_regression(
            test_name=test_name,
            current_time=execution_time,
            sample_size=timer.sample_size
        )
        
        if is_regression:
            warnings.warn(f"Performance regression in '{test_name}': {message}")