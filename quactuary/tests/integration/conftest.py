"""
Shared fixtures and utilities for integration tests.

This module provides pytest fixtures for different portfolio sizes, performance
measurement decorators, memory profiling utilities, and test data generators
for reproducible integration testing.
"""

import time
import psutil
import pytest
import numpy as np
from datetime import date
from typing import Dict, List, Optional, Tuple
from functools import wraps

import quactuary.book as book
from quactuary.book import LOB, PolicyTerms, Inforce, Portfolio
from quactuary.distributions.frequency import Poisson, NegativeBinomial, Geometric
from quactuary.distributions.severity import Lognormal, Pareto, Exponential
from quactuary.pricing import PricingModel
from quactuary.backend import set_backend


# Performance measurement decorators
def measure_performance(func):
    """Decorator to measure execution time and memory usage."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process()
        start_time = time.time()
        start_memory = process.memory_info().rss
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = process.memory_info().rss
        
        execution_time = end_time - start_time
        memory_delta = end_memory - start_memory
        peak_memory = process.memory_info().peak_wss if hasattr(process.memory_info(), 'peak_wss') else end_memory
        
        return {
            'result': result,
            'execution_time': execution_time,
            'memory_delta': memory_delta,
            'peak_memory': peak_memory
        }
    return wrapper


class PerformanceProfiler:
    """Utility class for detailed performance profiling."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.start_time = None
        self.start_memory = None
        self.measurements = []
    
    def start(self):
        """Start performance measurement."""
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss
        self.measurements = []
    
    def checkpoint(self, label: str):
        """Record a performance checkpoint."""
        current_time = time.time()
        current_memory = self.process.memory_info().rss
        
        self.measurements.append({
            'label': label,
            'timestamp': current_time,
            'elapsed_time': current_time - self.start_time,
            'memory_usage': current_memory,
            'memory_delta': current_memory - self.start_memory
        })
    
    def get_results(self) -> Dict:
        """Get comprehensive performance results."""
        if not self.measurements:
            return {}
        
        total_time = self.measurements[-1]['elapsed_time']
        peak_memory = max(m['memory_usage'] for m in self.measurements)
        memory_delta = self.measurements[-1]['memory_delta']
        
        return {
            'total_execution_time': total_time,
            'peak_memory_usage': peak_memory,
            'total_memory_delta': memory_delta,
            'checkpoints': self.measurements
        }


# Test data generators
def generate_deterministic_portfolio(
    size: str = "small",
    seed: int = 42,
    complexity: str = "simple"
) -> Portfolio:
    """Generate deterministic test portfolios for reproducible testing."""
    np.random.seed(seed)
    
    size_config = {
        "tiny": {"n_policies": 10, "buckets": 1},
        "small": {"n_policies": 100, "buckets": 2},
        "medium": {"n_policies": 1000, "buckets": 3},
        "large": {"n_policies": 5000, "buckets": 4},
        "extreme": {"n_policies": 10000, "buckets": 5}
    }
    
    config = size_config.get(size, size_config["small"])
    
    # Create multiple buckets for portfolio diversity
    inforces = []
    
    for i in range(config["buckets"]):
        # Vary distributions across buckets
        if i % 3 == 0:
            freq = Poisson(mu=2.5 + i * 0.5)
            sev = Lognormal(shape=1.2 + i * 0.1, loc=0, scale=10000 * (1 + i * 0.2))
            lob = LOB.GLPL
            exposure_base = book.SALES
            exposure_amount = 1_000_000_000 * (1 + i)
        elif i % 3 == 1:
            freq = NegativeBinomial(r=10 + i * 2, p=0.3 + i * 0.1)
            sev = Pareto(b=1.5 + i * 0.1, loc=0, scale=5000 * (1 + i * 0.3))
            lob = LOB.WC
            exposure_base = book.PAYROLL
            exposure_amount = 50_000_000 * (1 + i)
        else:
            freq = Geometric(p=1/(5 + i))
            sev = Exponential(scale=8000 * (1 + i * 0.4))
            lob = LOB.CAuto
            exposure_base = book.VEHICLES
            exposure_amount = 1000 * (1 + i)
        
        policy_terms = PolicyTerms(
            effective_date=date(2026, 1, 1),
            expiration_date=date(2027, 1, 1),
            lob=lob,
            exposure_base=exposure_base,
            exposure_amount=exposure_amount,
            retention_type="deductible",
            per_occ_retention=100_000 + i * 50_000,
            coverage="occ"
        )
        
        n_policies_bucket = config["n_policies"] // config["buckets"]
        if i == config["buckets"] - 1:  # Last bucket gets remainder
            n_policies_bucket += config["n_policies"] % config["buckets"]
        
        inforce = Inforce(
            n_policies=n_policies_bucket,
            terms=policy_terms,
            frequency=freq,
            severity=sev,
            name=f"{lob.name} Bucket {i+1}"
        )
        
        inforces.append(inforce)
    
    # Create portfolio
    portfolio = inforces[0]
    for inforce in inforces[1:]:
        portfolio += inforce
    
    return portfolio


# Fixtures for different portfolio sizes
@pytest.fixture(scope="session")
def tiny_portfolio():
    """Tiny portfolio for quick tests (10 policies)."""
    return generate_deterministic_portfolio("tiny", seed=42)


@pytest.fixture(scope="session")
def small_portfolio():
    """Small portfolio for basic integration tests (100 policies)."""
    return generate_deterministic_portfolio("small", seed=42)


@pytest.fixture(scope="session")
def medium_portfolio():
    """Medium portfolio for performance testing (1000 policies)."""
    return generate_deterministic_portfolio("medium", seed=42)


@pytest.fixture(scope="session")
def large_portfolio():
    """Large portfolio for stress testing (5000 policies)."""
    return generate_deterministic_portfolio("large", seed=42)


@pytest.fixture(scope="session") 
def extreme_portfolio():
    """Extreme portfolio for scalability testing (10000 policies)."""
    return generate_deterministic_portfolio("extreme", seed=42)


@pytest.fixture(scope="function")
def portfolio_copy(small_portfolio):
    """Provide a fresh copy of small portfolio for each test."""
    # Note: Portfolio objects should be immutable for testing
    return small_portfolio


@pytest.fixture(scope="function")
def performance_profiler():
    """Provide a performance profiler instance for each test."""
    return PerformanceProfiler()


@pytest.fixture(scope="function")
def baseline_backend():
    """Set backend to classical for baseline measurements."""
    set_backend("classical")
    yield
    # Cleanup happens automatically


# Optimization configuration fixtures
@pytest.fixture(scope="function", params=[
    {"jit": False, "qmc": False, "parallel": False, "vectorized": False},
    {"jit": True, "qmc": False, "parallel": False, "vectorized": False},
    {"jit": False, "qmc": True, "parallel": False, "vectorized": False},
    {"jit": False, "qmc": False, "parallel": True, "vectorized": False},
    {"jit": True, "qmc": True, "parallel": False, "vectorized": False},
    {"jit": True, "qmc": False, "parallel": True, "vectorized": False},
    {"jit": False, "qmc": True, "parallel": True, "vectorized": False},
    {"jit": True, "qmc": True, "parallel": True, "vectorized": False},
])
def optimization_config(request):
    """Parametrized fixture for different optimization combinations."""
    return request.param


# Simulation parameters
@pytest.fixture(scope="function", params=[100, 500, 1000, 5000])
def n_simulations(request):
    """Parametrized fixture for different simulation counts."""
    return request.param


@pytest.fixture(scope="function", params=[0.01, 0.05, 0.1])
def tail_alpha(request):
    """Parametrized fixture for different confidence levels."""
    return request.param


# Memory and resource fixtures
@pytest.fixture(scope="function")
def memory_monitor():
    """Monitor memory usage during test execution."""
    class MemoryMonitor:
        def __init__(self):
            self.process = psutil.Process()
            self.initial_memory = self.process.memory_info().rss
            self.peak_memory = self.initial_memory
            self.measurements = []
        
        def record(self, label: str = ""):
            current_memory = self.process.memory_info().rss
            self.peak_memory = max(self.peak_memory, current_memory)
            self.measurements.append({
                'label': label,
                'memory': current_memory,
                'delta': current_memory - self.initial_memory
            })
        
        def get_peak_usage_mb(self):
            return self.peak_memory / (1024 * 1024)
        
        def get_delta_mb(self):
            current = self.process.memory_info().rss
            return (current - self.initial_memory) / (1024 * 1024)
    
    return MemoryMonitor()


# Configuration for pytest markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "integration: integration tests")
    config.addinivalue_line("markers", "slow: tests that take > 1 minute")
    config.addinivalue_line("markers", "memory_intensive: tests requiring > 4GB RAM")
    config.addinivalue_line("markers", "performance: performance validation tests")
    config.addinivalue_line("markers", "accuracy: numerical accuracy tests")


# Helper functions for test validation
def assert_numerical_accuracy(result1, result2, tolerance_mean=1e-6, tolerance_quantiles=1e-4):
    """Assert numerical accuracy between two simulation results."""
    # Mean comparison
    mean_diff = abs(result1.estimates['mean'] - result2.estimates['mean'])
    mean_relative_error = mean_diff / abs(result1.estimates['mean'])
    assert mean_relative_error < tolerance_mean, f"Mean relative error {mean_relative_error} exceeds tolerance {tolerance_mean}"
    
    # Variance comparison  
    var_diff = abs(result1.estimates['variance'] - result2.estimates['variance'])
    var_relative_error = var_diff / abs(result1.estimates['variance'])
    assert var_relative_error < tolerance_quantiles, f"Variance relative error {var_relative_error} exceeds tolerance {tolerance_quantiles}"
    
    # VaR comparison
    var_diff = abs(result1.estimates['VaR'] - result2.estimates['VaR'])
    var_relative_error = var_diff / abs(result1.estimates['VaR'])
    assert var_relative_error < tolerance_quantiles, f"VaR relative error {var_relative_error} exceeds tolerance {tolerance_quantiles}"


def assert_performance_improvement(baseline_time, optimized_time, min_speedup=2.0):
    """Assert that optimization provides minimum performance improvement."""
    speedup = baseline_time / optimized_time
    assert speedup >= min_speedup, f"Speedup {speedup:.2f}x is below minimum required {min_speedup}x"


def assert_memory_efficiency(memory_usage_mb, max_memory_mb=4096):
    """Assert that memory usage is within acceptable limits."""
    assert memory_usage_mb < max_memory_mb, f"Memory usage {memory_usage_mb:.1f}MB exceeds limit {max_memory_mb}MB"