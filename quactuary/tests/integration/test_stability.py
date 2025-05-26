"""Test stability improvements and utilities."""

import pytest
import numpy as np
import functools
import time
import warnings
from typing import Callable, Any, Dict, Optional, List
import os


class TestStabilizer:
    """Utilities for improving test stability."""
    
    @staticmethod
    def with_retries(max_attempts: int = 3, delay: float = 1.0, 
                     exceptions: tuple = (AssertionError,)):
        """Decorator to retry flaky tests.
        
        Args:
            max_attempts: Maximum number of attempts
            delay: Delay between attempts in seconds
            exceptions: Tuple of exceptions to catch and retry
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                last_exception = None
                
                for attempt in range(max_attempts):
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e
                        if attempt < max_attempts - 1:
                            warnings.warn(
                                f"Test {func.__name__} failed on attempt {attempt + 1}, "
                                f"retrying in {delay} seconds..."
                            )
                            time.sleep(delay)
                        else:
                            # Add diagnostic info on final failure
                            print(f"\nTest {func.__name__} failed after {max_attempts} attempts")
                            print(f"Final error: {str(e)}")
                
                raise last_exception
            
            return wrapper
        return decorator
    
    @staticmethod
    def with_fixed_seed(seed: int = 42):
        """Decorator to ensure reproducible random state.
        
        Args:
            seed: Random seed to use
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Save current random state
                np_state = np.random.get_state()
                
                try:
                    # Set fixed seed
                    np.random.seed(seed)
                    return func(*args, **kwargs)
                finally:
                    # Restore random state
                    np.random.set_state(np_state)
            
            return wrapper
        return decorator
    
    @staticmethod
    def skip_on_ci(reason: str = "Test not suitable for CI environment"):
        """Skip test when running in CI environment."""
        return pytest.mark.skipif(
            os.environ.get('CI', '').lower() == 'true',
            reason=reason
        )
    
    @staticmethod
    def requires_stable_environment():
        """Skip test if system is under heavy load."""
        import psutil
        
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        
        if cpu_percent > 80:
            pytest.skip(f"CPU usage too high: {cpu_percent}%")
        if memory_percent > 90:
            pytest.skip(f"Memory usage too high: {memory_percent}%")
    
    @staticmethod
    def diagnostic_on_failure(func: Callable) -> Callable:
        """Add diagnostic information when test fails."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Add diagnostic information
                import psutil
                from datetime import datetime
                
                print("\n" + "="*60)
                print("TEST FAILURE DIAGNOSTICS")
                print("="*60)
                print(f"Test: {func.__name__}")
                print(f"Time: {datetime.now()}")
                print(f"Error: {str(e)}")
                print(f"\nSystem State:")
                print(f"  CPU Usage: {psutil.cpu_percent()}%")
                print(f"  Memory Usage: {psutil.virtual_memory().percent}%")
                print(f"  Available Memory: {psutil.virtual_memory().available / (1024**3):.1f} GB")
                
                # Get test environment info
                from .test_config import get_test_config
                config = get_test_config()
                print(f"\nTest Configuration:")
                print(f"  Profile: {config['environment']['profile']}")
                print(f"  CPU Count: {config['environment']['cpu_count']}")
                print(f"  Memory: {config['environment']['memory_gb']:.1f} GB")
                
                print("="*60)
                raise
        
        return wrapper


class StableRandomState:
    """Context manager for stable random state in tests."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.np_state = None
    
    def __enter__(self):
        # Save current state
        self.np_state = np.random.get_state()
        # Set fixed seed
        np.random.seed(self.seed)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore state
        if self.np_state is not None:
            np.random.set_state(self.np_state)


class TestDataCache:
    """Cache expensive test data generation."""
    
    _cache: Dict[str, Any] = {}
    
    @classmethod
    def get_or_create(cls, key: str, factory: Callable, 
                      ttl: Optional[float] = None) -> Any:
        """Get cached data or create if not exists.
        
        Args:
            key: Cache key
            factory: Function to create data if not cached
            ttl: Time to live in seconds (None = forever)
        """
        # Check if exists and not expired
        if key in cls._cache:
            data, timestamp = cls._cache[key]
            if ttl is None or (time.time() - timestamp) < ttl:
                return data
        
        # Create new data
        data = factory()
        cls._cache[key] = (data, time.time())
        return data
    
    @classmethod
    def clear(cls):
        """Clear all cached data."""
        cls._cache.clear()


# Stability fixtures

@pytest.fixture
def stable_random():
    """Fixture providing stable random state."""
    with StableRandomState(seed=42):
        yield


@pytest.fixture
def retry_on_failure():
    """Fixture to enable retries for a test."""
    def retry_wrapper(func, max_attempts=3):
        return TestStabilizer.with_retries(max_attempts=max_attempts)(func)
    return retry_wrapper


@pytest.fixture(autouse=True)
def test_isolation():
    """Ensure test isolation."""
    # Clear any caches
    TestDataCache.clear()
    
    # Reset numpy random state
    np.random.seed(None)
    
    yield
    
    # Cleanup after test
    TestDataCache.clear()


# Example stability patterns

def test_example_with_retries():
    """Example of test with retry logic."""
    
    @TestStabilizer.with_retries(max_attempts=3, delay=0.5)
    @TestStabilizer.diagnostic_on_failure
    def flaky_test():
        # Simulate flaky behavior
        if np.random.random() < 0.3:  # 30% failure rate
            raise AssertionError("Random failure")
        return True
    
    # This will retry up to 3 times
    assert flaky_test()


def test_example_with_stable_random():
    """Example of test with stable randomness."""
    
    @TestStabilizer.with_fixed_seed(seed=12345)
    def deterministic_test():
        # This will always produce same results
        values = [np.random.normal() for _ in range(10)]
        return np.mean(values)
    
    # Run multiple times - should get same result
    result1 = deterministic_test()
    result2 = deterministic_test()
    assert result1 == result2


def test_example_with_caching():
    """Example of test with expensive data caching."""
    
    def expensive_portfolio_generation():
        """Simulate expensive operation."""
        time.sleep(0.1)  # Simulate work
        return {"portfolio": "data"}
    
    # First call generates data
    data1 = TestDataCache.get_or_create(
        "test_portfolio",
        expensive_portfolio_generation,
        ttl=60  # Cache for 60 seconds
    )
    
    # Second call uses cache (fast)
    data2 = TestDataCache.get_or_create(
        "test_portfolio",
        expensive_portfolio_generation,
        ttl=60
    )
    
    assert data1 is data2  # Same object from cache


# Flaky test markers and utilities

def mark_flaky(reason: str, max_runs: int = 3):
    """Mark test as flaky with reason."""
    return pytest.mark.flaky(
        reruns=max_runs,
        reruns_delay=1,
        condition=reason
    )


def skip_if_unstable_environment():
    """Skip test if environment is unstable."""
    TestStabilizer.requires_stable_environment()


# Diagnostic utilities

class TestDiagnostics:
    """Collect diagnostics during test execution."""
    
    def __init__(self):
        self.metrics: List[Dict[str, Any]] = []
    
    def record(self, **kwargs):
        """Record diagnostic metrics."""
        self.metrics.append({
            'timestamp': time.time(),
            **kwargs
        })
    
    def summarize(self) -> Dict[str, Any]:
        """Summarize collected diagnostics."""
        if not self.metrics:
            return {}
        
        return {
            'count': len(self.metrics),
            'duration': self.metrics[-1]['timestamp'] - self.metrics[0]['timestamp'],
            'metrics': self.metrics
        }
    
    def assert_stable(self, metric: str, max_cv: float = 0.1):
        """Assert that a metric is stable (low coefficient of variation)."""
        values = [m.get(metric, 0) for m in self.metrics if metric in m]
        if len(values) < 2:
            return
        
        mean_val = np.mean(values)
        std_val = np.std(values)
        cv = std_val / mean_val if mean_val != 0 else 0
        
        assert cv <= max_cv, (
            f"Metric '{metric}' is unstable: CV={cv:.3f} > {max_cv:.3f} "
            f"(mean={mean_val:.3f}, std={std_val:.3f})"
        )