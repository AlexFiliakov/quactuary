"""Configuration for integration tests with environment-based adaptations."""

import os
import multiprocessing
import psutil
from typing import Dict, Any
import pytest


class TestEnvironment:
    """Detects and configures test environment capabilities."""
    
    def __init__(self):
        self.cpu_count = multiprocessing.cpu_count()
        self.memory_gb = psutil.virtual_memory().total / (1024**3)
        self.profile = self._detect_profile()
        
    def _detect_profile(self) -> str:
        """Detect environment profile based on available resources."""
        # Check for explicit profile override
        profile = os.environ.get('QUACTUARY_TEST_PROFILE', '').lower()
        if profile in ['minimal', 'standard', 'performance']:
            return profile
            
        # Auto-detect based on resources
        if self.cpu_count >= 8 and self.memory_gb >= 16:
            return 'performance'
        elif self.cpu_count >= 4 and self.memory_gb >= 8:
            return 'standard'
        else:
            return 'minimal'
    
    def get_performance_expectations(self) -> Dict[str, Any]:
        """Get performance expectations based on environment profile."""
        profiles = {
            'minimal': {
                'max_speedup': 2.0,
                'min_speedup': 1.2,
                'max_memory_mb': 2048,
                'max_test_duration': 300,  # 5 minutes
                'parallel_workers': min(2, self.cpu_count),
                'qmc_convergence_rate': -0.5,  # More lenient
                'statistical_tolerance': 0.2,   # 20% tolerance
            },
            'standard': {
                'max_speedup': 4.0,
                'min_speedup': 1.5,
                'max_memory_mb': 4096,
                'max_test_duration': 180,  # 3 minutes
                'parallel_workers': min(4, self.cpu_count),
                'qmc_convergence_rate': -0.6,
                'statistical_tolerance': 0.1,   # 10% tolerance
            },
            'performance': {
                'max_speedup': 8.0,
                'min_speedup': 2.0,
                'max_memory_mb': 8192,
                'max_test_duration': 120,  # 2 minutes
                'parallel_workers': min(8, self.cpu_count),
                'qmc_convergence_rate': -0.8,
                'statistical_tolerance': 0.05,  # 5% tolerance
            }
        }
        return profiles[self.profile]
    
    def should_skip_test(self, requirements: Dict[str, Any]) -> tuple[bool, str]:
        """Check if test should be skipped based on requirements."""
        # Check CPU requirements
        if 'min_cpus' in requirements and self.cpu_count < requirements['min_cpus']:
            return True, f"Requires at least {requirements['min_cpus']} CPUs, have {self.cpu_count}"
            
        # Check memory requirements
        if 'min_memory_gb' in requirements and self.memory_gb < requirements['min_memory_gb']:
            return True, f"Requires at least {requirements['min_memory_gb']}GB RAM, have {self.memory_gb:.1f}GB"
            
        # Check profile requirements
        if 'min_profile' in requirements:
            profile_order = ['minimal', 'standard', 'performance']
            current_idx = profile_order.index(self.profile)
            required_idx = profile_order.index(requirements['min_profile'])
            if current_idx < required_idx:
                return True, f"Requires {requirements['min_profile']} profile, running in {self.profile} mode"
                
        return False, ""


# Global test environment instance
TEST_ENV = TestEnvironment()


def get_test_config() -> Dict[str, Any]:
    """Get current test configuration."""
    return {
        'environment': {
            'cpu_count': TEST_ENV.cpu_count,
            'memory_gb': TEST_ENV.memory_gb,
            'profile': TEST_ENV.profile,
        },
        'expectations': TEST_ENV.get_performance_expectations(),
    }


def skip_if_insufficient_resources(**requirements):
    """Decorator to skip tests if resources are insufficient."""
    def decorator(test_func):
        def wrapper(*args, **kwargs):
            should_skip, reason = TEST_ENV.should_skip_test(requirements)
            if should_skip:
                pytest.skip(reason)
            return test_func(*args, **kwargs)
        return wrapper
    return decorator


def adapt_test_parameters(base_params: Dict[str, Any]) -> Dict[str, Any]:
    """Adapt test parameters based on environment capabilities."""
    expectations = TEST_ENV.get_performance_expectations()
    adapted = base_params.copy()
    
    # Adapt simulation counts based on profile
    if 'n_simulations' in adapted:
        if TEST_ENV.profile == 'minimal':
            adapted['n_simulations'] = min(adapted['n_simulations'], 10_000)
        elif TEST_ENV.profile == 'standard':
            adapted['n_simulations'] = min(adapted['n_simulations'], 100_000)
    
    # Adapt parallel workers
    if 'n_workers' in adapted:
        adapted['n_workers'] = min(adapted['n_workers'], expectations['parallel_workers'])
    
    # Adapt tolerances
    if 'tolerance' in adapted:
        adapted['tolerance'] = max(adapted['tolerance'], expectations['statistical_tolerance'])
        
    return adapted