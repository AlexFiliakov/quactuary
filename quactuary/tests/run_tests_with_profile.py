#!/usr/bin/env python3
"""
Run tests with appropriate profile based on system capabilities.

Usage:
    python run_tests_with_profile.py [profile] [additional pytest args]
    
    Profiles: minimal, standard, performance, ci, quick, auto (default)
    
Examples:
    python run_tests_with_profile.py              # Auto-detect profile
    python run_tests_with_profile.py minimal      # Force minimal profile  
    python run_tests_with_profile.py standard -k test_accuracy  # Run specific tests
"""

import os
import sys
import subprocess
import multiprocessing
import psutil
import argparse
from pathlib import Path


class TestProfileSelector:
    """Select appropriate test profile based on system capabilities."""
    
    def __init__(self):
        self.cpu_count = multiprocessing.cpu_count()
        self.memory_gb = psutil.virtual_memory().total / (1024**3)
        
    def detect_profile(self) -> str:
        """Auto-detect the best profile for the current system."""
        # Check for CI environment
        if os.environ.get('CI'):
            return 'ci'
            
        # Check system resources
        if self.cpu_count >= 8 and self.memory_gb >= 16:
            return 'performance'
        elif self.cpu_count >= 4 and self.memory_gb >= 8:
            return 'standard'
        else:
            return 'minimal'
    
    def get_profile_env(self, profile: str) -> dict:
        """Get environment variables for a profile."""
        env_vars = os.environ.copy()
        
        profile_envs = {
            'minimal': {
                'QUACTUARY_TEST_PROFILE': 'minimal',
                'QUACTUARY_MAX_WORKERS': '2',
                'QUACTUARY_MAX_MEMORY_MB': '2048',
                'QUACTUARY_MAX_TEST_DURATION': '300'
            },
            'standard': {
                'QUACTUARY_TEST_PROFILE': 'standard',
                'QUACTUARY_MAX_WORKERS': '4',
                'QUACTUARY_MAX_MEMORY_MB': '4096',
                'QUACTUARY_MAX_TEST_DURATION': '180'
            },
            'performance': {
                'QUACTUARY_TEST_PROFILE': 'performance',
                'QUACTUARY_MAX_WORKERS': '8',
                'QUACTUARY_MAX_MEMORY_MB': '8192',
                'QUACTUARY_MAX_TEST_DURATION': '120'
            },
            'ci': {
                'QUACTUARY_TEST_PROFILE': 'minimal',
                'QUACTUARY_MAX_WORKERS': '2',
                'QUACTUARY_MAX_MEMORY_MB': '2048',
                'QUACTUARY_MAX_TEST_DURATION': '300',
                'CI': 'true'
            },
            'quick': {
                'QUACTUARY_TEST_PROFILE': 'minimal',
                'QUACTUARY_QUICK_TEST': 'true'
            }
        }
        
        if profile in profile_envs:
            env_vars.update(profile_envs[profile])
        
        return env_vars
    
    def get_pytest_args(self, profile: str) -> list:
        """Get pytest arguments for a profile."""
        base_args = ['-v', '--tb=short', '--strict-markers']
        
        profile_args = {
            'minimal': [
                '-m', 'not slow and not memory_intensive and not hardware_dependent',
                '--durations=10',
                '--maxfail=10'
            ],
            'standard': [
                '-m', 'not hardware_dependent or not performance',
                '--durations=20',
                '--maxfail=5'
            ],
            'performance': [
                '--durations=50',
                '--maxfail=3'
            ],
            'ci': [
                '-m', 'not slow and not memory_intensive and not hardware_dependent and not flaky',
                '--durations=10',
                '--maxfail=1',
                '--cov=quactuary',
                '--cov-report=xml',
                '--cov-report=term-missing',
                '--cov-branch'
            ],
            'quick': [
                '-m', 'not integration and not slow',
                '--durations=5',
                '--maxfail=1',
                '-x'
            ]
        }
        
        return base_args + profile_args.get(profile, [])


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run tests with appropriate profile based on system capabilities.',
        epilog='Additional arguments are passed directly to pytest.'
    )
    parser.add_argument(
        'profile',
        nargs='?',
        default='auto',
        choices=['minimal', 'standard', 'performance', 'ci', 'quick', 'auto'],
        help='Test profile to use (default: auto-detect)'
    )
    
    args, pytest_args = parser.parse_known_args()
    
    # Initialize profile selector
    selector = TestProfileSelector()
    
    # Determine profile
    if args.profile == 'auto':
        profile = selector.detect_profile()
        print(f"Auto-detected profile: {profile}")
        print(f"System: {selector.cpu_count} CPUs, {selector.memory_gb:.1f}GB RAM")
    else:
        profile = args.profile
        print(f"Using profile: {profile}")
    
    # Get environment and arguments
    env = selector.get_profile_env(profile)
    pytest_cmd_args = selector.get_pytest_args(profile)
    
    # Add any additional user arguments
    pytest_cmd_args.extend(pytest_args)
    
    # Build command
    cmd = [sys.executable, '-m', 'pytest'] + pytest_cmd_args
    
    # Print command for debugging
    print(f"Running: {' '.join(cmd)}")
    print(f"Environment overrides: {', '.join(f'{k}={v}' for k, v in env.items() if k.startswith('QUACTUARY'))}")
    print()
    
    # Run pytest
    result = subprocess.run(cmd, env=env)
    
    # Exit with same code as pytest
    sys.exit(result.returncode)


if __name__ == '__main__':
    main()