"""
QMC Benchmarking Suite

This module provides benchmarking tools for testing QMC performance
and convergence characteristics.
"""

from .qmc_convergence_benchmark import (
    create_test_portfolio,
    run_convergence_test
)

# Import plot function only if matplotlib is available
try:
    from .qmc_convergence_benchmark import plot_convergence_results
    _has_plot = True
except ImportError:
    _has_plot = False
    plot_convergence_results = None

from .qmc_stress_test import QMCStressTester

# Import performance benchmarking classes from benchmarks.py
import sys
import os
# Add the parent directory to the path to import benchmarks.py
_parent_dir = os.path.dirname(os.path.dirname(__file__))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Import from benchmarks.py directly
import importlib.util
spec = importlib.util.spec_from_file_location("benchmarks_module", 
                                             os.path.join(_parent_dir, "benchmarks.py"))
benchmarks_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmarks_module)

PerformanceBenchmark = benchmarks_module.PerformanceBenchmark
BenchmarkResult = benchmarks_module.BenchmarkResult

__all__ = [
    'create_test_portfolio',
    'run_convergence_test',
    'QMCStressTester',
    'PerformanceBenchmark',
    'BenchmarkResult'
]

if _has_plot:
    __all__.append('plot_convergence_results')