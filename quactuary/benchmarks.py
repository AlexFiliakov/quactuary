"""
Comprehensive benchmarking suite for performance optimization.

This module provides tools to measure and profile the performance of various
simulation methods, establish baselines, and track improvements.
"""

import time
import gc
import psutil
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from contextlib import contextmanager
import json
import os
from datetime import datetime

from quactuary.book import Portfolio, Inforce, PolicyTerms
from quactuary.distributions.frequency import Poisson, NegativeBinomial
from quactuary.distributions.severity import Lognormal, Exponential, Gamma, Pareto
from quactuary.pricing import PricingModel
from quactuary.pricing_strategies import ClassicalPricingStrategy

try:
    import memory_profiler
    HAS_MEMORY_PROFILER = True
except ImportError:
    HAS_MEMORY_PROFILER = False

try:
    import line_profiler
    HAS_LINE_PROFILER = True
except ImportError:
    HAS_LINE_PROFILER = False


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    name: str
    portfolio_size: int
    n_simulations: int
    execution_time: float
    memory_used: float
    memory_peak: float
    samples_per_second: float
    metadata: Dict = None
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'portfolio_size': self.portfolio_size,
            'n_simulations': self.n_simulations,
            'execution_time': self.execution_time,
            'memory_used': self.memory_used,
            'memory_peak': self.memory_peak,
            'samples_per_second': self.samples_per_second,
            'metadata': self.metadata or {}
        }


class PerformanceBenchmark:
    """
    Comprehensive performance benchmarking framework.
    
    Measures execution time, memory usage, and computational efficiency
    for different simulation methods and portfolio sizes.
    """
    
    def __init__(self, output_dir: str = "./benchmark_results"):
        """Initialize benchmark framework."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results: List[BenchmarkResult] = []
        self.last_measurement = {}
        
    @contextmanager
    def measure_performance(self, name: str):
        """Context manager to measure execution time and memory usage."""
        gc.collect()
        gc.disable()
        
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        start_time = time.perf_counter()
        peak_mem = mem_before
        
        def track_memory():
            nonlocal peak_mem
            current = process.memory_info().rss / 1024 / 1024
            peak_mem = max(peak_mem, current)
        
        try:
            yield track_memory
        finally:
            end_time = time.perf_counter()
            gc.enable()
            gc.collect()
            
            mem_after = process.memory_info().rss / 1024 / 1024
            
            self.last_measurement = {
                'name': name,
                'execution_time': end_time - start_time,
                'memory_used': mem_after - mem_before,
                'memory_peak': peak_mem - mem_before
            }
    
    def create_test_portfolios(self) -> Dict[str, Portfolio]:
        """Create standardized test portfolios of various sizes."""
        portfolios = {}
        
        # Common policy terms
        terms = PolicyTerms(
            effective_date=pd.Timestamp('2024-01-01'),
            expiration_date=pd.Timestamp('2024-12-31'),
            per_occ_retention=1000.0,
            per_occ_limit=100000.0
        )
        
        # Small portfolio (10 policies)
        portfolios['small'] = Portfolio([
            Inforce(
                n_policies=10,
                terms=terms,
                frequency=Poisson(mu=2.0),
                severity=Lognormal(shape=1.0, scale=np.exp(8.0)),
                name="Small Portfolio"
            )
        ])
        
        # Medium portfolio (100 policies across 3 buckets)
        portfolios['medium'] = Portfolio([
            Inforce(
                n_policies=40,
                terms=terms,
                frequency=Poisson(mu=1.5),
                severity=Lognormal(shape=1.2, scale=np.exp(8.5)),
                name="Medium Bucket 1"
            ),
            Inforce(
                n_policies=30,
                terms=terms,
                frequency=Poisson(mu=0.8),
                severity=Exponential(scale=5000.0),
                name="Medium Bucket 2"
            ),
            Inforce(
                n_policies=30,
                terms=terms,
                frequency=NegativeBinomial(r=5, p=0.3),
                severity=Gamma(shape=2.0, scale=4000.0),
                name="Medium Bucket 3"
            )
        ])
        
        # Large portfolio (1000 policies across 10 buckets)
        large_buckets = []
        for i in range(10):
            large_buckets.append(
                Inforce(
                    n_policies=100,
                    terms=terms,
                    frequency=Poisson(mu=1.0 + i*0.1),
                    severity=Lognormal(shape=0.8 + i*0.05, scale=np.exp(8.0 + i*0.1)),
                    name=f"Large Bucket {i+1}"
                )
            )
        portfolios['large'] = Portfolio(large_buckets)
        
        # Extra large portfolio (10000 policies across 50 buckets)
        xl_buckets = []
        for i in range(50):
            xl_buckets.append(
                Inforce(
                    n_policies=200,
                    terms=terms,
                    frequency=Poisson(mu=0.5 + (i % 5)*0.2),
                    severity=Lognormal(shape=0.6 + (i % 3)*0.2, scale=np.exp(7.5 + (i % 4)*0.5)),
                    name=f"XL Bucket {i+1}"
                )
            )
        portfolios['xlarge'] = Portfolio(xl_buckets)
        
        return portfolios
    
    def benchmark_baseline(self, portfolio: Portfolio, n_sims: int, name: str) -> BenchmarkResult:
        """Benchmark the baseline (current) implementation."""
        model = PricingModel(portfolio, strategy=ClassicalPricingStrategy(use_jit=False))
        
        with self.measure_performance(f"baseline_{name}") as track_mem:
            # Run simulation
            result = model.simulate(
                mean=True,
                variance=True,
                value_at_risk=True,
                tail_value_at_risk=True,
                n_sims=n_sims
            )
            track_mem()
        
        total_policies = sum(bucket.n_policies for bucket in portfolio)
        samples_per_second = (n_sims * total_policies) / self.last_measurement['execution_time']
        
        return BenchmarkResult(
            name=f"baseline_{name}",
            portfolio_size=total_policies,
            n_simulations=n_sims,
            execution_time=self.last_measurement['execution_time'],
            memory_used=self.last_measurement['memory_used'],
            memory_peak=self.last_measurement['memory_peak'],
            samples_per_second=samples_per_second,
            metadata={
                'mean': result.estimates.get('mean', 0),
                'var_95': result.estimates.get('VaR', 0),
                'tvar_95': result.estimates.get('TVaR', 0)
            }
        )
    
    def benchmark_jit(self, portfolio: Portfolio, n_sims: int, name: str) -> BenchmarkResult:
        """Benchmark the JIT-compiled implementation."""
        model = PricingModel(portfolio, strategy=ClassicalPricingStrategy(use_jit=True))
        
        # Warm up JIT compilation
        _ = model.simulate(n_sims=10)
        
        with self.measure_performance(f"jit_{name}") as track_mem:
            # Run simulation
            result = model.simulate(
                mean=True,
                variance=True,
                value_at_risk=True,
                tail_value_at_risk=True,
                n_sims=n_sims
            )
            track_mem()
        
        total_policies = sum(bucket.n_policies for bucket in portfolio)
        samples_per_second = (n_sims * total_policies) / self.last_measurement['execution_time']
        
        return BenchmarkResult(
            name=f"jit_{name}",
            portfolio_size=total_policies,
            n_simulations=n_sims,
            execution_time=self.last_measurement['execution_time'],
            memory_used=self.last_measurement['memory_used'],
            memory_peak=self.last_measurement['memory_peak'],
            samples_per_second=samples_per_second,
            metadata={
                'mean': result.estimates.get('mean', 0),
                'var_95': result.estimates.get('VaR', 0),
                'tvar_95': result.estimates.get('TVaR', 0),
                'jit_enabled': True
            }
        )
    
    def benchmark_qmc(self, portfolio: Portfolio, n_sims: int, name: str) -> BenchmarkResult:
        """Benchmark quasi-Monte Carlo (Sobol) implementation."""
        model = PricingModel(portfolio)
        
        with self.measure_performance(f"qmc_{name}") as track_mem:
            # Run simulation with Sobol sequences
            result = model.simulate(
                mean=True,
                variance=True,
                value_at_risk=True,
                tail_value_at_risk=True,
                n_sims=n_sims,
                qmc_method="sobol",
                qmc_scramble=True,
                qmc_skip=1024
            )
            track_mem()
        
        total_policies = sum(bucket.n_policies for bucket in portfolio)
        samples_per_second = (n_sims * total_policies) / self.last_measurement['execution_time']
        
        return BenchmarkResult(
            name=f"qmc_{name}",
            portfolio_size=total_policies,
            n_simulations=n_sims,
            execution_time=self.last_measurement['execution_time'],
            memory_used=self.last_measurement['memory_used'],
            memory_peak=self.last_measurement['memory_peak'],
            samples_per_second=samples_per_second,
            metadata={
                'mean': result.estimates.get('mean', 0),
                'var_95': result.estimates.get('VaR', 0),
                'tvar_95': result.estimates.get('TVaR', 0),
                'qmc_method': 'sobol'
            }
        )
    
    def run_comprehensive_benchmark(self):
        """Run comprehensive benchmarks across all portfolio sizes and methods."""
        portfolios = self.create_test_portfolios()
        simulation_counts = {
            'small': [1000, 10000, 100000],
            'medium': [1000, 10000, 100000],
            'large': [1000, 10000, 50000],
            'xlarge': [1000, 5000, 10000]
        }
        
        print("=" * 80)
        print("COMPREHENSIVE PERFORMANCE BENCHMARK")
        print("=" * 80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        for portfolio_name, portfolio in portfolios.items():
            total_policies = sum(bucket.n_policies for bucket in portfolio)
            print(f"\n{portfolio_name.upper()} PORTFOLIO ({total_policies} policies)")
            print("-" * 60)
            
            for n_sims in simulation_counts[portfolio_name]:
                print(f"\nSimulations: {n_sims:,}")
                
                # Baseline
                try:
                    baseline_result = self.benchmark_baseline(portfolio, n_sims, f"{portfolio_name}_{n_sims}")
                    self.results.append(baseline_result)
                    print(f"  Baseline: {baseline_result.execution_time:.3f}s, "
                          f"{baseline_result.samples_per_second:,.0f} samples/sec")
                except Exception as e:
                    print(f"  Baseline: FAILED - {str(e)}")
                
                # JIT
                try:
                    jit_result = self.benchmark_jit(portfolio, n_sims, f"{portfolio_name}_{n_sims}")
                    self.results.append(jit_result)
                    speedup = baseline_result.execution_time / jit_result.execution_time
                    print(f"  JIT:      {jit_result.execution_time:.3f}s, "
                          f"{jit_result.samples_per_second:,.0f} samples/sec "
                          f"(speedup: {speedup:.1f}x)")
                except Exception as e:
                    print(f"  JIT:      FAILED - {str(e)}")
                
                # QMC
                try:
                    qmc_result = self.benchmark_qmc(portfolio, n_sims, f"{portfolio_name}_{n_sims}")
                    self.results.append(qmc_result)
                    print(f"  QMC:      {qmc_result.execution_time:.3f}s, "
                          f"{qmc_result.samples_per_second:,.0f} samples/sec")
                except Exception as e:
                    print(f"  QMC:      FAILED - {str(e)}")
        
        # Save results
        self.save_results()
        self.generate_report()
    
    def save_results(self):
        """Save benchmark results to JSON file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(self.output_dir, f"benchmark_results_{timestamp}.json")
        
        results_data = {
            'timestamp': timestamp,
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'total_memory_gb': psutil.virtual_memory().total / (1024**3),
                'python_version': pd.__version__
            },
            'results': [r.to_dict() for r in self.results]
        }
        
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\nResults saved to: {filename}")
    
    def generate_report(self):
        """Generate summary report of benchmark results."""
        if not self.results:
            print("No results to report")
            return
        
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)
        
        # Group results by portfolio size and simulation count
        df = pd.DataFrame([r.to_dict() for r in self.results])
        
        # Calculate speedup factors
        baseline_times = df[df['name'].str.contains('baseline')].set_index(
            ['portfolio_size', 'n_simulations'])['execution_time']
        
        for idx, row in df.iterrows():
            if 'baseline' not in row['name']:
                key = (row['portfolio_size'], row['n_simulations'])
                if key in baseline_times.index:
                    speedup = baseline_times[key] / row['execution_time']
                    df.at[idx, 'speedup'] = speedup
        
        # Print summary table
        print("\nPerformance Summary (execution time in seconds):")
        print("-" * 80)
        
        pivot = df.pivot_table(
            values='execution_time',
            index=['portfolio_size', 'n_simulations'],
            columns=lambda x: df[df.index == x]['name'].iloc[0].split('_')[0],
            aggfunc='first'
        )
        
        print(pivot.round(3))
        
        # Memory usage summary
        print("\nMemory Usage Summary (peak MB):")
        print("-" * 60)
        
        memory_pivot = df.pivot_table(
            values='memory_peak',
            index=['portfolio_size', 'n_simulations'],
            columns=lambda x: df[df.index == x]['name'].iloc[0].split('_')[0],
            aggfunc='first'
        )
        
        print(memory_pivot.round(1))
        
        # Average speedup by method
        print("\nAverage Speedup vs Baseline:")
        print("-" * 40)
        
        for method in ['jit', 'qmc']:
            method_df = df[df['name'].str.contains(method)]
            if 'speedup' in method_df.columns:
                avg_speedup = method_df['speedup'].mean()
                print(f"{method.upper()}: {avg_speedup:.2f}x")


def run_baseline_profiling():
    """Run detailed profiling of baseline implementation."""
    print("\nRunning baseline profiling...")
    
    # Create a medium test portfolio
    terms = PolicyTerms(
        effective_date=pd.Timestamp('2024-01-01'),
        expiration_date=pd.Timestamp('2024-12-31'),
        per_occ_retention=1000.0,
        per_occ_limit=100000.0
    )
    
    portfolio = Portfolio([
        Inforce(
            n_policies=100,
            terms=terms,
            frequency=Poisson(mu=1.5),
            severity=Lognormal(shape=1.0, scale=np.exp(8.0)),
            name="Test Bucket"
        )
    ])
    
    # Profile with cProfile
    import cProfile
    import pstats
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run simulation
    model = PricingModel(portfolio)
    result = model.simulate(n_sims=1000)
    
    profiler.disable()
    
    # Save profile results
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)
    
    return stats


if __name__ == "__main__":
    # Run comprehensive benchmark
    benchmark = PerformanceBenchmark()
    benchmark.run_comprehensive_benchmark()
    
    # Run profiling
    run_baseline_profiling()