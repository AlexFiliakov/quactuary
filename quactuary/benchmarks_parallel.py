"""
Performance benchmarks for parallel processing in quactuary.

This module provides comprehensive benchmarks for parallel processing
capabilities, comparing different approaches and configurations to
validate performance improvements.
"""

import time
import numpy as np
import pandas as pd
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
import psutil
from datetime import datetime

# Import quactuary components
from quactuary.book import PolicyTerms, Inforce, Portfolio
from quactuary.distributions.frequency import Poisson, NegativeBinomial
from quactuary.distributions.severity import Lognormal, Gamma, Pareto
from quactuary.vectorized_simulation import VectorizedSimulator
from quactuary.parallel_processing import ParallelSimulator, ParallelConfig


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    name: str
    method: str
    n_simulations: int
    n_buckets: int
    n_policies_per_bucket: int
    n_workers: int
    execution_time: float
    speedup: float
    efficiency: float
    memory_usage_mb: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'method': self.method,
            'n_simulations': self.n_simulations,
            'n_buckets': self.n_buckets,
            'n_policies_per_bucket': self.n_policies_per_bucket,
            'n_workers': self.n_workers,
            'execution_time': self.execution_time,
            'speedup': self.speedup,
            'efficiency': self.efficiency,
            'memory_usage_mb': self.memory_usage_mb,
            'timestamp': datetime.now().isoformat()
        }


class ParallelBenchmarkSuite:
    """Comprehensive benchmark suite for parallel processing."""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.baseline_times: Dict[str, float] = {}
    
    def create_test_portfolio(
        self,
        n_buckets: int = 5,
        n_policies_per_bucket: int = 100,
        complexity: str = 'medium'
    ) -> Portfolio:
        """Create a test portfolio with specified characteristics."""
        buckets = []
        
        # Define distribution parameters based on complexity
        if complexity == 'simple':
            # Simple: Poisson-Lognormal
            freq_params = {'mu': 1.5}
            sev_params = {'shape': 1.0, 'scale': np.exp(8.0)}
            FreqDist = Poisson
            SevDist = Lognormal
        elif complexity == 'medium':
            # Medium: NegativeBinomial-Gamma
            freq_params = {'r': 10, 'p': 0.3}
            sev_params = {'a': 2.0, 'scale': 10000.0}
            FreqDist = NegativeBinomial
            SevDist = Gamma
        else:  # complex
            # Complex: NegativeBinomial-Pareto
            freq_params = {'r': 20, 'p': 0.5}
            sev_params = {'a': 2.5, 'scale': 5000.0}
            FreqDist = NegativeBinomial
            SevDist = Pareto
        
        # Create buckets with varying characteristics
        for i in range(n_buckets):
            # Vary parameters slightly for each bucket
            bucket_freq_params = freq_params.copy()
            bucket_sev_params = sev_params.copy()
            
            # Add some variation
            if 'mu' in bucket_freq_params:
                bucket_freq_params['mu'] *= (1 + 0.1 * i)
            if 'r' in bucket_freq_params:
                bucket_freq_params['r'] = int(bucket_freq_params['r'] * (1 + 0.05 * i))
            
            terms = PolicyTerms(
                effective_date=pd.Timestamp('2024-01-01'),
                expiration_date=pd.Timestamp('2024-12-31')
            )
            
            bucket = Inforce(
                n_policies=n_policies_per_bucket,
                frequency=FreqDist(**bucket_freq_params),
                severity=SevDist(**bucket_sev_params),
                terms=terms,
                name=f"Bucket_{i+1}"
            )
            buckets.append(bucket)
        
        return Portfolio(buckets)
    
    def measure_memory_usage(self) -> float:
        """Measure current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def benchmark_serial_baseline(
        self,
        portfolio: Portfolio,
        n_simulations: int,
        method: str = 'standard'
    ) -> float:
        """Establish serial baseline for comparison."""
        print(f"\nEstablishing baseline ({method})...")
        
        mem_start = self.measure_memory_usage()
        start_time = time.time()
        
        if method == 'standard':
            results = portfolio.simulate(n_simulations, parallel=False)
        elif method == 'vectorized':
            # Use vectorized simulation for each bucket
            all_results = []
            for bucket in portfolio:
                bucket_results = VectorizedSimulator.simulate_inforce_vectorized(
                    bucket, n_simulations, parallel=False
                )
                all_results.append(bucket_results)
            # Sum across buckets
            results = pd.Series(np.sum(all_results, axis=0))
        else:
            raise ValueError(f"Unknown method: {method}")
        
        execution_time = time.time() - start_time
        memory_usage = self.measure_memory_usage() - mem_start
        
        print(f"  Baseline time: {execution_time:.2f}s")
        print(f"  Memory usage: {memory_usage:.1f} MB")
        print(f"  Mean result: {results.mean():,.2f}")
        
        return execution_time
    
    def benchmark_parallel_configuration(
        self,
        portfolio: Portfolio,
        n_simulations: int,
        n_workers: int,
        baseline_time: float,
        config_name: str
    ) -> BenchmarkResult:
        """Benchmark a specific parallel configuration."""
        print(f"\nBenchmarking {config_name} with {n_workers} workers...")
        
        mem_start = self.measure_memory_usage()
        start_time = time.time()
        
        # Run parallel simulation
        results = portfolio.simulate(
            n_simulations,
            parallel=True,
            n_workers=n_workers
        )
        
        execution_time = time.time() - start_time
        memory_usage = self.measure_memory_usage() - mem_start
        
        # Calculate metrics
        speedup = baseline_time / execution_time
        efficiency = speedup / n_workers if n_workers > 0 else 0
        
        print(f"  Execution time: {execution_time:.2f}s")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Efficiency: {efficiency:.1%}")
        print(f"  Memory usage: {memory_usage:.1f} MB")
        
        result = BenchmarkResult(
            name=config_name,
            method='parallel',
            n_simulations=n_simulations,
            n_buckets=len(portfolio),
            n_policies_per_bucket=portfolio[0].n_policies,
            n_workers=n_workers,
            execution_time=execution_time,
            speedup=speedup,
            efficiency=efficiency,
            memory_usage_mb=memory_usage
        )
        
        self.results.append(result)
        return result
    
    def run_scaling_benchmark(
        self,
        n_simulations_list: List[int] = None,
        n_workers_list: List[int] = None
    ):
        """Run benchmarks with different problem sizes and worker counts."""
        if n_simulations_list is None:
            n_simulations_list = [10000, 50000, 100000, 500000]
        
        if n_workers_list is None:
            n_workers_list = [1, 2, 4, 8, psutil.cpu_count()]
        
        print("=" * 80)
        print("PARALLEL PROCESSING SCALING BENCHMARK")
        print("=" * 80)
        print(f"CPU cores available: {psutil.cpu_count()}")
        print(f"Memory available: {psutil.virtual_memory().available / (1024**3):.1f} GB")
        
        # Test different problem sizes
        for n_sims in n_simulations_list:
            print(f"\n{'='*80}")
            print(f"Problem size: {n_sims:,} simulations")
            print(f"{'='*80}")
            
            # Create portfolio
            portfolio = self.create_test_portfolio(
                n_buckets=5,
                n_policies_per_bucket=100,
                complexity='medium'
            )
            
            # Establish baseline
            baseline_time = self.benchmark_serial_baseline(
                portfolio, n_sims, method='vectorized'
            )
            self.baseline_times[f"{n_sims}_sims"] = baseline_time
            
            # Test different worker counts
            for n_workers in n_workers_list:
                if n_workers > psutil.cpu_count():
                    continue
                
                self.benchmark_parallel_configuration(
                    portfolio,
                    n_sims,
                    n_workers,
                    baseline_time,
                    f"{n_sims}_sims_{n_workers}_workers"
                )
    
    def run_complexity_benchmark(self):
        """Benchmark different complexity levels."""
        print("\n" + "=" * 80)
        print("COMPLEXITY BENCHMARK")
        print("=" * 80)
        
        n_simulations = 100000
        complexity_levels = ['simple', 'medium', 'complex']
        
        for complexity in complexity_levels:
            print(f"\n{'='*60}")
            print(f"Complexity: {complexity}")
            print(f"{'='*60}")
            
            # Create portfolio
            portfolio = self.create_test_portfolio(
                n_buckets=10,
                n_policies_per_bucket=50,
                complexity=complexity
            )
            
            # Baseline
            baseline_time = self.benchmark_serial_baseline(
                portfolio, n_simulations, method='vectorized'
            )
            
            # Parallel with optimal workers
            optimal_workers = min(8, psutil.cpu_count())
            self.benchmark_parallel_configuration(
                portfolio,
                n_simulations,
                optimal_workers,
                baseline_time,
                f"complexity_{complexity}"
            )
    
    def run_portfolio_size_benchmark(self):
        """Benchmark different portfolio sizes."""
        print("\n" + "=" * 80)
        print("PORTFOLIO SIZE BENCHMARK")
        print("=" * 80)
        
        n_simulations = 50000
        portfolio_configs = [
            (1, 1000),   # 1 bucket, 1000 policies
            (10, 100),   # 10 buckets, 100 policies each
            (100, 10),   # 100 buckets, 10 policies each
            (20, 50),    # 20 buckets, 50 policies each
        ]
        
        for n_buckets, n_policies in portfolio_configs:
            print(f"\n{'='*60}")
            print(f"Portfolio: {n_buckets} buckets × {n_policies} policies = {n_buckets * n_policies} total")
            print(f"{'='*60}")
            
            # Create portfolio
            portfolio = self.create_test_portfolio(
                n_buckets=n_buckets,
                n_policies_per_bucket=n_policies,
                complexity='medium'
            )
            
            # Baseline
            baseline_time = self.benchmark_serial_baseline(
                portfolio, n_simulations, method='vectorized'
            )
            
            # Parallel
            optimal_workers = min(8, psutil.cpu_count())
            self.benchmark_parallel_configuration(
                portfolio,
                n_simulations,
                optimal_workers,
                baseline_time,
                f"portfolio_{n_buckets}x{n_policies}"
            )
    
    def generate_report(self, output_file: str = "parallel_benchmark_report.json"):
        """Generate comprehensive benchmark report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'cpu_count_physical': psutil.cpu_count(logical=False),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'python_version': pd.__version__
            },
            'benchmarks': [r.to_dict() for r in self.results],
            'summary': self._generate_summary()
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nReport saved to: {output_file}")
        return report
    
    def _generate_summary(self) -> Dict:
        """Generate summary statistics."""
        if not self.results:
            return {}
        
        # Group by worker count
        by_workers = {}
        for result in self.results:
            n_workers = result.n_workers
            if n_workers not in by_workers:
                by_workers[n_workers] = []
            by_workers[n_workers].append(result)
        
        # Calculate average metrics
        summary = {
            'total_benchmarks': len(self.results),
            'avg_speedup_by_workers': {},
            'avg_efficiency_by_workers': {},
            'best_configuration': None,
            'recommendations': []
        }
        
        best_efficiency = 0
        best_config = None
        
        for n_workers, results in by_workers.items():
            avg_speedup = np.mean([r.speedup for r in results])
            avg_efficiency = np.mean([r.efficiency for r in results])
            
            summary['avg_speedup_by_workers'][n_workers] = round(avg_speedup, 2)
            summary['avg_efficiency_by_workers'][n_workers] = round(avg_efficiency, 3)
            
            if avg_efficiency > best_efficiency:
                best_efficiency = avg_efficiency
                best_config = n_workers
        
        summary['best_configuration'] = best_config
        
        # Generate recommendations
        if best_efficiency > 0.7:
            summary['recommendations'].append(
                f"Excellent parallel efficiency achieved! Use {best_config} workers for optimal performance."
            )
        elif best_efficiency > 0.5:
            summary['recommendations'].append(
                f"Good parallel efficiency. Consider using {best_config} workers."
            )
        else:
            summary['recommendations'].append(
                "Parallel efficiency is below expectations. Consider optimizing work distribution."
            )
        
        # Check if efficiency drops with more workers
        efficiencies = list(summary['avg_efficiency_by_workers'].values())
        if len(efficiencies) > 2 and efficiencies[-1] < efficiencies[-2]:
            summary['recommendations'].append(
                "Efficiency decreases with too many workers. Use fewer workers for better efficiency."
            )
        
        return summary
    
    def plot_results(self, output_file: str = "parallel_benchmark_plots.png"):
        """Generate visualization of benchmark results."""
        if not HAS_MATPLOTLIB:
            print("Matplotlib not available, skipping plots")
            return
        
        if not self.results:
            print("No results to plot")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Speedup vs Workers
        workers_data = {}
        for r in self.results:
            if r.n_workers not in workers_data:
                workers_data[r.n_workers] = []
            workers_data[r.n_workers].append(r.speedup)
        
        workers = sorted(workers_data.keys())
        avg_speedups = [np.mean(workers_data[w]) for w in workers]
        
        ax1.plot(workers, avg_speedups, 'b-o', label='Actual')
        ax1.plot(workers, workers, 'r--', label='Ideal')
        ax1.set_xlabel('Number of Workers')
        ax1.set_ylabel('Speedup')
        ax1.set_title('Speedup vs Number of Workers')
        ax1.legend()
        ax1.grid(True)
        
        # 2. Efficiency vs Workers
        avg_efficiencies = [np.mean([r.efficiency for r in self.results if r.n_workers == w]) 
                           for w in workers]
        
        ax2.plot(workers, avg_efficiencies, 'g-o')
        ax2.axhline(y=0.7, color='r', linestyle='--', label='Target (70%)')
        ax2.set_xlabel('Number of Workers')
        ax2.set_ylabel('Efficiency')
        ax2.set_title('Efficiency vs Number of Workers')
        ax2.legend()
        ax2.grid(True)
        
        # 3. Execution Time by Problem Size
        size_data = {}
        for r in self.results:
            size = r.n_simulations
            if size not in size_data:
                size_data[size] = {}
            size_data[size][r.n_workers] = r.execution_time
        
        for size, worker_times in sorted(size_data.items()):
            workers_sorted = sorted(worker_times.keys())
            times = [worker_times[w] for w in workers_sorted]
            ax3.plot(workers_sorted, times, '-o', label=f'{size:,} sims')
        
        ax3.set_xlabel('Number of Workers')
        ax3.set_ylabel('Execution Time (s)')
        ax3.set_title('Execution Time by Problem Size')
        ax3.legend()
        ax3.grid(True)
        ax3.set_yscale('log')
        
        # 4. Memory Usage
        memory_by_workers = {}
        for r in self.results:
            if r.n_workers not in memory_by_workers:
                memory_by_workers[r.n_workers] = []
            memory_by_workers[r.n_workers].append(r.memory_usage_mb)
        
        workers = sorted(memory_by_workers.keys())
        avg_memory = [np.mean(memory_by_workers[w]) for w in workers]
        
        ax4.bar(workers, avg_memory)
        ax4.set_xlabel('Number of Workers')
        ax4.set_ylabel('Memory Usage (MB)')
        ax4.set_title('Average Memory Usage by Workers')
        ax4.grid(True, axis='y')
        
        plt.suptitle('Parallel Processing Performance Benchmarks', fontsize=16)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nPlots saved to: {output_file}")


def run_quick_benchmark():
    """Run a quick benchmark for testing."""
    print("Running quick parallel processing benchmark...")
    
    suite = ParallelBenchmarkSuite()
    
    # Quick scaling test
    suite.run_scaling_benchmark(
        n_simulations_list=[10000, 50000],
        n_workers_list=[1, 2, 4]
    )
    
    # Generate report
    report = suite.generate_report("parallel_benchmark_quick.json")
    suite.plot_results("parallel_benchmark_quick.png")
    
    # Print summary
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    
    summary = report['summary']
    print(f"Total benchmarks run: {summary['total_benchmarks']}")
    print(f"Best configuration: {summary['best_configuration']} workers")
    print("\nAverage speedup by workers:")
    for w, s in summary['avg_speedup_by_workers'].items():
        print(f"  {w} workers: {s:.2f}x")
    print("\nRecommendations:")
    for rec in summary['recommendations']:
        print(f"  - {rec}")


def run_full_benchmark():
    """Run comprehensive benchmark suite."""
    print("Running comprehensive parallel processing benchmark...")
    print("This may take several minutes...\n")
    
    suite = ParallelBenchmarkSuite()
    
    # Run all benchmark types
    suite.run_scaling_benchmark()
    suite.run_complexity_benchmark()
    suite.run_portfolio_size_benchmark()
    
    # Generate report and plots
    report = suite.generate_report("parallel_benchmark_full.json")
    suite.plot_results("parallel_benchmark_full.png")
    
    # Print detailed summary
    print("\n" + "=" * 80)
    print("COMPREHENSIVE BENCHMARK SUMMARY")
    print("=" * 80)
    
    summary = report['summary']
    print(f"Total benchmarks run: {summary['total_benchmarks']}")
    print(f"Best configuration: {summary['best_configuration']} workers")
    
    print("\nAverage speedup by workers:")
    for w, s in sorted(summary['avg_speedup_by_workers'].items()):
        e = summary['avg_efficiency_by_workers'][w]
        print(f"  {w:2d} workers: {s:5.2f}x speedup ({e:.1%} efficiency)")
    
    print("\nRecommendations:")
    for i, rec in enumerate(summary['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    # Find best results
    print("\nBest results:")
    best_speedup = max(suite.results, key=lambda r: r.speedup)
    print(f"  Highest speedup: {best_speedup.speedup:.2f}x "
          f"({best_speedup.name})")
    
    best_efficiency = max(suite.results, key=lambda r: r.efficiency)
    print(f"  Best efficiency: {best_efficiency.efficiency:.1%} "
          f"({best_efficiency.name})")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        run_full_benchmark()
    else:
        run_quick_benchmark()