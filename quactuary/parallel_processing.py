"""
Parallel processing optimizations for Monte Carlo simulations.

This module implements various parallelization strategies including
multiprocessing, thread pools, and work stealing for optimal CPU
utilization.
"""

import os
import time
import numpy as np
from typing import Optional, List, Callable, Tuple, Any
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count, Pool
import warnings
from dataclasses import dataclass
import psutil

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Simple progress indicator fallback
    class tqdm:
        def __init__(self, total=None, desc=None):
            self.total = total
            self.desc = desc
            self.n = 0
            
        def update(self, n=1):
            self.n += n
            if self.total:
                pct = int(100 * self.n / self.total)
                print(f"\r{self.desc}: {pct}%", end="", flush=True)
                
        def close(self):
            if self.total:
                print()  # New line after progress

try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False


@dataclass
class ParallelConfig:
    """Configuration for parallel processing."""
    n_workers: Optional[int] = None  # None = auto-detect
    backend: str = 'multiprocessing'  # 'multiprocessing', 'threading', 'joblib'
    chunk_size: Optional[int] = None  # None = auto
    show_progress: bool = True
    work_stealing: bool = True
    prefetch_batches: int = 2


class ParallelSimulator:
    """
    Implements parallel simulation strategies for portfolios.
    
    Features:
    - Multiple backend support (multiprocessing, threading, joblib)
    - Work stealing for load balancing
    - Progress monitoring
    - Adaptive chunk sizing
    """
    
    def __init__(self, config: Optional[ParallelConfig] = None):
        """Initialize with parallel configuration."""
        self.config = config or ParallelConfig()
        
        # Auto-detect workers if not specified
        if self.config.n_workers is None:
            self.config.n_workers = min(cpu_count(), 8)  # Cap at 8 for stability
    
    def _calculate_chunk_size(self, total_work: int, n_workers: int) -> int:
        """Calculate optimal chunk size for work distribution."""
        if self.config.chunk_size is not None:
            return self.config.chunk_size
        
        # Heuristic: aim for at least 10 chunks per worker
        # but not too small (min 100) or too large (max 10000)
        ideal_chunks = n_workers * 10
        chunk_size = max(100, min(10000, total_work // ideal_chunks))
        
        return chunk_size
    
    def simulate_parallel_multiprocessing(
        self,
        simulate_func: Callable,
        n_simulations: int,
        n_policies: int,
        **kwargs
    ) -> np.ndarray:
        """
        Simulate using multiprocessing for CPU-bound tasks.
        
        Args:
            simulate_func: Function to simulate a batch
            n_simulations: Total simulations
            n_policies: Number of policies
            **kwargs: Additional arguments for simulate_func
            
        Returns:
            Array of simulation results
        """
        chunk_size = self._calculate_chunk_size(n_simulations, self.config.n_workers)
        chunks = []
        
        # Prepare work chunks
        for i in range(0, n_simulations, chunk_size):
            chunk_end = min(i + chunk_size, n_simulations)
            chunks.append((i, chunk_end - i))
        
        results = np.zeros(n_simulations)
        
        # Create progress bar if requested
        pbar = None
        if self.config.show_progress:
            pbar = tqdm(total=n_simulations, desc="Simulating")
        
        try:
            with ProcessPoolExecutor(max_workers=self.config.n_workers) as executor:
                # Submit all tasks
                future_to_chunk = {}
                for start_idx, chunk_n in chunks:
                    future = executor.submit(
                        simulate_func, 
                        chunk_n, 
                        n_policies,
                        **kwargs
                    )
                    future_to_chunk[future] = (start_idx, chunk_n)
                
                # Collect results as they complete
                for future in as_completed(future_to_chunk):
                    start_idx, chunk_n = future_to_chunk[future]
                    try:
                        chunk_results = future.result()
                        results[start_idx:start_idx + chunk_n] = chunk_results
                        
                        if pbar:
                            pbar.update(chunk_n)
                    except Exception as e:
                        warnings.warn(f"Chunk failed: {e}")
                        # Fill with zeros for failed chunk
                        results[start_idx:start_idx + chunk_n] = 0
        finally:
            if pbar:
                pbar.close()
        
        return results
    
    def simulate_parallel_joblib(
        self,
        simulate_func: Callable,
        n_simulations: int,
        n_policies: int,
        **kwargs
    ) -> np.ndarray:
        """
        Simulate using joblib for robust parallel execution.
        
        Joblib provides:
        - Better error handling
        - Memory mapping for large arrays
        - Choice of backends (loky, threading, multiprocessing)
        """
        if not HAS_JOBLIB:
            warnings.warn("joblib not available, falling back to multiprocessing")
            return self.simulate_parallel_multiprocessing(
                simulate_func, n_simulations, n_policies, **kwargs
            )
        
        chunk_size = self._calculate_chunk_size(n_simulations, self.config.n_workers)
        
        # Prepare work chunks
        chunks = []
        for i in range(0, n_simulations, chunk_size):
            chunk_end = min(i + chunk_size, n_simulations)
            chunks.append(chunk_end - i)
        
        # Run parallel simulation
        if self.config.show_progress:
            results_list = Parallel(
                n_jobs=self.config.n_workers,
                backend='loky',  # 'loky' is more robust than 'multiprocessing'
                verbose=10
            )(
                delayed(simulate_func)(chunk_n, n_policies, **kwargs)
                for chunk_n in chunks
            )
        else:
            results_list = Parallel(
                n_jobs=self.config.n_workers,
                backend='loky'
            )(
                delayed(simulate_func)(chunk_n, n_policies, **kwargs)
                for chunk_n in chunks
            )
        
        # Combine results
        results = np.concatenate(results_list)
        return results
    
    def simulate_work_stealing(
        self,
        simulate_func: Callable,
        n_simulations: int,
        n_policies: int,
        **kwargs
    ) -> np.ndarray:
        """
        Implement work-stealing algorithm for better load balancing.
        
        Work stealing helps when some simulations take longer than others,
        ensuring all cores stay busy.
        """
        # Create smaller work units for stealing
        steal_size = max(10, n_simulations // (self.config.n_workers * 20))
        work_queue = []
        
        for i in range(0, n_simulations, steal_size):
            chunk_size = min(steal_size, n_simulations - i)
            work_queue.append((i, chunk_size))
        
        results = np.zeros(n_simulations)
        completed = 0
        
        # Progress bar
        pbar = None
        if self.config.show_progress:
            pbar = tqdm(total=n_simulations, desc="Work-stealing simulation")
        
        try:
            with ProcessPoolExecutor(max_workers=self.config.n_workers) as executor:
                # Keep workers busy
                futures = set()
                queue_idx = 0
                
                # Initial work distribution
                for _ in range(min(self.config.n_workers * 2, len(work_queue))):
                    if queue_idx < len(work_queue):
                        start_idx, chunk_size = work_queue[queue_idx]
                        future = executor.submit(
                            simulate_func, chunk_size, n_policies, **kwargs
                        )
                        futures.add((future, start_idx, chunk_size))
                        queue_idx += 1
                
                # Process results and distribute new work
                while futures or queue_idx < len(work_queue):
                    # Wait for any future to complete
                    done_futures = []
                    for future_info in futures:
                        future, start_idx, chunk_size = future_info
                        if future.done():
                            done_futures.append(future_info)
                    
                    # Process completed futures
                    for future_info in done_futures:
                        future, start_idx, chunk_size = future_info
                        futures.remove(future_info)
                        
                        try:
                            chunk_results = future.result()
                            results[start_idx:start_idx + chunk_size] = chunk_results
                            completed += chunk_size
                            
                            if pbar:
                                pbar.update(chunk_size)
                        except Exception as e:
                            warnings.warn(f"Work unit failed: {e}")
                        
                        # Submit new work if available
                        if queue_idx < len(work_queue):
                            start_idx, chunk_size = work_queue[queue_idx]
                            future = executor.submit(
                                simulate_func, chunk_size, n_policies, **kwargs
                            )
                            futures.add((future, start_idx, chunk_size))
                            queue_idx += 1
                    
                    # Small sleep to avoid busy waiting
                    if not done_futures:
                        time.sleep(0.001)
        
        finally:
            if pbar:
                pbar.close()
        
        return results


def parallel_worker(args: Tuple[Any, ...]) -> np.ndarray:
    """Worker function for parallel simulation."""
    simulate_func, n_sims, n_policies, kwargs = args
    return simulate_func(n_sims, n_policies, **kwargs)


def benchmark_parallel_methods():
    """Benchmark different parallel processing methods."""
    from quactuary.book import Portfolio, Inforce, PolicyTerms
    from quactuary.distributions.frequency import Poisson
    from quactuary.distributions.severity import Lognormal
    from quactuary.vectorized_simulation import VectorizedSimulator
    import pandas as pd
    
    print("PARALLEL PROCESSING BENCHMARK")
    print("=" * 60)
    print(f"CPU cores available: {cpu_count()}")
    print(f"Memory available: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    print()
    
    # Create test portfolio
    terms = PolicyTerms(
        effective_date=pd.Timestamp('2024-01-01'),
        expiration_date=pd.Timestamp('2024-12-31')
    )
    
    inforce = Inforce(
        n_policies=100,
        terms=terms,
        frequency=Poisson(mu=1.5),
        severity=Lognormal(shape=1.0, scale=np.exp(8.0))
    )
    
    # Define simulation function
    def simulate_batch(n_sims, n_policies):
        return VectorizedSimulator.simulate_inforce_vectorized(inforce, n_sims)
    
    n_simulations = 100000
    configs = [
        ParallelConfig(n_workers=1, show_progress=False),
        ParallelConfig(n_workers=2, show_progress=False),
        ParallelConfig(n_workers=4, show_progress=False),
        ParallelConfig(n_workers=cpu_count(), show_progress=False),
    ]
    
    results = {}
    
    for config in configs:
        simulator = ParallelSimulator(config)
        
        # Multiprocessing
        start = time.perf_counter()
        mp_results = simulator.simulate_parallel_multiprocessing(
            simulate_batch, n_simulations, inforce.n_policies
        )
        mp_time = time.perf_counter() - start
        results[f'multiprocessing_{config.n_workers}'] = mp_time
        
        # Joblib (if available)
        if HAS_JOBLIB:
            start = time.perf_counter()
            jl_results = simulator.simulate_parallel_joblib(
                simulate_batch, n_simulations, inforce.n_policies
            )
            jl_time = time.perf_counter() - start
            results[f'joblib_{config.n_workers}'] = jl_time
        
        # Work stealing (only for multi-worker)
        if config.n_workers > 1:
            config.work_stealing = True
            simulator_ws = ParallelSimulator(config)
            start = time.perf_counter()
            ws_results = simulator_ws.simulate_work_stealing(
                simulate_batch, n_simulations, inforce.n_policies
            )
            ws_time = time.perf_counter() - start
            results[f'work_stealing_{config.n_workers}'] = ws_time
    
    # Print results
    print("\nResults (seconds):")
    print("-" * 40)
    baseline = results.get('multiprocessing_1', 1.0)
    
    for method, time_taken in sorted(results.items()):
        speedup = baseline / time_taken
        print(f"{method:25} {time_taken:8.3f}s (speedup: {speedup:5.2f}x)")


if __name__ == "__main__":
    benchmark_parallel_methods()