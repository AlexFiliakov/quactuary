"""
Optimized Sobol sequence generation with parallel processing support.

This module provides performance optimizations for Sobol sequence generation
including multi-threaded generation and memory-efficient batch processing.
"""

import numpy as np
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from scipy.stats import qmc

from quactuary.sobol import SobolEngine


class OptimizedSobolEngine:
    """
    Optimized Sobol sequence generator with parallel processing capabilities.
    
    Features:
    - Multi-threaded batch generation
    - Memory-efficient streaming for large sequences
    - Dimension-wise parallelization
    - Cache-friendly access patterns
    """
    
    def __init__(
        self,
        dimension: int,
        scramble: bool = True,
        seed: Optional[int] = None,
        n_threads: Optional[int] = None
    ):
        """
        Initialize optimized Sobol engine.
        
        Args:
            dimension: Number of dimensions
            scramble: Whether to use scrambling
            seed: Random seed for scrambling
            n_threads: Number of threads (default: CPU count)
        """
        self.dimension = dimension
        self.scramble = scramble
        self.seed = seed
        self.n_threads = n_threads or mp.cpu_count()
        
        # Pre-allocate engines for each thread
        self._engines = []
        for i in range(self.n_threads):
            engine_seed = None if seed is None else seed + i
            engine = qmc.Sobol(
                d=dimension,
                scramble=scramble,
                seed=engine_seed
            )
            self._engines.append(engine)
    
    def generate_parallel(self, n_samples: int) -> np.ndarray:
        """
        Generate Sobol points using parallel processing.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Array of shape (n_samples, dimension)
        """
        # Determine optimal batch size
        batch_size = self._optimal_batch_size(n_samples)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        # Generate batches in parallel
        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            futures = []
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)
                batch_samples = end_idx - start_idx
                
                future = executor.submit(
                    self._generate_batch,
                    engine_idx=i % self.n_threads,
                    n_samples=batch_samples,
                    skip=start_idx
                )
                futures.append(future)
            
            # Collect results
            results = [future.result() for future in futures]
        
        # Concatenate batches
        return np.vstack(results)[:n_samples]
    
    def generate_streaming(
        self,
        n_samples: int,
        chunk_size: int = 10000
    ) -> np.ndarray:
        """
        Generate Sobol points with memory-efficient streaming.
        
        Useful for very large sample sizes that might not fit in memory.
        
        Args:
            n_samples: Total number of samples
            chunk_size: Size of each chunk to process
            
        Yields:
            Arrays of shape (chunk_size, dimension)
        """
        n_chunks = (n_samples + chunk_size - 1) // chunk_size
        
        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, n_samples)
            chunk_samples = end_idx - start_idx
            
            # Generate chunk
            chunk = self._engines[0].random(chunk_samples)
            yield chunk
    
    def generate_dimension_parallel(
        self,
        n_samples: int,
        dim_batch_size: int = 100
    ) -> np.ndarray:
        """
        Generate Sobol points with dimension-wise parallelization.
        
        Useful for very high-dimensional problems where dimension > 1000.
        
        Args:
            n_samples: Number of samples
            dim_batch_size: Number of dimensions per batch
            
        Returns:
            Array of shape (n_samples, dimension)
        """
        n_dim_batches = (self.dimension + dim_batch_size - 1) // dim_batch_size
        
        # Initialize result array
        result = np.zeros((n_samples, self.dimension))
        
        with ProcessPoolExecutor(max_workers=self.n_threads) as executor:
            futures = []
            
            for i in range(n_dim_batches):
                start_dim = i * dim_batch_size
                end_dim = min((i + 1) * dim_batch_size, self.dimension)
                n_dims = end_dim - start_dim
                
                future = executor.submit(
                    self._generate_dim_batch,
                    n_samples=n_samples,
                    n_dims=n_dims,
                    seed=self.seed + i if self.seed else None
                )
                futures.append((future, start_dim, end_dim))
            
            # Collect results
            for future, start_dim, end_dim in futures:
                dim_data = future.result()
                result[:, start_dim:end_dim] = dim_data
        
        return result
    
    def _generate_batch(
        self,
        engine_idx: int,
        n_samples: int,
        skip: int = 0
    ) -> np.ndarray:
        """Generate a batch of samples using specified engine."""
        engine = self._engines[engine_idx]
        
        # Skip to appropriate position
        if skip > 0:
            _ = engine.random(skip)
        
        return engine.random(n_samples)
    
    def _generate_dim_batch(
        self,
        n_samples: int,
        n_dims: int,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """Generate samples for a subset of dimensions."""
        engine = qmc.Sobol(d=n_dims, scramble=self.scramble, seed=seed)
        return engine.random(n_samples)
    
    def _optimal_batch_size(self, n_samples: int) -> int:
        """Determine optimal batch size based on system capabilities."""
        # Balance between parallelization overhead and memory usage
        min_batch = 1000
        max_batch = 100000
        
        # Aim for ~10-20 batches per thread
        ideal_batch = n_samples // (self.n_threads * 15)
        
        return max(min_batch, min(ideal_batch, max_batch))


class GPUSobolEngine:
    """
    GPU-accelerated Sobol sequence generation (requires CuPy).
    
    Note: This is a placeholder for future GPU implementation.
    Actual implementation would require CuPy or similar GPU library.
    """
    
    def __init__(self, dimension: int, device: int = 0):
        """
        Initialize GPU Sobol engine.
        
        Args:
            dimension: Number of dimensions
            device: GPU device ID
        """
        self.dimension = dimension
        self.device = device
        
        # Check if GPU acceleration is available
        try:
            import cupy as cp
            self.cp = cp
            self.gpu_available = True
        except ImportError:
            self.gpu_available = False
            print("Warning: CuPy not available, falling back to CPU")
    
    def generate(self, n_samples: int) -> np.ndarray:
        """
        Generate Sobol points on GPU.
        
        Args:
            n_samples: Number of samples
            
        Returns:
            Array of shape (n_samples, dimension)
        """
        if not self.gpu_available:
            # Fallback to CPU
            engine = SobolEngine(self.dimension)
            return engine.generate(n_samples)
        
        # Placeholder for GPU implementation
        # Would implement Gray code generation and direction numbers on GPU
        raise NotImplementedError("GPU Sobol generation not yet implemented")


def benchmark_sobol_optimizations(
    dimensions: List[int],
    sample_sizes: List[int]
) -> dict:
    """
    Benchmark different Sobol generation strategies.
    
    Args:
        dimensions: List of dimension counts to test
        sample_sizes: List of sample sizes to test
        
    Returns:
        Dictionary with benchmark results
    """
    import time
    
    results = {
        'standard': [],
        'parallel': [],
        'streaming': [],
        'dimension_parallel': []
    }
    
    for dim in dimensions:
        for n_samples in sample_sizes:
            print(f"\nBenchmarking d={dim}, n={n_samples}")
            
            # Standard generation
            engine = SobolEngine(dimension=dim)
            start = time.time()
            _ = engine.generate(n_samples)
            standard_time = time.time() - start
            results['standard'].append({
                'dimension': dim,
                'n_samples': n_samples,
                'time': standard_time,
                'samples_per_second': n_samples / standard_time
            })
            
            # Parallel generation
            opt_engine = OptimizedSobolEngine(dimension=dim)
            start = time.time()
            _ = opt_engine.generate_parallel(n_samples)
            parallel_time = time.time() - start
            results['parallel'].append({
                'dimension': dim,
                'n_samples': n_samples,
                'time': parallel_time,
                'samples_per_second': n_samples / parallel_time,
                'speedup': standard_time / parallel_time
            })
            
            # Streaming generation (just time first chunk)
            start = time.time()
            for i, chunk in enumerate(opt_engine.generate_streaming(n_samples)):
                if i == 0:
                    streaming_time = time.time() - start
                    break
            results['streaming'].append({
                'dimension': dim,
                'n_samples': n_samples,
                'first_chunk_time': streaming_time
            })
            
            # Dimension-parallel (only for high dimensions)
            if dim >= 100:
                start = time.time()
                _ = opt_engine.generate_dimension_parallel(min(n_samples, 1000))
                dim_parallel_time = time.time() - start
                results['dimension_parallel'].append({
                    'dimension': dim,
                    'n_samples': min(n_samples, 1000),
                    'time': dim_parallel_time,
                    'speedup': standard_time / dim_parallel_time if n_samples <= 1000 else None
                })
    
    return results


def optimize_dimension_allocation(
    n_policies: int,
    claim_distributions: List[Tuple[float, float]],
    total_dimensions: int = 10000
) -> List[Tuple[int, List[int]]]:
    """
    Optimize dimension allocation for a portfolio.
    
    Uses policy characteristics to allocate dimensions more efficiently.
    
    Args:
        n_policies: Number of policies
        claim_distributions: List of (mean, std) for each policy's claim count
        total_dimensions: Total available dimensions
        
    Returns:
        List of (freq_dim, sev_dims) for each policy
    """
    # Calculate dimension needs based on claim volatility
    dimension_needs = []
    
    for i, (mean_claims, std_claims) in enumerate(claim_distributions):
        # Policies with higher claim volatility need more dimensions
        volatility = std_claims / (mean_claims + 1e-6)
        max_claims = int(mean_claims + 3 * std_claims)
        
        # Weight by volatility
        weight = 1.0 + volatility
        dimension_needs.append((i, max_claims, weight))
    
    # Sort by weight (highest need first)
    dimension_needs.sort(key=lambda x: x[2], reverse=True)
    
    # Allocate dimensions proportionally
    allocations = []
    used_dims = 0
    
    for policy_idx, max_claims, weight in dimension_needs:
        # Ensure we don't exceed total dimensions
        remaining_dims = total_dimensions - used_dims
        remaining_policies = n_policies - len(allocations)
        
        if remaining_policies > 0:
            # Reserve at least 10 dimensions per remaining policy
            max_for_this_policy = remaining_dims - 10 * (remaining_policies - 1)
            
            # Allocate based on need but within constraints
            dims_for_policy = min(max_claims + 1, max_for_this_policy)
            dims_for_policy = max(dims_for_policy, 2)  # At least freq + 1 severity
            
            freq_dim = used_dims
            sev_dims = list(range(used_dims + 1, used_dims + dims_for_policy))
            
            allocations.append((policy_idx, (freq_dim, sev_dims)))
            used_dims += dims_for_policy
    
    # Sort back to original order
    allocations.sort(key=lambda x: x[0])
    
    return [alloc[1] for alloc in allocations]


if __name__ == '__main__':
    # Example usage and benchmarks
    print("Optimized Sobol Generation Benchmark\n")
    
    # Test configurations
    dimensions = [10, 100, 1000]
    sample_sizes = [10000, 50000, 100000]
    
    # Run benchmarks
    results = benchmark_sobol_optimizations(dimensions, sample_sizes)
    
    # Print results
    print("\nBenchmark Results:")
    print("-" * 60)
    
    for method, method_results in results.items():
        print(f"\n{method.upper()}:")
        for result in method_results:
            print(f"  d={result['dimension']}, n={result.get('n_samples', 'N/A')}: "
                  f"{result.get('time', 'N/A'):.3f}s")
            if 'speedup' in result and result['speedup']:
                print(f"    Speedup: {result['speedup']:.2f}x")
    
    # Test dimension allocation optimization
    print("\n\nDimension Allocation Optimization:")
    print("-" * 60)
    
    # Example portfolio with varying claim characteristics
    claim_dists = [
        (5, 2),    # Low volatility
        (10, 8),   # High volatility
        (3, 1),    # Very low volatility
        (20, 15),  # Very high volatility
        (8, 4),    # Medium volatility
    ]
    
    allocations = optimize_dimension_allocation(5, claim_dists, total_dimensions=100)
    
    for i, (freq_dim, sev_dims) in enumerate(allocations):
        mean, std = claim_dists[i]
        print(f"Policy {i}: mean={mean}, std={std}")
        print(f"  Allocated: freq_dim={freq_dim}, n_sev_dims={len(sev_dims)}")
        print(f"  Total dimensions: {1 + len(sev_dims)}")