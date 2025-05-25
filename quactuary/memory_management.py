"""
Memory management optimizations for large-scale simulations.

This module provides adaptive memory management, streaming capabilities,
and memory-efficient algorithms for handling large portfolios and 
high simulation counts.
"""

import gc
import os
import psutil
import numpy as np
import warnings
from typing import Optional, Tuple, Iterator, Callable
from dataclasses import dataclass
import tempfile
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


@dataclass
class MemoryConfig:
    """Configuration for memory management."""
    max_memory_gb: float = None  # Maximum memory to use (None = auto)
    safety_factor: float = 0.8   # Use only 80% of available memory
    min_batch_size: int = 100    # Minimum batch size
    max_batch_size: int = 100000 # Maximum batch size
    use_disk_cache: bool = True  # Use disk for very large simulations
    temp_dir: str = None         # Directory for temporary files


class MemoryManager:
    """
    Manages memory usage for large-scale simulations.
    
    Key features:
    - Adaptive batch sizing based on available memory
    - Memory usage monitoring
    - Disk-based caching for extreme cases
    - Garbage collection optimization
    """
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        """Initialize memory manager with configuration."""
        self.config = config or MemoryConfig()
        self.process = psutil.Process()
        
        # Set maximum memory if not specified
        if self.config.max_memory_gb is None:
            total_memory_gb = psutil.virtual_memory().total / (1024**3)
            self.config.max_memory_gb = total_memory_gb * self.config.safety_factor
    
    def get_available_memory(self) -> float:
        """Get available memory in GB."""
        return psutil.virtual_memory().available / (1024**3)
    
    def get_used_memory(self) -> float:
        """Get memory used by current process in GB."""
        return self.process.memory_info().rss / (1024**3)
    
    def estimate_memory_usage(
        self,
        n_policies: int,
        n_simulations: int,
        bytes_per_element: int = 8  # float64
    ) -> float:
        """
        Estimate memory usage for a simulation in GB.
        
        Args:
            n_policies: Number of policies
            n_simulations: Number of simulations
            bytes_per_element: Bytes per numeric element
            
        Returns:
            Estimated memory usage in GB
        """
        # Basic estimate: frequency matrix + severity samples + results
        freq_memory = n_policies * n_simulations * bytes_per_element
        # Assume average 2 claims per policy
        sev_memory = 2 * n_policies * n_simulations * bytes_per_element
        result_memory = n_simulations * bytes_per_element
        
        total_bytes = freq_memory + sev_memory + result_memory
        # Add 50% overhead for temporary arrays and Python objects
        total_bytes *= 1.5
        
        return total_bytes / (1024**3)
    
    def calculate_optimal_batch_size(
        self,
        n_policies: int,
        n_simulations: int,
        target_memory_gb: Optional[float] = None
    ) -> int:
        """
        Calculate optimal batch size for memory efficiency.
        
        Args:
            n_policies: Number of policies
            n_simulations: Total simulations needed
            target_memory_gb: Target memory usage (None = auto)
            
        Returns:
            Optimal batch size
        """
        if target_memory_gb is None:
            available = self.get_available_memory()
            target_memory_gb = min(
                available * self.config.safety_factor,
                self.config.max_memory_gb - self.get_used_memory()
            )
        
        # Estimate memory per simulation
        memory_per_sim = self.estimate_memory_usage(n_policies, 1)
        
        # Calculate batch size
        optimal_batch = int(target_memory_gb / memory_per_sim)
        
        # Apply constraints
        optimal_batch = max(self.config.min_batch_size, optimal_batch)
        optimal_batch = min(self.config.max_batch_size, optimal_batch)
        optimal_batch = min(n_simulations, optimal_batch)
        
        return optimal_batch
    
    def optimize_gc(self):
        """Optimize garbage collection for batch processing."""
        # Disable automatic GC during batch processing
        gc.collect()
        gc.disable()
        
    def restore_gc(self):
        """Restore normal garbage collection."""
        gc.enable()
        gc.collect()
    
    def create_memory_map(
        self,
        shape: Tuple[int, ...],
        dtype: np.dtype = np.float64,
        mode: str = 'w+'
    ) -> np.memmap:
        """
        Create a memory-mapped array for out-of-core computation.
        
        Args:
            shape: Shape of the array
            dtype: Data type
            mode: File mode ('w+' for read/write)
            
        Returns:
            Memory-mapped numpy array
        """
        if self.config.temp_dir:
            temp_dir = self.config.temp_dir
        else:
            temp_dir = tempfile.gettempdir()
        
        temp_file = tempfile.NamedTemporaryFile(
            dir=temp_dir,
            delete=False,
            prefix='quactuary_mmap_'
        )
        temp_file.close()
        
        mmap = np.memmap(
            temp_file.name,
            dtype=dtype,
            mode=mode,
            shape=shape
        )
        
        # Store filename for cleanup
        if not hasattr(self, '_temp_files'):
            self._temp_files = []
        self._temp_files.append(temp_file.name)
        
        return mmap
    
    def cleanup_temp_files(self):
        """Clean up temporary memory-mapped files."""
        if hasattr(self, '_temp_files'):
            for filename in self._temp_files:
                try:
                    os.unlink(filename)
                except:
                    pass
            self._temp_files = []


class StreamingSimulator:
    """
    Implements streaming simulation for extreme memory constraints.
    
    Processes simulations in small chunks and aggregates results
    incrementally, allowing simulation of arbitrarily large portfolios.
    """
    
    def __init__(self, memory_manager: MemoryManager):
        """Initialize with memory manager."""
        self.memory_manager = memory_manager
    
    def simulate_streaming(
        self,
        simulate_func: Callable,
        n_simulations: int,
        n_policies: int,
        output_file: Optional[str] = None,
        callback: Optional[Callable] = None
    ) -> Iterator[np.ndarray]:
        """
        Stream simulation results in batches.
        
        Args:
            simulate_func: Function that simulates a batch
            n_simulations: Total simulations
            n_policies: Number of policies  
            output_file: Optional HDF5 file for results
            callback: Optional progress callback
            
        Yields:
            Batches of simulation results
        """
        batch_size = self.memory_manager.calculate_optimal_batch_size(
            n_policies, n_simulations
        )
        
        print(f"Streaming simulation with batch size: {batch_size:,}")
        
        # Optional HDF5 output
        h5file = None
        h5dataset = None
        if output_file and HAS_H5PY:
            h5file = h5py.File(output_file, 'w')
            h5dataset = h5file.create_dataset(
                'simulations',
                shape=(n_simulations,),
                dtype=np.float64,
                chunks=True,
                compression='gzip'
            )
        elif output_file and not HAS_H5PY:
            warnings.warn("h5py not available, output_file will be ignored")
        
        try:
            self.memory_manager.optimize_gc()
            
            processed = 0
            while processed < n_simulations:
                # Calculate batch size for this iteration
                current_batch = min(batch_size, n_simulations - processed)
                
                # Run simulation batch
                batch_results = simulate_func(current_batch)
                
                # Save to HDF5 if specified
                if h5dataset is not None:
                    h5dataset[processed:processed + current_batch] = batch_results
                
                # Yield results
                yield batch_results
                
                processed += current_batch
                
                # Progress callback
                if callback:
                    callback(processed, n_simulations)
                
                # Explicit garbage collection every 10 batches
                if processed % (batch_size * 10) == 0:
                    gc.collect()
        
        finally:
            self.memory_manager.restore_gc()
            if h5file:
                h5file.close()
    
    def calculate_streaming_statistics(
        self,
        data_iterator: Iterator[np.ndarray],
        confidence_levels: list = None
    ) -> dict:
        """
        Calculate statistics from streaming data.
        
        Uses online algorithms to compute statistics without
        loading all data into memory at once.
        
        Args:
            data_iterator: Iterator yielding data batches
            confidence_levels: Confidence levels for quantiles
            
        Returns:
            Dictionary of computed statistics
        """
        if confidence_levels is None:
            confidence_levels = [0.90, 0.95, 0.99]
        
        # Online statistics
        n = 0
        mean = 0.0
        m2 = 0.0  # For variance calculation
        min_val = float('inf')
        max_val = float('-inf')
        
        # For quantile estimation, we need to keep some data
        # Use reservoir sampling for a subset
        reservoir_size = min(100000, int(1e6))  # Keep up to 100k samples
        reservoir = np.empty(reservoir_size)
        
        for batch in data_iterator:
            batch_n = len(batch)
            
            # Update count
            old_n = n
            n += batch_n
            
            # Update min/max
            min_val = min(min_val, np.min(batch))
            max_val = max(max_val, np.max(batch))
            
            # Update mean and variance (Welford's algorithm)
            for x in batch:
                delta = x - mean
                mean += delta / n
                m2 += delta * (x - mean)
            
            # Reservoir sampling for quantiles
            if old_n < reservoir_size:
                # Still filling reservoir
                available = reservoir_size - old_n
                to_add = min(available, batch_n)
                reservoir[old_n:old_n + to_add] = batch[:to_add]
            else:
                # Reservoir full, randomly replace
                for i, x in enumerate(batch):
                    j = np.random.randint(0, old_n + i + 1)
                    if j < reservoir_size:
                        reservoir[j] = x
        
        # Calculate final statistics
        variance = m2 / (n - 1) if n > 1 else 0.0
        std = np.sqrt(variance)
        
        results = {
            'count': n,
            'mean': mean,
            'std': std,
            'variance': variance,
            'min': min_val,
            'max': max_val
        }
        
        # Estimate quantiles from reservoir
        if n > 0:
            sample_size = min(n, reservoir_size)
            sorted_reservoir = np.sort(reservoir[:sample_size])
            
            for cl in confidence_levels:
                idx = int(cl * sample_size)
                idx = max(0, min(idx, sample_size - 1))
                results[f'var_{cl:.0%}'] = sorted_reservoir[idx]
        
        return results


def demonstrate_memory_management():
    """Demonstrate memory management capabilities."""
    print("MEMORY MANAGEMENT DEMONSTRATION")
    print("=" * 60)
    
    # Create memory manager
    config = MemoryConfig(max_memory_gb=2.0)  # Limit to 2GB
    mem_manager = MemoryManager(config)
    
    print(f"System memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print(f"Available memory: {mem_manager.get_available_memory():.1f} GB")
    print(f"Max allowed: {config.max_memory_gb:.1f} GB")
    print()
    
    # Test batch size calculation
    test_cases = [
        (100, 10000),      # Small
        (1000, 100000),    # Medium
        (10000, 1000000),  # Large
    ]
    
    for n_policies, n_sims in test_cases:
        batch_size = mem_manager.calculate_optimal_batch_size(n_policies, n_sims)
        memory_est = mem_manager.estimate_memory_usage(n_policies, batch_size)
        
        print(f"Portfolio: {n_policies:,} policies, {n_sims:,} simulations")
        print(f"  Optimal batch size: {batch_size:,}")
        print(f"  Memory per batch: {memory_est:.2f} GB")
        print(f"  Number of batches: {(n_sims + batch_size - 1) // batch_size}")
        print()


if __name__ == "__main__":
    demonstrate_memory_management()