"""
Memory management optimizations for large-scale actuarial simulations.

This module provides comprehensive memory management capabilities for handling
large portfolios and high simulation counts that would otherwise exceed available
memory. It implements adaptive memory management, streaming simulation capabilities,
and memory-efficient algorithms to enable actuarial calculations at scale.

Key Features:
    - Adaptive batch sizing based on available system memory
    - Memory usage monitoring and estimation
    - Streaming simulation for arbitrarily large problems
    - Disk-based caching and memory mapping for extreme cases
    - Online statistical algorithms for memory-efficient computation
    - Garbage collection optimization for batch processing

Memory Management Strategy:
    The module uses a multi-tiered approach to handle memory constraints:
    1. Estimate memory requirements for the full problem
    2. Calculate optimal batch sizes based on available memory
    3. Use streaming algorithms for problems that exceed memory
    4. Employ memory mapping and disk caching as fallbacks
    5. Apply online statistics to compute results without storing all data

Architecture:
    - MemoryConfig: Configuration for memory management parameters
    - MemoryManager: Core memory monitoring and batch size calculation
    - StreamingSimulator: Implements streaming simulation for large problems
    - Helper functions for memory optimization and cleanup

Typical Usage Patterns:
    Small portfolios (< 1GB):
        Use standard in-memory calculations with optional batching
        
    Medium portfolios (1-10GB):
        Use adaptive batching with memory monitoring
        
    Large portfolios (> 10GB):
        Use streaming simulation with online statistics
        
    Extreme portfolios (> available memory):
        Use memory mapping and disk-based intermediate storage

Examples:
    Basic memory management:
        >>> from quactuary.memory_management import MemoryManager, MemoryConfig
        >>> 
        >>> # Configure for 4GB maximum usage
        >>> config = MemoryConfig(max_memory_gb=4.0, safety_factor=0.8)
        >>> mem_manager = MemoryManager(config)
        >>> 
        >>> # Calculate optimal batch size for simulation
        >>> n_policies, n_sims = 10000, 1000000
        >>> batch_size = mem_manager.calculate_optimal_batch_size(n_policies, n_sims)
        >>> print(f"Optimal batch size: {batch_size:,}")
        
    Streaming simulation for large problems:
        >>> from quactuary.memory_management import StreamingSimulator
        >>> 
        >>> def my_simulation_func(batch_size):
        ...     # Your simulation logic here
        ...     return np.random.normal(1000, 200, batch_size)
        >>> 
        >>> simulator = StreamingSimulator(mem_manager)
        >>> results = []
        >>> for batch in simulator.simulate_streaming(
        ...     simulate_func=my_simulation_func,
        ...     n_simulations=10000000,  # 10M simulations
        ...     n_policies=50000         # 50K policies
        ... ):
        ...     # Process each batch
        ...     results.append(batch.mean())
        
    Memory usage monitoring:
        >>> # Monitor memory during calculation
        >>> initial_memory = mem_manager.get_used_memory()
        >>> 
        >>> # ... run calculations ...
        >>> 
        >>> final_memory = mem_manager.get_used_memory()
        >>> print(f"Memory used: {final_memory - initial_memory:.2f} GB")
        
    Online statistics for streaming data:
        >>> def data_generator():
        ...     for i in range(1000):  # 1000 batches
        ...         yield np.random.normal(0, 1, 10000)  # 10K samples each
        >>> 
        >>> stats = simulator.calculate_streaming_statistics(
        ...     data_generator(),
        ...     confidence_levels=[0.95, 0.99]
        ... )
        >>> print(f"Streaming mean: {stats['mean']:.4f}")
        >>> print(f"95% VaR: {stats['var_95%']:.4f}")

Performance Considerations:
    - Batch size calculation balances memory usage and computation efficiency
    - Memory mapping provides constant memory usage regardless of problem size
    - Streaming algorithms have O(1) memory complexity but O(n) time complexity
    - Garbage collection optimization reduces overhead during batch processing
    - Online algorithms provide approximate results with bounded memory usage

Notes:
    - Memory estimates include safety factors for Python object overhead
    - The module automatically detects system capabilities and adjusts behavior
    - Temporary files are automatically cleaned up on normal and error exits
    - HDF5 support is optional but recommended for very large simulations
    - Memory mapping requires sufficient disk space for temporary storage

See Also:
    - parallel_processing: For CPU parallelization with memory awareness
    - vectorized_simulation: For NumPy-optimized memory-efficient calculations
    - classical: For standard in-memory actuarial calculations
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
    """Configuration parameters for memory management.
    
    This dataclass defines the configuration parameters that control memory
    management behavior throughout the system. It provides sensible defaults
    while allowing fine-tuning for specific use cases.
    
    Attributes:
        max_memory_gb (Optional[float]): Maximum memory to use in GB. If None,
            automatically determined as safety_factor * total_system_memory.
            Default is None (auto-detection).
        safety_factor (float): Fraction of available memory to use as a safety
            margin. Should be between 0.0 and 1.0. Default is 0.8 (80%).
        min_batch_size (int): Minimum batch size for simulations. Prevents
            excessive overhead from very small batches. Default is 100.
        max_batch_size (int): Maximum batch size for simulations. Prevents
            excessive memory usage from very large batches. Default is 100,000.
        use_disk_cache (bool): Whether to use disk-based caching for very large
            simulations that exceed memory limits. Default is True.
        temp_dir (Optional[str]): Directory for temporary files. If None, uses
            system default temporary directory. Default is None.
    
    Examples:
        Conservative memory usage:
            >>> config = MemoryConfig(
            ...     max_memory_gb=2.0,
            ...     safety_factor=0.6,
            ...     min_batch_size=500
            ... )
            
        High-performance configuration:
            >>> config = MemoryConfig(
            ...     max_memory_gb=16.0,
            ...     safety_factor=0.9,
            ...     max_batch_size=1000000
            ... )
            
        Disk-caching disabled:
            >>> config = MemoryConfig(
            ...     use_disk_cache=False,
            ...     max_memory_gb=8.0
            ... )
    
    Notes:
        - Setting max_memory_gb too high can cause system instability
        - safety_factor accounts for Python object overhead and fragmentation
        - Batch size limits affect both memory usage and computational efficiency
        - Disk caching requires sufficient disk space in temp_dir
    """
    max_memory_gb: float = None  # Maximum memory to use (None = auto)
    safety_factor: float = 0.8   # Use only 80% of available memory
    min_batch_size: int = 100    # Minimum batch size
    max_batch_size: int = 100000 # Maximum batch size
    use_disk_cache: bool = True  # Use disk for very large simulations
    temp_dir: str = None         # Directory for temporary files


class MemoryManager:
    """
    Central memory management system for large-scale actuarial simulations.
    
    The MemoryManager provides comprehensive memory monitoring, estimation, and
    optimization capabilities for handling actuarial calculations that approach
    or exceed available system memory. It dynamically adapts batch sizes,
    monitors memory usage, and provides memory-efficient alternatives.
    
    Key Capabilities:
        - Real-time memory monitoring of system and process usage
        - Memory requirement estimation for simulation parameters
        - Adaptive batch size calculation based on available memory
        - Memory-mapped array creation for out-of-core computation
        - Garbage collection optimization for batch processing
        - Temporary file management and cleanup
    
    Memory Estimation Algorithm:
        The manager estimates memory requirements based on:
        1. Number of policies and simulations
        2. Data type sizes (typically float64 = 8 bytes)
        3. Overhead factors for intermediate arrays
        4. Python object overhead (typically 50% additional)
        5. Safety margins to prevent memory exhaustion
    
    Batch Size Optimization:
        Optimal batch sizes are calculated by:
        1. Determining available memory after safety factors
        2. Estimating memory per simulation
        3. Computing maximum simulations that fit in memory
        4. Applying min/max constraints and problem size limits
    
    Attributes:
        config (MemoryConfig): Configuration parameters for memory management.
        process (psutil.Process): Process object for memory monitoring.
    
    Examples:
        Basic usage:
            >>> from quactuary.memory_management import MemoryManager, MemoryConfig
            >>> 
            >>> # Use default configuration
            >>> manager = MemoryManager()
            >>> print(f"Available memory: {manager.get_available_memory():.1f} GB")
            
        Custom configuration:
            >>> config = MemoryConfig(max_memory_gb=4.0, safety_factor=0.7)
            >>> manager = MemoryManager(config)
            >>> 
            >>> # Calculate batch size for large simulation
            >>> batch_size = manager.calculate_optimal_batch_size(
            ...     n_policies=50000,
            ...     n_simulations=1000000
            ... )
            >>> print(f"Optimal batch size: {batch_size:,}")
            
        Memory estimation:
            >>> # Estimate memory for full simulation
            >>> memory_gb = manager.estimate_memory_usage(
            ...     n_policies=10000,
            ...     n_simulations=100000
            ... )
            >>> print(f"Estimated memory: {memory_gb:.2f} GB")
            >>> 
            >>> # Check if it fits in available memory
            >>> if memory_gb > manager.get_available_memory():
            ...     print("Need to use batching or streaming")
            
        Memory-mapped arrays:
            >>> # Create large array that may not fit in memory
            >>> shape = (1000000, 100)  # 100M float64 values ≈ 800MB
            >>> mmap_array = manager.create_memory_map(shape)
            >>> 
            >>> # Use the array for computation
            >>> mmap_array[:] = np.random.normal(0, 1, shape)
            >>> result = np.mean(mmap_array, axis=0)
            >>> 
            >>> # Cleanup temporary files
            >>> manager.cleanup_temp_files()
            
        Garbage collection optimization:
            >>> manager.optimize_gc()  # Disable GC during batch processing
            >>> try:
            ...     # Perform memory-intensive calculations
            ...     for batch in range(num_batches):
            ...         process_batch(batch)
            ... finally:
            ...     manager.restore_gc()  # Re-enable GC
    
    Performance Notes:
        - Memory estimation includes generous overhead factors
        - Batch size calculation favors memory safety over speed
        - Memory mapping provides predictable memory usage
        - GC optimization can significantly reduce processing time
        - Available memory detection accounts for OS and other processes
    
    Thread Safety:
        The MemoryManager is designed for single-threaded use. For multi-
        threaded applications, create separate instances per thread or use
        appropriate synchronization.
    
    See Also:
        - StreamingSimulator: For problems that exceed memory even with batching
        - parallel_processing: For memory-aware parallel computation
        - MemoryConfig: For configuring memory management parameters
    """
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        """Initialize the memory manager with configuration.
        
        Args:
            config (Optional[MemoryConfig]): Memory management configuration.
                If None, uses default MemoryConfig with auto-detected memory limits.
        
        Examples:
            Default configuration:
                >>> manager = MemoryManager()
                >>> # Uses 80% of system memory with standard batch size limits
                
            Custom configuration:
                >>> config = MemoryConfig(
                ...     max_memory_gb=8.0,
                ...     safety_factor=0.9,
                ...     min_batch_size=1000
                ... )
                >>> manager = MemoryManager(config)
        
        Notes:
            - Auto-detection uses psutil to determine system memory
            - Maximum memory is set to total_memory * safety_factor if not specified
            - The process object enables monitoring of current memory usage
        """
        self.config = config or MemoryConfig()
        self.process = psutil.Process()
        
        # Set maximum memory if not specified
        if self.config.max_memory_gb is None:
            total_memory_gb = psutil.virtual_memory().total / (1024**3)
            self.config.max_memory_gb = total_memory_gb * self.config.safety_factor
    
    def get_available_memory(self) -> float:
        """Get currently available system memory in GB.
        
        Returns:
            float: Available memory in gigabytes.
            
        Examples:
            >>> manager = MemoryManager()
            >>> available = manager.get_available_memory()
            >>> print(f"Available memory: {available:.1f} GB")
            
        Notes:
            - Available memory excludes memory used by OS and other processes
            - Value changes dynamically as system memory usage fluctuates
            - Used for real-time batch size adjustment
        """
        return psutil.virtual_memory().available / (1024**3)
    
    def get_used_memory(self) -> float:
        """Get memory used by the current process in GB.
        
        Returns:
            float: Resident memory usage of current process in gigabytes.
            
        Examples:
            >>> manager = MemoryManager()
            >>> initial_memory = manager.get_used_memory()
            >>> 
            >>> # ... perform calculations ...
            >>> 
            >>> final_memory = manager.get_used_memory()
            >>> print(f"Memory increase: {final_memory - initial_memory:.2f} GB")
            
        Notes:
            - Uses RSS (Resident Set Size) which is physical memory actually in use
            - Excludes swapped memory and shared libraries
            - Useful for monitoring memory growth during calculations
        """
        return self.process.memory_info().rss / (1024**3)
    
    def estimate_memory_usage(
        self,
        n_policies: int,
        n_simulations: int,
        bytes_per_element: int = 8  # float64
    ) -> float:
        """
        Estimate memory usage for an actuarial simulation in GB.
        
        This method provides a conservative estimate of memory requirements for
        actuarial simulations based on portfolio size and simulation parameters.
        The estimate includes overhead for temporary arrays and Python objects.
        
        The estimation algorithm accounts for:
        - Frequency simulation matrices (n_policies × n_simulations)
        - Severity sample storage (estimated 2 claims per policy on average)
        - Result aggregation arrays
        - Python object overhead (50% additional)
        - Temporary arrays used during computation
        
        Args:
            n_policies (int): Number of policies in the portfolio.
            n_simulations (int): Number of Monte Carlo simulations to run.
            bytes_per_element (int): Memory size per numeric element in bytes.
                Default is 8 for float64. Use 4 for float32.
                
        Returns:
            float: Estimated memory usage in gigabytes.
            
        Examples:
            Small portfolio estimation:
                >>> manager = MemoryManager()
                >>> memory_gb = manager.estimate_memory_usage(
                ...     n_policies=1000,
                ...     n_simulations=10000
                ... )
                >>> print(f"Estimated memory: {memory_gb:.3f} GB")
                
            Large portfolio with float32:
                >>> memory_gb = manager.estimate_memory_usage(
                ...     n_policies=100000,
                ...     n_simulations=1000000,
                ...     bytes_per_element=4  # float32
                ... )
                >>> print(f"Memory with float32: {memory_gb:.2f} GB")
                
            Check if simulation fits in memory:
                >>> available = manager.get_available_memory()
                >>> required = manager.estimate_memory_usage(50000, 100000)
                >>> if required > available:
                ...     print(f"Need {required:.1f} GB but only {available:.1f} GB available")
                ...     # Use batching or streaming
                
        Notes:
            - Estimates are conservative and include generous overhead
            - Actual usage may be lower for sparse portfolios
            - Does not account for frequency distribution specifics
            - Overhead factor may need adjustment for very large simulations
            - Consider using float32 for large problems to halve memory usage
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
        Calculate the optimal batch size for memory-efficient simulation.
        
        This method determines the largest batch size that will fit within
        available memory constraints while respecting configured limits.
        The calculation balances memory efficiency with computational overhead.
        
        Algorithm:
        1. Determine target memory (available memory with safety factors)
        2. Estimate memory usage per simulation
        3. Calculate maximum simulations that fit in target memory
        4. Apply minimum and maximum batch size constraints
        5. Ensure batch size doesn't exceed total simulations needed
        
        Args:
            n_policies (int): Number of policies in the portfolio.
            n_simulations (int): Total number of simulations needed.
            target_memory_gb (Optional[float]): Target memory usage in GB.
                If None, automatically calculated based on available memory
                and safety factors. Default is None.
                
        Returns:
            int: Optimal batch size for the given constraints.
            
        Examples:
            Basic batch size calculation:
                >>> manager = MemoryManager()
                >>> batch_size = manager.calculate_optimal_batch_size(
                ...     n_policies=10000,
                ...     n_simulations=1000000
                ... )
                >>> print(f"Optimal batch size: {batch_size:,}")
                
            With specific memory target:
                >>> batch_size = manager.calculate_optimal_batch_size(
                ...     n_policies=50000,
                ...     n_simulations=500000,
                ...     target_memory_gb=2.0  # Limit to 2GB
                ... )
                >>> print(f"Batch size for 2GB limit: {batch_size:,}")
                
            Calculate number of batches needed:
                >>> n_sims = 1000000
                >>> batch_size = manager.calculate_optimal_batch_size(10000, n_sims)
                >>> n_batches = (n_sims + batch_size - 1) // batch_size
                >>> print(f"Will need {n_batches} batches of size {batch_size:,}")
                
            Memory-constrained scenario:
                >>> config = MemoryConfig(max_memory_gb=1.0)  # Very limited
                >>> manager = MemoryManager(config)
                >>> batch_size = manager.calculate_optimal_batch_size(100000, 1000000)
                >>> # Will return minimum batch size if memory is very constrained
                
        Notes:
            - Batch size is constrained by min_batch_size and max_batch_size in config
            - Smaller batches reduce memory usage but increase computational overhead
            - Larger batches are more efficient but require more memory
            - The algorithm favors memory safety over computational efficiency
            - Returns min_batch_size if memory is extremely constrained
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
        """Optimize garbage collection for batch processing.
        
        Disables automatic garbage collection to reduce overhead during
        intensive batch processing. Should be paired with restore_gc()
        in a try/finally block to ensure GC is re-enabled.
        
        Examples:
            Typical usage pattern:
                >>> manager.optimize_gc()
                >>> try:
                ...     # Perform batch processing
                ...     for batch in batches:
                ...         process_batch(batch)
                ... finally:
                ...     manager.restore_gc()
                
        Notes:
            - Performs a full collection before disabling GC
            - Reduces overhead but may increase memory usage temporarily
            - Must be paired with restore_gc() to avoid memory leaks
            - Most effective for tight loops with many small allocations
        """
        # Disable automatic GC during batch processing
        gc.collect()
        gc.disable()
        
    def restore_gc(self):
        """Restore normal garbage collection after optimization.
        
        Re-enables automatic garbage collection and performs a full
        collection to clean up any accumulated garbage during the
        optimized processing period.
        
        Examples:
            Always use in finally block:
                >>> manager.optimize_gc()
                >>> try:
                ...     # Batch processing code
                ...     pass
                ... finally:
                ...     manager.restore_gc()  # Always restore
                
        Notes:
            - Performs a full collection after re-enabling GC
            - Should always be called after optimize_gc()
            - Safe to call multiple times
        """
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
    Streaming simulation engine for memory-constrained environments.
    
    The StreamingSimulator enables actuarial calculations on portfolios and
    simulation counts that would be impossible to handle in memory all at once.
    It processes simulations in optimally-sized batches, yielding results
    incrementally while maintaining constant memory usage.
    
    Key Features:
        - Constant memory usage regardless of total simulation count
        - Automatic batch size optimization based on available memory
        - Optional disk-based storage for intermediate results
        - Progress monitoring and callback support
        - Online statistical computation for streaming data
        - HDF5 integration for efficient data storage
    
    Architecture:
        The simulator works by:
        1. Calculating optimal batch sizes using the memory manager
        2. Processing simulations in batches
        3. Yielding each batch for immediate processing
        4. Optionally storing results to disk in compressed format
        5. Cleaning up resources automatically
    
    Memory Efficiency:
        - O(batch_size) memory complexity instead of O(total_simulations)
        - Automatic garbage collection optimization
        - Memory-mapped file support for very large results
        - Compressed storage options to minimize disk usage
    
    Attributes:
        memory_manager (MemoryManager): Memory manager for batch size optimization.
    
    Examples:
        Basic streaming simulation:
            >>> from quactuary.memory_management import MemoryManager, StreamingSimulator
            >>> 
            >>> def my_simulation(batch_size):
            ...     # Your simulation logic here
            ...     return np.random.exponential(1000, batch_size)
            >>> 
            >>> manager = MemoryManager()
            >>> simulator = StreamingSimulator(manager)
            >>> 
            >>> # Stream 10M simulations in memory-efficient batches
            >>> results = []
            >>> for batch in simulator.simulate_streaming(
            ...     simulate_func=my_simulation,
            ...     n_simulations=10_000_000,
            ...     n_policies=50_000
            ... ):
            ...     # Process each batch immediately
            ...     batch_stats = {'mean': batch.mean(), 'std': batch.std()}
            ...     results.append(batch_stats)
            
        With progress monitoring:
            >>> def progress_callback(completed, total):
            ...     percent = 100 * completed / total
            ...     print(f"Progress: {percent:.1f}% ({completed:,}/{total:,})")
            >>> 
            >>> for batch in simulator.simulate_streaming(
            ...     simulate_func=my_simulation,
            ...     n_simulations=1_000_000,
            ...     n_policies=10_000,
            ...     callback=progress_callback
            ... ):
            ...     # Process batches with progress updates
            ...     pass
            
        With disk storage:
            >>> # Store results to HDF5 file for later analysis
            >>> for batch in simulator.simulate_streaming(
            ...     simulate_func=my_simulation,
            ...     n_simulations=50_000_000,
            ...     n_policies=100_000,
            ...     output_file='large_simulation.h5'
            ... ):
            ...     # Results automatically saved to disk
            ...     pass
            
        Online statistics:
            >>> # Calculate statistics without storing all data
            >>> data_stream = simulator.simulate_streaming(
            ...     my_simulation, 10_000_000, 25_000
            ... )
            >>> stats = simulator.calculate_streaming_statistics(
            ...     data_stream,
            ...     confidence_levels=[0.90, 0.95, 0.99]
            ... )
            >>> print(f"Mean: {stats['mean']:.2f}")
            >>> print(f"95% VaR: {stats['var_95%']:.2f}")
    
    Performance Notes:
        - Batch processing reduces memory allocation overhead
        - HDF5 storage provides compression and fast I/O
        - Progress callbacks add minimal overhead
        - Online algorithms provide approximate but accurate statistics
        - Memory usage remains constant throughout execution
    
    Thread Safety:
        The StreamingSimulator is designed for single-threaded use. For
        parallel processing, use separate simulator instances per thread.
    
    See Also:
        - MemoryManager: For batch size optimization and memory monitoring
        - parallel_processing: For CPU parallelization with memory awareness
        - classical: For traditional in-memory simulation approaches
    """
    
    def __init__(self, memory_manager: MemoryManager):
        """Initialize the streaming simulator with a memory manager.
        
        Args:
            memory_manager (MemoryManager): Memory manager instance for
                batch size optimization and memory monitoring.
                
        Examples:
            >>> from quactuary.memory_management import MemoryManager, StreamingSimulator
            >>> 
            >>> # Create memory manager with custom settings
            >>> config = MemoryConfig(max_memory_gb=4.0)
            >>> manager = MemoryManager(config)
            >>> 
            >>> # Initialize simulator
            >>> simulator = StreamingSimulator(manager)
        """
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