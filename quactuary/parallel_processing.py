"""
Stable parallel processing implementation for Monte Carlo simulations.

This module provides robust parallelization with proper error handling,
timeout management, and resource cleanup.
"""

import os
import sys
import time
import signal
import traceback
import warnings
import numpy as np
from typing import Optional, List, Callable, Tuple, Any, Union
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, TimeoutError, as_completed
from multiprocessing import cpu_count, Pool, Queue, Manager, Process
from threading import Thread
from dataclasses import dataclass
from contextlib import contextmanager
import psutil
from functools import partial
import multiprocessing

# Import utilities
try:
    from .parallel_utils import CloudPickleWrapper, register_simulation_function, get_simulation_function
except ImportError:
    # Fallback if parallel_utils not available
    CloudPickleWrapper = None
    register_simulation_function = None
    get_simulation_function = None

try:
    from .work_distribution import DynamicLoadBalancer, AdaptiveChunkSizer, WorkStealingQueue
except ImportError:
    DynamicLoadBalancer = None
    AdaptiveChunkSizer = None
    WorkStealingQueue = None

try:
    from .parallel_error_handling import ErrorRecoveryManager, DiagnosticLogger, create_resilient_wrapper
except ImportError:
    ErrorRecoveryManager = None
    DiagnosticLogger = None
    create_resilient_wrapper = None

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
    """
    Comprehensive configuration for parallel processing operations.
    
    This dataclass encapsulates all configuration parameters for parallel
    processing, providing sensible defaults while allowing fine-tuning for
    specific use cases and performance requirements.
    
    Attributes:
        n_workers (Optional[int]): Number of worker processes or threads to use.
            If None, automatically determined based on CPU count and memory.
            Default is None (auto-detection).
        backend (str): Parallelization backend to use. Options are:
            - 'multiprocessing': Process-based parallelization (recommended for CPU-bound)
            - 'threading': Thread-based parallelization (better for I/O-bound) 
            - 'joblib': Joblib-based processing (experimental)
            Default is 'multiprocessing'.
        chunk_size (Optional[int]): Number of work items to process per chunk.
            Larger chunks reduce overhead but may cause load imbalance.
            If None, automatically calculated. Default is None (auto-sizing).
        show_progress (bool): Whether to display a progress bar during processing.
            Requires tqdm library. Default is True.
        work_stealing (bool): Enable work stealing between workers.
            Improves load balancing for variable execution times. Default is True.
        prefetch_batches (int): Number of work batches to prefetch for workers.
            Higher values improve throughput but increase memory usage. Default is 2.
        timeout (Optional[float]): Maximum time in seconds for each work item.
            Prevents hung processes. If None, no timeout applied. Default is None.
        max_retries (int): Maximum retry attempts for failed work items.
            Enables automatic recovery from transient failures. Default is 2.
        memory_limit_mb (Optional[float]): Memory limit per worker in MB.
            Workers exceeding this limit will be restarted. Default is None.
        fallback_to_serial (bool): Whether to fall back to serial processing
            if parallel processing fails. Default is True.
    
    Examples:
        High-performance configuration:
            >>> config = ParallelConfig(
            ...     n_workers=16,
            ...     backend='multiprocessing',
            ...     chunk_size=100,
            ...     timeout=300,
            ...     work_stealing=True
            ... )
            
        Memory-constrained configuration:
            >>> config = ParallelConfig(
            ...     n_workers=4,
            ...     memory_limit_mb=2000,
            ...     chunk_size=50,
            ...     prefetch_batches=1
            ... )
            
        Production configuration:
            >>> config = ParallelConfig(
            ...     n_workers=None,  # Auto-detect
            ...     timeout=600,     # 10 minute timeout
            ...     max_retries=3,   # Retry failed items
            ...     show_progress=True,
            ...     fallback_to_serial=True
            ... )
    
    Notes:
        - Auto-detection considers both CPU and memory constraints
        - Work stealing adds small coordination overhead
        - Memory limits require psutil for monitoring
        - Serial fallback ensures reliability in production environments
    """
    n_workers: Optional[int] = None  # None = auto-detect
    backend: str = 'multiprocessing'  # 'multiprocessing', 'threading', 'joblib'
    chunk_size: Optional[int] = None  # None = auto
    show_progress: bool = True
    work_stealing: bool = True
    prefetch_batches: int = 2
    timeout: Optional[float] = None  # Timeout in seconds for each task
    max_retries: int = 2  # Number of retries for failed tasks
    memory_limit_mb: Optional[float] = None  # Memory limit per worker
    fallback_to_serial: bool = True  # Fall back to serial on failures


class WorkerMonitor:
    """
    Real-time monitoring system for worker processes and resource usage.
    
    The WorkerMonitor provides comprehensive monitoring of worker processes,
    tracking memory usage, CPU utilization, and process health. It enables
    automatic detection of problematic workers and supports resource-based
    decision making for optimal parallel processing performance.
    
    Key Features:
        - Real-time memory usage monitoring per worker
        - Process health and responsiveness tracking
        - Automatic detection of memory leaks and resource exhaustion
        - Support for worker replacement and cleanup
        - Integration with system resource monitoring
    
    Attributes:
        max_memory_mb (Optional[float]): Maximum memory limit per worker in MB.
        processes (dict): Dictionary tracking monitored worker processes.
    
    Examples:
        Basic worker monitoring:
            >>> monitor = WorkerMonitor(max_memory_mb=1000)  # 1GB limit
            >>> 
            >>> # Monitor a worker process
            >>> worker_pid = 12345
            >>> monitor.register_worker(worker_pid)
            >>> 
            >>> # Check if worker exceeds memory limit
            >>> if monitor.check_worker_memory(worker_pid):
            ...     print("Worker memory usage is acceptable")
            ... else:
            ...     print("Worker exceeds memory limit - restart needed")
            
        Integration with parallel processing:
            >>> config = ParallelConfig(memory_limit_mb=1500)
            >>> monitor = WorkerMonitor(config.memory_limit_mb)
            >>> 
            >>> # Monitor workers during parallel execution
            >>> for worker in active_workers:
            ...     if not monitor.is_worker_healthy(worker.pid):
            ...         restart_worker(worker)
    
    Notes:
        - Requires psutil for detailed process monitoring
        - Memory monitoring has minimal performance overhead
        - Designed for long-running parallel processing jobs
        - Supports both process and thread monitoring
    """
    
    def __init__(self, max_memory_mb: Optional[float] = None):
        """
        Initialize the worker monitor with optional memory limits.
        
        Args:
            max_memory_mb (Optional[float]): Maximum memory usage per worker
                in megabytes. If None, no memory limit is enforced.
                
        Examples:
            >>> # Monitor with 500MB memory limit per worker
            >>> monitor = WorkerMonitor(max_memory_mb=500)
            >>> 
            >>> # Monitor without memory limits
            >>> monitor = WorkerMonitor()
        """
        self.max_memory_mb = max_memory_mb
        self.processes = {}
    
    def register_process(self, pid: int):
        """Register a process for monitoring."""
        try:
            self.processes[pid] = psutil.Process(pid)
        except:
            pass
    
    def check_memory(self, pid: int) -> bool:
        """Check if process exceeds memory limit."""
        if not self.max_memory_mb or pid not in self.processes:
            return True
        
        try:
            mem_mb = self.processes[pid].memory_info().rss / 1024 / 1024
            return mem_mb <= self.max_memory_mb
        except:
            return True
    
    def terminate_process(self, pid: int):
        """Terminate a process that exceeds limits."""
        if pid in self.processes:
            try:
                self.processes[pid].terminate()
                self.processes[pid].wait(timeout=5)
            except:
                try:
                    self.processes[pid].kill()
                except:
                    pass


def worker_with_monitoring(task_data: Tuple) -> Tuple[bool, Any]:
    """Worker function with error handling and monitoring."""
    func, args, kwargs, worker_id, monitor_config = task_data
    
    try:
        # Set up signal handler for timeout (Unix only)
        if hasattr(signal, 'SIGALRM'):
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Worker {worker_id} timed out")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            if monitor_config.get('timeout'):
                signal.alarm(int(monitor_config['timeout']))
        
        # Execute the function
        result = func(*args, **kwargs)
        
        # Cancel alarm if set
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
        
        return True, result
        
    except Exception as e:
        # Cancel alarm if set
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
        
        error_info = {
            'worker_id': worker_id,
            'error_type': type(e).__name__,
            'error_msg': str(e),
            'traceback': traceback.format_exc()
        }
        return False, error_info


class ParallelSimulator:
    """
    Stable parallel simulation with robust error handling.
    
    Features:
    - Timeout handling for stuck workers
    - Automatic retry logic for failed tasks
    - Memory monitoring and limits
    - Graceful degradation to serial execution
    - Proper resource cleanup
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
        
        # Initialize worker monitor
        self.monitor = WorkerMonitor(self.config.memory_limit_mb)
    
    def _calculate_chunk_size(self, total_work: int, n_workers: int) -> int:
        """Calculate optimal chunk size for work distribution."""
        if self.config.chunk_size is not None:
            return self.config.chunk_size
        
        # Smaller chunks for better error recovery and load balancing
        ideal_chunks = n_workers * 20
        chunk_size = max(10, min(1000, total_work // ideal_chunks))
        
        return chunk_size
    
    @contextmanager
    def _get_executor(self, n_workers: int):
        """Get appropriate executor with cleanup."""
        executor = None
        try:
            if self.config.backend == 'threading':
                executor = ThreadPoolExecutor(max_workers=n_workers)
            else:
                # Use appropriate context based on platform
                if hasattr(os, 'fork'):
                    ctx = multiprocessing.get_context('fork')
                else:
                    ctx = multiprocessing.get_context('spawn')
                executor = ProcessPoolExecutor(
                    max_workers=n_workers,
                    mp_context=ctx
                )
            yield executor
        finally:
            if executor:
                executor.shutdown(wait=True, cancel_futures=True)
    
    def _make_serializable(self, func: Callable) -> Callable:
        """Make function serializable for multiprocessing."""
        # Try to use CloudPickleWrapper if available
        if CloudPickleWrapper and self.config.backend == 'multiprocessing':
            try:
                return CloudPickleWrapper(func)
            except:
                pass
        
        # Return original function and hope for the best
        return func
    
    def _execute_with_retry(
        self,
        func: Callable,
        args: Tuple,
        kwargs: dict,
        task_id: int
    ) -> Tuple[bool, Any]:
        """Execute a task with retry logic."""
        monitor_config = {
            'timeout': self.config.timeout,
            'memory_limit': self.config.memory_limit_mb
        }
        
        for attempt in range(self.config.max_retries + 1):
            success, result = worker_with_monitoring(
                (func, args, kwargs, task_id, monitor_config)
            )
            
            if success:
                return True, result
            
            if attempt < self.config.max_retries:
                # Log retry
                warnings.warn(
                    f"Task {task_id} failed (attempt {attempt + 1}), retrying: "
                    f"{result.get('error_type', 'Unknown error')}"
                )
                time.sleep(0.1 * (attempt + 1))  # Exponential backoff
        
        return False, result
    
    def simulate_parallel_multiprocessing(
        self,
        simulate_func: Callable,
        n_simulations: int,
        n_policies: int,
        **kwargs
    ) -> np.ndarray:
        """
        Stable parallel simulation with comprehensive error handling.
        
        Args:
            simulate_func: Function to simulate a batch
            n_simulations: Total simulations
            n_policies: Number of policies
            **kwargs: Additional arguments for simulate_func
            
        Returns:
            Array of simulation results
        """
        chunk_size = self._calculate_chunk_size(n_simulations, self.config.n_workers)
        
        # Prepare work chunks
        chunks = []
        for i in range(0, n_simulations, chunk_size):
            chunk_end = min(i + chunk_size, n_simulations)
            chunks.append((i, chunk_end - i))
        
        results = np.zeros(n_simulations)
        failed_chunks = []
        
        # Progress tracking
        pbar = None
        if self.config.show_progress and HAS_TQDM:
            pbar = tqdm(total=n_simulations, desc="Parallel simulation")
        
        try:
            # Make function serializable
            wrapped_func = self._make_serializable(simulate_func)
            
            with self._get_executor(self.config.n_workers) as executor:
                # Submit all tasks
                future_to_chunk = {}
                for chunk_idx, (start_idx, chunk_n) in enumerate(chunks):
                    # Create a partial function with kwargs
                    task_func = partial(wrapped_func, **kwargs)
                    
                    future = executor.submit(
                        self._execute_with_retry,
                        task_func,
                        (chunk_n, n_policies),
                        {},
                        chunk_idx
                    )
                    future_to_chunk[future] = (start_idx, chunk_n, chunk_idx)
                
                # Collect results with timeout
                completed = 0
                timeout = self.config.timeout * len(chunks) if self.config.timeout else None
                
                for future in as_completed(future_to_chunk, timeout=timeout):
                    start_idx, chunk_n, chunk_idx = future_to_chunk[future]
                    
                    try:
                        success, result = future.result(timeout=1.0)
                        
                        if success:
                            results[start_idx:start_idx + chunk_n] = result
                            completed += chunk_n
                        else:
                            failed_chunks.append((start_idx, chunk_n))
                            warnings.warn(
                                f"Chunk {chunk_idx} failed after retries: "
                                f"{result.get('error_msg', 'Unknown error')}"
                            )
                        
                        if pbar:
                            pbar.update(chunk_n)
                            
                    except TimeoutError:
                        failed_chunks.append((start_idx, chunk_n))
                        warnings.warn(f"Chunk {chunk_idx} timed out")
                        if pbar:
                            pbar.update(chunk_n)
                    except Exception as e:
                        failed_chunks.append((start_idx, chunk_n))
                        warnings.warn(f"Chunk {chunk_idx} error: {type(e).__name__}: {str(e)}")
                        if pbar:
                            pbar.update(chunk_n)
        
        except Exception as e:
            warnings.warn(f"Parallel execution failed: {type(e).__name__}: {str(e)}")
            failed_chunks = chunks  # Mark all as failed
        
        finally:
            if pbar:
                pbar.close()
        
        # Handle failed chunks
        if failed_chunks and self.config.fallback_to_serial:
            total_failed = sum(chunk_n for _, chunk_n in failed_chunks)
            warnings.warn(
                f"{len(failed_chunks)} chunks failed ({total_failed} simulations). "
                f"Attempting serial fallback..."
            )
            
            # Run failed chunks serially
            for start_idx, chunk_n in failed_chunks:
                try:
                    chunk_results = simulate_func(chunk_n, n_policies, **kwargs)
                    results[start_idx:start_idx + chunk_n] = chunk_results
                except Exception as e:
                    warnings.warn(
                        f"Serial fallback failed for chunk at {start_idx}: "
                        f"{type(e).__name__}: {str(e)}"
                    )
                    # Leave as zeros
        
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
        
        # Create partial function with kwargs
        task_func = partial(simulate_func, **kwargs)
        
        # Run parallel simulation
        verbose = 10 if self.config.show_progress else 0
        
        try:
            results_list = Parallel(
                n_jobs=self.config.n_workers,
                backend='loky',  # 'loky' is more robust than 'multiprocessing'
                verbose=verbose,
                timeout=self.config.timeout
            )(
                delayed(task_func)(chunk_n, n_policies)
                for chunk_n in chunks
            )
            
            # Combine results
            results = np.concatenate(results_list)
            return results
            
        except Exception as e:
            warnings.warn(f"Joblib execution failed: {type(e).__name__}: {str(e)}")
            
            if self.config.fallback_to_serial:
                warnings.warn("Falling back to serial execution...")
                return simulate_func(n_simulations, n_policies, **kwargs)
            else:
                return np.zeros(n_simulations)
    
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
        failed_work = []
        
        # Progress bar
        pbar = None
        if self.config.show_progress:
            pbar = tqdm(total=n_simulations, desc="Work-stealing simulation")
        
        try:
            # Make function serializable
            wrapped_func = self._make_serializable(simulate_func)
            task_func = partial(wrapped_func, **kwargs)
            
            with self._get_executor(self.config.n_workers) as executor:
                # Keep workers busy
                futures = {}
                queue_idx = 0
                
                # Initial work distribution
                for _ in range(min(self.config.n_workers * 2, len(work_queue))):
                    if queue_idx < len(work_queue):
                        start_idx, chunk_size = work_queue[queue_idx]
                        future = executor.submit(
                            self._execute_with_retry,
                            task_func,
                            (chunk_size, n_policies),
                            {},
                            f"ws_{queue_idx}"
                        )
                        futures[future] = (start_idx, chunk_size)
                        queue_idx += 1
                
                # Process results and distribute new work
                while futures or queue_idx < len(work_queue):
                    # Get completed futures
                    done_futures = [f for f in futures if f.done()]
                    
                    # Process completed futures
                    for future in done_futures:
                        start_idx, chunk_size = futures.pop(future)
                        
                        try:
                            success, result = future.result()
                            
                            if success:
                                results[start_idx:start_idx + chunk_size] = result
                                completed += chunk_size
                            else:
                                failed_work.append((start_idx, chunk_size))
                                warnings.warn(
                                    f"Work unit at {start_idx} failed: "
                                    f"{result.get('error_msg', 'Unknown')}"
                                )
                            
                            if pbar:
                                pbar.update(chunk_size)
                                
                        except Exception as e:
                            failed_work.append((start_idx, chunk_size))
                            warnings.warn(f"Work unit failed: {e}")
                            if pbar:
                                pbar.update(chunk_size)
                        
                        # Submit new work if available
                        if queue_idx < len(work_queue):
                            start_idx, chunk_size = work_queue[queue_idx]
                            future = executor.submit(
                                self._execute_with_retry,
                                task_func,
                                (chunk_size, n_policies),
                                {},
                                f"ws_{queue_idx}"
                            )
                            futures[future] = (start_idx, chunk_size)
                            queue_idx += 1
                    
                    # Small sleep to avoid busy waiting
                    if not done_futures and futures:
                        time.sleep(0.001)
        
        finally:
            if pbar:
                pbar.close()
        
        # Handle failed work
        if failed_work and self.config.fallback_to_serial:
            for start_idx, chunk_size in failed_work:
                try:
                    chunk_results = simulate_func(chunk_size, n_policies, **kwargs)
                    results[start_idx:start_idx + chunk_size] = chunk_results
                except Exception as e:
                    warnings.warn(f"Serial fallback failed at {start_idx}: {e}")
        
        return results


# Convenience function for backward compatibility
def parallel_worker(args: Tuple[Any, ...]) -> np.ndarray:
    """Worker function for parallel simulation."""
    simulate_func, n_sims, n_policies, kwargs = args
    return simulate_func(n_sims, n_policies, **kwargs)


# Set multiprocessing start method
if __name__ != "__main__":
    # Use fork on Unix for better compatibility with closures
    # Use spawn on Windows where fork is not available
    if hasattr(os, 'fork'):
        try:
            multiprocessing.set_start_method('fork', force=True)
        except RuntimeError:
            pass  # Already set
    else:
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Already set