"""
Utilities for parallel processing with better serialization support.
"""

import numpy as np
from typing import Dict, Any, Callable, Tuple
import cloudpickle
import pickle
import base64


# Global registry for simulation functions
_SIMULATION_REGISTRY: Dict[str, Callable] = {}


def register_simulation_function(name: str, func: Callable) -> None:
    """Register a simulation function for parallel execution."""
    _SIMULATION_REGISTRY[name] = func


def get_simulation_function(name: str) -> Callable:
    """Get a registered simulation function."""
    if name not in _SIMULATION_REGISTRY:
        raise KeyError(f"Simulation function '{name}' not registered")
    return _SIMULATION_REGISTRY[name]


def cloudpickle_wrapper(func_data: bytes, args: Tuple, kwargs: dict) -> Any:
    """Execute a cloudpickled function."""
    func = cloudpickle.loads(func_data)
    return func(*args, **kwargs)


class CloudPickleWrapper:
    """Wrapper that uses cloudpickle for better serialization support."""
    
    def __init__(self, func: Callable):
        self.func_data = cloudpickle.dumps(func)
    
    def __call__(self, *args, **kwargs):
        return cloudpickle_wrapper(self.func_data, args, kwargs)


def create_shared_memory_array(shape: Tuple[int, ...], dtype=np.float64) -> np.ndarray:
    """Create a shared memory array for multiprocessing."""
    import multiprocessing as mp
    from multiprocessing import shared_memory
    
    # Calculate size
    size = int(np.prod(shape)) * np.dtype(dtype).itemsize
    
    # Create shared memory
    shm = shared_memory.SharedMemory(create=True, size=size)
    
    # Create numpy array backed by shared memory
    array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    array[:] = 0  # Initialize
    
    return array, shm


def parallel_map_with_shared_memory(
    func: Callable,
    tasks: list,
    n_workers: int = None,
    shared_arrays: dict = None
) -> list:
    """
    Parallel map with shared memory support.
    
    Args:
        func: Function to execute
        tasks: List of task arguments
        n_workers: Number of workers
        shared_arrays: Dict of shared memory arrays
        
    Returns:
        List of results
    """
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    
    if n_workers is None:
        n_workers = mp.cpu_count()
    
    # Wrap function to handle shared memory
    def worker_func(task_args):
        # Reconstruct shared arrays if needed
        if shared_arrays:
            # Pass shared array info to function
            return func(task_args, shared_arrays)
        else:
            return func(task_args)
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(worker_func, tasks))
    
    return results


# Pre-register common simulation functions
def _monte_carlo_simulation(n_sims: int, n_policies: int, **params) -> np.ndarray:
    """Standard Monte Carlo simulation."""
    # Get distribution parameters
    mu = params.get('mu', 1.0)
    sigma = params.get('sigma', 0.5)
    
    # Generate random values
    return np.random.normal(mu, sigma, size=n_sims)


def _vectorized_simulation(n_sims: int, n_policies: int, **params) -> np.ndarray:
    """Vectorized simulation for compound distributions."""
    # Frequency parameters
    freq_mu = params.get('freq_mu', 1.5)
    
    # Severity parameters  
    sev_shape = params.get('sev_shape', 1.0)
    sev_scale = params.get('sev_scale', np.exp(8.0))
    
    # Generate frequencies
    frequencies = np.random.poisson(freq_mu, size=n_sims)
    
    # Generate severities
    results = np.zeros(n_sims)
    for i in range(n_sims):
        if frequencies[i] > 0:
            severities = np.random.lognormal(
                mean=np.log(sev_scale),
                sigma=sev_shape,
                size=frequencies[i]
            )
            results[i] = np.sum(severities)
    
    return results


# Register default functions
register_simulation_function('monte_carlo', _monte_carlo_simulation)
register_simulation_function('vectorized', _vectorized_simulation)