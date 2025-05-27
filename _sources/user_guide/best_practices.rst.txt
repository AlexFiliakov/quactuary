.. _best_practices:

=============================
Optimization Best Practices
=============================

This guide presents best practices for using quActuary's optimization features effectively,
based on real-world use cases and performance testing.

.. contents:: Table of Contents
   :local:
   :depth: 2

General Principles
==================

1. Start Simple, Optimize Incrementally
---------------------------------------

Begin with default settings and add optimizations based on profiling:

.. code-block:: python

   # Step 1: Baseline with defaults
   results = model.simulate(n_simulations=10_000)
   baseline_time = results.execution_time
   
   # Step 2: Profile to identify bottlenecks
   results = model.simulate(n_simulations=10_000, profile=True)
   results.show_profile()
   
   # Step 3: Apply targeted optimizations
   if results.numerical_operations_percent > 50:
       results = model.simulate(n_simulations=10_000, use_jit=True)
   
   # Step 4: Measure improvement
   speedup = baseline_time / results.execution_time
   print(f"Optimization speedup: {speedup:.1f}x")

2. Match Optimization to Problem Size
-------------------------------------

Use this decision matrix:

.. code-block:: python

   def select_optimizations(portfolio_size, n_simulations):
       config = {"vectorized": True}  # Always beneficial
       
       # JIT compilation threshold
       if portfolio_size > 100 or n_simulations > 50_000:
           config["use_jit"] = True
           
       # Parallel processing threshold
       if portfolio_size > 50 and n_simulations > 10_000:
           config["parallel"] = True
           
       # Memory management threshold
       memory_required_gb = (portfolio_size * n_simulations * 8) / 1e9
       if memory_required_gb > 4:
           config["memory_limit_gb"] = 4
           config["checkpoint_interval"] = n_simulations // 100
           
       # QMC for high-precision requirements
       if n_simulations > 100_000:
           config["use_qmc"] = True
           
       return config
   
   # Apply intelligent defaults
   config = select_optimizations(len(portfolio), 1_000_000)
   results = model.simulate(n_simulations=1_000_000, **config)

3. Consider Total Time, Not Just Computation
--------------------------------------------

Include setup and compilation time in benchmarks:

.. code-block:: python

   import time
   
   # Include all overhead
   total_start = time.time()
   
   # First run includes JIT compilation
   results = model.simulate(n_simulations=100_000, use_jit=True)
   
   total_time = time.time() - total_start
   computation_time = results.execution_time
   overhead_time = total_time - computation_time
   
   print(f"Total time: {total_time:.2f}s")
   print(f"Computation: {computation_time:.2f}s ({computation_time/total_time*100:.1f}%)")
   print(f"Overhead: {overhead_time:.2f}s ({overhead_time/total_time*100:.1f}%)")

JIT Compilation Best Practices
==============================

Design JIT-Friendly Functions
-----------------------------

Write functions that compile efficiently:

.. code-block:: python

   # Good: JIT-friendly function
   @numba.jit(nopython=True)
   def calculate_losses_good(frequencies, severities, limits):
       n = len(frequencies)
       losses = np.zeros(n)
       for i in range(n):
           loss = frequencies[i] * severities[i]
           losses[i] = min(loss, limits[i])
       return losses
   
   # Bad: JIT-unfriendly function
   def calculate_losses_bad(policies):
       losses = []
       for policy in policies:  # Python objects don't compile
           loss = policy.frequency * policy.severity
           losses.append(min(loss, policy.limit))
       return losses

Use Type Annotations for Better Performance
-------------------------------------------

Help the JIT compiler with type hints:

.. code-block:: python

   from numba import float64, int32
   
   @numba.jit(
       float64[:](float64[:], float64[:], float64),
       nopython=True,
       cache=True
   )
   def apply_deductible(losses, deductibles, aggregate_deductible):
       """Apply per-occurrence and aggregate deductibles."""
       n = len(losses)
       net_losses = np.maximum(losses - deductibles, 0.0)
       total_retained = np.sum(losses - net_losses)
       
       if total_retained < aggregate_deductible:
           # Apply additional aggregate deductible
           remaining = aggregate_deductible - total_retained
           for i in range(n):
               if net_losses[i] > 0:
                   reduction = min(net_losses[i], remaining)
                   net_losses[i] -= reduction
                   remaining -= reduction
                   if remaining <= 0:
                       break
                       
       return net_losses

Avoid JIT Compilation Pitfalls
------------------------------

Common issues and solutions:

.. code-block:: python

   # Pitfall 1: Python objects in JIT functions
   # Bad
   @numba.jit
   def process_policies_bad(policies):  # List of objects won't compile
       return [p.calculate() for p in policies]
   
   # Good
   @numba.jit
   def process_policies_good(values, limits):  # Use arrays instead
       return np.minimum(values, limits)
   
   # Pitfall 2: Dynamic types
   # Bad
   @numba.jit
   def dynamic_bad(x):
       if x > 0:
           return x  # returns float
       else:
           return "negative"  # returns string - type inconsistency!
   
   # Good
   @numba.jit
   def dynamic_good(x):
       if x > 0:
           return x
       else:
           return 0.0  # consistent type

Parallel Processing Best Practices
==================================

Optimize Work Distribution
--------------------------

Balance work across processes:

.. code-block:: python

   def optimize_parallel_chunks(portfolio_size, n_simulations, n_workers):
       """Calculate optimal chunk sizes for parallel processing."""
       
       # Minimum chunk size to avoid overhead
       min_chunk = max(1000, n_simulations // (n_workers * 100))
       
       # Maximum chunk size for good load balancing
       max_chunk = n_simulations // (n_workers * 4)
       
       # Optimal chunk considering cache efficiency
       optimal_chunk = int(np.sqrt(n_simulations / n_workers) * 10)
       
       return np.clip(optimal_chunk, min_chunk, max_chunk)
   
   # Use optimized chunks
   chunk_size = optimize_parallel_chunks(
       len(portfolio), 
       1_000_000, 
       os.cpu_count()
   )
   
   results = model.simulate(
       n_simulations=1_000_000,
       parallel=True,
       chunk_size=chunk_size
   )

Handle Platform Differences
---------------------------

Account for OS-specific behavior:

.. code-block:: python

   import platform
   import multiprocessing as mp
   
   def get_platform_config():
       """Platform-specific optimization settings."""
       system = platform.system()
       
       if system == "Windows":
           # Windows has higher process overhead
           return {
               "parallel_threshold": 100,  # Higher threshold
               "start_method": "spawn",
               "max_workers": min(4, mp.cpu_count())  # Limit workers
           }
       elif system == "Darwin":  # macOS
           # macOS has fork safety issues
           return {
               "parallel_threshold": 50,
               "start_method": "spawn",
               "max_workers": mp.cpu_count()
           }
       else:  # Linux
           # Linux has efficient forking
           return {
               "parallel_threshold": 25,
               "start_method": "fork",
               "max_workers": mp.cpu_count()
           }
   
   # Apply platform-specific settings
   config = get_platform_config()
   if len(portfolio) > config["parallel_threshold"]:
       results = model.simulate(
           n_simulations=100_000,
           parallel=True,
           max_workers=config["max_workers"]
       )

Memory Management Best Practices
================================

Estimate Memory Requirements
----------------------------

Calculate memory needs before running:

.. code-block:: python

   def estimate_memory_requirements(portfolio_size, n_simulations):
       """Estimate memory requirements in GB."""
       
       # Base memory per simulation result
       bytes_per_result = 8  # float64
       
       # Account for different arrays needed
       arrays_needed = {
           "frequencies": portfolio_size,
           "severities": portfolio_size, 
           "losses": portfolio_size,
           "net_losses": portfolio_size,
           "aggregated": 1
       }
       
       total_elements = sum(arrays_needed.values()) * n_simulations
       base_memory = total_elements * bytes_per_result
       
       # Add overhead (temporary arrays, Python objects)
       overhead_factor = 2.5
       total_memory_bytes = base_memory * overhead_factor
       
       return total_memory_bytes / 1e9
   
   # Check before running
   required_gb = estimate_memory_requirements(len(portfolio), 10_000_000)
   available_gb = psutil.virtual_memory().available / 1e9
   
   if required_gb > available_gb * 0.8:
       print(f"Warning: Need {required_gb:.1f}GB, have {available_gb:.1f}GB")
       print("Enabling memory optimization...")
       results = model.simulate(
           n_simulations=10_000_000,
           memory_limit_gb=available_gb * 0.7,
           checkpoint_interval=100_000
       )
   else:
       results = model.simulate(n_simulations=10_000_000)

Implement Streaming for Large Datasets
--------------------------------------

Process data in chunks to manage memory:

.. code-block:: python

   def simulate_streaming(model, n_simulations, chunk_size=100_000):
       """Run simulation in memory-efficient chunks."""
       
       all_results = []
       n_chunks = (n_simulations + chunk_size - 1) // chunk_size
       
       for i in range(n_chunks):
           start_idx = i * chunk_size
           end_idx = min((i + 1) * chunk_size, n_simulations)
           chunk_sims = end_idx - start_idx
           
           # Process chunk
           chunk_results = model.simulate(
               n_simulations=chunk_sims,
               progress_bar=False  # Avoid multiple progress bars
           )
           
           # Extract only essential statistics
           all_results.append({
               'mean': chunk_results.mean(),
               'std': chunk_results.std(),
               'percentiles': chunk_results.percentiles([0.95, 0.99])
           })
           
           # Force garbage collection between chunks
           import gc
           gc.collect()
       
       # Combine results
       return combine_chunk_results(all_results)

QMC Best Practices
==================

Choose the Right QMC Engine
---------------------------

Select based on problem characteristics:

.. code-block:: python

   def select_qmc_engine(portfolio):
       """Choose QMC engine based on portfolio characteristics."""
       
       # Calculate effective dimension
       n_random_variables = len(portfolio) * 2  # frequency + severity
       has_correlation = portfolio.correlation_matrix is not None
       
       if n_random_variables <= 10:
           # Low dimension - Halton is efficient
           return 'halton'
       elif n_random_variables <= 50 and not has_correlation:
           # Medium dimension without correlation - Sobol
           return 'sobol'
       elif has_correlation:
           # Correlation requires special handling
           return 'sobol'  # With scrambling
       else:
           # High dimension - Sobol with dimension reduction
           return 'sobol'
   
   # Apply intelligent QMC selection
   qmc_engine = select_qmc_engine(portfolio)
   results = model.simulate(
       n_simulations=50_000,
       use_qmc=True,
       qmc_engine=qmc_engine
   )

Monitor QMC Convergence
-----------------------

Track convergence to optimize simulation count:

.. code-block:: python

   def monitor_qmc_convergence(model, target_precision=0.001):
       """Run QMC with convergence monitoring."""
       
       n_base = 1000
       results_history = []
       
       for power in range(10):  # Up to 2^10 * 1000 simulations
           n_sims = n_base * (2 ** power)
           
           results = model.simulate(
               n_simulations=n_sims,
               use_qmc=True,
               random_state=42  # Consistent sequence
           )
           
           results_history.append({
               'n': n_sims,
               'mean': results.mean(),
               'std_error': results.standard_error()
           })
           
           # Check convergence
           if len(results_history) > 1:
               prev_mean = results_history[-2]['mean']
               curr_mean = results_history[-1]['mean']
               relative_change = abs(curr_mean - prev_mean) / abs(prev_mean)
               
               if relative_change < target_precision:
                   print(f"Converged at {n_sims} simulations")
                   break
       
       return results, results_history

Production Deployment
=====================

Configuration Management
------------------------

Use configuration files for production:

.. code-block:: yaml

   # config/optimization.yaml
   development:
     use_jit: false
     parallel: false
     n_simulations: 1000
     random_state: 42
   
   staging:
     use_jit: true
     parallel: true
     max_workers: 4
     n_simulations: 100_000
     memory_limit_gb: 8
   
   production:
     use_jit: true
     parallel: true
     max_workers: null  # Use all cores
     n_simulations: 1_000_000
     memory_limit_gb: null  # Auto-detect
     use_qmc: true
     checkpoint_interval: 50_000

Load and apply configuration:

.. code-block:: python

   import yaml
   import os
   
   def load_optimization_config():
       env = os.environ.get('QUACTUARY_ENV', 'development')
       
       with open('config/optimization.yaml', 'r') as f:
           config = yaml.safe_load(f)
       
       return config.get(env, config['development'])
   
   # Use environment-specific configuration
   config = load_optimization_config()
   results = model.simulate(**config)

Error Handling and Recovery
---------------------------

Implement robust error handling:

.. code-block:: python

   import logging
   from contextlib import contextmanager
   
   @contextmanager
   def safe_simulation(model, n_simulations, **kwargs):
       """Run simulation with comprehensive error handling."""
       
       logger = logging.getLogger(__name__)
       checkpoint_file = f"checkpoint_{model.id}_{n_simulations}.pkl"
       
       try:
           # Try optimal configuration
           logger.info(f"Starting simulation with {n_simulations} paths")
           yield model.simulate(n_simulations=n_simulations, **kwargs)
           
       except MemoryError:
           logger.warning("Memory error, switching to streaming mode")
           yield simulate_streaming(model, n_simulations)
           
       except mp.ProcessError:
           logger.warning("Parallel processing failed, using single process")
           kwargs['parallel'] = False
           yield model.simulate(n_simulations=n_simulations, **kwargs)
           
       except KeyboardInterrupt:
           logger.info("Simulation interrupted, saving checkpoint")
           if os.path.exists(checkpoint_file):
               logger.info(f"Checkpoint saved to {checkpoint_file}")
           raise
           
       finally:
           # Cleanup
           if os.path.exists(checkpoint_file):
               os.remove(checkpoint_file)
   
   # Use safe simulation
   with safe_simulation(model, 1_000_000, use_jit=True, parallel=True) as results:
       print(f"Simulation completed: {results.execution_time:.2f}s")

Performance Monitoring
----------------------

Track performance metrics in production:

.. code-block:: python

   class PerformanceTracker:
       def __init__(self):
           self.metrics = []
       
       def track_simulation(self, model, config):
           """Track simulation performance metrics."""
           
           start_time = time.time()
           start_memory = psutil.Process().memory_info().rss / 1e9
           
           results = model.simulate(**config)
           
           end_time = time.time()
           peak_memory = psutil.Process().memory_info().rss / 1e9
           
           metrics = {
               'timestamp': datetime.now().isoformat(),
               'portfolio_size': len(model.portfolio),
               'n_simulations': config['n_simulations'],
               'execution_time': end_time - start_time,
               'memory_used_gb': peak_memory - start_memory,
               'simulations_per_second': config['n_simulations'] / (end_time - start_time),
               'optimizations': {
                   'jit': config.get('use_jit', False),
                   'parallel': config.get('parallel', False),
                   'qmc': config.get('use_qmc', False)
               }
           }
           
           self.metrics.append(metrics)
           self.save_metrics()
           
           return results
       
       def save_metrics(self):
           """Save metrics for analysis."""
           with open('performance_metrics.json', 'w') as f:
               json.dump(self.metrics, f, indent=2)
   
   # Track production performance
   tracker = PerformanceTracker()
   results = tracker.track_simulation(model, production_config)

Next Steps
==========

* :doc:`../performance/tuning_guide` - Advanced performance tuning
* :doc:`../performance/benchmarks` - Performance benchmarks
* :doc:`../performance/tuning_guide` - Performance tuning guide
* :doc:`../performance/benchmarks` - Performance benchmarks