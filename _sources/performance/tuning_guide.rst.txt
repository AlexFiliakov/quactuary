.. _performance_tuning_guide:

============================
Advanced Performance Tuning
============================

This guide provides advanced techniques for optimizing quActuary performance beyond
the standard optimization features.

.. contents:: Table of Contents
   :local:
   :depth: 2

Hardware Optimization
=====================

CPU Optimization
----------------

**Intel-Specific Optimizations:**

.. code-block:: python

   import os
   import numpy as np
   
   # Enable Intel MKL optimizations
   os.environ['MKL_NUM_THREADS'] = str(os.cpu_count())
   os.environ['MKL_DYNAMIC'] = 'TRUE'
   
   # For Intel CPUs with AVX-512
   os.environ['NPY_NUM_BUILD_JOBS'] = str(os.cpu_count())
   
   # Configure NumPy for Intel MKL
   np.show_config()  # Verify MKL is being used

**AMD-Specific Optimizations:**

.. code-block:: python

   # AMD BLIS/OpenBLAS optimizations
   os.environ['OPENBLAS_NUM_THREADS'] = str(os.cpu_count())
   os.environ['OPENBLAS_CORETYPE'] = 'ZEN'  # For Zen architecture
   
   # AMD-specific compiler flags
   os.environ['CFLAGS'] = '-march=znver2 -mtune=znver2'

NUMA Awareness
--------------

For multi-socket systems, optimize NUMA locality:

.. code-block:: python

   import psutil
   import subprocess
   
   def get_numa_info():
       """Get NUMA topology information."""
       try:
           result = subprocess.run(['numactl', '--hardware'], 
                                 capture_output=True, text=True)
           return result.stdout
       except FileNotFoundError:
           return "NUMA tools not available"
   
   def optimize_for_numa(model, n_simulations):
       """Optimize simulation for NUMA systems."""
       numa_nodes = psutil.cpu_count(logical=False) // 2  # Rough estimate
       
       if numa_nodes > 1:
           # Bind workers to NUMA nodes
           workers_per_node = os.cpu_count() // numa_nodes
           
           results = model.simulate(
               n_simulations=n_simulations,
               parallel=True,
               max_workers=workers_per_node * numa_nodes,
               numa_policy='local'  # Custom parameter
           )
       else:
           # Standard optimization for single-socket
           results = model.simulate(n_simulations=n_simulations)
       
       return results

Memory Hierarchy Optimization
-----------------------------

**Cache-Friendly Data Structures:**

.. code-block:: python

   import numba
   import numpy as np
   
   @numba.jit(nopython=True, cache=True)
   def cache_friendly_aggregation(losses, limits, cache_line_size=64):
       """Aggregate losses with cache-friendly access patterns."""
       n = len(losses)
       # Process in chunks that fit in cache
       chunk_size = cache_line_size // 8  # 8 bytes per float64
       
       total_loss = 0.0
       for start in range(0, n, chunk_size):
           end = min(start + chunk_size, n)
           chunk_sum = 0.0
           
           # Sequential access within chunk
           for i in range(start, end):
               chunk_sum += min(losses[i], limits[i])
           
           total_loss += chunk_sum
       
       return total_loss

**Memory Prefetching:**

.. code-block:: python

   @numba.jit(nopython=True, cache=True)
   def prefetch_optimized_processing(data, indices):
       """Process data with manual prefetching."""
       n = len(indices)
       results = np.zeros(n)
       
       for i in range(n):
           # Prefetch next few elements
           if i + 4 < n:
               numba.literally(data[indices[i + 4]])  # Hint to prefetch
           
           results[i] = expensive_calculation(data[indices[i]])
       
       return results

JIT Optimization Techniques
===========================

Advanced JIT Configuration
--------------------------

**Compilation Control:**

.. code-block:: python

   from numba import config, jit
   import logging
   
   # Enable JIT debugging
   config.NUMBA_DEBUG = 1
   config.NUMBA_DEBUG_FRONTEND = 1
   
   # Configure compilation logging
   logging.getLogger('numba').setLevel(logging.DEBUG)
   
   @jit(nopython=True, cache=True, nogil=True, 
        parallel=True, fastmath=True)
   def optimized_simulation_kernel(frequencies, severities, limits):
       """Highly optimized simulation kernel."""
       n = len(frequencies)
       losses = np.zeros(n)
       
       # Use parallel loop
       for i in numba.prange(n):
           total_loss = 0.0
           n_claims = frequencies[i]
           
           for claim in range(int(n_claims)):
               claim_amount = severities[i * 1000 + claim]  # Pre-generated
               total_loss += min(claim_amount, limits[i])
           
           losses[i] = total_loss
       
       return losses

**Type Specialization:**

.. code-block:: python

   from numba import types, typed
   
   # Create specialized signatures for common types
   float64_array = types.float64[:]
   int64_array = types.int64[:]
   
   @jit([
       float64_array(float64_array, float64_array, float64_array),
       float64_array(types.float32[:], types.float32[:], types.float32[:])
   ], cache=True)
   def type_specialized_function(freq, sev, limits):
       """Function with multiple type specializations."""
       return compute_losses(freq, sev, limits)

**Custom Compilation Pipeline:**

.. code-block:: python

   from numba.core import types
   from numba.core.extending import overload
   from numba.core.imputils import lower_builtin
   
   @overload(np.clip)
   def custom_clip_implementation(a, a_min, a_max):
       """Custom implementation of np.clip for better performance."""
       def clip_impl(a, a_min, a_max):
           return max(a_min, min(a, a_max))
       return clip_impl

Parallel Processing Optimization
================================

Advanced Worker Management
--------------------------

**Dynamic Worker Scaling:**

.. code-block:: python

   import time
   import multiprocessing as mp
   from concurrent.futures import ProcessPoolExecutor
   
   class AdaptiveWorkerPool:
       def __init__(self, min_workers=2, max_workers=None):
           self.min_workers = min_workers
           self.max_workers = max_workers or mp.cpu_count()
           self.current_workers = min_workers
           self.performance_history = []
       
       def optimize_worker_count(self, task_duration, throughput):
           """Dynamically adjust worker count based on performance."""
           self.performance_history.append({
               'workers': self.current_workers,
               'duration': task_duration,
               'throughput': throughput
           })
           
           if len(self.performance_history) >= 3:
               # Analyze trend
               recent = self.performance_history[-3:]
               if all(r['throughput'] < recent[0]['throughput'] for r in recent[1:]):
                   # Decreasing performance, reduce workers
                   self.current_workers = max(self.min_workers, 
                                            self.current_workers - 1)
               elif recent[-1]['throughput'] > recent[0]['throughput'] * 1.1:
                   # Increasing performance, add workers
                   self.current_workers = min(self.max_workers,
                                            self.current_workers + 1)
           
           return self.current_workers

**Custom Process Executor:**

.. code-block:: python

   from multiprocessing import Process, Queue, shared_memory
   import numpy as np
   
   class SharedMemoryExecutor:
       """Process executor using shared memory for large arrays."""
       
       def __init__(self, max_workers=None):
           self.max_workers = max_workers or mp.cpu_count()
           self.workers = []
           self.shared_arrays = {}
       
       def create_shared_array(self, name, shape, dtype=np.float64):
           """Create shared memory array."""
           size = np.prod(shape) * np.dtype(dtype).itemsize
           shm = shared_memory.SharedMemory(create=True, size=size, name=name)
           
           array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
           self.shared_arrays[name] = (shm, array)
           return array
       
       def worker_function(self, task_queue, result_queue, shared_info):
           """Worker function using shared memory."""
           while True:
               task = task_queue.get()
               if task is None:
                   break
               
               # Access shared arrays
               arrays = {}
               for name, (shape, dtype) in shared_info.items():
                   shm = shared_memory.SharedMemory(name=name)
                   arrays[name] = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
               
               # Process task
               result = process_chunk(task, arrays)
               result_queue.put(result)

Communication Optimization
--------------------------

**Efficient Data Serialization:**

.. code-block:: python

   import pickle
   import lz4.frame
   import numpy as np
   
   class OptimizedSerializer:
       """Optimized serialization for numerical data."""
       
       @staticmethod
       def serialize_array(array):
           """Compress and serialize numpy array."""
           # Use efficient binary format
           data = array.tobytes()
           # Compress for transmission
           compressed = lz4.frame.compress(data)
           metadata = {
               'shape': array.shape,
               'dtype': str(array.dtype),
               'compressed_size': len(compressed),
               'original_size': len(data)
           }
           return metadata, compressed
       
       @staticmethod
       def deserialize_array(metadata, compressed_data):
           """Decompress and deserialize numpy array."""
           data = lz4.frame.decompress(compressed_data)
           array = np.frombuffer(data, dtype=metadata['dtype'])
           return array.reshape(metadata['shape'])

Memory Optimization Strategies
==============================

Advanced Memory Management
--------------------------

**Memory Pools:**

.. code-block:: python

   import gc
   from collections import deque
   
   class ArrayMemoryPool:
       """Memory pool for reusing numpy arrays."""
       
       def __init__(self, max_size=100):
           self.pools = {}  # {(shape, dtype): deque of arrays}
           self.max_size = max_size
       
       def get_array(self, shape, dtype=np.float64):
           """Get array from pool or create new one."""
           key = (tuple(shape), dtype)
           
           if key in self.pools and self.pools[key]:
               return self.pools[key].popleft()
           else:
               return np.zeros(shape, dtype=dtype)
       
       def return_array(self, array):
           """Return array to pool."""
           key = (tuple(array.shape), array.dtype)
           
           if key not in self.pools:
               self.pools[key] = deque()
           
           if len(self.pools[key]) < self.max_size:
               # Clear array and return to pool
               array.fill(0)
               self.pools[key].append(array)
       
       def clear(self):
           """Clear all pools and force garbage collection."""
           self.pools.clear()
           gc.collect()
   
   # Global memory pool
   memory_pool = ArrayMemoryPool()
   
   def efficient_simulation(model, n_simulations):
       """Simulation using memory pool."""
       try:
           # Get arrays from pool
           results = memory_pool.get_array((n_simulations,))
           temp_array = memory_pool.get_array((n_simulations, 10))
           
           # Run simulation
           simulation_results = model.simulate_with_arrays(
               results, temp_array, n_simulations
           )
           
           return simulation_results.copy()  # Return copy
           
       finally:
           # Return arrays to pool
           memory_pool.return_array(results)
           memory_pool.return_array(temp_array)

**Memory Mapping for Large Datasets:**

.. code-block:: python

   import mmap
   import os
   
   class MemoryMappedSimulation:
       """Use memory mapping for very large simulations."""
       
       def __init__(self, simulation_file, mode='w+'):
           self.simulation_file = simulation_file
           self.mode = mode
           self.mmap_obj = None
           self.array = None
       
       def create_simulation_space(self, n_simulations, portfolio_size):
           """Create memory-mapped space for simulation."""
           # Calculate required space
           bytes_needed = n_simulations * portfolio_size * 8  # float64
           
           # Create file
           with open(self.simulation_file, 'wb') as f:
               f.write(b'\x00' * bytes_needed)
           
           # Open memory map
           with open(self.simulation_file, 'r+b') as f:
               self.mmap_obj = mmap.mmap(f.fileno(), 0)
               self.array = np.frombuffer(self.mmap_obj, dtype=np.float64)
               self.array = self.array.reshape((n_simulations, portfolio_size))
       
       def simulate_chunk(self, start_idx, end_idx):
           """Simulate chunk directly in memory-mapped space."""
           chunk = self.array[start_idx:end_idx]
           # Perform simulation directly on memory-mapped array
           return self.process_simulation_chunk(chunk)
       
       def cleanup(self):
           """Cleanup memory map and file."""
           if self.mmap_obj:
               self.mmap_obj.close()
           if os.path.exists(self.simulation_file):
               os.remove(self.simulation_file)

Algorithm-Level Optimizations
=============================

Distribution-Specific Optimizations
-----------------------------------

**Optimized Gamma Sampling:**

.. code-block:: python

   @numba.jit(nopython=True, cache=True)
   def fast_gamma_sampler(shape, scale, size):
       """Optimized gamma distribution sampling."""
       if shape >= 1.0:
           # Use Marsaglia-Tsang method for shape >= 1
           return marsaglia_tsang_gamma(shape, scale, size)
       else:
           # Use Ahrens-Dieter method for shape < 1
           return ahrens_dieter_gamma(shape, scale, size)
   
   @numba.jit(nopython=True, cache=True)
   def marsaglia_tsang_gamma(shape, scale, size):
       """Marsaglia-Tsang gamma sampler."""
       results = np.zeros(size)
       d = shape - 1.0/3.0
       c = 1.0 / np.sqrt(9.0 * d)
       
       for i in range(size):
           while True:
               x = np.random.normal()
               v = (1.0 + c * x) ** 3
               
               if v > 0:
                   u = np.random.random()
                   if u < 1.0 - 0.0331 * (x * x) ** 2:
                       results[i] = d * v * scale
                       break
                   if np.log(u) < 0.5 * x * x + d * (1.0 - v + np.log(v)):
                       results[i] = d * v * scale
                       break
       
       return results

**Compound Distribution Fast Paths:**

.. code-block:: python

   @numba.jit(nopython=True, cache=True)
   def fast_compound_poisson_gamma(lambda_param, alpha, beta, n_simulations):
       """Fast path for Poisson-Gamma compound distribution."""
       results = np.zeros(n_simulations)
       
       for i in range(n_simulations):
           # Sample frequency
           n_claims = np.random.poisson(lambda_param)
           
           if n_claims == 0:
               results[i] = 0.0
           else:
               # Use gamma sum property
               total_alpha = alpha * n_claims
               results[i] = np.random.gamma(total_alpha, 1.0/beta)
       
       return results

**Vectorized Risk Measures:**

.. code-block:: python

   @numba.jit(nopython=True, parallel=True, cache=True)
   def fast_var_tvar(losses, confidence_levels):
       """Vectorized VaR and TVaR calculation."""
       n_sims = len(losses)
       n_levels = len(confidence_levels)
       
       # Sort once
       sorted_losses = np.sort(losses)
       
       vars = np.zeros(n_levels)
       tvars = np.zeros(n_levels)
       
       for i in numba.prange(n_levels):
           alpha = confidence_levels[i]
           var_index = int(alpha * n_sims)
           
           if var_index >= n_sims:
               var_index = n_sims - 1
           
           # VaR
           vars[i] = sorted_losses[var_index]
           
           # TVaR (tail average)
           if var_index < n_sims - 1:
               tail_sum = 0.0
               tail_count = 0
               for j in range(var_index, n_sims):
                   tail_sum += sorted_losses[j]
                   tail_count += 1
               tvars[i] = tail_sum / tail_count
           else:
               tvars[i] = vars[i]
       
       return vars, tvars

Monitoring and Profiling
========================

Performance Profiling Tools
---------------------------

**Custom Profiler Integration:**

.. code-block:: python

   import cProfile
   import pstats
   import line_profiler
   import memory_profiler
   
   class ComprehensiveProfiler:
       """Comprehensive profiling for quActuary simulations."""
       
       def __init__(self):
           self.cpu_profiler = cProfile.Profile()
           self.memory_tracker = []
           self.line_profiler = line_profiler.LineProfiler()
       
       def profile_simulation(self, model, n_simulations, **kwargs):
           """Profile simulation with multiple tools."""
           
           # CPU profiling
           self.cpu_profiler.enable()
           
           # Memory profiling
           @memory_profiler.profile
           def run_simulation():
               return model.simulate(n_simulations=n_simulations, **kwargs)
           
           # Line profiling for critical functions
           self.line_profiler.add_function(model.simulate)
           self.line_profiler.enable()
           
           try:
               results = run_simulation()
           finally:
               self.cpu_profiler.disable()
               self.line_profiler.disable()
           
           return results, self.generate_report()
       
       def generate_report(self):
           """Generate comprehensive performance report."""
           report = {}
           
           # CPU profiling report
           cpu_stats = pstats.Stats(self.cpu_profiler)
           cpu_stats.sort_stats('cumulative')
           report['cpu'] = cpu_stats
           
           # Line profiling report
           self.line_profiler.print_stats()
           
           return report

**Real-time Performance Monitoring:**

.. code-block:: python

   import threading
   import time
   import psutil
   
   class PerformanceMonitor:
       """Real-time performance monitoring."""
       
       def __init__(self, sample_interval=1.0):
           self.sample_interval = sample_interval
           self.monitoring = False
           self.metrics = []
           self.monitor_thread = None
       
       def start_monitoring(self):
           """Start performance monitoring."""
           self.monitoring = True
           self.monitor_thread = threading.Thread(target=self._monitor_loop)
           self.monitor_thread.start()
       
       def stop_monitoring(self):
           """Stop performance monitoring."""
           self.monitoring = False
           if self.monitor_thread:
               self.monitor_thread.join()
       
       def _monitor_loop(self):
           """Monitoring loop."""
           process = psutil.Process()
           
           while self.monitoring:
               timestamp = time.time()
               cpu_percent = process.cpu_percent()
               memory_info = process.memory_info()
               
               self.metrics.append({
                   'timestamp': timestamp,
                   'cpu_percent': cpu_percent,
                   'memory_rss': memory_info.rss / 1e6,  # MB
                   'memory_vms': memory_info.vms / 1e6,  # MB
               })
               
               time.sleep(self.sample_interval)
       
       def get_summary(self):
           """Get performance summary."""
           if not self.metrics:
               return {}
           
           cpu_values = [m['cpu_percent'] for m in self.metrics]
           memory_values = [m['memory_rss'] for m in self.metrics]
           
           return {
               'duration': self.metrics[-1]['timestamp'] - self.metrics[0]['timestamp'],
               'avg_cpu': np.mean(cpu_values),
               'max_cpu': np.max(cpu_values),
               'avg_memory_mb': np.mean(memory_values),
               'peak_memory_mb': np.max(memory_values)
           }

Configuration Optimization
==========================

Adaptive Configuration
----------------------

**Machine Learning for Configuration:**

.. code-block:: python

   import pickle
   from sklearn.ensemble import RandomForestRegressor
   import numpy as np
   
   class ConfigurationOptimizer:
       """ML-based configuration optimizer."""
       
       def __init__(self):
           self.model = RandomForestRegressor(n_estimators=100)
           self.feature_history = []
           self.performance_history = []
           self.is_trained = False
       
       def extract_features(self, portfolio, n_simulations):
           """Extract features for configuration prediction."""
           return np.array([
               len(portfolio),
               n_simulations,
               portfolio.complexity_score(),
               portfolio.correlation_factor(),
               psutil.cpu_count(),
               psutil.virtual_memory().available / 1e9,
               self.get_system_load()
           ])
       
       def record_performance(self, features, config, execution_time):
           """Record performance for training."""
           # Encode configuration
           config_vector = [
               float(config.get('use_jit', False)),
               float(config.get('parallel', False)),
               config.get('max_workers', 0),
               config.get('memory_limit_gb', 0) or 0,
               float(config.get('use_qmc', False))
           ]
           
           combined_features = np.concatenate([features, config_vector])
           
           self.feature_history.append(combined_features)
           self.performance_history.append(execution_time)
       
       def train_model(self):
           """Train configuration optimization model."""
           if len(self.feature_history) >= 10:
               X = np.array(self.feature_history)
               y = np.array(self.performance_history)
               
               self.model.fit(X, y)
               self.is_trained = True
       
       def predict_optimal_config(self, portfolio, n_simulations):
           """Predict optimal configuration."""
           if not self.is_trained:
               return self.get_default_config(portfolio, n_simulations)
           
           base_features = self.extract_features(portfolio, n_simulations)
           best_config = None
           best_time = float('inf')
           
           # Test different configurations
           for config in self.generate_config_candidates():
               config_vector = self.encode_config(config)
               features = np.concatenate([base_features, config_vector]).reshape(1, -1)
               predicted_time = self.model.predict(features)[0]
               
               if predicted_time < best_time:
                   best_time = predicted_time
                   best_config = config
           
           return best_config

Environment-Specific Tuning
---------------------------

**Cloud Platform Optimization:**

.. code-block:: python

   import boto3
   import platform
   
   class CloudOptimizer:
       """Cloud platform-specific optimizations."""
       
       def __init__(self):
           self.platform = self.detect_platform()
           self.instance_type = self.get_instance_type()
       
       def detect_platform(self):
           """Detect cloud platform."""
           try:
               # AWS detection
               response = requests.get(
                   'http://169.254.169.254/latest/meta-data/instance-type',
                   timeout=1
               )
               if response.status_code == 200:
                   return 'aws'
           except:
               pass
           
           # Add other cloud detection logic
           return 'unknown'
       
       def get_optimized_config(self, portfolio_size, n_simulations):
           """Get cloud-optimized configuration."""
           if self.platform == 'aws':
               return self.aws_optimization(portfolio_size, n_simulations)
           else:
               return self.default_optimization(portfolio_size, n_simulations)
       
       def aws_optimization(self, portfolio_size, n_simulations):
           """AWS-specific optimization."""
           config = {}
           
           if self.instance_type.startswith('c5'):
               # Compute-optimized instances
               config.update({
                   'use_jit': True,
                   'parallel': True,
                   'max_workers': psutil.cpu_count(),
                   'numa_policy': 'local'
               })
           elif self.instance_type.startswith('r5'):
               # Memory-optimized instances
               config.update({
                   'memory_limit_gb': psutil.virtual_memory().total / 1e9 * 0.9,
                   'checkpoint_interval': n_simulations // 50
               })
           
           return config

Next Steps
==========

* :doc:`benchmarks` - Performance benchmarks
* :doc:`../user_guide/best_practices` - General best practices
* :doc:`../user_guide/index` - User guide with examples
* `Performance Repository <https://github.com/quactuary/performance-optimization>`_ - Advanced optimization examples