.. _configuration_guide:

================================
Optimization Configuration Guide
================================

This guide provides detailed information about all optimization parameters available in quActuary's
``PricingModel.simulate()`` method.

.. contents:: Table of Contents
   :local:
   :depth: 3

Overview
========

The ``simulate()`` method accepts numerous parameters to control optimization behavior:

.. code-block:: python

   results = model.simulate(
       n_simulations: int = 10000,
       use_jit: bool = None,
       parallel: bool = None,
       max_workers: int = None,
       vectorized: bool = True,
       memory_limit_gb: float = None,
       use_qmc: bool = False,
       qmc_engine: str = 'sobol',
       progress_bar: bool = True,
       checkpoint_interval: int = None,
       random_state: int = None
   )

Core Parameters
===============

n_simulations
-------------

Number of Monte Carlo simulation paths to generate.

:Type: ``int``
:Default: ``10000``
:Range: ``1`` to ``10^9`` (system memory permitting)

.. code-block:: python

   # Standard simulation
   results = model.simulate(n_simulations=100_000)
   
   # Large-scale simulation
   results = model.simulate(
       n_simulations=10_000_000,
       memory_limit_gb=8  # Manage memory for large runs
   )

.. note::
   Larger simulation counts provide better convergence but require more time and memory.
   Use the rule of thumb: double simulations for half the standard error.

random_state
------------

Seed for random number generation to ensure reproducibility.

:Type: ``int`` or ``None``
:Default: ``None`` (random seed)

.. code-block:: python

   # Reproducible results
   results1 = model.simulate(n_simulations=10_000, random_state=42)
   results2 = model.simulate(n_simulations=10_000, random_state=42)
   # results1 and results2 will be identical
   
   # Different runs
   results3 = model.simulate(n_simulations=10_000, random_state=123)
   # results3 will differ from results1/results2

JIT Compilation Parameters
==========================

use_jit
-------

Enable Just-In-Time compilation using Numba for numerical operations.

:Type: ``bool`` or ``None``
:Default: ``None`` (auto-detect based on portfolio size)
:Auto-detection: Enabled when portfolio size > 100

.. code-block:: python

   # Automatic detection (recommended)
   results = model.simulate(n_simulations=100_000)
   
   # Force JIT on
   results = model.simulate(n_simulations=100_000, use_jit=True)
   
   # Force JIT off (e.g., for debugging)
   results = model.simulate(n_simulations=100_000, use_jit=False)

**Performance characteristics:**

* First run includes compilation time (typically 0.5-2 seconds)
* Subsequent runs use cached compiled code
* Speedup: 10-50x for numerical operations
* Memory overhead: Minimal

**Best practices:**

.. code-block:: python

   # Warm-up for production use
   if first_run:
       model.simulate(n_simulations=100, use_jit=True)  # Compile
   
   # Production runs benefit from compiled code
   results = model.simulate(n_simulations=1_000_000, use_jit=True)

Parallel Processing Parameters
==============================

parallel
--------

Enable parallel processing across multiple CPU cores.

:Type: ``bool`` or ``None``
:Default: ``None`` (auto-detect based on portfolio size)
:Auto-detection: Enabled when portfolio size > 50 and CPU count > 1

.. code-block:: python

   # Automatic detection
   results = model.simulate(n_simulations=100_000)
   
   # Force parallel processing
   results = model.simulate(n_simulations=100_000, parallel=True)
   
   # Force single process (debugging, memory constraints)
   results = model.simulate(n_simulations=100_000, parallel=False)

max_workers
-----------

Maximum number of worker processes for parallel execution.

:Type: ``int`` or ``None``
:Default: ``None`` (uses CPU count)
:Range: ``1`` to ``os.cpu_count()``

.. code-block:: python

   import os
   
   # Use all available cores (default)
   results = model.simulate(n_simulations=1_000_000, parallel=True)
   
   # Use half the cores (leave resources for other tasks)
   results = model.simulate(
       n_simulations=1_000_000,
       parallel=True,
       max_workers=os.cpu_count() // 2
   )
   
   # Fixed worker count
   results = model.simulate(
       n_simulations=1_000_000,
       parallel=True,
       max_workers=4
   )

**Optimal worker calculation:**

.. code-block:: python

   def optimal_workers(portfolio_size, available_memory_gb):
       cpu_based = os.cpu_count()
       portfolio_based = max(1, portfolio_size // 50)
       memory_based = int(available_memory_gb // 2)
       
       return min(cpu_based, portfolio_based, memory_based)

Memory Management Parameters
============================

memory_limit_gb
---------------

Maximum memory usage limit in gigabytes.

:Type: ``float`` or ``None``
:Default: ``None`` (auto-detect available memory)
:Range: ``0.1`` to system memory

.. code-block:: python

   # Auto-detect available memory
   results = model.simulate(n_simulations=10_000_000)
   
   # Strict memory limit
   results = model.simulate(
       n_simulations=10_000_000,
       memory_limit_gb=4.0  # Limit to 4GB
   )
   
   # Conservative limit for shared systems
   import psutil
   available_gb = psutil.virtual_memory().available / 1e9
   results = model.simulate(
       n_simulations=10_000_000,
       memory_limit_gb=available_gb * 0.5  # Use 50% of available
   )

**Memory estimation:**

.. code-block:: python

   def estimate_memory_gb(n_simulations, portfolio_size):
       # Rough estimation formula
       bytes_per_simulation = 8 * portfolio_size  # 8 bytes per float
       total_bytes = n_simulations * bytes_per_simulation * 3  # 3x for overhead
       return total_bytes / 1e9

checkpoint_interval
-------------------

Save progress at regular intervals for recovery from interruptions.

:Type: ``int`` or ``None``
:Default: ``None`` (no checkpointing)
:Recommended: ``n_simulations // 100`` for long runs

.. code-block:: python

   # No checkpointing (default for small runs)
   results = model.simulate(n_simulations=10_000)
   
   # Regular checkpoints for large runs
   results = model.simulate(
       n_simulations=10_000_000,
       checkpoint_interval=100_000  # Save every 100k simulations
   )
   
   # Recover from checkpoint after interruption
   try:
       results = model.simulate(
           n_simulations=10_000_000,
           checkpoint_interval=100_000,
           resume_from_checkpoint=True  # Continue from last checkpoint
       )
   except KeyboardInterrupt:
       print("Simulation interrupted, progress saved")

Vectorization Parameters
========================

vectorized
----------

Enable NumPy vectorization for array operations.

:Type: ``bool``
:Default: ``True``

.. code-block:: python

   # Vectorized operations (default, recommended)
   results = model.simulate(n_simulations=100_000, vectorized=True)
   
   # Disable for debugging or special cases
   results = model.simulate(n_simulations=100_000, vectorized=False)

.. warning::
   Disabling vectorization significantly impacts performance. 
   Only disable for debugging or when required by custom distributions.

Quasi-Monte Carlo Parameters
============================

use_qmc
-------

Enable Quasi-Monte Carlo using low-discrepancy sequences.

:Type: ``bool``
:Default: ``False``

.. code-block:: python

   # Standard Monte Carlo (default)
   results_mc = model.simulate(n_simulations=10_000, use_qmc=False)
   
   # Quasi-Monte Carlo for better convergence
   results_qmc = model.simulate(n_simulations=10_000, use_qmc=True)
   
   # Compare convergence
   print(f"MC std error: {results_mc.standard_error:.4f}")
   print(f"QMC std error: {results_qmc.standard_error:.4f}")

qmc_engine
----------

Type of low-discrepancy sequence to use.

:Type: ``str``
:Default: ``'sobol'``
:Options: ``'sobol'``, ``'halton'``, ``'latin_hypercube'``

.. code-block:: python

   # Sobol sequence (recommended for most cases)
   results = model.simulate(
       n_simulations=10_000,
       use_qmc=True,
       qmc_engine='sobol'
   )
   
   # Halton sequence (better for low dimensions)
   results = model.simulate(
       n_simulations=10_000,
       use_qmc=True,
       qmc_engine='halton'
   )
   
   # Latin Hypercube (good for sensitivity analysis)
   results = model.simulate(
       n_simulations=10_000,
       use_qmc=True,
       qmc_engine='latin_hypercube'
   )

**QMC engine selection guide:**

.. list-table:: QMC Engine Comparison
   :header-rows: 1
   :widths: 20 20 20 40

   * - Engine
     - Best Dimension Range
     - Convergence Rate
     - Use Case
   * - Sobol
     - 1-1000
     - O(log(N)^d/N)
     - General purpose, high dimensions
   * - Halton
     - 1-20
     - O(log(N)^d/N)
     - Low dimensions, simple problems
   * - Latin Hypercube
     - 1-100
     - O(1/N)
     - Design of experiments, sensitivity

User Interface Parameters
=========================

progress_bar
------------

Display progress bar during simulation.

:Type: ``bool``
:Default: ``True``

.. code-block:: python

   # With progress bar (default)
   results = model.simulate(n_simulations=1_000_000, progress_bar=True)
   # Output: [████████████████████] 100% | 1000000/1000000 | ETA: 00:00
   
   # Disable for non-interactive environments
   results = model.simulate(n_simulations=1_000_000, progress_bar=False)
   
   # Custom progress callback
   def custom_progress(current, total):
       percent = 100 * current / total
       print(f"Progress: {percent:.1f}% ({current}/{total})")
   
   results = model.simulate(
       n_simulations=1_000_000,
       progress_callback=custom_progress
   )

Configuration Presets
=====================

quActuary provides configuration presets for common scenarios:

.. code-block:: python

   from quactuary.optimization import OptimizationPresets
   
   # Memory-constrained environment
   config = OptimizationPresets.MEMORY_CONSTRAINED
   results = model.simulate(n_simulations=10_000_000, **config)
   
   # Maximum performance
   config = OptimizationPresets.MAX_PERFORMANCE
   results = model.simulate(n_simulations=1_000_000, **config)
   
   # Balanced (default)
   config = OptimizationPresets.BALANCED
   results = model.simulate(n_simulations=100_000, **config)
   
   # Development/debugging
   config = OptimizationPresets.DEBUG
   results = model.simulate(n_simulations=1_000, **config)

**Preset definitions:**

.. code-block:: python

   class OptimizationPresets:
       MEMORY_CONSTRAINED = {
           "use_jit": False,
           "parallel": False,
           "memory_limit_gb": 2,
           "checkpoint_interval": 10_000
       }
       
       MAX_PERFORMANCE = {
           "use_jit": True,
           "parallel": True,
           "max_workers": None,
           "use_qmc": True,
           "vectorized": True
       }
       
       BALANCED = {
           "use_jit": None,
           "parallel": None,
           "memory_limit_gb": None
       }
       
       DEBUG = {
           "use_jit": False,
           "parallel": False,
           "progress_bar": True,
           "random_state": 42
       }

Environment Variables
=====================

Configuration via environment variables for deployment:

.. code-block:: bash

   # Disable JIT globally
   export QUACTUARY_DISABLE_JIT=1
   
   # Set default worker count
   export QUACTUARY_MAX_WORKERS=4
   
   # Set memory limit
   export QUACTUARY_MEMORY_LIMIT_GB=8
   
   # Disable progress bars
   export QUACTUARY_NO_PROGRESS=1

Access in Python:

.. code-block:: python

   import os
   
   # Override defaults with environment variables
   config = {
       "use_jit": not os.environ.get("QUACTUARY_DISABLE_JIT"),
       "max_workers": int(os.environ.get("QUACTUARY_MAX_WORKERS", 0)) or None,
       "memory_limit_gb": float(os.environ.get("QUACTUARY_MEMORY_LIMIT_GB", 0)) or None,
       "progress_bar": not os.environ.get("QUACTUARY_NO_PROGRESS")
   }
   
   results = model.simulate(n_simulations=100_000, **config)

Advanced Configuration
======================

Custom Optimization Strategy
----------------------------

Implement custom optimization logic:

.. code-block:: python

   class CustomOptimizer:
       def __init__(self, portfolio):
           self.portfolio = portfolio
           
       def should_use_jit(self):
           # Custom heuristic based on portfolio characteristics
           if self.portfolio.size < 50:
               return False
           if self.portfolio.has_complex_dependencies:
               return True
           return self.portfolio.size > 100
           
       def optimal_workers(self):
           # Dynamic worker allocation
           if self.portfolio.is_correlated:
               return max(2, os.cpu_count() // 2)
           return os.cpu_count()
           
       def memory_limit(self):
           # Adaptive memory limit
           base_memory = 0.1 * self.portfolio.size / 1000  # GB per 1k policies
           return min(base_memory * 2, 16)  # Cap at 16GB
   
   # Use custom optimizer
   optimizer = CustomOptimizer(portfolio)
   results = model.simulate(
       n_simulations=100_000,
       use_jit=optimizer.should_use_jit(),
       max_workers=optimizer.optimal_workers(),
       memory_limit_gb=optimizer.memory_limit()
   )

Performance Monitoring Integration
----------------------------------

.. code-block:: python

   from quactuary.monitoring import PerformanceMonitor
   
   # Wrap simulation with monitoring
   with PerformanceMonitor() as monitor:
       results = model.simulate(
           n_simulations=1_000_000,
           use_jit=True,
           parallel=True
       )
   
   # Access detailed metrics
   monitor.plot_timeline()
   monitor.show_resource_usage()
   monitor.export_metrics("simulation_metrics.json")

Next Steps
==========

* :doc:`best_practices` - Optimization best practices
* :doc:`../performance/tuning_guide` - Performance tuning
* :doc:`../api_reference/pricing_model` - Full API reference
* :doc:`../performance/benchmarks` - Performance benchmarks