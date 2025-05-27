Optimization Configuration Guide
================================

This guide explains how to use ``OptimizationConfig`` to control performance optimizations in quActuary.

.. versionadded:: 0.2.0
   The unified ``OptimizationConfig`` replaced individual optimization parameters.

Overview
--------

quActuary provides automatic and manual optimization configuration through the ``OptimizationConfig`` class. The system can automatically select optimal settings based on your data size and system resources, or you can manually configure specific optimizations.

Quick Start
-----------

Using Auto-Optimization
~~~~~~~~~~~~~~~~~~~~~~~

The simplest way to get optimal performance is to use ``auto_optimize=True``:

.. code-block:: python

   from quactuary.pricing import PricingModel
   
   model = PricingModel(portfolio)
   result = model.simulate(
       auto_optimize=True,
       n_sims=100000
   )

The system will automatically:

* Analyze your portfolio size and complexity
* Check available system resources (memory, CPU cores)
* Select the best combination of optimizations
* Apply fallback strategies if needed

Manual Configuration
~~~~~~~~~~~~~~~~~~~~

For fine-grained control, create an ``OptimizationConfig`` instance:

.. code-block:: python

   from quactuary.optimization_selector import OptimizationConfig
   
   config = OptimizationConfig(
       use_jit=True,
       use_parallel=True,
       n_workers=4,
       use_qmc=True,
       qmc_method="sobol"
   )
   
   result = model.simulate(
       optimization_config=config,
       n_sims=100000
   )

Configuration Options
---------------------

The ``OptimizationConfig`` class supports the following options:

.. list-table:: OptimizationConfig Parameters
   :widths: 25 15 60
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``use_jit``
     - ``False``
     - Enable Just-In-Time compilation for faster execution
   * - ``use_parallel``
     - ``False``
     - Enable parallel processing across multiple CPU cores
   * - ``use_qmc``
     - ``False``
     - Use Quasi-Monte Carlo sequences for better convergence
   * - ``qmc_method``
     - ``None``
     - QMC sequence type: ``"sobol"`` or ``"halton"``
   * - ``use_vectorization``
     - ``True``
     - Use NumPy vectorized operations (recommended)
   * - ``use_memory_optimization``
     - ``False``
     - Enable batch processing for memory-constrained systems
   * - ``parallel_backend``
     - ``None``
     - Backend for parallelization: ``"joblib"`` or ``"multiprocessing"``
   * - ``batch_size``
     - ``None``
     - Batch size for memory optimization (auto-calculated if None)
   * - ``n_workers``
     - ``None``
     - Number of parallel workers (defaults to CPU count - 1)
   * - ``fallback_chain``
     - ``[]``
     - List of fallback strategies if primary fails
   * - ``user_preferences``
     - ``None``
     - Custom preferences for optimization selection

How Auto-Optimization Works
---------------------------

When ``auto_optimize=True``, the system follows this decision process:

1. **Portfolio Analysis**
   
   * Number of policies and simulations
   * Distribution complexity
   * Total memory requirements

2. **Resource Assessment**
   
   * Available system memory
   * Number of CPU cores
   * Memory safety margins

3. **Strategy Selection**
   
   .. code-block:: text
   
      Data Points = n_policies × n_simulations
      
      If Data Points < 1M:
          → Use vectorization only (fast for small data)
      
      If 1M ≤ Data Points < 100M:
          → Use JIT + Vectorization + QMC
      
      If Data Points ≥ 100M:
          → Use all optimizations (JIT, Parallel, QMC, Vectorization)
      
      If Memory Required > 80% Available:
          → Enable memory optimization with batching

Common Optimization Combinations
--------------------------------

Small Portfolios (< 1,000 policies)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Let auto-optimize handle it
   result = model.simulate(auto_optimize=True, n_sims=10000)
   
   # Or manually optimize for speed
   config = OptimizationConfig(
       use_jit=True,
       use_vectorization=True
   )

Medium Portfolios (1,000 - 100,000 policies)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Good balance of speed and accuracy
   config = OptimizationConfig(
       use_jit=True,
       use_qmc=True,
       qmc_method="sobol",
       use_vectorization=True
   )

Large Portfolios (> 100,000 policies)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Maximum performance
   config = OptimizationConfig(
       use_jit=True,
       use_parallel=True,
       n_workers=8,  # Adjust based on your CPU
       use_qmc=True,
       qmc_method="sobol",
       use_memory_optimization=True,
       batch_size=50000
   )

Memory-Constrained Systems
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Optimize for low memory usage
   config = OptimizationConfig(
       use_memory_optimization=True,
       batch_size=10000,  # Small batches
       use_parallel=False,  # Save memory
       use_vectorization=True
   )

Advanced Usage
--------------

Combining Auto and Manual Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can provide hints to the auto-optimizer:

.. code-block:: python

   # Force parallel but let system choose workers
   config = OptimizationConfig(
       use_parallel=True
   )
   
   result = model.simulate(
       auto_optimize=True,
       optimization_config=config,  # Used as hints
       n_sims=100000
   )

Custom Fallback Chains
~~~~~~~~~~~~~~~~~~~~~~

Define fallback strategies if primary optimization fails:

.. code-block:: python

   config = OptimizationConfig(
       use_parallel=True,
       use_jit=True,
       fallback_chain=["jit_only", "vectorized_only", "basic"]
   )

Runtime Monitoring
~~~~~~~~~~~~~~~~~~

The system monitors performance and can adapt:

.. code-block:: python

   # Enable verbose output to see optimization decisions
   import logging
   logging.getLogger('quactuary.optimization').setLevel(logging.INFO)
   
   result = model.simulate(
       auto_optimize=True,
       n_sims=100000
   )

Performance Guidelines
----------------------

1. **Start with auto-optimization** - It handles most cases well
2. **Use QMC for better convergence** - Reduces simulations needed by 10-100x
3. **Enable JIT for repeated calculations** - Compilation overhead pays off
4. **Use parallel for large portfolios** - Scales well up to CPU core count
5. **Enable memory optimization only when needed** - Adds overhead

Troubleshooting
---------------

Common Issues and Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Issue**: Auto-optimization chooses suboptimal strategy

.. code-block:: python

   # Solution: Provide hints
   config = OptimizationConfig(use_qmc=True)  # Force QMC
   result = model.simulate(auto_optimize=True, optimization_config=config)

**Issue**: Out of memory errors

.. code-block:: python

   # Solution: Enable memory optimization
   config = OptimizationConfig(
       use_memory_optimization=True,
       batch_size=1000,  # Very small batches
       use_parallel=False  # Disable parallel to save memory
   )

**Issue**: Slow performance on small data

.. code-block:: python

   # Solution: Disable heavy optimizations
   config = OptimizationConfig(
       use_jit=False,  # JIT has compilation overhead
       use_parallel=False,  # Parallel has communication overhead
       use_vectorization=True  # Keep only vectorization
   )

Best Practices
--------------

1. **Profile before optimizing manually**
   
   .. code-block:: python
   
      # Use built-in profiling
      result = model.simulate(
          auto_optimize=True,
          profile=True  # Shows timing breakdown
      )

2. **Consider your use case**
   
   * One-time calculations: Skip JIT (compilation overhead)
   * Repeated calculations: Enable JIT
   * High accuracy needed: Use QMC
   * Large portfolios: Enable parallel

3. **Monitor resource usage**
   
   .. code-block:: python
   
      from quactuary.utils import get_system_info
      
      info = get_system_info()
      print(f"Available memory: {info['memory_available_gb']:.1f} GB")
      print(f"CPU cores: {info['cpu_count']}")

Examples
--------

Example 1: Insurance Portfolio Pricing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from quactuary.pricing import PricingModel
   from quactuary.entities import Portfolio
   from quactuary.optimization_selector import OptimizationConfig
   
   # Large commercial insurance portfolio
   portfolio = Portfolio.from_file("commercial_portfolio.pkl")
   print(f"Portfolio size: {len(portfolio)} policies")
   
   # Let system optimize
   model = PricingModel(portfolio)
   result = model.simulate(
       auto_optimize=True,
       n_sims=1000000,
       confidence_levels=[0.95, 0.99, 0.999]
   )
   
   print(f"Optimization used: {result.optimization_info}")

Example 2: Reinsurance Treaty Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # High-accuracy simulation for treaty pricing
   config = OptimizationConfig(
       use_qmc=True,
       qmc_method="sobol",
       use_jit=True,
       use_parallel=True,
       n_workers=16  # High-end server
   )
   
   model = PricingModel(portfolio, reinsurance=treaty)
   result = model.simulate(
       optimization_config=config,
       n_sims=10000000  # 10M simulations
   )

Example 3: Quick Exploratory Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Fast iteration during model development
   config = OptimizationConfig(
       use_vectorization=True,  # Fast for small data
       use_jit=False,  # Skip compilation
       use_parallel=False  # Skip overhead
   )
   
   for threshold in [1000, 5000, 10000]:
       model = PricingModel(portfolio, excess_threshold=threshold)
       result = model.simulate(
           optimization_config=config,
           n_sims=10000  # Small sample
       )
       print(f"Threshold {threshold}: Mean = {result.mean:.2f}")

See Also
--------

* :doc:`/performance/tuning_guide` - Performance tuning and optimization guide
* :doc:`/user_guide/optimization_overview` - Optimization overview and concepts
* :doc:`/performance/benchmarks` - Performance benchmarks and comparisons