.. _optimization_quick_start:

========================
Optimization Quick Start
========================

Get started with quActuary's optimization features in minutes. This guide shows you how to 
enable optimizations with minimal configuration.

.. contents:: Table of Contents
   :local:
   :depth: 2

Installation
============

Ensure you have the optimization dependencies installed:

.. code-block:: bash

   pip install quactuary[optimize]
   
   # Or install individual components
   pip install numba  # For JIT compilation
   pip install scipy  # For QMC sequences

Basic Usage
===========

Auto-Optimization (Recommended)
-------------------------------

Let quActuary automatically choose optimizations based on your portfolio:

.. code-block:: python

   from quactuary import PricingModel, Portfolio
   
   # Create your portfolio
   portfolio = Portfolio()
   # ... add policies ...
   
   # Create model - optimizations auto-selected
   model = PricingModel(portfolio)
   
   # Run simulation with automatic optimization
   results = model.simulate(n_simulations=100_000)
   
   print(f"Simulation completed in {results.execution_time:.2f} seconds")
   print(f"Optimizations used: {results.optimizations_applied}")

Manual Optimization Control
---------------------------

Override automatic optimization selection:

.. code-block:: python

   # Force specific optimizations
   results = model.simulate(
       n_simulations=100_000,
       use_jit=True,        # Force JIT compilation
       parallel=True,       # Force parallel processing
       max_workers=4,       # Use 4 CPU cores
       use_qmc=True        # Use quasi-Monte Carlo
   )

Common Scenarios
================

Scenario 1: Maximum Speed
-------------------------

When performance is critical and resources are available:

.. code-block:: python

   # Maximum performance configuration
   results = model.simulate(
       n_simulations=1_000_000,
       use_jit=True,
       parallel=True,
       max_workers=None,     # Use all available cores
       vectorized=True,
       use_qmc=True,
       qmc_engine='sobol'
   )

Scenario 2: Limited Memory
--------------------------

For memory-constrained environments:

.. code-block:: python

   # Memory-efficient configuration
   results = model.simulate(
       n_simulations=10_000_000,
       memory_limit_gb=2,            # Limit to 2GB
       checkpoint_interval=100_000,   # Save progress frequently
       parallel=False                # Single process uses less memory
   )

Scenario 3: Development/Testing
-------------------------------

For quick iterations during development:

.. code-block:: python

   # Fast feedback configuration
   results = model.simulate(
       n_simulations=1_000,    # Small sample
       use_jit=False,          # Skip compilation
       progress_bar=True       # Visual feedback
   )

Monitoring Performance
======================

Progress Tracking
-----------------

Enable progress bars for long-running simulations:

.. code-block:: python

   results = model.simulate(
       n_simulations=1_000_000,
       progress_bar=True
   )
   # Shows: [████████████████████] 100% | 1000000/1000000 | ETA: 00:00

Performance Metrics
-------------------

Access detailed performance information:

.. code-block:: python

   results = model.simulate(n_simulations=100_000)
   
   # View performance metrics
   print(f"Total time: {results.execution_time:.2f}s")
   print(f"Simulations/second: {results.simulations_per_second:,.0f}")
   print(f"Memory peak: {results.peak_memory_mb:.1f} MB")
   
   # Breakdown by component
   for component, time in results.timing_breakdown.items():
       print(f"{component}: {time:.2f}s")

Comparing Configurations
------------------------

Benchmark different optimization strategies:

.. code-block:: python

   import time
   
   configurations = {
       "baseline": {"use_jit": False, "parallel": False},
       "jit_only": {"use_jit": True, "parallel": False},
       "parallel_only": {"use_jit": False, "parallel": True},
       "full_optimization": {"use_jit": True, "parallel": True, "use_qmc": True}
   }
   
   for name, config in configurations.items():
       start = time.time()
       results = model.simulate(n_simulations=100_000, **config)
       elapsed = time.time() - start
       print(f"{name}: {elapsed:.2f}s ({results.simulations_per_second:,.0f} sims/s)")

Best Practices
==============

1. **Start with Auto-Optimization**
   
   Let quActuary choose optimizations initially:
   
   .. code-block:: python
   
      # Good - automatic optimization
      results = model.simulate(n_simulations=100_000)
   
   Only override if you have specific requirements.

2. **Profile Before Optimizing**
   
   Understand where time is spent:
   
   .. code-block:: python
   
      # Enable detailed profiling
      results = model.simulate(
          n_simulations=10_000,
          profile=True
      )
      results.show_profile()

3. **Test Optimization Impact**
   
   Verify optimizations improve your specific use case:
   
   .. code-block:: python
   
      # Compare with and without optimization
      baseline = model.simulate(n_simulations=10_000, use_jit=False)
      optimized = model.simulate(n_simulations=10_000, use_jit=True)
      
      speedup = baseline.execution_time / optimized.execution_time
      print(f"JIT speedup: {speedup:.1f}x")

4. **Consider Warm-up for JIT**
   
   First JIT run includes compilation time:
   
   .. code-block:: python
   
      # Warm-up run (small)
      _ = model.simulate(n_simulations=100, use_jit=True)
      
      # Actual run (benefits from compiled code)
      results = model.simulate(n_simulations=1_000_000, use_jit=True)

Common Issues
=============

Slower Than Expected
--------------------

If optimizations make code slower:

1. **Check portfolio size** - Small portfolios may not benefit
2. **Verify no memory swapping** - Use ``memory_limit_gb``
3. **Consider compilation overhead** - JIT has startup cost

.. code-block:: python

   # Diagnose performance issues
   results = model.simulate(
       n_simulations=10_000,
       profile=True,
       verbose=True  # Show optimization decisions
   )

Memory Errors
-------------

For out-of-memory errors:

.. code-block:: python

   # Fix memory issues
   results = model.simulate(
       n_simulations=large_number,
       memory_limit_gb=available_memory * 0.8,  # Leave headroom
       checkpoint_interval=10_000               # Smaller batches
   )

Parallel Processing Issues
--------------------------

If parallel processing fails:

.. code-block:: python

   # Fallback configuration
   try:
       results = model.simulate(n_simulations=100_000, parallel=True)
   except Exception as e:
       print(f"Parallel failed: {e}, using single process")
       results = model.simulate(n_simulations=100_000, parallel=False)

Next Steps
==========

* :doc:`optimization_overview` - Detailed optimization guide
* :doc:`configuration_guide` - Full configuration reference
* :doc:`../performance/tuning_guide` - Advanced tuning
* :doc:`../performance/benchmarks` - Performance benchmarks