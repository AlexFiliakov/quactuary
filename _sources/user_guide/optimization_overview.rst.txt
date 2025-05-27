.. _optimization_overview:

=======================
Optimization Overview
=======================

The quActuary package provides multiple optimization strategies to accelerate Monte Carlo simulations
for actuarial computations. This guide helps you choose the right optimization approach for your use case.

.. contents:: Table of Contents
   :local:
   :depth: 2

Quick Decision Guide
====================

.. .. figure:: _static/optimization_decision_tree.png
..    :alt: Optimization Decision Tree
..    :align: center
..    
..    Decision tree for selecting optimization strategies based on portfolio size

For quick reference:

* **< 100 policies**: Use vectorization only
* **100-1,000 policies**: Add JIT compilation
* **> 1,000 policies**: Enable all optimizations including parallel processing

Available Optimizations
=======================

JIT Compilation
---------------

Just-In-Time (JIT) compilation using Numba accelerates numerical computations by compiling Python
functions to machine code.

**When to use:**
   - Portfolio size > 100 policies
   - Complex mathematical operations
   - Repeated function calls

**Benefits:**
   - 5-20x speedup for numerical operations
   - No code changes required
   - Automatic type inference

**Example:**

.. code-block:: python

   from quactuary import PricingModel
   
   # JIT compilation is auto-enabled for larger portfolios
   model = PricingModel(portfolio)
   results = model.simulate(n_simulations=100_000, use_jit=True)

Vectorization
-------------

NumPy-based vectorization processes entire arrays at once instead of using loops.

**When to use:**
   - Always enabled by default
   - Especially effective for simple operations
   - Works well with all portfolio sizes

**Benefits:**
   - 5-10x speedup for array operations
   - Memory efficient
   - No compilation overhead

Parallel Processing
-------------------

Distributes simulation work across multiple CPU cores using multiprocessing.

**When to use:**
   - Portfolio size > 50 policies
   - Multiple CPU cores available
   - Non-memory-constrained environment

**Benefits:**
   - Near-linear scaling with CPU cores
   - Automatic work distribution
   - Progress tracking support

**Example:**

.. code-block:: python

   # Parallel processing with custom worker count
   results = model.simulate(
       n_simulations=1_000_000,
       parallel=True,
       max_workers=8  # Use 8 CPU cores
   )

Memory Optimization
-------------------

Intelligent memory management for large-scale simulations.

**When to use:**
   - Limited RAM available
   - Very large simulation counts (> 10M)
   - Large portfolios (> 10K policies)

**Features:**
   - Automatic batch processing
   - Memory limit enforcement
   - Checkpoint support for recovery

**Example:**

.. code-block:: python

   # Memory-constrained simulation
   results = model.simulate(
       n_simulations=10_000_000,
       memory_limit_gb=4,  # Limit to 4GB RAM
       checkpoint_interval=100_000  # Save progress every 100K simulations
   )

Quasi-Monte Carlo (QMC)
-----------------------

Low-discrepancy sequences for improved convergence rates.

**When to use:**
   - High-dimensional problems
   - Need better convergence than standard Monte Carlo
   - Smooth integrand functions

**Benefits:**
   - Better convergence rate (O(1/N) vs O(1/√N))
   - Deterministic sequences
   - Improved tail accuracy

**Example:**

.. code-block:: python

   # QMC with Sobol sequences
   results = model.simulate(
       n_simulations=50_000,
       use_qmc=True,
       qmc_engine='sobol'
   )

Performance vs Accuracy Trade-offs
==================================

Speed vs Memory Usage
---------------------

.. list-table:: Optimization Trade-offs
   :header-rows: 1
   :widths: 30 20 20 30

   * - Optimization
     - Speed Gain
     - Memory Usage
     - Best For
   * - Vectorization
     - 5-10x
     - Moderate
     - All scenarios
   * - JIT Compilation
     - 10-50x
     - Low
     - Complex calculations
   * - Parallel Processing
     - Nx (N cores)
     - N × base
     - Large portfolios
   * - Memory Optimization
     - 0.8-1x
     - Controlled
     - Memory-constrained

Parallelization Overhead
------------------------

Parallel processing has startup overhead. Use this guide:

* < 50 policies: Overhead exceeds benefit
* 50-500 policies: Moderate benefit (2-4x speedup)
* > 500 policies: Full benefit (near-linear scaling)

Configuration Examples
======================

Small Portfolio (< 100 policies)
---------------------------------

.. code-block:: python

   # Optimal for small portfolios
   model = PricingModel(small_portfolio)
   results = model.simulate(
       n_simulations=10_000,
       use_jit=False,  # Avoid compilation overhead
       parallel=False,  # Avoid parallelization overhead
       vectorized=True  # Always beneficial
   )

Medium Portfolio (100-1,000 policies)
-------------------------------------

.. code-block:: python

   # Balanced optimization for medium portfolios
   model = PricingModel(medium_portfolio)
   results = model.simulate(
       n_simulations=100_000,
       use_jit=True,    # Significant benefit
       parallel=True,   # Moderate benefit
       max_workers=4    # Limit worker count
   )

Large Portfolio (> 1,000 policies)
----------------------------------

.. code-block:: python

   # Maximum performance for large portfolios
   model = PricingModel(large_portfolio)
   results = model.simulate(
       n_simulations=1_000_000,
       use_jit=True,           # Essential
       parallel=True,          # Essential
       max_workers=None,       # Use all cores
       memory_limit_gb=None,   # Auto-detect
       use_qmc=True           # Better convergence
   )

Memory-Constrained Environment
------------------------------

.. code-block:: python

   # Configuration for limited memory
   model = PricingModel(portfolio)
   results = model.simulate(
       n_simulations=10_000_000,
       memory_limit_gb=2,          # Strict limit
       checkpoint_interval=50_000,  # Frequent saves
       use_jit=False,              # Save memory
       parallel=False              # Single process
   )

Industry-Specific Use Cases
===========================

Property Catastrophe Modeling
-----------------------------

.. code-block:: python

   # Optimize for many locations, complex dependencies
   cat_model = PricingModel(property_portfolio)
   results = cat_model.simulate(
       n_simulations=100_000,
       use_jit=True,      # Complex calculations
       parallel=True,     # Many locations
       use_qmc=True      # Better tail estimates
   )

Life Insurance Valuations
-------------------------

.. code-block:: python

   # Optimize for long time horizons
   life_model = PricingModel(life_portfolio)
   results = life_model.simulate(
       n_simulations=50_000,
       use_jit=True,      # Mortality calculations
       memory_limit_gb=8,  # Control memory growth
       checkpoint_interval=10_000  # Regular saves
   )

Next Steps
==========

* :doc:`quick_start` - Get started with optimization
* :doc:`configuration_guide` - Detailed configuration options
* :doc:`best_practices` - Optimization best practices
* :doc:`../performance/tuning_guide` - Advanced performance tuning