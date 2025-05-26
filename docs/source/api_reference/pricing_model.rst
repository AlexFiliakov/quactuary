.. _api_pricing_model:

===================
PricingModel API
===================

The ``PricingModel`` class is the primary interface for running actuarial simulations
with optimization features.

.. contents:: Table of Contents
   :local:
   :depth: 2

Class Overview
==============

.. currentmodule:: quactuary

.. autoclass:: PricingModel
   :members:
   :undoc-members:
   :show-inheritance:

Core Methods
============

simulate()
----------

.. automethod:: PricingModel.simulate
   :no-index:

**Method Signature:**

.. code-block:: python

   def simulate(
       self,
       n_simulations: int = 10000,
       use_jit: Optional[bool] = None,
       parallel: Optional[bool] = None,
       max_workers: Optional[int] = None,
       vectorized: bool = True,
       memory_limit_gb: Optional[float] = None,
       use_qmc: bool = False,
       qmc_engine: str = 'sobol',
       progress_bar: bool = True,
       checkpoint_interval: Optional[int] = None,
       random_state: Optional[int] = None,
       **kwargs
   ) -> SimulationResults

**Parameters:**

.. list-table:: Simulation Parameters
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Type
     - Description
   * - ``n_simulations``
     - ``int``
     - Number of simulation paths to generate (default: 10,000)
   * - ``use_jit``
     - ``bool | None``
     - Enable JIT compilation. Auto-detected if None based on portfolio size
   * - ``parallel``
     - ``bool | None``
     - Enable parallel processing. Auto-detected if None
   * - ``max_workers``
     - ``int | None``
     - Maximum worker processes. Uses CPU count if None
   * - ``vectorized``
     - ``bool``
     - Enable vectorization (default: True, recommended)
   * - ``memory_limit_gb``
     - ``float | None``
     - Memory limit in GB. Auto-detected if None
   * - ``use_qmc``
     - ``bool``
     - Use quasi-Monte Carlo sequences (default: False)
   * - ``qmc_engine``
     - ``str``
     - QMC engine: 'sobol', 'halton', 'latin_hypercube'
   * - ``progress_bar``
     - ``bool``
     - Show progress bar during simulation (default: True)
   * - ``checkpoint_interval``
     - ``int | None``
     - Save progress every N simulations. None disables checkpointing
   * - ``random_state``
     - ``int | None``
     - Random seed for reproducibility

**Returns:**

* :class:`SimulationResults` - Object containing simulation results and metadata

**Examples:**

Basic usage with automatic optimization:

.. code-block:: python

   from quactuary import PricingModel, Portfolio
   
   # Create portfolio and model
   portfolio = Portfolio()
   # ... add policies to portfolio ...
   
   model = PricingModel(portfolio)
   
   # Run simulation with automatic optimization
   results = model.simulate(n_simulations=100_000)
   
   print(f"Mean loss: ${results.mean():,.0f}")
   print(f"95% VaR: ${results.var(0.95):,.0f}")
   print(f"99% TVaR: ${results.tvar(0.99):,.0f}")

Explicit optimization configuration:

.. code-block:: python

   # High-performance configuration
   results = model.simulate(
       n_simulations=1_000_000,
       use_jit=True,
       parallel=True,
       max_workers=8,
       use_qmc=True,
       qmc_engine='sobol'
   )

Memory-constrained environment:

.. code-block:: python

   # Memory-efficient configuration
   results = model.simulate(
       n_simulations=10_000_000,
       memory_limit_gb=4,
       checkpoint_interval=100_000,
       parallel=False  # Reduce memory usage
   )

**Auto-Detection Logic:**

The ``simulate()`` method automatically selects optimizations when parameters are ``None``:

.. code-block:: python

   def auto_detect_optimizations(portfolio, n_simulations):
       """Logic for automatic optimization detection."""
       
       # JIT compilation
       if len(portfolio) > 100 or n_simulations > 50_000:
           use_jit = True
       else:
           use_jit = False
       
       # Parallel processing  
       if len(portfolio) > 50 and n_simulations > 10_000:
           parallel = True
           max_workers = min(os.cpu_count(), len(portfolio) // 10)
       else:
           parallel = False
           max_workers = 1
       
       # Memory management
       estimated_memory = estimate_memory_gb(len(portfolio), n_simulations)
       available_memory = psutil.virtual_memory().available / 1e9
       
       if estimated_memory > available_memory * 0.8:
           memory_limit_gb = available_memory * 0.7
           checkpoint_interval = n_simulations // 100
       else:
           memory_limit_gb = None
           checkpoint_interval = None
       
       return {
           'use_jit': use_jit,
           'parallel': parallel,
           'max_workers': max_workers,
           'memory_limit_gb': memory_limit_gb,
           'checkpoint_interval': checkpoint_interval
       }

**Performance Considerations:**

* **JIT Compilation**: First run includes compilation overhead (0.5-2 seconds)
* **Parallel Processing**: Overhead makes it unsuitable for small portfolios (< 50 policies)
* **Memory Management**: Automatic batching when memory limit is approached
* **QMC Convergence**: Often achieves better precision with fewer simulations

.. validate_portfolio()
.. ---------------------
.. 
.. .. automethod:: PricingModel.validate_portfolio
.. 
.. Validates portfolio consistency and completeness before simulation.
.. 
.. .. code-block:: python
.. 
..    # Validate portfolio before simulation
..    validation_result = model.validate_portfolio()
..    
..    if validation_result.is_valid:
..        results = model.simulate(n_simulations=100_000)
..    else:
..        print("Portfolio validation errors:")
..        for error in validation_result.errors:
..            print(f"- {error}")

.. estimate_memory_requirements()
.. ------------------------------
.. 
.. .. automethod:: PricingModel.estimate_memory_requirements
.. 
.. Estimates memory requirements for a given simulation configuration.
.. 
.. .. code-block:: python
.. 
..    # Check memory requirements before large simulation
..    memory_gb = model.estimate_memory_requirements(
..        n_simulations=10_000_000
..    )
..    
..    available_gb = psutil.virtual_memory().available / 1e9
..    
..    if memory_gb > available_gb * 0.8:
..        print(f"Warning: Simulation needs {memory_gb:.1f}GB, "
..              f"available: {available_gb:.1f}GB")
..        # Use memory optimization
..        results = model.simulate(
..            n_simulations=10_000_000,
..            memory_limit_gb=available_gb * 0.7
..        )

.. Properties
.. ==========
.. 
.. portfolio
.. ---------
.. 
.. .. autoproperty:: PricingModel.portfolio
.. 
.. The :class:`Portfolio` object containing policies to be simulated.

.. .. code-block:: python
.. 
..    model = PricingModel(portfolio)
..    
..    print(f"Portfolio size: {len(model.portfolio)}")
..    print(f"Total exposure: ${model.portfolio.total_exposure:,.0f}")

.. backend
.. -------
.. 
.. .. autoproperty:: PricingModel.backend

.. The computational backend used for simulations.
.. 
.. .. code-block:: python
.. 
..    from quactuary import Backend
..    
..    # Set quantum backend
..    model.backend = Backend.QUANTUM_QISKIT
..    
..    # Check current backend
..    if model.backend == Backend.CLASSICAL_NUMPY:
..        print("Using classical NumPy backend")

.. optimization_config
.. -------------------
.. 
.. .. autoproperty:: PricingModel.optimization_config

.. Current optimization configuration as a dictionary.
.. 
.. .. code-block:: python
.. 
..    # View current optimization settings
..    config = model.optimization_config
..    print(f"JIT enabled: {config['use_jit']}")
..    print(f"Parallel enabled: {config['parallel']}")
..    print(f"Max workers: {config['max_workers']}")
..    
..    # Modify configuration
..    model.optimization_config.update({
..        'use_jit': True,
..        'parallel': True,
..        'max_workers': 8
..    })

Advanced Usage
==============

Custom Optimization Strategies
------------------------------

Implement custom optimization logic:

.. code-block:: python

   class CustomPricingModel(PricingModel):
       """Custom model with specialized optimization."""
       
       def custom_optimize(self, n_simulations):
           """Custom optimization based on portfolio characteristics."""
           
           # Analyze portfolio complexity
           complexity = self.analyze_complexity()
           
           if complexity['has_dependencies']:
               # Correlated portfolios benefit from QMC
               config = {
                   'use_qmc': True,
                   'qmc_engine': 'sobol',
                   'use_jit': True
               }
           elif complexity['high_frequency']:
               # High-frequency portfolios need parallel processing
               config = {
                   'parallel': True,
                   'max_workers': os.cpu_count(),
                   'use_jit': True
               }
           else:
               # Simple portfolios use basic optimization
               config = {
                   'use_jit': True,
                   'parallel': False
               }
           
           return self.simulate(n_simulations=n_simulations, **config)
       
       def analyze_complexity(self):
           """Analyze portfolio complexity."""
           return {
               'has_dependencies': self.portfolio.correlation_matrix is not None,
               'high_frequency': any(p.frequency > 10 for p in self.portfolio),
               'complex_terms': any(p.has_complex_terms() for p in self.portfolio)
           }

Batch Processing
----------------

Process multiple scenarios efficiently:

.. code-block:: python

   def batch_simulate(model, scenarios, n_simulations=100_000):
       """Simulate multiple scenarios efficiently."""
       
       results = {}
       
       # Warm up JIT compilation once
       model.simulate(n_simulations=100, use_jit=True)
       
       for scenario_name, scenario_params in scenarios.items():
           # Apply scenario parameters
           model.apply_scenario(scenario_params)
           
           # Run optimized simulation
           scenario_results = model.simulate(
               n_simulations=n_simulations,
               use_jit=True,  # Already compiled
               parallel=True
           )
           
           results[scenario_name] = scenario_results
       
       return results
   
   # Define scenarios
   scenarios = {
       'base_case': {'loss_ratio': 1.0},
       'adverse': {'loss_ratio': 1.2},
       'severe': {'loss_ratio': 1.5}
   }
   
   # Run batch simulation
   batch_results = batch_simulate(model, scenarios)

Streaming Results
-----------------

Handle very large simulations with streaming:

.. code-block:: python

   def streaming_simulate(model, total_simulations, chunk_size=100_000):
       """Simulate in chunks and stream results."""
       
       aggregated_stats = SimulationAggregator()
       n_chunks = (total_simulations + chunk_size - 1) // chunk_size
       
       for i in range(n_chunks):
           chunk_sims = min(chunk_size, total_simulations - i * chunk_size)
           
           # Simulate chunk
           chunk_results = model.simulate(
               n_simulations=chunk_sims,
               progress_bar=False  # Avoid multiple progress bars
           )
           
           # Update aggregated statistics
           aggregated_stats.add_chunk(chunk_results)
           
           # Optional: yield intermediate results
           yield aggregated_stats.current_stats()
       
       return aggregated_stats.final_results()

Error Handling
==============

Common Exceptions
-----------------

The ``PricingModel`` class raises specific exceptions for different error conditions:

.. code-block:: python

   from quactuary.exceptions import (
       PortfolioValidationError,
       MemoryLimitExceededError,
       CompilationError,
       ParallelProcessingError
   )
   
   try:
       results = model.simulate(n_simulations=1_000_000)
   
   except PortfolioValidationError as e:
       print(f"Portfolio validation failed: {e}")
       # Fix portfolio issues
       
   except MemoryLimitExceededError as e:
       print(f"Memory limit exceeded: {e}")
       # Reduce simulation size or enable memory optimization
       results = model.simulate(
           n_simulations=500_000,
           memory_limit_gb=4
       )
       
   except CompilationError as e:
       print(f"JIT compilation failed: {e}")
       # Fallback to non-JIT simulation
       results = model.simulate(
           n_simulations=1_000_000,
           use_jit=False
       )
       
   except ParallelProcessingError as e:
       print(f"Parallel processing failed: {e}")
       # Fallback to single-threaded
       results = model.simulate(
           n_simulations=1_000_000,
           parallel=False
       )

Debugging Mode
--------------

Enable debugging for troubleshooting:

.. code-block:: python

   # Enable debug mode
   model.debug = True
   
   # Run simulation with detailed logging
   results = model.simulate(
       n_simulations=10_000,
       verbose=True  # Additional parameter for debug output
   )
   
   # Access debug information
   print("Optimization decisions:")
   for decision, reason in results.debug_info['optimization_decisions'].items():
       print(f"  {decision}: {reason}")
   
   print(f"Compilation time: {results.debug_info['compilation_time']:.2f}s")
   print(f"Memory usage: {results.debug_info['peak_memory_mb']:.1f}MB")

See Also
========

* :class:`SimulationResults` - Results object returned by ``simulate()``
* :class:`Portfolio` - Portfolio construction and management  
* :doc:`../user_guide/optimization_overview` - Optimization strategies guide
* :doc:`../user_guide/configuration_guide` - Detailed parameter reference
* :doc:`../performance/benchmarks` - Performance benchmarks