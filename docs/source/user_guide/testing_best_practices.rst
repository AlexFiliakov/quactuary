Testing Best Practices
======================

This guide provides best practices for writing robust tests in quActuary, especially when dealing with stochastic methods and numerical computations.

.. important::

   Stochastic methods require careful tolerance selection. Tests that are too strict will fail randomly, 
   while tests that are too loose won't catch real issues.

Tolerance Guidelines
--------------------

Choosing Appropriate Tolerances
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Different types of calculations require different tolerance levels:

.. list-table:: Recommended Tolerances by Calculation Type
   :widths: 40 20 40
   :header-rows: 1

   * - Calculation Type
     - Relative Tolerance
     - Notes
   * - Deterministic calculations
     - ``1e-10``
     - Should match exactly (accounting for floating point)
   * - Small sample Monte Carlo (n < 1,000)
     - ``0.1`` (10%)
     - High variance expected
   * - Medium sample Monte Carlo (n ~ 10,000)
     - ``0.05`` (5%)
     - Reasonable for most tests
   * - Large sample Monte Carlo (n > 100,000)
     - ``0.01`` (1%)
     - More stable results
   * - Quasi-Monte Carlo convergence
     - ``0.05`` (5%)
     - Better convergence than standard MC
   * - Moment calculations (mean, variance)
     - ``0.02`` (2%)
     - Central limit theorem helps
   * - Tail statistics (VaR 99.5%)
     - ``0.1`` (10%)
     - Fewer samples in tails
   * - Correlation estimates
     - ``0.05`` (5%)
     - Depends on sample size

Setting Random Seeds
~~~~~~~~~~~~~~~~~~~~

Always set random seeds for reproducible tests:

.. code-block:: python

   import numpy as np
   import pytest
   
   @pytest.fixture
   def set_random_seed():
       """Ensure reproducible random numbers."""
       np.random.seed(42)
       yield
       # Reset to random state after test
       np.random.seed(None)
   
   def test_monte_carlo_mean(set_random_seed):
       # Test will use same random numbers each run
       result = simulate_portfolio(n_sims=10000)
       assert abs(result.mean - expected_mean) / expected_mean < 0.02

Testing Stochastic Methods
--------------------------

Basic Pattern
~~~~~~~~~~~~~

.. code-block:: python

   def test_stochastic_calculation():
       # Set up
       np.random.seed(42)
       n_sims = 10000
       
       # Run simulation
       result = run_simulation(n_sims=n_sims)
       
       # Check with appropriate tolerance
       expected_mean = 1000.0
       relative_error = abs(result.mean - expected_mean) / expected_mean
       
       # Use generous tolerance for stochastic results
       assert relative_error < 0.05, f"Mean {result.mean} not within 5% of {expected_mean}"

Testing Convergence
~~~~~~~~~~~~~~~~~~~

For methods that should converge, test the trend rather than absolute values:

.. code-block:: python

   def test_qmc_convergence():
       """Test that QMC converges faster than standard MC."""
       sample_sizes = [100, 1000, 10000]
       qmc_errors = []
       mc_errors = []
       
       true_value = 1000.0
       
       for n in sample_sizes:
           # QMC simulation
           qmc_result = simulate_with_qmc(n_sims=n)
           qmc_error = abs(qmc_result.mean - true_value) / true_value
           qmc_errors.append(qmc_error)
           
           # Standard MC simulation  
           mc_result = simulate_with_mc(n_sims=n)
           mc_error = abs(mc_result.mean - true_value) / true_value
           mc_errors.append(mc_error)
       
       # QMC should converge faster (errors decrease more)
       qmc_improvement = qmc_errors[0] / qmc_errors[-1]
       mc_improvement = mc_errors[0] / mc_errors[-1]
       
       assert qmc_improvement > mc_improvement * 1.5  # QMC at least 50% better

Hardware-Dependent Tests
------------------------

When to Skip Tests
~~~~~~~~~~~~~~~~~~

Some tests depend on hardware or environment. Use ``pytest.mark.skip`` appropriately:

.. code-block:: python

   import pytest
   import psutil
   
   @pytest.mark.skip(reason="Requires stable baseline data from CI environment")
   def test_performance_regression():
       """This test should only run in controlled CI environment."""
       result = benchmark_function()
       assert result.time < baseline_time * 1.1
   
   @pytest.mark.skipif(
       psutil.virtual_memory().available < 8 * 1024**3,
       reason="Requires at least 8GB free memory"
   )
   def test_large_portfolio():
       """Test handling of large portfolios."""
       portfolio = generate_large_portfolio(n_policies=1_000_000)
       result = process_portfolio(portfolio)
       assert result.success

Conditional Testing
~~~~~~~~~~~~~~~~~~~

For tests that may behave differently on different hardware:

.. code-block:: python

   def test_parallel_speedup():
       """Test parallel processing provides speedup."""
       import multiprocessing
       
       n_cores = multiprocessing.cpu_count()
       
       if n_cores < 4:
           pytest.skip("Parallel speedup test requires at least 4 cores")
       
       # Time sequential processing
       start = time.time()
       sequential_result = process_sequential(data)
       sequential_time = time.time() - start
       
       # Time parallel processing
       start = time.time()
       parallel_result = process_parallel(data, n_workers=n_cores)
       parallel_time = time.time() - start
       
       # Expect speedup, but not perfect scaling
       min_speedup = min(2.0, n_cores * 0.5)  # At least 50% efficiency
       actual_speedup = sequential_time / parallel_time
       
       assert actual_speedup > min_speedup

Numerical Accuracy Testing
--------------------------

Testing Floating Point Calculations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use appropriate comparison methods for floating point:

.. code-block:: python

   import numpy as np
   
   def test_numerical_stability():
       """Test calculations are numerically stable."""
       
       # Don't use exact equality for floats
       result = complex_calculation()
       expected = 1.23456789
       
       # BAD: May fail due to floating point precision
       # assert result == expected
       
       # GOOD: Use np.allclose for arrays
       assert np.allclose(result, expected, rtol=1e-7, atol=1e-10)
       
       # GOOD: Use pytest.approx for scalars
       assert result == pytest.approx(expected, rel=1e-7, abs=1e-10)

Testing Edge Cases
~~~~~~~~~~~~~~~~~~

Always test numerical edge cases:

.. code-block:: python

   def test_distribution_edge_cases():
       """Test distribution behavior at extremes."""
       
       # Test near-zero values
       dist = Exponential(scale=1.0)
       
       # Small probabilities
       assert dist.ppf(1e-10) == pytest.approx(1e-10, rel=0.01)
       
       # Large probabilities  
       assert dist.ppf(0.99999) < 20  # Reasonable upper bound
       
       # Test parameter boundaries
       with pytest.raises(ValueError):
           Exponential(scale=-1.0)  # Invalid parameter

Integration Test Patterns
-------------------------

Testing End-to-End Workflows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def test_portfolio_pricing_workflow():
       """Test complete pricing workflow with appropriate tolerances."""
       
       # Create test portfolio
       portfolio = Portfolio([
           Policy(
               frequency=Poisson(mu=2.0),
               severity=Lognormal(shape=1.0, scale=1000)
           )
           for _ in range(100)
       ])
       
       # Configure for testing (faster but less accurate)
       config = OptimizationConfig(
           use_vectorization=True,
           use_qmc=True,
           qmc_method="sobol"
       )
       
       # Run simulation with reasonable sample size
       model = PricingModel(portfolio)
       result = model.simulate(
           n_sims=10000,
           optimization_config=config
       )
       
       # Check results with appropriate tolerances
       assert result.mean == pytest.approx(2000, rel=0.05)  # 5% tolerance
       assert result.std == pytest.approx(1500, rel=0.10)   # 10% for std
       assert result.var_95 == pytest.approx(4000, rel=0.10)  # 10% for VaR

Testing Different Optimization Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   @pytest.mark.parametrize("use_jit,use_parallel,use_qmc", [
       (False, False, False),  # Baseline
       (True, False, False),   # JIT only
       (False, True, False),   # Parallel only
       (False, False, True),   # QMC only
       (True, True, True),     # All optimizations
   ])
   def test_optimization_consistency(use_jit, use_parallel, use_qmc):
       """Test that all optimization strategies give consistent results."""
       
       np.random.seed(42)
       
       config = OptimizationConfig(
           use_jit=use_jit,
           use_parallel=use_parallel,
           use_qmc=use_qmc,
           n_workers=2  # Limit for testing
       )
       
       result = model.simulate(
           n_sims=5000,
           optimization_config=config
       )
       
       # All strategies should give similar results
       assert result.mean == pytest.approx(expected_mean, rel=0.1)

Common Pitfalls and Solutions
-----------------------------

Issue: Flaky Tests
~~~~~~~~~~~~~~~~~~

**Problem**: Tests pass sometimes but fail randomly.

**Solution**: 

.. code-block:: python

   # BAD: No seed, tight tolerance
   def test_simulation():
       result = simulate(n_sims=100)
       assert result.mean == pytest.approx(1000, rel=0.001)
   
   # GOOD: Seed set, appropriate tolerance
   def test_simulation():
       np.random.seed(42)
       result = simulate(n_sims=10000)  # Larger sample
       assert result.mean == pytest.approx(1000, rel=0.05)

Issue: Slow Tests
~~~~~~~~~~~~~~~~~

**Problem**: Tests take too long to run regularly.

**Solution**:

.. code-block:: python

   # Mark slow tests
   @pytest.mark.slow
   def test_large_portfolio_performance():
       """Full performance test - only run before release."""
       portfolio = generate_portfolio(n_policies=100000)
       # ... expensive test
   
   # Create a fast version for regular testing
   def test_small_portfolio_performance():
       """Quick performance check for CI."""
       portfolio = generate_portfolio(n_policies=100)
       # ... fast test

Run regular tests with: ``pytest -m "not slow"``

Issue: Platform-Dependent Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Tests pass on one platform but fail on another.

**Solution**:

.. code-block:: python

   import platform
   
   def test_numerical_precision():
       """Test with platform-appropriate tolerance."""
       
       result = complex_calculation()
       
       # Different platforms may have different precision
       if platform.system() == "Windows":
           tolerance = 1e-10
       else:
           tolerance = 1e-12
           
       assert abs(result - expected) < tolerance

Best Practices Summary
----------------------

1. **Set random seeds** for reproducibility
2. **Use appropriate tolerances** based on calculation type
3. **Test trends** rather than absolute values for convergence
4. **Skip hardware-dependent tests** when environment isn't suitable
5. **Use larger sample sizes** for more stable test results
6. **Parameterize tests** to cover multiple scenarios efficiently
7. **Mark slow tests** to keep regular test runs fast
8. **Document why** specific tolerances were chosen

Example: Complete Test Module
-----------------------------

.. code-block:: python

   """
   Tests for portfolio pricing with best practices.
   """
   import numpy as np
   import pytest
   from quactuary.pricing import PricingModel
   from quactuary.distributions import Poisson, Lognormal
   
   
   @pytest.fixture
   def random_seed():
       """Ensure reproducible randomness."""
       np.random.seed(42)
       yield
       np.random.seed(None)
   
   
   @pytest.fixture
   def small_portfolio():
       """Create a small test portfolio."""
       return Portfolio([
           Policy(
               frequency=Poisson(mu=2.0),
               severity=Lognormal(shape=1.0, scale=1000)
           )
           for _ in range(10)
       ])
   
   
   class TestPricingModel:
       """Test pricing model with appropriate tolerances."""
       
       def test_mean_estimation(self, small_portfolio, random_seed):
           """Test mean estimation with stochastic tolerance."""
           model = PricingModel(small_portfolio)
           result = model.simulate(n_sims=10000)
           
           expected_mean = 2.0 * 1000  # frequency * severity mean
           assert result.mean == pytest.approx(expected_mean, rel=0.05)
       
       def test_convergence(self, small_portfolio, random_seed):
           """Test that results converge with more simulations."""
           model = PricingModel(small_portfolio)
           
           # Run with increasing sample sizes
           results = []
           for n_sims in [1000, 10000, 100000]:
               result = model.simulate(n_sims=n_sims)
               results.append(result.mean)
           
           # Later results should be more stable
           diff_small = abs(results[1] - results[0]) / results[0]
           diff_large = abs(results[2] - results[1]) / results[1]
           
           assert diff_large < diff_small * 0.5  # Convergence
       
       @pytest.mark.slow
       def test_large_portfolio_memory(self):
           """Test memory handling for large portfolios."""
           large_portfolio = generate_portfolio(n_policies=100000)
           
           model = PricingModel(large_portfolio)
           result = model.simulate(
               n_sims=1000,
               optimization_config=OptimizationConfig(
                   use_memory_optimization=True,
                   batch_size=10000
               )
           )
           
           assert result.success
       
       @pytest.mark.skipif(
           not is_ci_environment(),
           reason="Performance regression tests only run in CI"
       )
       def test_performance_regression(self, small_portfolio):
           """Test performance hasn't regressed."""
           import time
           
           model = PricingModel(small_portfolio)
           
           start = time.time()
           model.simulate(n_sims=10000)
           duration = time.time() - start
           
           baseline_duration = load_baseline_duration()
           assert duration < baseline_duration * 1.2  # 20% margin

See Also
--------

* :doc:`/development/contributing` - General contribution guidelines
* :doc:`/api_reference/testing` - Testing utilities and fixtures
* `pytest documentation <https://docs.pytest.org/>`_ - Official pytest docs