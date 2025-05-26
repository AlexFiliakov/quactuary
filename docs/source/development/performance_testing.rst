Performance Testing Guide
========================

This guide describes the performance testing infrastructure in quActuary, including
how to write performance tests, manage baselines, and detect regressions.

Overview
--------

The quActuary performance testing system provides:

- **Adaptive Baselines**: Hardware-aware baseline management that normalizes performance across different environments
- **Regression Detection**: Automatic detection of performance regressions with configurable thresholds
- **CI/CD Integration**: Automated performance testing in pull requests and baseline updates on releases
- **CLI Tools**: Command-line utilities for baseline management and analysis

Key Components
-------------

Performance Baseline System
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The adaptive baseline system consists of three main classes:

1. **HardwareProfile**: Captures system characteristics and calculates a performance score
2. **PerformanceBaseline**: Stores performance measurements with hardware normalization
3. **AdaptiveBaselineManager**: Manages baselines and detects regressions

.. code-block:: python

   from quactuary.performance_baseline import AdaptiveBaselineManager
   
   # Create baseline manager
   manager = AdaptiveBaselineManager()
   
   # Record a performance measurement
   baseline = manager.record_performance(
       test_name="my_algorithm",
       execution_time=1.234,
       sample_size=10000
   )
   
   # Check for regression
   is_regression, expected_time, message = manager.check_regression(
       test_name="my_algorithm",
       current_time=1.456,
       sample_size=10000
   )

Performance Test Framework
~~~~~~~~~~~~~~~~~~~~~~~~~

The framework provides base classes and decorators for writing performance tests:

.. code-block:: python

   from quactuary.performance_testing import PerformanceTestCase, performance_test
   
   class TestMyAlgorithm(PerformanceTestCase):
       
       @performance_test("matrix_multiplication", sample_size=1000)
       def test_matrix_mult_performance(self):
           # Your test code here
           result = expensive_matrix_operation()
           return result
       
       def test_custom_performance(self):
           # Manual performance assertion
           self.assertPerformance(
               test_func=lambda: my_algorithm(data),
               test_name="my_algorithm",
               sample_size=len(data),
               max_time=2.0,  # Maximum allowed time
               check_regression=True
           )

Writing Performance Tests
------------------------

Basic Performance Test
~~~~~~~~~~~~~~~~~~~~~

Here's a simple example of a performance test:

.. code-block:: python

   import unittest
   from quactuary.performance_testing import PerformanceTestCase
   from quactuary.pricing import PricingModel
   from quactuary.book import Portfolio, Inforce
   
   class TestPricingPerformance(PerformanceTestCase):
       
       def setUp(self):
           # Create test portfolio
           self.portfolio = create_test_portfolio()
       
       def test_simulation_performance(self):
           """Test that simulation meets performance requirements."""
           model = PricingModel(self.portfolio)
           
           # Define test function
           def run_simulation():
               return model.simulate(n_sims=10000)
           
           # Assert performance
           self.assertPerformance(
               test_func=run_simulation,
               test_name="pricing_simulation_10k",
               sample_size=10000,
               check_regression=True
           )

Using the Performance Decorator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For simpler tests, use the decorator:

.. code-block:: python

   @performance_test("fibonacci_recursive", sample_size=35)
   def test_fibonacci_performance(self):
       # This will automatically track performance
       result = fibonacci_recursive(35)
       self.assertEqual(result, 9227465)

Hardware Normalization
~~~~~~~~~~~~~~~~~~~~~

The system automatically normalizes performance based on hardware capabilities:

- CPU count and frequency
- Available memory
- Platform characteristics

This allows meaningful comparisons across different environments:

.. code-block:: python

   # On a fast machine (performance score 2.0):
   # Raw time: 1.0s, Normalized time: 0.5s
   
   # On a slow machine (performance score 0.5):
   # Raw time: 4.0s, Normalized time: 4.0s
   
   # Both are considered equivalent performance

Managing Baselines
-----------------

CLI Commands
~~~~~~~~~~~

The performance baseline CLI provides several commands:

.. code-block:: bash

   # Show current hardware profile
   python -m quactuary.cli.performance_baseline_cli show --hardware
   
   # Show all baselines
   python -m quactuary.cli.performance_baseline_cli show
   
   # Update baselines by running benchmarks
   python -m quactuary.cli.performance_baseline_cli update
   
   # Check for regression
   python -m quactuary.cli.performance_baseline_cli compare \
       --test my_test --time 1.234 --sample-size 1000
   
   # Export baselines for backup
   python -m quactuary.cli.performance_baseline_cli export \
       -o baselines_backup.json
   
   # Import baselines
   python -m quactuary.cli.performance_baseline_cli import \
       -i baselines_backup.json

Baseline Storage
~~~~~~~~~~~~~~~

Baselines are stored in JSON format in the ``performance_baselines/`` directory:

.. code-block:: json

   {
     "test_name": [
       {
         "timestamp": "2025-05-26T13:00:00",
         "hardware_profile": {
           "cpu_model": "Intel Core i7-9750H",
           "cpu_count": 6,
           "performance_score": 1.2
         },
         "raw_time": 1.234,
         "normalized_time": 1.028,
         "sample_size": 10000
       }
     ]
   }

CI/CD Integration
----------------

GitHub Actions Workflow
~~~~~~~~~~~~~~~~~~~~~~

The repository includes a GitHub Actions workflow that:

1. Runs performance tests on pull requests
2. Checks for regressions
3. Updates baselines on merges to main
4. Posts performance reports as PR comments

Manual Baseline Updates
~~~~~~~~~~~~~~~~~~~~~~

To manually update baselines:

.. code-block:: bash

   # Trigger workflow with baseline update
   gh workflow run performance-testing.yml \
       -f update_baselines=true

Regression Detection
-------------------

Regression Thresholds
~~~~~~~~~~~~~~~~~~~~

The system uses adaptive thresholds for regression detection:

- **Default threshold**: 20% slower than baseline
- **Statistical threshold**: baseline + 2 standard deviations
- **Dynamic adjustment**: Uses the larger of the two thresholds

Handling Regressions
~~~~~~~~~~~~~~~~~~~

When a regression is detected:

1. **In development**: Tests fail with detailed error message
2. **In CI (PRs)**: Tests warn but don't block (configurable)
3. **In production**: Baselines are updated after review

Example regression output:

.. code-block:: text

   Performance regression detected: 35.2% slower than baseline.
   Expected: 1.234s (normalized), Got: 1.668s (normalized)

Best Practices
-------------

1. **Isolate Performance Tests**
   
   Keep performance tests separate from functional tests:
   
   .. code-block:: python
   
      # Good: Dedicated performance test file
      # tests/performance/test_pricing_performance.py
      
      # Bad: Mixed with unit tests
      # tests/test_pricing.py

2. **Use Appropriate Sample Sizes**
   
   Choose sample sizes that provide stable measurements:
   
   .. code-block:: python
   
      # Good: Large enough for stable timing
      @performance_test("algorithm", sample_size=10000)
      
      # Bad: Too small, high variance
      @performance_test("algorithm", sample_size=10)

3. **Include Warm-up Runs**
   
   For JIT-compiled code, include warm-up runs:
   
   .. code-block:: python
   
      def test_jit_performance(self):
          # Warm up JIT
          for _ in range(5):
              jit_function(small_data)
          
          # Actual performance test
          self.assertPerformance(
              test_func=lambda: jit_function(large_data),
              test_name="jit_function",
              sample_size=len(large_data)
          )

4. **Document Performance Requirements**
   
   Clearly document expected performance characteristics:
   
   .. code-block:: python
   
      def test_real_time_pricing(self):
          """Pricing must complete within 100ms for real-time applications."""
          self.assertPerformance(
              test_func=lambda: model.price(),
              test_name="real_time_pricing",
              sample_size=1,
              max_time=0.1  # 100ms requirement
          )

Troubleshooting
--------------

Common Issues
~~~~~~~~~~~~

1. **High Variance in Measurements**
   
   - Increase sample size
   - Add more warm-up runs
   - Check for background processes

2. **False Positives on CI**
   
   - CI servers may have variable performance
   - Adjust regression thresholds for CI
   - Use performance scores for normalization

3. **Missing Baselines**
   
   - Run baseline update command
   - Import baselines from another environment
   - Let tests establish initial baselines

Debug Output
~~~~~~~~~~~

Enable verbose output for debugging:

.. code-block:: python

   import logging
   logging.basicConfig(level=logging.DEBUG)
   
   # Run tests with detailed output
   python -m pytest tests/performance -v -s

Performance Reports
~~~~~~~~~~~~~~~~~~

Generate detailed performance reports:

.. code-block:: bash

   # Generate JSON report
   python scripts/check_performance_regressions.py \
       --output-json performance_report.json
   
   # View regression details
   cat performance_report.json | jq '.regressions'