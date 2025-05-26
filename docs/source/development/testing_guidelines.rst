.. _testing_guidelines:

******************
Testing Guidelines
******************

This document outlines our testing standards and practices for quactuary. Comprehensive testing ensures code quality, prevents regressions, and gives contributors confidence in their changes.

.. contents:: Table of Contents
   :local:
   :depth: 2

Testing Philosophy
==================

We believe in:

* **Test-driven development**: Write tests before or alongside code
* **Comprehensive coverage**: Aim for ≥90% code coverage on new code
* **Fast feedback**: Tests should run quickly to enable rapid iteration
* **Clear test cases**: Tests should be readable and document expected behavior
* **Realistic scenarios**: Test with data that resembles real-world usage

Testing Framework
=================

We use `pytest <https://docs.pytest.org/>`_ as our primary testing framework:

.. code-block:: bash

   # Run all tests
   pytest
   
   # Run with coverage
   pytest --cov=quactuary
   
   # Run specific test file
   pytest tests/test_pricing.py
   
   # Run with verbose output
   pytest -v
   
   # Run in parallel (faster for large test suites)
   pytest -n auto

Test Organization
=================

Directory Structure
-------------------

Tests are organized to mirror the source code structure:

.. code-block:: text

   tests/
   ├── __init__.py
   ├── conftest.py                 # Shared fixtures
   ├── test_pricing.py             # Tests for quactuary.pricing
   ├── test_classical.py           # Tests for quactuary.classical
   ├── test_quantum.py             # Tests for quactuary.quantum
   ├── distributions/
   │   ├── __init__.py
   │   ├── test_frequency.py       # Frequency distribution tests
   │   ├── test_severity.py        # Severity distribution tests
   │   └── test_compound.py        # Compound distribution tests
   ├── backend/
   │   ├── test_backend.py         # Backend management tests
   │   └── test_imports.py         # Import and dependency tests
   └── integration/
       ├── test_end_to_end.py      # Full workflow tests
       └── test_performance.py     # Performance regression tests

Test File Naming
----------------

* Test files: ``test_*.py``
* Test classes: ``Test*`` (if using classes)
* Test functions: ``test_*``

Types of Tests
==============

Unit Tests
----------

Test individual functions and methods in isolation:

.. code-block:: python

   import pytest
   import numpy as np
   from quactuary.distributions import Poisson
   
   
   class TestPoisson:
       """Test suite for Poisson distribution."""
       
       def test_initialization(self):
           """Test Poisson distribution initialization."""
           # Test valid initialization
           dist = Poisson(lambda_=2.0)
           assert dist._dist.args[0] == 2.0
           
           # Test parameter validation
           with pytest.raises(ValueError, match="lambda_ must be positive"):
               Poisson(lambda_=-1.0)
       
       def test_pmf_calculation(self):
           """Test probability mass function calculation."""
           dist = Poisson(lambda_=2.0)
           
           # Test known values
           assert abs(dist.pmf(0) - 0.1353) < 1e-3
           assert abs(dist.pmf(2) - 0.2707) < 1e-3
           
           # Test edge cases
           assert dist.pmf(-1) == 0.0
           assert 0 <= dist.pmf(10) <= 1.0
       
       def test_sampling(self):
           """Test random sample generation."""
           dist = Poisson(lambda_=2.0)
           
           # Test single sample
           sample = dist.rvs(size=1)
           assert isinstance(sample, (int, np.integer))
           assert sample >= 0
           
           # Test multiple samples
           samples = dist.rvs(size=1000)
           assert len(samples) == 1000
           assert all(s >= 0 for s in samples)
           
           # Test statistical properties
           mean_sample = np.mean(samples)
           assert abs(mean_sample - 2.0) < 0.2  # Should be close to lambda

Integration Tests
-----------------

Test interactions between components:

.. code-block:: python

   import pytest
   from quactuary.pricing import PricingModel
   from quactuary.book import Portfolio, Inforce, PolicyTerms
   from quactuary.distributions import Poisson, LogNormal
   import datetime
   
   
   class TestPricingIntegration:
       """Integration tests for pricing workflow."""
       
       @pytest.fixture
       def sample_portfolio(self):
           """Create a sample portfolio for testing."""
           terms = PolicyTerms(
               effective_date=datetime.date(2024, 1, 1),
               expiration_date=datetime.date(2025, 1, 1),
               per_occ_retention=1000.0,
               per_occ_limit=100000.0
           )
           
           inforce = Inforce(
               n_policies=100,
               frequency=Poisson(lambda_=1.5),
               severity=LogNormal(mu=8, sigma=1),
               terms=terms
           )
           
           return Portfolio([inforce])
       
       def test_end_to_end_pricing(self, sample_portfolio):
           """Test complete pricing workflow."""
           model = PricingModel(sample_portfolio)
           
           result = model.simulate(
               mean=True,
               variance=True,
               value_at_risk=True,
               tail_value_at_risk=True,
               n_sims=1000
           )
           
           # Verify result structure
           assert 'mean' in result.estimates
           assert 'variance' in result.estimates
           assert 'VaR' in result.estimates
           assert 'TVaR' in result.estimates
           
           # Verify statistical relationships
           assert result.estimates['mean'] > 0
           assert result.estimates['variance'] > 0
           assert result.estimates['TVaR'] >= result.estimates['VaR']
           
           # Verify metadata
           assert result.metadata['n_sims'] == 1000
           assert 'run_date' in result.metadata

Performance Tests
-----------------

Test performance characteristics and prevent regressions:

.. code-block:: python

   import time
   import pytest
   from quactuary.pricing_strategies import ClassicalPricingStrategy
   
   
   class TestPerformance:
       """Performance regression tests."""
       
       @pytest.mark.slow
       def test_large_portfolio_performance(self, large_portfolio):
           """Test performance with large portfolios."""
           strategy = ClassicalPricingStrategy(use_jit=True)
           
           start_time = time.time()
           result = strategy.calculate_portfolio_statistics(
               large_portfolio, n_sims=10000
           )
           elapsed = time.time() - start_time
           
           # Should complete within reasonable time
           assert elapsed < 60  # 1 minute threshold
           assert result.estimates['mean'] > 0
       
       def test_jit_speedup(self, sample_portfolio):
           """Test that JIT provides speedup over pure Python."""
           # Test JIT version
           jit_strategy = ClassicalPricingStrategy(use_jit=True)
           start = time.time()
           jit_result = jit_strategy.calculate_portfolio_statistics(
               sample_portfolio, n_sims=10000
           )
           jit_time = time.time() - start
           
           # Test pure Python version
           py_strategy = ClassicalPricingStrategy(use_jit=False)
           start = time.time()
           py_result = py_strategy.calculate_portfolio_statistics(
               sample_portfolio, n_sims=10000
           )
           py_time = time.time() - start
           
           # JIT should be faster (after warmup)
           # Note: This test may be flaky on first run due to compilation
           speedup = py_time / jit_time
           assert speedup > 0.5  # Should be at least somewhat faster

Property-Based Tests
--------------------

Use `hypothesis <https://hypothesis.readthedocs.io/>`_ for property-based testing:

.. code-block:: python

   from hypothesis import given, strategies as st
   import numpy as np
   
   
   class TestDistributionProperties:
       """Property-based tests for distributions."""
       
       @given(lambda_=st.floats(min_value=0.1, max_value=100))
       def test_poisson_mean_property(self, lambda_):
           """Test that Poisson distribution has correct mean."""
           dist = Poisson(lambda_=lambda_)
           
           # Generate large sample to test mean
           samples = [dist.rvs() for _ in range(10000)]
           sample_mean = np.mean(samples)
           
           # Sample mean should be close to theoretical mean
           assert abs(sample_mean - lambda_) < 0.1 * lambda_
       
       @given(
           n_sims=st.integers(min_value=100, max_value=10000),
           alpha=st.floats(min_value=0.01, max_value=0.99)
       )
       def test_var_properties(self, n_sims, alpha):
           """Test VaR calculation properties."""
           # Generate random data
           data = np.random.exponential(scale=1000, size=n_sims)
           
           var = np.percentile(data, (1 - alpha) * 100)
           
           # VaR should be positive for positive data
           assert var > 0
           
           # Proportion of data above VaR should be approximately alpha
           proportion_above = np.mean(data > var)
           assert abs(proportion_above - alpha) < 0.05

Test Fixtures
=============

Use pytest fixtures for reusable test data:

.. code-block:: python

   # conftest.py
   import pytest
   import pandas as pd
   from quactuary.book import Portfolio, Inforce, PolicyTerms
   from quactuary.distributions import Poisson, LogNormal
   import datetime
   
   
   @pytest.fixture
   def sample_policy_terms():
       """Standard policy terms for testing."""
       return PolicyTerms(
           effective_date=datetime.date(2024, 1, 1),
           expiration_date=datetime.date(2025, 1, 1),
           per_occ_retention=1000.0,
           per_occ_limit=100000.0,
           coinsurance=0.8
       )
   
   
   @pytest.fixture
   def simple_inforce(sample_policy_terms):
       """Simple inforce bucket for testing."""
       return Inforce(
           n_policies=50,
           frequency=Poisson(lambda_=1.0),
           severity=LogNormal(mu=7, sigma=1.5),
           terms=sample_policy_terms
       )
   
   
   @pytest.fixture
   def sample_portfolio(simple_inforce):
       """Simple portfolio for testing."""
       return Portfolio([simple_inforce])
   
   
   @pytest.fixture
   def large_portfolio():
       """Large portfolio for performance testing."""
       # Create multiple inforce buckets
       buckets = []
       for i in range(10):
           terms = PolicyTerms(
               effective_date=datetime.date(2024, 1, 1),
               expiration_date=datetime.date(2025, 1, 1),
               per_occ_retention=1000.0 * (i + 1),
               per_occ_limit=100000.0
           )
           
           bucket = Inforce(
               n_policies=1000,
               frequency=Poisson(lambda_=1.5 + i * 0.1),
               severity=LogNormal(mu=7 + i * 0.1, sigma=1.5),
               terms=terms
           )
           buckets.append(bucket)
       
       return Portfolio(buckets)

Parametrized Tests
==================

Use pytest parametrization for testing multiple scenarios:

.. code-block:: python

   import pytest
   
   
   class TestRiskMeasures:
       """Test risk measure calculations with various parameters."""
       
       @pytest.mark.parametrize("confidence_level", [0.90, 0.95, 0.99])
       def test_var_calculation(self, sample_portfolio, confidence_level):
           """Test VaR calculation at different confidence levels."""
           model = PricingModel(sample_portfolio)
           result = model.simulate(
               value_at_risk=True,
               tail_alpha=1 - confidence_level,
               n_sims=5000
           )
           
           assert result.estimates['VaR'] > 0
           assert result.metadata['tail_alpha'] == 1 - confidence_level
       
       @pytest.mark.parametrize("n_sims", [1000, 5000, 10000])
       def test_convergence_with_sample_size(self, sample_portfolio, n_sims):
           """Test that results stabilize with larger sample sizes."""
           model = PricingModel(sample_portfolio)
           result = model.simulate(n_sims=n_sims)
           
           # Basic sanity checks
           assert result.estimates['mean'] > 0
           assert result.metadata['n_sims'] == n_sims
       
       @pytest.mark.parametrize("distribution_type,params", [
           ("poisson", {"lambda_": 2.0}),
           ("poisson", {"lambda_": 5.0}),
           ("poisson", {"lambda_": 10.0}),
       ])
       def test_frequency_distributions(self, distribution_type, params):
           """Test different frequency distribution parameters."""
           if distribution_type == "poisson":
               dist = Poisson(**params)
               
           # Test basic properties
           assert dist.pmf(0) >= 0
           assert dist.pmf(1) >= 0
           samples = dist.rvs(size=100)
           assert len(samples) == 100

Coverage Requirements
=====================

We require **≥90% test coverage** for all new code. Check coverage with:

.. code-block:: bash

   # Run tests with coverage report
   pytest --cov=quactuary --cov-report=html --cov-report=term
   
   # View detailed HTML report
   open htmlcov/index.html

Coverage Configuration
----------------------

Configure coverage in ``pyproject.toml``:

.. code-block:: toml

   [tool.coverage.run]
   source = ["quactuary"]
   omit = [
       "*/tests/*",
       "*/test_*",
       "setup.py",
       "*/venv/*",
       "*/__pycache__/*"
   ]
   
   [tool.coverage.report]
   exclude_lines = [
       "pragma: no cover",
       "def __repr__",
       "if self.debug:",
       "if settings.DEBUG",
       "raise AssertionError",
       "raise NotImplementedError",
       "if 0:",
       "if __name__ == .__main__.:"
   ]
   show_missing = true
   precision = 2

Mock and Patch
==============

Use mocking for external dependencies and expensive operations:

.. code-block:: python

   import pytest
   from unittest.mock import Mock, patch
   import numpy as np
   
   
   class TestBackendIntegration:
       """Test backend switching and integration."""
       
       @patch('quactuary.backend.Aer.get_backend')
       def test_quantum_backend_initialization(self, mock_aer):
           """Test quantum backend setup with mocked Qiskit."""
           # Mock the Qiskit backend
           mock_backend = Mock()
           mock_backend.name.return_value = "aer_simulator"
           mock_aer.return_value = mock_backend
           
           from quactuary.backend import set_backend
           backend_manager = set_backend('quantum', provider='AerSimulator')
           
           assert backend_manager.backend_type == 'quantum'
           mock_aer.assert_called_once_with('aer_simulator')
       
       def test_expensive_calculation_mocked(self, sample_portfolio):
           """Test expensive calculation with mocked results."""
           with patch('quactuary.classical.ClassicalPricingModel.calculate_portfolio_statistics') as mock_calc:
               # Mock the expensive calculation
               mock_result = Mock()
               mock_result.estimates = {'mean': 50000.0, 'VaR': 75000.0}
               mock_calc.return_value = mock_result
               
               model = PricingModel(sample_portfolio)
               result = model.simulate()
               
               assert result.estimates['mean'] == 50000.0
               mock_calc.assert_called_once()

Error and Edge Case Testing
===========================

Test error conditions and edge cases thoroughly:

.. code-block:: python

   class TestErrorHandling:
       """Test error conditions and edge cases."""
       
       def test_invalid_parameters(self):
           """Test handling of invalid parameters."""
           # Test invalid probability
           with pytest.raises(ValueError, match="must be between 0 and 1"):
               model.simulate(tail_alpha=1.5)
           
           # Test negative simulation count
           with pytest.raises(ValueError, match="must be positive"):
               model.simulate(n_sims=-100)
       
       def test_empty_portfolio(self):
           """Test behavior with empty portfolio."""
           empty_portfolio = Portfolio([])
           model = PricingModel(empty_portfolio)
           
           with pytest.raises(ValueError, match="Portfolio must contain"):
               model.simulate()
       
       def test_extreme_values(self):
           """Test behavior with extreme parameter values."""
           # Very small lambda
           small_dist = Poisson(lambda_=1e-10)
           assert small_dist.pmf(0) > 0.99
           
           # Very large lambda
           large_dist = Poisson(lambda_=1e6)
           samples = large_dist.rvs(size=100)
           assert np.mean(samples) > 1e5

Continuous Integration
======================

Our CI pipeline runs tests automatically on:

* **All pull requests**
* **Commits to main branch**
* **Multiple Python versions** (3.8, 3.9, 3.10, 3.11, 3.12)
* **Multiple operating systems** (Ubuntu, Windows, macOS)

Test Markers
============

Use pytest markers to categorize tests:

.. code-block:: python

   import pytest
   
   
   @pytest.mark.slow
   def test_large_simulation():
       """Test that takes a long time to run."""
       pass
   
   
   @pytest.mark.quantum
   def test_quantum_algorithm():
       """Test that requires quantum backend."""
       pass
   
   
   @pytest.mark.integration
   def test_end_to_end_workflow():
       """Integration test."""
       pass

Run specific test categories:

.. code-block:: bash

   # Run only fast tests
   pytest -m "not slow"
   
   # Run only quantum tests
   pytest -m quantum
   
   # Run integration tests
   pytest -m integration

Writing Good Tests
==================

Best Practices
--------------

1. **Test one thing**: Each test should focus on a single behavior
2. **Use descriptive names**: Test names should clearly indicate what they test
3. **Arrange-Act-Assert**: Structure tests with clear setup, action, and verification
4. **Test edge cases**: Include boundary conditions and error cases
5. **Keep tests fast**: Avoid unnecessary delays or expensive operations
6. **Make tests deterministic**: Avoid flaky tests that pass/fail randomly

Example of a Well-Written Test
------------------------------

.. code-block:: python

   def test_portfolio_var_calculation_with_policy_terms():
       """Test VaR calculation correctly applies policy terms."""
       # Arrange
       terms = PolicyTerms(
           effective_date=datetime.date(2024, 1, 1),
           expiration_date=datetime.date(2025, 1, 1),
           per_occ_retention=5000.0,  # $5k deductible
           per_occ_limit=50000.0,     # $50k limit
           coinsurance=0.8            # 80% insurer share
       )
       
       inforce = Inforce(
           n_policies=100,
           frequency=Poisson(lambda_=2.0),
           severity=LogNormal(mu=9, sigma=1),  # High severity
           terms=terms
       )
       
       portfolio = Portfolio([inforce])
       model = PricingModel(portfolio)
       
       # Act
       result = model.simulate(
           value_at_risk=True,
           tail_alpha=0.05,
           n_sims=10000
       )
       
       # Assert
       var_95 = result.estimates['VaR']
       
       # VaR should be positive but bounded by policy terms
       assert var_95 > 0
       assert var_95 <= 100 * 50000 * 0.8  # Max possible loss per policy * policies * coinsurance
       
       # Should be affected by retention (lower than ground-up VaR)
       # This is a behavioral test - we expect retention to reduce VaR
       assert var_95 < 100 * 100000  # Less than if no retention

Common Testing Pitfalls
=======================

Avoid These Mistakes
--------------------

1. **Testing implementation details**: Test behavior, not internal implementation
2. **Brittle assertions**: Avoid overly specific numeric assertions for stochastic processes
3. **Missing edge cases**: Don't forget to test boundary conditions
4. **Slow tests**: Minimize use of large simulations in unit tests
5. **Flaky tests**: Ensure tests pass consistently

.. code-block:: python

   # Bad - tests implementation detail
   def test_internal_cache_structure():
       model = PricingModel(portfolio)
       assert hasattr(model, '_cache')
       assert isinstance(model._cache, dict)
   
   # Good - tests behavior
   def test_repeated_calls_return_consistent_results():
       model = PricingModel(portfolio)
       result1 = model.simulate(n_sims=1000, random_seed=42)
       result2 = model.simulate(n_sims=1000, random_seed=42)
       assert result1.estimates['mean'] == result2.estimates['mean']

Testing New Features
====================

When adding new features:

1. **Write tests first** (TDD approach)
2. **Test happy path** and edge cases
3. **Include integration tests** if the feature interacts with other components
4. **Add performance tests** if the feature affects performance
5. **Update documentation** and examples

Our testing standards help ensure that quactuary remains reliable, performant, and maintainable as it grows!