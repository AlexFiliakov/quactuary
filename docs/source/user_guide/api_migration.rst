API Migration Guide
===================

This guide helps you migrate from older versions of quActuary to the current API (v0.2.0+).

.. important::

   Version 0.2.0 introduced several breaking changes to improve API consistency and usability.
   This guide covers all major changes and provides migration examples.

Distribution Parameter Changes
------------------------------

Location Parameter Standardization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Change**: All distributions now use ``loc`` for location parameters.

.. code-block:: python

   # Old API (pre-0.2.0)
   lognormal = Lognormal(shape=1.5, location=0.0, scale=1.0)  # DEPRECATED
   pareto = Pareto(b=2.0, threshold=1000.0, scale=1.0)        # DEPRECATED
   
   # New API (0.2.0+)
   lognormal = Lognormal(shape=1.5, loc=0.0, scale=1.0)       # CORRECT
   pareto = Pareto(b=2.0, loc=1000.0, scale=1.0)             # CORRECT

**Affected Distributions**:

- ``Lognormal``: ``location`` → ``loc``
- ``Pareto``: ``threshold`` → ``loc``
- All other severity distributions: Already used ``loc``

Optimization API Changes
------------------------

OptimizationConfig Introduction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Change**: Individual optimization parameters replaced with ``OptimizationConfig`` class.

.. code-block:: python

   # Old API (pre-0.2.0)
   result = model.simulate(
       n_sims=100000,
       use_jit=True,           # DEPRECATED
       use_parallel=True,      # DEPRECATED
       n_workers=4,           # DEPRECATED
       use_qmc=True           # DEPRECATED
   )
   
   # New API (0.2.0+)
   from quactuary.optimization_selector import OptimizationConfig
   
   config = OptimizationConfig(
       use_jit=True,
       use_parallel=True,
       n_workers=4,
       use_qmc=True
   )
   
   result = model.simulate(
       n_sims=100000,
       optimization_config=config  # CORRECT
   )

Auto-Optimization
~~~~~~~~~~~~~~~~~

**New Feature**: Automatic optimization selection based on data size and system resources.

.. code-block:: python

   # New in 0.2.0
   result = model.simulate(
       n_sims=100000,
       auto_optimize=True  # Let system choose optimal settings
   )

Testing Tolerance Changes
-------------------------

Stochastic Method Tolerances
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Change**: Recommended tolerances for stochastic tests have been relaxed.

.. code-block:: python

   # Old approach (often failed randomly)
   assert result.mean == pytest.approx(expected, rel=0.001)  # Too strict
   
   # New approach (stable tests)
   assert result.mean == pytest.approx(expected, rel=0.05)   # 5% for small samples
   assert result.mean == pytest.approx(expected, rel=0.02)   # 2% for large samples

Hardware-Dependent Test Handling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Change**: Use ``pytest.mark.skip`` for tests that depend on specific hardware.

.. code-block:: python

   # New approach
   @pytest.mark.skip(reason="Requires stable baseline from CI environment")
   def test_performance_regression():
       # Performance-sensitive test
       pass

Complete Migration Examples
---------------------------

Example 1: Basic Portfolio Pricing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Old Code**:

.. code-block:: python

   from quactuary.distributions import Poisson, Lognormal
   from quactuary.pricing import PricingModel
   
   # Create distributions (old parameter names)
   freq = Poisson(5.0)  # Positional argument
   sev = Lognormal(1.5, 0.0, 1000.0)  # Positional arguments
   
   # Run simulation (old optimization API)
   model = PricingModel(portfolio)
   result = model.simulate(
       n_sims=10000,
       use_jit=True,
       use_parallel=True
   )

**New Code**:

.. code-block:: python

   from quactuary.distributions import Poisson, Lognormal
   from quactuary.pricing import PricingModel
   from quactuary.optimization_selector import OptimizationConfig
   
   # Create distributions (explicit parameter names)
   freq = Poisson(mu=5.0)
   sev = Lognormal(shape=1.5, loc=0.0, scale=1000.0)
   
   # Run simulation (new optimization API)
   model = PricingModel(portfolio)
   
   # Option 1: Manual configuration
   config = OptimizationConfig(use_jit=True, use_parallel=True)
   result = model.simulate(n_sims=10000, optimization_config=config)
   
   # Option 2: Auto-optimization (recommended)
   result = model.simulate(n_sims=10000, auto_optimize=True)

Example 2: Complex Distribution Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Old Code**:

.. code-block:: python

   # Mixed parameter styles
   pareto = Pareto(2.0, 1000.0, 1.0)  # Positional
   beta = Beta(a=2, b=3, location=0, scale=1)  # Mixed named
   uniform = ContinuousUniformSeverity(0, 1000)  # Min/max style

**New Code**:

.. code-block:: python

   # Consistent named parameters
   pareto = Pareto(b=2.0, loc=1000.0, scale=1.0)
   beta = Beta(a=2, b=3, loc=0, scale=1)
   uniform = ContinuousUniformSeverity(loc=0, scale=1000)  # scale is width

Example 3: Test Migration
~~~~~~~~~~~~~~~~~~~~~~~~~

**Old Test**:

.. code-block:: python

   def test_simulation_mean():
       result = model.simulate(n_sims=1000)
       assert abs(result.mean - 1000) < 1  # Absolute tolerance

**New Test**:

.. code-block:: python

   import pytest
   import numpy as np
   
   def test_simulation_mean():
       np.random.seed(42)  # Ensure reproducibility
       result = model.simulate(n_sims=10000)  # Larger sample
       assert result.mean == pytest.approx(1000, rel=0.05)  # Relative tolerance

Automated Migration Script
--------------------------

For large codebases, use this script to help identify areas needing updates:

.. code-block:: python

   #!/usr/bin/env python
   """
   Migration helper script for quActuary 0.2.0
   Identifies potential API usage that needs updating.
   """
   import re
   import glob
   import sys
   
   def check_file(filename):
       """Check a file for old API patterns."""
       issues = []
       
       with open(filename, 'r') as f:
           content = f.read()
           lines = content.split('\n')
       
       # Check for old parameter names
       patterns = [
           (r'location\s*=', 'Use "loc" instead of "location"'),
           (r'threshold\s*=', 'Use "loc" instead of "threshold"'),
           (r'\.simulate\([^)]*use_jit\s*=', 'Use OptimizationConfig instead'),
           (r'\.simulate\([^)]*use_parallel\s*=', 'Use OptimizationConfig instead'),
           (r'Lognormal\s*\(\s*[\d.]+\s*,\s*[\d.]+\s*,', 'Use named parameters'),
           (r'Pareto\s*\(\s*[\d.]+\s*,\s*[\d.]+\s*,', 'Use named parameters'),
       ]
       
       for i, line in enumerate(lines, 1):
           for pattern, message in patterns:
               if re.search(pattern, line):
                   issues.append(f"{filename}:{i}: {message}")
       
       return issues
   
   def main():
       # Find all Python files
       files = glob.glob('**/*.py', recursive=True)
       files.extend(glob.glob('**/*.ipynb', recursive=True))
       
       all_issues = []
       for file in files:
           issues = check_file(file)
           all_issues.extend(issues)
       
       if all_issues:
           print("Migration issues found:")
           for issue in all_issues:
               print(f"  {issue}")
           sys.exit(1)
       else:
           print("No migration issues found!")
   
   if __name__ == '__main__':
       main()

Troubleshooting
---------------

Common Migration Issues
~~~~~~~~~~~~~~~~~~~~~~~

**Issue**: ``TypeError: __init__() got an unexpected keyword argument 'location'``

**Solution**: Change ``location`` to ``loc`` in distribution initialization.

**Issue**: ``TypeError: simulate() got an unexpected keyword argument 'use_jit'``

**Solution**: Create an ``OptimizationConfig`` object with your settings.

**Issue**: Tests failing with ``AssertionError`` on numerical comparisons

**Solution**: Update to use ``pytest.approx()`` with appropriate tolerances.

Getting Help
~~~~~~~~~~~~

If you encounter issues not covered in this guide:

1. Check the :doc:`/api_reference/index` for current parameter names
2. Review the :doc:`/user_guide/distribution_parameters` for distribution specifics
3. See :doc:`/user_guide/optimization_config` for optimization details
4. File an issue on `GitHub <https://github.com/quactuary/quactuary/issues>`_

Version History
---------------

**v0.2.0** (2025-05-26)

- Standardized distribution location parameters to ``loc``
- Introduced ``OptimizationConfig`` for unified optimization control
- Added ``auto_optimize`` feature for automatic optimization selection
- Updated test tolerance recommendations
- Improved API consistency across all modules

**v0.1.x**

- Initial release with mixed parameter naming conventions
- Individual optimization parameters in ``simulate()`` method