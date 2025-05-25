.. _code_standards:

**************
Code Standards
**************

This document outlines the coding standards and style guidelines for quactuary. Following these standards helps ensure code quality, readability, and maintainability across the project.

.. contents:: Table of Contents
   :local:
   :depth: 2

Style Guide
===========

We follow the `PEP 8 Style Guide for Python Code <https://peps.python.org/pep-0008/>`_ with some project-specific additions and clarifications.

Code Formatting
================

Black Formatter
---------------

We use `Black <https://black.readthedocs.io/>`_ for automatic code formatting:

.. code-block:: bash

   # Format all Python files
   black .
   
   # Check formatting without changes
   black --check .
   
   # Format specific file
   black quactuary/pricing.py

**Black configuration** (in ``pyproject.toml``):

.. code-block:: toml

   [tool.black]
   line-length = 88
   target-version = ['py38']
   include = '\.pyi?$'
   extend-exclude = '''
   /(
       \.eggs
     | \.git
     | \.venv
     | build
     | dist
   )/
   '''

Line Length
-----------

* **Maximum line length**: 88 characters (Black default)
* **Docstring line length**: 79 characters for readability
* **Comments**: 72 characters for inline comments

Import Organization
===================

Organize imports in the following order:

1. **Standard library imports**
2. **Third-party imports** 
3. **Local application imports**

Use `isort <https://pycqa.github.io/isort/>`_ for automatic import sorting:

.. code-block:: bash

   # Sort imports
   isort .
   
   # Check import order
   isort --check-only .

**Example import organization:**

.. code-block:: python

   # Standard library
   import datetime
   import warnings
   from typing import Optional, Union, List
   
   # Third-party
   import numpy as np
   import pandas as pd
   from scipy import stats
   
   # Local imports
   from quactuary.backend import BackendManager
   from quactuary.distributions.frequency import FrequencyModel
   from quactuary.utils.validation import validate_positive

Naming Conventions
==================

Variables and Functions
-----------------------

* Use ``snake_case`` for variables and functions
* Use descriptive names that clearly indicate purpose
* Avoid abbreviations unless they're standard in the domain

.. code-block:: python

   # Good
   portfolio_mean_loss = calculate_portfolio_statistics()
   frequency_distribution = Poisson(lambda_=2.0)
   
   # Avoid
   pml = calc_stats()
   freq_dist = Poisson(lambda_=2.0)

Constants
---------

* Use ``UPPER_CASE`` for module-level constants
* Group related constants together

.. code-block:: python

   # Good
   DEFAULT_N_SIMULATIONS = 10000
   MAX_QUANTUM_QUBITS = 50
   SUPPORTED_BACKENDS = ['classical', 'quantum']

Classes
-------

* Use ``PascalCase`` for class names
* Choose names that clearly indicate the class purpose
* Prefer composition over inheritance

.. code-block:: python

   # Good
   class PricingModel:
       """Portfolio pricing and risk analysis model."""
   
   class ClassicalPricingStrategy:
       """Strategy for classical Monte Carlo calculations."""

Private Methods and Variables
-----------------------------

* Use single leading underscore for internal use
* Use double leading underscore only for name mangling when necessary

.. code-block:: python

   class ExampleClass:
       def __init__(self):
           self.public_attribute = "visible"
           self._internal_attribute = "internal use"
       
       def public_method(self):
           """Public interface method."""
           return self._internal_method()
       
       def _internal_method(self):
           """Internal helper method."""
           pass

Type Hints
==========

Use type hints for all public functions, methods, and class attributes:

.. code-block:: python

   from typing import Optional, Union, List, Dict, Any
   import numpy as np
   import pandas as pd
   
   def calculate_statistics(
       portfolio: Portfolio,
       n_sims: int = 10000,
       confidence_level: float = 0.95,
       include_samples: bool = False
   ) -> PricingResult:
       """Calculate portfolio risk statistics."""
       pass
   
   class PricingModel:
       """Portfolio pricing model."""
       
       def __init__(self, portfolio: Portfolio) -> None:
           self.portfolio: Portfolio = portfolio
           self.results: Optional[PricingResult] = None

**Type hint guidelines:**

* Always include return type annotations
* Use ``Optional[T]`` instead of ``Union[T, None]``
* Import types from ``typing`` module
* Use ``Any`` sparingly - prefer specific types

Documentation Standards
========================

Docstring Format
----------------

We use `Google docstring format <https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings>`_:

.. code-block:: python

   def calculate_portfolio_var(
       portfolio: Portfolio,
       confidence_level: float = 0.95,
       n_sims: int = 10000
   ) -> float:
       """
       Calculate Value at Risk for a portfolio using Monte Carlo simulation.
   
       This function estimates the VaR by simulating portfolio losses and
       computing the specified quantile. The calculation uses either classical
       Monte Carlo or quantum algorithms depending on the current backend.
   
       Args:
           portfolio (Portfolio): Portfolio containing policy information and
               loss distributions. Must have at least one inforce bucket.
           confidence_level (float): Confidence level for VaR calculation.
               Must be between 0 and 1. Default is 0.95 (95% VaR).
           n_sims (int): Number of Monte Carlo simulations to perform.
               Higher values improve accuracy but increase computation time.
               Default is 10,000.
   
       Returns:
           float: Value at Risk at the specified confidence level in the same
               currency units as the portfolio.
   
       Raises:
           ValueError: If confidence_level is not between 0 and 1.
           ValueError: If portfolio is empty or invalid.
           RuntimeError: If simulation fails to converge.
   
       Examples:
           Basic VaR calculation:
               >>> portfolio = Portfolio(policies_df)
               >>> var_95 = calculate_portfolio_var(portfolio, 0.95)
               >>> print(f"95% VaR: ${var_95:,.2f}")
   
           High-precision calculation:
               >>> var_99 = calculate_portfolio_var(
               ...     portfolio,
               ...     confidence_level=0.99,
               ...     n_sims=100000
               ... )
   
       Notes:
           - VaR represents the loss threshold that will not be exceeded
             with the specified probability
           - Uses the current global backend (classical or quantum)
           - For quantum calculations, n_sims may be adjusted internally
       """

**Docstring sections to include:**

* **Brief description**: One-line summary
* **Detailed description**: Multi-paragraph explanation if needed
* **Args**: All parameters with types and descriptions
* **Returns**: Return value type and description
* **Raises**: Exceptions that may be raised
* **Examples**: Code examples showing usage
* **Notes**: Additional important information

Class Documentation
-------------------

.. code-block:: python

   class PricingModel:
       """
       Portfolio pricing and risk analysis model.
   
       This class provides a unified interface for calculating portfolio risk
       measures using various computational backends. It supports both classical
       Monte Carlo simulation and quantum-accelerated algorithms.
   
       The model uses a strategy pattern to delegate calculations to different
       implementations while maintaining a consistent interface. This allows
       for runtime switching between computational approaches.
   
       Attributes:
           portfolio (Portfolio): The portfolio being analyzed.
           strategy (PricingStrategy): Current calculation strategy.
           backend_type (str): Type of computational backend in use.
   
       Examples:
           Basic usage:
               >>> portfolio = Portfolio(policies_df)
               >>> model = PricingModel(portfolio)
               >>> result = model.simulate(n_sims=10000)
   
           With custom strategy:
               >>> strategy = ClassicalPricingStrategy(use_jit=True)
               >>> model = PricingModel(portfolio, strategy=strategy)
   
       Notes:
           - Strategies can be swapped at runtime
           - The model is stateless by design for thread safety
           - All monetary values should be in consistent units
       """

Comments
========

Inline Comments
---------------

Use comments sparingly - prefer self-documenting code:

.. code-block:: python

   # Good - explains non-obvious business logic
   tail_alpha = 1 - confidence_level  # VaR uses tail probability
   
   # Good - explains complex calculation
   # Apply Owen scrambling to improve QMC uniformity
   scrambled_points = sobol_generator.random_scrambled()
   
   # Avoid - states the obvious
   n_sims = 10000  # Set number of simulations to 10000

TODO Comments
-------------

.. code-block:: python

   # TODO(username): Add support for conditional VaR calculation
   # FIXME: This approximation fails for heavy-tailed distributions
   # NOTE: This uses a simplified approach - see issue #123

Error Handling
==============

Exception Handling
------------------

* Use specific exception types
* Provide helpful error messages
* Include context in error messages

.. code-block:: python

   def validate_portfolio(portfolio: Portfolio) -> None:
       """Validate portfolio before calculations."""
       if not portfolio.policies:
           raise ValueError(
               "Portfolio must contain at least one policy. "
               "Received empty portfolio."
           )
       
       if portfolio.total_exposure <= 0:
           raise ValueError(
               f"Portfolio exposure must be positive. "
               f"Got {portfolio.total_exposure}"
           )

Custom Exceptions
-----------------

Create specific exception types for domain errors:

.. code-block:: python

   class QuactuaryError(Exception):
       """Base exception for quactuary package."""
       pass
   
   class CalculationError(QuactuaryError):
       """Raised when numerical calculations fail."""
       pass
   
   class BackendError(QuactuaryError):
       """Raised when backend operations fail."""
       pass

Performance Considerations
==========================

Efficient Code Patterns
------------------------

.. code-block:: python

   # Good - use NumPy operations
   losses = np.sum(frequency_samples * severity_samples, axis=1)
   
   # Avoid - Python loops for large arrays
   losses = []
   for i in range(len(frequency_samples)):
       losses.append(frequency_samples[i] * severity_samples[i])

Memory Management
-----------------

.. code-block:: python

   # Good - process in chunks for large datasets
   def process_large_portfolio(portfolio: Portfolio, chunk_size: int = 10000):
       for chunk in portfolio.iter_chunks(chunk_size):
           yield process_chunk(chunk)
   
   # Good - use generators for large sequences
   def simulate_losses(n_sims: int):
       for i in range(n_sims):
           yield simulate_single_loss()

Testing Code Quality
====================

Static Analysis
---------------

Use these tools to check code quality:

.. code-block:: bash

   # Style checking
   flake8 quactuary/
   
   # Type checking
   mypy quactuary/
   
   # Security scanning
   bandit -r quactuary/
   
   # Complexity analysis
   radon cc quactuary/ -a

Configuration files for these tools should be in ``pyproject.toml`` or dedicated config files.

Pre-commit Hooks
----------------

Set up pre-commit hooks to automatically check code quality:

.. code-block:: bash

   # Install hooks
   pre-commit install
   
   # Run on all files
   pre-commit run --all-files

Git Commit Standards
====================

Commit Message Format
---------------------

.. code-block:: text

   type(scope): brief description
   
   Longer description explaining what changed and why.
   Include any breaking changes or migration notes.
   
   Closes #123
   Fixes #456

**Types:**
* ``feat``: New feature
* ``fix``: Bug fix
* ``docs``: Documentation changes
* ``style``: Code style changes (no logic changes)
* ``refactor``: Code refactoring
* ``test``: Adding or updating tests
* ``perf``: Performance improvements

**Examples:**

.. code-block:: text

   feat(quantum): add quantum amplitude estimation for VaR
   
   Implement QAE algorithm for Value at Risk calculations, providing
   quadratic speedup over classical Monte Carlo for certain problem
   structures. Includes comprehensive test suite and benchmarks.
   
   Closes #123

.. code-block:: text

   fix(pricing): handle edge case in compound distribution sampling
   
   Fix issue where extremely small frequency parameters caused
   numerical instability in Poisson sampling. Add validation
   and improved error messages.
   
   Fixes #456

Code Review Guidelines
======================

For Reviewers
-------------

* **Focus on correctness**: Does the code do what it's supposed to do?
* **Check performance**: Are there obvious performance issues?
* **Verify tests**: Is the code adequately tested?
* **Review documentation**: Are docstrings clear and complete?
* **Consider maintainability**: Is the code easy to understand and modify?

For Authors
-----------

* **Keep PRs focused**: One feature or fix per PR
* **Write clear descriptions**: Explain what changed and why
* **Include tests**: All new code should have tests
* **Update documentation**: Keep docs in sync with code changes
* **Be responsive**: Address review feedback promptly

Common Code Patterns
=====================

Error Checking
--------------

.. code-block:: python

   def calculate_risk_measure(data: np.ndarray, alpha: float) -> float:
       """Calculate risk measure with proper validation."""
       # Input validation
       if not isinstance(data, np.ndarray):
           data = np.asarray(data)
       
       if len(data) == 0:
           raise ValueError("Data array cannot be empty")
       
       if not 0 < alpha < 1:
           raise ValueError(f"Alpha must be between 0 and 1, got {alpha}")
       
       # Calculation
       return np.percentile(data, (1 - alpha) * 100)

Caching Results
---------------

.. code-block:: python

   class ExpensiveCalculation:
       def __init__(self):
           self._cache: Dict[str, Any] = {}
       
       def expensive_method(self, param: float) -> float:
           """Method with caching for expensive calculations."""
           cache_key = f"expensive_{param}"
           
           if cache_key not in self._cache:
               # Perform expensive calculation
               result = self._do_expensive_calculation(param)
               self._cache[cache_key] = result
           
           return self._cache[cache_key]

Backward Compatibility
======================

When making changes:

* **Deprecate before removing**: Use ``warnings.warn`` for deprecated features
* **Maintain API compatibility**: Avoid breaking changes in minor versions
* **Document breaking changes**: Include migration notes in release notes

.. code-block:: python

   import warnings
   
   def old_function_name(*args, **kwargs):
       """Deprecated function - use new_function_name instead."""
       warnings.warn(
           "old_function_name is deprecated and will be removed in v2.0. "
           "Use new_function_name instead.",
           DeprecationWarning,
           stacklevel=2
       )
       return new_function_name(*args, **kwargs)

Following these standards helps ensure that quactuary remains a high-quality, maintainable codebase that's enjoyable for everyone to work with!