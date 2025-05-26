.. _documentation_guidelines:

***********************
Documentation Guidelines
***********************

Good documentation is essential for making quactuary accessible and useful to both users and contributors. This guide covers our documentation standards, tools, and best practices.

.. contents:: Table of Contents
   :local:
   :depth: 2

Documentation Philosophy
========================

We believe documentation should be:

* **User-focused**: Written from the user's perspective with clear use cases
* **Example-driven**: Include practical examples that users can run
* **Comprehensive**: Cover both basic usage and advanced scenarios
* **Maintainable**: Easy to keep in sync with code changes
* **Accessible**: Clear language that doesn't assume deep domain knowledge

Types of Documentation
======================

We maintain several types of documentation:

**API Documentation**
  Docstrings in code that document individual functions, classes, and modules

**User Guide**
  Tutorial-style documentation that walks users through common tasks

**Examples**
  Jupyter notebooks demonstrating real-world usage scenarios

**Developer Documentation**
  This section - guides for contributors and maintainers

**Reference Documentation**
  Comprehensive API reference generated from docstrings

Docstring Standards
===================

We use `Google docstring format <https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings>`_ for all code documentation.

Basic Function Docstring
-------------------------

.. code-block:: python

   def calculate_var(data: np.ndarray, alpha: float = 0.05) -> float:
       """
       Calculate Value at Risk from loss data.
   
       Value at Risk (VaR) represents the loss threshold that will not be 
       exceeded with a given probability. This function computes VaR using
       the empirical quantile method.
   
       Args:
           data (np.ndarray): Array of loss values. Must contain at least
               one element and all values should be non-negative.
           alpha (float): Tail probability for VaR calculation. Must be
               between 0 and 1. Default is 0.05 (95% VaR).
   
       Returns:
           float: Value at Risk at the specified confidence level.
   
       Raises:
           ValueError: If data is empty or alpha is not in (0, 1).
           TypeError: If data cannot be converted to numeric array.
   
       Examples:
           Calculate 95% VaR from simulated losses:
               >>> losses = np.random.exponential(1000, 10000)
               >>> var_95 = calculate_var(losses, alpha=0.05)
               >>> print(f"95% VaR: ${var_95:.2f}")
   
           Calculate 99% VaR:
               >>> var_99 = calculate_var(losses, alpha=0.01)
               >>> print(f"99% VaR: ${var_99:.2f}")
   
       Notes:
           - VaR is a quantile-based risk measure
           - For heavy-tailed distributions, consider using TVaR instead
           - Results depend on the empirical distribution of input data
       """

Class Docstring
---------------

.. code-block:: python

   class PricingModel:
       """
       Portfolio pricing and risk analysis model.
   
       This class provides a unified interface for calculating portfolio risk
       measures using various computational backends. It supports classical
       Monte Carlo simulation and experimental quantum acceleration.
   
       The model uses the strategy pattern to delegate calculations to different
       implementations while maintaining a consistent API. This design allows
       for runtime backend switching and easy extension with new algorithms.
   
       Attributes:
           portfolio (Portfolio): The insurance portfolio being analyzed.
           strategy (PricingStrategy): Current calculation strategy (classical/quantum).
           backend_type (str): Type of computational backend ('classical' or 'quantum').
   
       Examples:
           Basic portfolio analysis:
               >>> from quactuary import PricingModel, Portfolio
               >>> portfolio = Portfolio.from_dataframe(policies_df)
               >>> model = PricingModel(portfolio)
               >>> result = model.simulate(n_sims=10000)
               >>> print(f"Expected loss: ${result.estimates['mean']:,.2f}")
   
           Using quantum backend:
               >>> import quactuary as qa
               >>> qa.set_backend('quantum', provider='AerSimulator')
               >>> result = model.simulate(n_sims=1000)  # Quantum calculation
   
           Advanced configuration:
               >>> from quactuary.pricing_strategies import ClassicalPricingStrategy
               >>> strategy = ClassicalPricingStrategy(use_jit=True)
               >>> model = PricingModel(portfolio, strategy=strategy)
   
       Notes:
           - Models are stateless and thread-safe
           - Backend switching affects all subsequent calculations
           - Quantum features are experimental and under development
       """

Module Docstring
-----------------

.. code-block:: python

   """
   Portfolio pricing and risk analysis module.
   
   This module provides the main interface for actuarial pricing calculations
   in the quactuary framework. It implements both classical Monte Carlo methods
   and experimental quantum algorithms for computing portfolio risk measures.
   
   Key Components:
       - PricingModel: Main interface for portfolio analysis
       - Risk measure calculations (VaR, TVaR, mean, variance)
       - Strategy pattern for different computational approaches
       - Integration with various probability distributions
   
   The module supports:
       - Classical Monte Carlo simulation
       - Quasi-Monte Carlo methods (Sobol, Halton sequences)
       - Quantum amplitude estimation (experimental)
       - JIT compilation for performance optimization
   
   Examples:
       Basic usage:
           >>> from quactuary.pricing import PricingModel
           >>> from quactuary.book import Portfolio
           >>> 
           >>> portfolio = Portfolio.from_csv("policies.csv")
           >>> model = PricingModel(portfolio)
           >>> result = model.simulate(n_sims=10000)
   
       Advanced risk analysis:
           >>> # Calculate multiple risk measures
           >>> result = model.simulate(
           ...     mean=True,
           ...     variance=True,
           ...     value_at_risk=True,
           ...     tail_value_at_risk=True,
           ...     tail_alpha=0.01,  # 99% confidence
           ...     n_sims=50000
           ... )
           >>> 
           >>> # Access results
           >>> print(f"Expected Loss: ${result.estimates['mean']:,.2f}")
           >>> print(f"99% VaR: ${result.estimates['VaR']:,.2f}")
           >>> print(f"99% TVaR: ${result.estimates['TVaR']:,.2f}")
   
   See Also:
       - :mod:`quactuary.classical`: Classical Monte Carlo implementations
       - :mod:`quactuary.quantum`: Quantum algorithm implementations
       - :mod:`quactuary.distributions`: Probability distributions
       - :mod:`quactuary.book`: Portfolio and policy modeling
   """

Docstring Elements
==================

Required Elements
-----------------

All public functions and classes must include:

* **Brief description**: One-line summary of purpose
* **Args section**: All parameters with types and descriptions
* **Returns section**: Return value type and description
* **Examples section**: At least one working example

Optional Elements
-----------------

Include when relevant:

* **Raises section**: Exceptions that may be raised
* **Notes section**: Additional important information
* **See Also section**: Links to related functions/classes
* **References section**: Academic papers or external resources

Mathematical Documentation
--------------------------

For mathematical functions, include formulas:

.. code-block:: python

   def tail_value_at_risk(data: np.ndarray, alpha: float = 0.05) -> float:
       """
       Calculate Tail Value at Risk (Conditional Value at Risk).
   
       TVaR is defined as the expected loss given that the loss exceeds VaR:
   
       .. math::
           \\text{TVaR}_{\\alpha} = E[X | X > \\text{VaR}_{\\alpha}]
   
       where VaR_α is the Value at Risk at confidence level (1-α).
   
       Args:
           data (np.ndarray): Array of loss values.
           alpha (float): Tail probability. Default is 0.05.
   
       Returns:
           float: Tail Value at Risk (also known as Expected Shortfall).
   
       Examples:
           >>> losses = np.array([100, 200, 500, 1000, 2000])
           >>> tvar = tail_value_at_risk(losses, alpha=0.2)
           >>> print(f"TVaR: {tvar}")
   
       References:
           Artzner, P., et al. (1999). Coherent measures of risk.
           Mathematical Finance, 9(3), 203-228.
       """

Examples in Docstrings
======================

Good Examples
-------------

Examples should be:

* **Runnable**: Users should be able to copy and execute them
* **Realistic**: Use realistic parameter values and data
* **Progressive**: Start simple, then show more advanced usage
* **Complete**: Include necessary imports and setup

.. code-block:: python

   def create_compound_distribution(frequency, severity):
       """
       Create a compound distribution from frequency and severity models.
   
       Examples:
           Basic compound Poisson-LogNormal:
               >>> from quactuary.distributions import Poisson, LogNormal
               >>> freq = Poisson(lambda_=50)  # 50 claims per year
               >>> sev = LogNormal(mu=8, sigma=1.5)  # ~$3k average severity
               >>> compound = create_compound_distribution(freq, sev)
               >>> print(f"Expected annual loss: ${compound.mean():,.2f}")
   
           Heavy-tailed severity distribution:
               >>> from quactuary.distributions import NegativeBinomial, Pareto
               >>> freq = NegativeBinomial(n=10, p=0.8)
               >>> sev = Pareto(alpha=1.5, scale=1000)  # Heavy tail
               >>> compound = create_compound_distribution(freq, sev)
               >>> 
               >>> # Calculate risk measures
               >>> var_99 = compound.ppf(0.99)
               >>> print(f"99% VaR: ${var_99:,.2f}")
   
           Using empirical data:
               >>> import pandas as pd
               >>> claims_data = pd.read_csv("historical_claims.csv")
               >>> freq = Empirical(claims_data['claim_counts'])
               >>> sev = LogNormal.fit(claims_data['claim_amounts'])
               >>> compound = create_compound_distribution(freq, sev)
       """

Bad Examples
------------

Avoid examples that:

* **Don't run**: Missing imports or undefined variables
* **Are trivial**: ``>>> result = function(1, 2)``
* **Are unrealistic**: Using toy data that doesn't reflect real usage
* **Are incomplete**: Not showing how to use the results

User Guide Documentation
========================

Structure
---------

User guides should follow this structure:

1. **Overview**: What the guide covers and prerequisites
2. **Setup**: Any necessary configuration or imports
3. **Basic Usage**: Simple examples to get started
4. **Common Patterns**: Typical use cases with explanations
5. **Advanced Topics**: More complex scenarios
6. **Troubleshooting**: Common issues and solutions

Example User Guide Section
---------------------------

.. code-block:: rst

   Getting Started with Portfolio Pricing
   ======================================
   
   This guide shows you how to price an insurance portfolio using quactuary.
   We'll start with basic risk measures and progress to advanced techniques.
   
   Prerequisites
   -------------
   
   * Basic understanding of insurance concepts (VaR, deductibles, limits)
   * Familiarity with Python and pandas
   * quactuary installed with: ``pip install quactuary``
   
   Basic Portfolio Analysis
   ------------------------
   
   Let's start by creating a simple portfolio and calculating risk measures:
   
   .. code-block:: python
   
      import quactuary as qa
      import pandas as pd
      import numpy as np
      
      # Create sample policy data
      policies = pd.DataFrame({
          'policy_id': range(100),
          'premium': np.random.normal(5000, 1000, 100),
          'deductible': np.random.choice([1000, 2500, 5000], 100),
          'limit': np.random.choice([100000, 250000, 500000], 100)
      })
      
      # Build portfolio
      portfolio = qa.Portfolio.from_dataframe(policies)
      
      # Create pricing model
      model = qa.PricingModel(portfolio)
      
      # Calculate risk measures
      result = model.simulate(n_sims=10000)
      
      # Display results
      print(f"Expected Loss: ${result.estimates['mean']:,.2f}")
      print(f"95% VaR: ${result.estimates['VaR']:,.2f}")
      print(f"95% TVaR: ${result.estimates['TVaR']:,.2f}")
   
   This example creates a portfolio of 100 policies with varying terms and
   calculates key risk measures using Monte Carlo simulation.

Jupyter Notebook Documentation
==============================

Notebook Guidelines
-------------------

For tutorial notebooks:

* **Clear narrative**: Tell a story that guides users through concepts
* **Runnable code**: All code cells should execute without errors
* **Visualizations**: Include plots and charts to illustrate results
* **Real data**: Use realistic datasets when possible
* **Checkpoints**: Break complex workflows into digestible sections

Example Notebook Structure
---------------------------

.. code-block:: text

   # Introduction to Portfolio Risk Analysis
   
   ## Overview
   This notebook demonstrates how to use quactuary for portfolio risk analysis...
   
   ## Setup
   ```python
   import quactuary as qa
   import pandas as pd
   import matplotlib.pyplot as plt
   ```
   
   ## Loading Data
   ```python
   # Load historical policy data
   policies = pd.read_csv("sample_portfolio.csv")
   policies.head()
   ```
   
   ## Creating the Portfolio
   ```python
   portfolio = qa.Portfolio.from_dataframe(policies)
   print(f"Portfolio contains {len(portfolio)} policies")
   ```
   
   ## Risk Analysis
   ```python
   model = qa.PricingModel(portfolio)
   result = model.simulate(n_sims=50000)
   ```
   
   ## Visualizing Results
   ```python
   plt.figure(figsize=(10, 6))
   plt.hist(result.samples, bins=50, alpha=0.7)
   plt.xlabel("Portfolio Loss ($)")
   plt.ylabel("Frequency")
   plt.title("Distribution of Portfolio Losses")
   plt.show()
   ```

API Reference Documentation
===========================

We use Sphinx with autodoc to generate API documentation from docstrings:

Configuration
-------------

In ``docs/source/conf.py``:

.. code-block:: python

   extensions = [
       'sphinx.ext.autodoc',
       'sphinx.ext.autosummary',
       'sphinx.ext.napoleon',  # For Google-style docstrings
       'sphinx.ext.viewcode',
       'sphinx.ext.intersphinx',
       'nbsphinx',  # For Jupyter notebooks
   ]
   
   # Napoleon settings for Google docstrings
   napoleon_google_docstring = True
   napoleon_numpy_docstring = False
   napoleon_include_init_with_doc = False
   napoleon_include_private_with_doc = False

Building Documentation
======================

Local Building
--------------

.. code-block:: bash

   # Navigate to docs directory
   cd docs/
   
   # Build HTML documentation
   make html
   
   # View documentation
   open build/html/index.html

Continuous Integration
----------------------

Documentation is built automatically on:

* Pull requests (to check for errors)
* Merges to main branch (published to docs site)
* Tagged releases (versioned documentation)

Documentation Standards
=======================

Writing Style
-------------

* **Active voice**: "Calculate the mean" not "The mean is calculated"
* **Present tense**: "Returns the result" not "Will return the result"
* **Clear language**: Avoid jargon, explain technical terms
* **Consistent terminology**: Use the same terms throughout

Formatting
----------

* Use **bold** for UI elements and important concepts
* Use ``code formatting`` for function names, parameters, and code snippets
* Use *italics* sparingly for emphasis
* Include blank lines for readability in long docstrings

Cross-References
----------------

Link to related documentation:

.. code-block:: python

   """
   Calculate portfolio statistics.
   
   See Also:
       :func:`calculate_var`: For VaR-only calculations
       :class:`Portfolio`: Portfolio data structure
       :mod:`quactuary.distributions`: Available distributions
   """

External Links
--------------

.. code-block:: python

   """
   Implement quantum amplitude estimation algorithm.
   
   References:
       Brassard, G., et al. (2002). Quantum amplitude amplification and estimation.
       Contemporary Mathematics, 305, 53-74.
       https://arxiv.org/abs/quant-ph/0005055
   """

Documentation Maintenance
=========================

Keeping Docs Updated
--------------------

* **Update with code changes**: Modify docs when changing APIs
* **Version compatibility**: Note version requirements for features
* **Deprecation warnings**: Document deprecated features clearly
* **Migration guides**: Help users adapt to breaking changes

Review Process
--------------

Documentation changes should be reviewed for:

* **Accuracy**: Does it correctly describe the code behavior?
* **Clarity**: Is it easy to understand for the target audience?
* **Completeness**: Are all important aspects covered?
* **Examples**: Do the examples work and illustrate key points?

Common Documentation Issues
===========================

Issues to Avoid
---------------

* **Outdated examples**: Code that doesn't work with current version
* **Missing imports**: Examples that can't be run as-is
* **Unclear parameter descriptions**: Vague or incomplete arg documentation
* **No examples**: Public functions without usage examples
* **Inconsistent formatting**: Mixed docstring styles within the project

Quality Checklist
------------------

Before submitting documentation:

- [ ] All examples are runnable
- [ ] Docstrings follow Google format
- [ ] Public functions have examples
- [ ] Cross-references are working
- [ ] Spelling and grammar are correct
- [ ] Mathematical notation is properly formatted
- [ ] Code formatting is consistent

Tools and Resources
===================

Documentation Tools
-------------------

* **Sphinx**: Documentation generation framework
* **Napoleon**: Sphinx extension for Google docstrings
* **nbsphinx**: Include Jupyter notebooks in docs
* **autodoc**: Automatic API documentation from docstrings

Writing Resources
-----------------

* `Google Style Guide <https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings>`_
* `Sphinx Documentation <https://www.sphinx-doc.org/>`_
* `Write the Docs <https://www.writethedocs.org/>`_

Good documentation is one of the most valuable contributions you can make to quactuary. It helps users get started quickly and makes the project more accessible to everyone!