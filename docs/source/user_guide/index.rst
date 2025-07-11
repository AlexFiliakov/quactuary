User Guide
==========

This guide provides detailed information on using the quactuary package effectively.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   optimization_overview
   quick_start
   configuration_guide
   best_practices
   numerical_stability
   distribution_parameters
   optimization_config
   testing_best_practices
   api_migration
   mcp_integration

Getting Started
---------------

The quactuary package provides tools for actuarial loss modeling with both classical and quantum computing backends. Key features include:

* Frequency and severity distributions
* Compound distribution modeling
* Portfolio-based pricing
* Risk measure calculations (VaR, TVaR)
* Performance optimization (JIT, parallel, QMC)
* Quantum computing integration
* MCP Server for LLM integration (Claude, etc.)

Performance Optimization
------------------------

The quactuary package includes comprehensive optimization features to accelerate Monte Carlo simulations:

* **Optimization Overview** (:doc:`optimization_overview`) - Understand available optimization strategies and when to use them
* **Quick Start Guide** (:doc:`quick_start`) - Get started with optimization in minutes
* **Configuration Guide** (:doc:`configuration_guide`) - Detailed reference for all optimization parameters
* **Best Practices** (:doc:`best_practices`) - Learn from real-world optimization patterns

Advanced Topics
---------------

Numerical Stability
~~~~~~~~~~~~~~~~~~~

Working with actuarial calculations often involves extreme values and complex operations. The :doc:`numerical_stability` guide provides best practices for ensuring accurate results even with challenging numerical conditions.

API Reference
~~~~~~~~~~~~~

* **Distribution Parameters** (:doc:`distribution_parameters`) - Complete reference for all distribution parameter names and conventions
* **Optimization Configuration** (:doc:`optimization_config`) - Guide to using OptimizationConfig for performance tuning
* **API Migration Guide** (:doc:`api_migration`) - Migrate from older versions to the current API

Development
~~~~~~~~~~~

* **Testing Best Practices** (:doc:`testing_best_practices`) - Write robust tests for stochastic methods and numerical computations

LLM Integration
~~~~~~~~~~~~~~~

* **MCP Integration Guide** (:doc:`mcp_integration`) - Use quActuary tools directly in Claude and other LLM assistants via Model Context Protocol

Basic Usage
-----------

See the main documentation page for basic usage examples. For optimization-specific examples, refer to the :doc:`quick_start` guide.