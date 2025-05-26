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

Getting Started
---------------

The quactuary package provides tools for actuarial loss modeling with both classical and quantum computing backends. Key features include:

* Frequency and severity distributions
* Compound distribution modeling
* Portfolio-based pricing
* Risk measure calculations (VaR, TVaR)
* Performance optimization (JIT, parallel, QMC)
* Quantum computing integration

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

Basic Usage
-----------

See the main documentation page for basic usage examples. For optimization-specific examples, refer to the :doc:`quick_start` guide.