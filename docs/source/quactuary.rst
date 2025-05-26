quactuary package
=================

Subpackages
-----------

.. toctree::
   :maxdepth: 4

   quactuary.distributions
   quactuary.future
   quactuary.utils

Package Overview
~~~~~~~~~~~~~~~~

The quactuary package provides a comprehensive framework for quantum-accelerated
actuarial modeling. For detailed documentation of individual modules, see the
API reference sections below.

Key Features:
    - Classical and quantum-accelerated pricing models
    - Comprehensive frequency and severity distributions
    - High-performance parallel and vectorized simulations
    - Memory-efficient algorithms for large-scale portfolios
    - Integration with quantum computing frameworks

Quick Start
-----------

The main entry points for users are:

* :class:`~quactuary.PricingModel` - For portfolio pricing and risk calculations
* :class:`~quactuary.Portfolio` - For managing collections of insurance policies  
* :class:`~quactuary.PolicyTerms` - For defining individual policy characteristics
* :class:`~quactuary.Inforce` - For grouping policies with similar characteristics

For more details, see the individual module documentation below.
