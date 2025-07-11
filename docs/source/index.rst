.. quActuary documentation master file.

===============================
quActuary Documentation
===============================

quActuary is a Python library for classical and Quantum-accelerated actuarial loss modeling. It provides frequency and severity distribution models, excess-loss and risk measure pricing, and integration with Qiskit for quantum-enhanced simulations.

Features
--------

- Extensive frequency distributions: Poisson, Binomial, Negative Binomial, Geometric, etc.
- Support for the generalized Panjer (a, b, k) frequency distribution class.
- Comprehensive severity distributions: Exponential, Gamma, Lognormal, Pareto, Continuous Uniform, etc.
- Mixture models, empirical distributions, and QMC wrappers.
- Classical and quantum pricing models with backend switching.
- **MCP Server Integration**: Use quActuary tools directly in Claude via Model Context Protocol.

Installation
------------

.. code-block:: powershell

   pip install quactuary

To use the MCP Server integration:

.. code-block:: powershell

   pip install quactuary[mcp]

Getting Started
---------------

.. code-block:: python

   import quactuary as qa
   import quactuary.book as book

   from datetime import date
   from quactuary.book import (
      LOB, PolicyTerms, Inforce, Portfolio)
   from quactuary.distributions.frequency import Poisson
   from quactuary.distributions.severity import Exponential
   from quactuary.pricing import PricingModel

   # Frequency model example
   freq = Poisson(mu=3.5)
   print(freq.pmf(2))

   # Severity model example
   sev = Exponential(scale=1000.0)
   print(sev.pdf(500.0))

   ### Classical pricing model example ###

   # Define the policy terms
   glpl_policy = PolicyTerms(
      effective_date=date(2026, 1, 1),
      expiration_date=date(2027, 1, 1),
      lob=LOB.GLPL,
      exposure_base=book.SALES,
      exposure_amount=10_000_000_000,
      retention_type="deductible",
      per_occ_retention=1_000_000,
      coverage="occ"
   )

   # Define the inforce block model
   glpl_inforce = Inforce(
      n_policies=700,
      terms=glpl_policy,
      frequency=freq,
      severity=sev,
      name = "GLPL 2026 Bucket"
   )

   portfolio = Portfolio(glpl_inforce)

   # Initialize the pricing model
   pricing = PricingModel(portfolio)

   # Run the classical simulation
   result = pricing.simulate()
   result.estimates

   ### Output: ###
   # 0.18495897346170082
   # 0.0006065306597126335
   # {'mean': np.float64(2449951.874110688),
   #  'variance': np.float64(4860457942.058481),
   #  'VaR': 2565751.9589995327,
   #  'TVaR': 2596372.4576630797}

Documentation
-------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide:

   user_guide/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   quactuary
   quactuary.entities
   quactuary.backend
   quactuary.pricing
   api_reference/pricing_model
   quactuary.quantum_pricing
   quactuary.distributions
   quactuary.mcp
   quactuary.memory_management
   quactuary.parallel_processing
   quactuary.vectorized_simulation
   quactuary.utils
   quactuary.future

.. toctree::
   :maxdepth: 2
   :caption: Performance:

   performance/benchmarks
   performance/tuning_guide

.. toctree::
   :maxdepth: 2
   :caption: Development:

   development/index

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

