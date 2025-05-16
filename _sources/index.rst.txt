.. quActuary documentation master file.

===============================
quActuary Documentation
===============================

quActuary is a Python library for classical and Quantum-accelerated actuarial loss modeling. It provides frequency and severity distribution models, excess-loss and risk measure pricing, and integration with Qiskit for quantum-enhanced simulations.

Features
--------

- Extensive frequency distributions: Poisson, Binomial, Negative Binomial, etc.
- Support for the generalized Panjer (a, b, k) frequency distribution class.
- Comprehensive severity distributions: Exponential, Gamma, Lognormal, Pareto, etc.
- Mixture models and empirical distributions.

Installation
------------

.. code-block:: powershell

   pip install quactuary

Getting Started
---------------

.. code-block:: python

   from quactuary.distributions.frequency import Poisson
   from quactuary.distributions.severity import Exponential
   from quactuary.pricing import ExcessLossModel

   # Frequency model example
   freq = Poisson(mu=3.5)
   print(freq.pmf(2))

   # Severity model example
   sev = Exponential(scale=1000.0)
   print(sev.pdf(500.0))

   # Pricing model example (requires valid Inforce data)
   # pricing = ExcessLossModel(inforce, deductible=1000, limit=10000)
   # result = pricing.compute_excess_loss()

Actuarial Modules
-----------------

.. toctree::
   :maxdepth: 2
   :caption: Actuarial Modules:

   quactuary.entities
   quactuary.pricing
   quactuary.distributions
   quactuary.quantum
   quactuary.backend
   quactuary.utils
   quactuary.future

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

