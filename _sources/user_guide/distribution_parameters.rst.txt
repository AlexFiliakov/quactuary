Distribution Parameter Reference
================================

This guide documents the parameter naming conventions used across all distribution classes in quActuary.

.. important::

   As of version 0.2.0, all distributions use ``loc`` for location parameters (not ``location`` or ``threshold``).
   This was standardized to ensure consistency across the API.

Overview
--------

quActuary uses consistent parameter naming conventions across its distribution classes:

* **Location parameter**: Always use ``loc``
* **Scale parameter**: Always use ``scale``
* **Shape parameters**: Vary by distribution type but follow statistical conventions

Parameter Naming by Distribution Type
-------------------------------------

Frequency Distributions
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Frequency Distribution Parameters
   :widths: 30 70
   :header-rows: 1

   * - Distribution
     - Parameters
   * - Binomial
     - ``n`` (number of trials), ``p`` (probability of success)
   * - DeterministicFrequency
     - ``value`` (fixed frequency value)
   * - DiscreteUniformFrequency
     - ``low`` (minimum value), ``high`` (maximum value)
   * - EmpiricalFrequency
     - ``pmf_values`` (dictionary of value: probability pairs)
   * - Geometric
     - ``p`` (probability of success)
   * - Hypergeometric
     - ``M`` (population size), ``n`` (number of draws), ``N`` (success states)
   * - MixedFrequency
     - ``components`` (list of distributions), ``weights`` (mixing weights)
   * - NegativeBinomial
     - ``r`` (number of successes), ``p`` (probability of success)
   * - PanjerABk
     - ``a``, ``b``, ``k`` (Panjer parameters), ``tol`` (tolerance), ``max_iter`` (max iterations)
   * - Poisson
     - ``mu`` (mean/rate parameter)
   * - TriangularFrequency
     - ``c`` (mode), ``loc`` (minimum), ``scale`` (maximum - minimum)

Severity Distributions
~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Severity Distribution Parameters
   :widths: 30 70
   :header-rows: 1

   * - Distribution
     - Parameters
   * - Beta
     - ``a``, ``b`` (shape parameters), ``loc``, ``scale``
   * - ChiSquared
     - ``df`` (degrees of freedom), ``loc``, ``scale``
   * - ConstantSeverity
     - ``value`` (fixed severity value)
   * - ContinuousUniformSeverity
     - ``loc`` (minimum), ``scale`` (maximum - minimum)
   * - DiscretizedSeverity
     - ``sev_dist`` (underlying distribution), ``min_val``, ``max_val``, ``bins``
   * - EmpiricalSeverity
     - ``values`` (data values), ``probs`` (probabilities)
   * - Exponential
     - ``scale`` (rate = 1/scale), ``loc``
   * - Gamma
     - ``shape`` (alpha), ``loc``, ``scale``
   * - InverseGamma
     - ``shape`` (alpha), ``loc``, ``scale``
   * - InverseGaussian
     - ``shape`` (lambda), ``loc``, ``scale`` (mu)
   * - InverseWeibull
     - ``shape`` (c), ``loc``, ``scale``
   * - Lognormal
     - ``shape`` (sigma), ``loc``, ``scale``
   * - MixedSeverity
     - ``components`` (list of distributions), ``weights`` (mixing weights)
   * - Pareto
     - ``b`` (shape/alpha), ``loc``, ``scale``
   * - StudentsT
     - ``df`` (degrees of freedom), ``loc``, ``scale``
   * - TriangularSeverity
     - ``c`` (mode), ``loc`` (minimum), ``scale`` (maximum - minimum)
   * - Weibull
     - ``shape`` (c), ``loc``, ``scale``

Compound Distributions
~~~~~~~~~~~~~~~~~~~~~~

All compound distributions use the same parameter structure:

* ``frequency``: A frequency distribution instance
* ``severity``: A severity distribution instance

Mixed Poisson Distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Mixed Poisson Distribution Parameters
   :widths: 30 70
   :header-rows: 1

   * - Distribution
     - Parameters
   * - PoissonGammaMixture
     - ``alpha``, ``beta`` (Gamma mixing parameters)
   * - PoissonInverseGaussianMixture
     - ``mu`` (mean), ``lambda_shape`` (shape parameter)
   * - HierarchicalPoissonMixture
     - ``portfolio_alpha``, ``portfolio_beta``, ``group_alpha``, ``n_groups``
   * - TimeVaryingPoissonMixture
     - ``base_rate``, ``intensity_func``, ``param_dist``, ``time_horizon``

Zero-Inflated Distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

All zero-inflated distributions use:

* ``frequency``: Base frequency distribution
* ``severity``: Base severity distribution  
* ``zero_prob``: Probability of zero claims

Migration Guide
---------------

If you're updating from an older version, here are the key parameter changes:

.. code-block:: python

   # Old (pre-0.2.0)
   lognormal = Lognormal(shape=1.5, location=0.0, scale=1.0)  # WRONG
   pareto = Pareto(b=2.0, threshold=1000.0, scale=1.0)        # WRONG
   
   # New (0.2.0+)
   lognormal = Lognormal(shape=1.5, loc=0.0, scale=1.0)       # CORRECT
   pareto = Pareto(b=2.0, loc=1000.0, scale=1.0)             # CORRECT

Best Practices
--------------

1. **Always use keyword arguments** when creating distributions to avoid confusion:

   .. code-block:: python

      # Good
      dist = Gamma(shape=2.0, loc=0.0, scale=1.0)
      
      # Avoid
      dist = Gamma(2.0, 0.0, 1.0)

2. **Check parameter bounds** - some distributions have restrictions:

   .. code-block:: python

      # Pareto requires loc > 0 (minimum value)
      pareto = Pareto(b=2.0, loc=1000.0, scale=1.0)  # loc must be positive
      
      # Beta requires 0 <= x <= 1 after transformation
      beta = Beta(a=2.0, b=3.0, loc=0.0, scale=1.0)

3. **Use consistent units** across frequency and severity when creating compounds:

   .. code-block:: python

      freq = Poisson(mu=10)  # 10 claims per period
      sev = Lognormal(shape=1.0, loc=0.0, scale=1000)  # severity in dollars
      compound = CompoundPoisson(frequency=freq, severity=sev)

Examples
--------

Creating Common Distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from quactuary.distributions import (
       Poisson, NegativeBinomial, 
       Lognormal, Pareto, Gamma,
       CompoundPoisson
   )
   
   # Frequency distributions
   poisson_freq = Poisson(mu=5.0)
   negbin_freq = NegativeBinomial(r=10, p=0.3)
   
   # Severity distributions  
   lognormal_sev = Lognormal(shape=1.5, loc=0.0, scale=1000.0)
   pareto_sev = Pareto(b=2.0, loc=1000.0, scale=1.0)
   gamma_sev = Gamma(shape=2.0, loc=0.0, scale=500.0)
   
   # Compound distribution
   compound = CompoundPoisson(
       frequency=poisson_freq,
       severity=lognormal_sev
   )

Working with Zero-Inflated Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from quactuary.distributions import ZeroInflatedCompound
   
   # 20% chance of zero claims
   zi_compound = ZeroInflatedCompound(
       frequency=Poisson(mu=5.0),
       severity=Gamma(shape=2.0, scale=1000.0),
       zero_prob=0.2
   )

See Also
--------

* :doc:`/quactuary.distributions` - Full API documentation
* :doc:`/user_guide/testing_best_practices` - Testing with stochastic distributions
* :doc:`/user_guide/best_practices` - Best practices for using distributions