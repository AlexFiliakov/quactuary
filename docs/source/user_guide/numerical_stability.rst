Numerical Stability Guide
=========================

This guide provides best practices for numerical stability in actuarial computations, particularly when working with extreme values, compound distributions, and aggregate loss calculations.

Why Numerical Stability Matters
-------------------------------

Actuarial calculations often involve:

* **Extreme values**: Very large claim amounts or very small probabilities
* **Compound operations**: Products and sums of many terms
* **Log-space calculations**: Working with log-probabilities to avoid underflow
* **Tail calculations**: Computing values in distribution tails where precision matters

Without proper numerical handling, these calculations can lead to:

* Overflow (values too large to represent)
* Underflow (values rounded to zero)
* Loss of precision in intermediate calculations
* Incorrect results that may not be obviously wrong

Using the Numerical Utilities
-----------------------------

The ``quactuary.utils.numerical`` module provides stable implementations of common operations.

Log-Space Operations
~~~~~~~~~~~~~~~~~~~~

When working with probabilities or likelihood calculations, use log-space operations:

.. code-block:: python

    from quactuary.utils.numerical import logsumexp, logaddexp, stable_log
    
    # Instead of: sum(exp(log_probs))
    log_probs = np.array([-1000, -1001, -1002])
    total_prob = np.exp(logsumexp(log_probs))
    
    # For pairwise operations
    log_p1 = -500
    log_p2 = -501
    log_sum = logaddexp(log_p1, log_p2)

Stable Exponentials and Logarithms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Protect against overflow/underflow in exponential calculations:

.. code-block:: python

    from quactuary.utils.numerical import stable_exp, stable_log
    
    # Handles large values gracefully
    large_value = 800
    result = stable_exp(large_value)  # Won't overflow
    
    # Handles near-zero values
    small_value = 1e-400
    log_result = stable_log(small_value)  # Won't underflow

Input Validation
~~~~~~~~~~~~~~~~

Validate numerical inputs to catch issues early:

.. code-block:: python

    from quactuary.utils.numerical import check_finite, clip_to_valid_range
    
    # Check for NaN/inf values
    data = check_finite(input_array, name="claim amounts")
    
    # Ensure probabilities are in [0, 1]
    probs = clip_to_valid_range(raw_probs, 0.0, 1.0, name="probabilities")

Detecting Numerical Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~

Proactively detect potential numerical problems:

.. code-block:: python

    from quactuary.utils.numerical import detect_numerical_issues
    
    # Check for potential problems
    issues = detect_numerical_issues(data, "portfolio losses")
    if issues:
        print(f"Warning: {issues}")

Best Practices for Specific Calculations
-----------------------------------------

Compound Distributions
~~~~~~~~~~~~~~~~~~~~~~

When implementing compound distributions:

1. **Use log-space for probability calculations**:

   .. code-block:: python

       # In Poisson initialization
       p0 = stable_exp(-lambda_param)  # Instead of np.exp(-lambda_param)

2. **Handle extreme parameters**:

   .. code-block:: python

       # For large powers
       p_n = stable_exp(n * stable_log(p))  # Instead of p**n

3. **Normalize carefully**:

   .. code-block:: python

       # Convert to log-space for normalization
       log_probs = stable_log(raw_probs)
       normalized = stable_probability_calculation(log_probs)

Moment Calculations
~~~~~~~~~~~~~~~~~~~

For stable moment calculations:

.. code-block:: python

    from quactuary.utils.numerical import stable_moment_calculation
    
    # Calculate moments with numerical stability
    mean = stable_moment_calculation(values, probs, moment=1)
    variance = stable_moment_calculation(values, probs, moment=2, central=True)
    
    # For high moments, use log-space
    fourth_moment = stable_moment_calculation(
        values, probs, moment=4, log_space=True
    )

Aggregate Loss Calculations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

When computing aggregate losses:

1. **Monitor numerical range**: Check intermediate results for potential issues
2. **Use stable summation**: For many terms, consider log-space summation
3. **Validate results**: Ensure outputs are within reasonable bounds

Example: Stable Tweedie Implementation
---------------------------------------

Here's an example of how the numerical utilities improve the Tweedie distribution implementation:

.. code-block:: python

    # Before: Manual clipping and direct exp
    log_weight = np.clip(log_term1 - log_factorial - log_term3, -100, 100)
    weight = np.exp(log_weight)
    
    # After: Using stable utilities
    log_weight = log_term1 - log_factorial - log_term3
    weight = stable_exp(log_weight)  # Handles extreme values automatically

Common Pitfalls to Avoid
------------------------

1. **Direct exponentiation of large values**:

   .. code-block:: python

       # Bad: Can overflow
       p = (1 - p) ** n
       
       # Good: Use log-space
       p = stable_exp(n * stable_log(1 - p))

2. **Log of potentially zero values**:

   .. code-block:: python

       # Bad: log(0) is -inf
       log_val = np.log(probability)
       
       # Good: Protected against zero
       log_val = stable_log(probability)

3. **Normalization without checking**:

   .. code-block:: python

       # Bad: Division without checking
       normalized = values / values.sum()
       
       # Good: Check for numerical issues
       if detect_numerical_issues(values):
           # Use stable normalization
           log_vals = stable_log(values)
           normalized = stable_probability_calculation(log_vals)

Testing for Numerical Stability
-------------------------------

When developing new algorithms:

1. **Test extreme inputs**: Use very large and very small parameter values
2. **Check edge cases**: Test with zero, one, and many terms
3. **Verify conservation**: Ensure probabilities sum to 1, etc.
4. **Compare methods**: Test against known stable implementations

Example test:

.. code-block:: python

    def test_extreme_parameters():
        # Test with extreme lambda
        lambda_large = 1000
        p0 = stable_exp(-lambda_large)
        assert np.isfinite(p0)
        assert p0 >= 0
        
        # Test with many terms
        n = 10000
        p = 0.001
        result = stable_exp(n * stable_log(1 - p))
        assert np.isfinite(result)

Conclusion
----------

Numerical stability is crucial for accurate actuarial calculations. By using the utilities in ``quactuary.utils.numerical`` and following these best practices, you can ensure your calculations remain accurate even with extreme values or complex operations.

Remember:

* Work in log-space when dealing with probabilities
* Use stable versions of exp and log operations
* Validate inputs and detect numerical issues early
* Test with extreme values during development

For more details, see the API documentation for :mod:`quactuary.utils.numerical`.