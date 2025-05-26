.. _issue_reporting:

***************
Issue Reporting
***************

This guide helps you effectively report bugs, request features, and contribute to discussions in the quactuary project. Well-written issues help maintainers understand problems quickly and provide better support.

.. contents:: Table of Contents
   :local:
   :depth: 2

Where to Report Issues
======================

**GitHub Issues**: https://github.com/AlexFiliakov/quactuary/issues

This is our central hub for:

* **Bug reports**: Something isn't working correctly
* **Feature requests**: Ideas for new functionality
* **Documentation issues**: Problems with guides or examples
* **Performance problems**: Slow or memory-intensive operations
* **Questions**: Usage questions and clarifications

Before Creating an Issue
=========================

Search Existing Issues
-----------------------

Before reporting a new issue:

1. **Search open issues**: Someone may have already reported it
2. **Check closed issues**: The problem might be already fixed
3. **Review discussions**: Look for related conversations

Use these search strategies:

.. code-block:: text

   # Search for specific errors
   "ValueError: invalid parameter"
   
   # Search for feature areas
   label:quantum label:enhancement
   
   # Search for keywords
   VaR calculation Monte Carlo

Read the Documentation
----------------------

Before asking questions:

* **Check the user guide**: https://docs.quactuary.com
* **Review API documentation**: Look for usage examples
* **Browse example notebooks**: See real-world usage patterns
* **Check FAQ section**: Common questions and answers

Update to Latest Version
-------------------------

Many issues are fixed in newer versions:

.. code-block:: bash

   # Check your version
   python -c "import quactuary; print(quactuary.__version__)"
   
   # Update to latest
   pip install --upgrade quactuary
   
   # Test if issue persists

Types of Issues
===============

Bug Reports
-----------

Use the bug report template for:

* **Incorrect calculations**: Wrong results from functions
* **Crashes or exceptions**: Unexpected errors
* **Performance regressions**: Slower than expected
* **Documentation errors**: Incorrect examples or information

**Bug Report Template:**

.. code-block:: markdown

   ## Bug Description
   Clear description of what went wrong and what you expected.
   
   ## Steps to Reproduce
   1. Import quactuary
   2. Create portfolio with...
   3. Run model.simulate()
   4. Error occurs
   
   ## Code Example
   ```python
   # Minimal code that reproduces the issue
   import quactuary as qa
   
   portfolio = qa.Portfolio(...)
   model = qa.PricingModel(portfolio)
   result = model.simulate()  # Error here
   ```
   
   ## Error Message
   ```
   Full error traceback here
   ```
   
   ## Environment
   - quactuary version: 1.2.3
   - Python version: 3.9.7
   - Operating system: macOS 12.6
   - Dependencies: numpy 1.21.0, pandas 1.3.3
   
   ## Expected Behavior
   What you expected to happen instead.
   
   ## Additional Context
   Any other relevant information.

Feature Requests
----------------

Use for new functionality suggestions:

* **New distributions**: Additional probability distributions
* **New algorithms**: Quantum or classical calculation methods
* **API improvements**: Better interfaces or convenience functions
* **Performance enhancements**: Faster or more memory-efficient code

**Feature Request Template:**

.. code-block:: markdown

   ## Feature Description
   Clear description of the requested feature and its purpose.
   
   ## Motivation
   Why is this feature needed? What problem does it solve?
   
   ## Proposed Solution
   How do you envision this feature working?
   
   ## Code Example
   ```python
   # How the feature might be used
   from quactuary.distributions import NewDistribution
   
   dist = NewDistribution(param1=5, param2=0.8)
   result = dist.calculate_something()
   ```
   
   ## Alternatives Considered
   Other approaches you've considered and why they don't work.
   
   ## Implementation Notes
   Any technical considerations or suggestions.
   
   ## Breaking Changes
   Would this require breaking existing APIs?

Documentation Issues
--------------------

Report problems with:

* **Incorrect examples**: Code that doesn't work
* **Missing documentation**: Undocumented features
* **Unclear explanations**: Confusing descriptions
* **Broken links**: Non-working references

Performance Issues
------------------

For performance problems:

* **Include benchmarks**: Show timing comparisons
* **Specify hardware**: CPU, memory, etc.
* **Provide scale**: Data size that causes issues
* **Compare versions**: Was it faster in previous versions?

Questions and Discussions
-------------------------

For usage questions:

* **Be specific**: Include code examples
* **Show what you tried**: Demonstrate your attempts
* **Explain the goal**: What are you trying to achieve?
* **Provide context**: Domain-specific background if relevant

Writing Effective Issues
========================

Good Title
----------

Titles should be:

* **Specific**: Describe the exact problem
* **Concise**: 50-80 characters when possible
* **Searchable**: Include key terms others might search for

**Good titles:**
- "VaR calculation returns NaN with Pareto distribution"
- "Memory leak in large portfolio simulation with JIT"
- "Feature request: Add Tweedie distribution support"
- "Documentation: Missing example for quantum backend setup"

**Poor titles:**
- "Bug in pricing model"
- "Feature request"
- "Help needed"
- "Question about distributions"

Clear Description
-----------------

Descriptions should:

* **Start with summary**: One-paragraph overview
* **Provide context**: Why you're doing this
* **Include details**: All relevant information
* **Show examples**: Code that demonstrates the issue

Minimal Reproducible Example
-----------------------------

Create the smallest possible code example that shows the issue:

**Good example:**

.. code-block:: python

   import quactuary as qa
   import numpy as np
   
   # Create simple portfolio
   portfolio = qa.Portfolio([
       qa.Inforce(
           n_policies=10,
           frequency=qa.distributions.Poisson(lambda_=1.0),
           severity=qa.distributions.Pareto(alpha=1.5, scale=1000),
           terms=qa.PolicyTerms(
               effective_date='2024-01-01',
               expiration_date='2024-12-31'
           )
       )
   ])
   
   # This should work but raises ValueError
   model = qa.PricingModel(portfolio)
   result = model.simulate(n_sims=1000)

**Poor example:**

.. code-block:: python

   # My complex portfolio setup (100 lines of code)
   # Various data processing steps
   # Multiple model configurations
   # Error happens somewhere in here

Environment Information
-----------------------

Always include:

.. code-block:: python

   import sys
   import quactuary
   import numpy
   import pandas
   import scipy
   
   print(f"Python: {sys.version}")
   print(f"quactuary: {quactuary.__version__}")
   print(f"numpy: {numpy.__version__}")
   print(f"pandas: {pandas.__version__}")
   print(f"scipy: {scipy.__version__}")

.. code-block:: bash

   # Also helpful
   pip list | grep qiskit  # For quantum-related issues
   python -m platform  # System information

Error Messages
--------------

Include the **complete** error traceback:

.. code-block:: text

   Traceback (most recent call last):
     File "test_script.py", line 15, in <module>
       result = model.simulate(n_sims=1000)
     File "/path/to/quactuary/pricing.py", line 234, in simulate
       return self.strategy.calculate_portfolio_statistics(...)
     File "/path/to/quactuary/pricing_strategies.py", line 156, in calculate_portfolio_statistics
       raise ValueError(f"Invalid parameter: {param}")
   ValueError: Invalid parameter: -1.5

Don't just include the final error message - the full traceback helps identify where the problem occurs.

Issue Labels and Categories
===========================

Common Labels
-------------

Issues are categorized with labels:

**Type Labels:**
- ``bug``: Something is broken
- ``enhancement``: New feature request
- ``documentation``: Docs issue
- ``question``: Usage question
- ``performance``: Speed or memory issue

**Component Labels:**
- ``pricing``: Portfolio pricing functionality
- ``distributions``: Probability distributions
- ``quantum``: Quantum computing features
- ``backend``: Backend management
- ``testing``: Test-related issues

**Priority Labels:**
- ``critical``: Crashes, data corruption
- ``high``: Major functionality affected
- ``medium``: Standard priority
- ``low``: Nice to have improvements

**Status Labels:**
- ``good first issue``: Great for new contributors
- ``help wanted``: Community assistance welcomed
- ``blocked``: Waiting on external dependency
- ``wontfix``: Won't be addressed

Special Issue Types
===================

Security Issues
---------------

For security vulnerabilities:

* **Don't open public issues** for security problems
* **Email maintainers directly**: Use contact info in README
* **Provide details privately**: Include reproduction steps
* **Allow time for fixes**: Coordinate responsible disclosure

Performance Regressions
-----------------------

For performance problems:

.. code-block:: python

   import time
   import quactuary as qa
   
   # Benchmark setup
   portfolio = create_large_portfolio()  # Your setup
   model = qa.PricingModel(portfolio)
   
   # Timing test
   start = time.time()
   result = model.simulate(n_sims=10000)
   duration = time.time() - start
   
   print(f"Calculation took {duration:.2f} seconds")
   # Previous version took 5.2 seconds
   # Current version takes 15.8 seconds

Integration Issues
------------------

When quactuary doesn't work with other packages:

.. code-block:: python

   # Show the integration attempt
   import quactuary as qa
   import other_package as op
   
   # This combination doesn't work
   qa_result = qa.calculate_something()
   op_result = op.process(qa_result)  # Error here

Research Questions
------------------

For academic or research-related questions:

* **Provide background**: Explain the mathematical context
* **Include references**: Cite relevant papers
* **Show current approach**: What you've tried so far
* **Specify goals**: What accuracy or performance you need

Following Up on Issues
======================

Providing Additional Information
--------------------------------

When maintainers ask for more information:

* **Respond promptly**: Within a few days when possible
* **Answer completely**: Address all questions asked
* **Update environment**: Test with latest versions
* **Try suggestions**: Attempt proposed solutions

Closing Issues
--------------

Close issues when:

* **Problem is resolved**: Update works or workaround found
* **Question is answered**: Your question was resolved
* **No longer relevant**: You've changed approach
* **Duplicate found**: Reference the original issue

Contributing Solutions
----------------------

If you figure out a fix:

1. **Share the solution**: Help others with same problem
2. **Consider contributing**: Turn fix into a pull request
3. **Update documentation**: If it was a usage issue
4. **Thank contributors**: Acknowledge helpful responses

Issue Etiquette
===============

Being Respectful
----------------

* **Be patient**: Maintainers are often volunteers
* **Be grateful**: Thank people for their help
* **Be collaborative**: Work together toward solutions
* **Be understanding**: Not every request can be implemented

Managing Expectations
---------------------

* **Bug fixes**: Usually addressed quickly
* **Feature requests**: May take longer or require discussion
* **Questions**: Often answered within a few days
* **Complex issues**: May need back-and-forth to resolve

Contributing Back
-----------------

Ways to help the project:

* **Answer questions**: Help other users with issues
* **Test fixes**: Verify that fixes work in your environment
* **Improve documentation**: Clarify confusing sections
* **Submit pull requests**: Fix bugs or implement features

Quality Checklist
=================

Before submitting an issue:

- [ ] Searched existing issues
- [ ] Updated to latest version
- [ ] Created minimal reproducible example
- [ ] Included complete error messages
- [ ] Provided environment information
- [ ] Used clear, descriptive title
- [ ] Followed appropriate template
- [ ] Added relevant context and motivation

Good issues help maintainers help you more effectively. Thank you for taking the time to report problems and suggest improvements!