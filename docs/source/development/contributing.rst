.. _contributing:

***************************
Contributing to quactuary
***************************

Thank you for your interest in contributing to quactuary! This guide will walk you through the entire process of contributing to the project, from finding something to work on to getting your changes merged.

.. contents:: Table of Contents
   :local:
   :depth: 2

Getting Started
===============

Before You Begin
----------------

1. **Familiarize yourself with quactuary**: Spend some time using the library and reading the documentation
2. **Review existing issues**: Check our `issue tracker <https://github.com/AlexFiliakov/quactuary/issues>`_ to see what's already being worked on
3. **Join the community**: Follow the repository on GitHub to stay updated with discussions

Finding Something to Work On
============================

Issue Tracker
-------------

Our `GitHub issue tracker <https://github.com/AlexFiliakov/quactuary/issues>`_ is the central place to find work. Issues are labeled to help you find something that matches your interests and skill level:

* **good first issue**: Perfect for new contributors
* **help wanted**: We'd especially like community help on these
* **bug**: Something isn't working correctly
* **enhancement**: New features or improvements
* **documentation**: Improvements to docs, examples, or tutorials
* **quantum**: Related to quantum computing features
* **performance**: Speed or memory optimization opportunities

Types of Contributions
----------------------

**Code Contributions**

* **Bug fixes**: Fix existing functionality that isn't working correctly
* **New features**: Add new distributions, algorithms, or capabilities
* **Performance**: Optimize existing code for speed or memory usage
* **Quantum algorithms**: Implement new quantum approaches for actuarial problems

**Documentation Contributions**

* **API documentation**: Improve docstrings and reference material
* **Tutorials**: Write step-by-step guides for common use cases
* **Examples**: Create Jupyter notebooks demonstrating features
* **Theory**: Document the mathematical foundations

**Testing Contributions**

* **Increase coverage**: Add tests for untested code paths
* **Edge cases**: Test boundary conditions and error cases
* **Performance tests**: Benchmark and regression testing
* **Integration tests**: Test component interactions

Contribution Workflow
======================

Step 1: Fork the Repository
----------------------------

1. Navigate to https://github.com/AlexFiliakov/quactuary
2. Click the "Fork" button in the top-right corner
3. This creates your own copy of the repository

Step 2: Clone Your Fork
-----------------------

.. code-block:: bash

   # Clone your fork
   git clone https://github.com/YOUR_USERNAME/quactuary.git
   cd quactuary
   
   # Add the original repository as 'upstream'
   git remote add upstream https://github.com/AlexFiliakov/quactuary.git

Step 3: Set Up Development Environment
--------------------------------------

See our detailed :doc:`setting_up_environment` guide for complete instructions.

Quick setup:

.. code-block:: bash

   # Create virtual environment
   python -m venv quactuary-dev
   source quactuary-dev/bin/activate  # On Windows: quactuary-dev\Scripts\activate
   
   # Install in development mode
   pip install -e .[dev]

Step 4: Create a Feature Branch
--------------------------------

Always create a new branch for your work:

.. code-block:: bash

   # Make sure you're on main and it's up to date
   git checkout main
   git pull upstream main
   
   # Create and switch to your feature branch
   git checkout -b feature/your-feature-name
   
   # Or for bug fixes
   git checkout -b bugfix/issue-number-description

**Branch Naming Conventions:**

* ``feature/description-of-feature``
* ``bugfix/issue-number-brief-description``
* ``docs/description-of-changes``
* ``perf/description-of-optimization``

Step 5: Make Your Changes
-------------------------

Follow our :doc:`code_standards` and :doc:`documentation_guidelines` while implementing your changes.

**Key Points:**

* Write clean, readable code following PEP 8
* Add comprehensive docstrings using Google format
* Include type hints where appropriate
* Consider performance implications
* Think about backward compatibility

Step 6: Write Tests
-------------------

All new code must include tests. See our :doc:`testing_guidelines` for details.

**Requirements:**

* ≥90% test coverage for new code (aim for higher)
* Test both happy paths and edge cases
* Include integration tests for new features
* Add performance tests for optimizations

.. code-block:: bash

   # Run tests with coverage
   pytest --cov=quactuary --cov-report=html
   
   # Check coverage report
   open htmlcov/index.html

Step 7: Commit Your Changes
---------------------------

Write clear, descriptive commit messages:

.. code-block:: bash

   # Stage your changes
   git add .
   
   # Commit with a clear message
   git commit -m "Add feature: quantum amplitude estimation for VaR calculation
   
   - Implement QAE algorithm for Value at Risk estimation
   - Add comprehensive tests with 95% coverage
   - Include performance benchmarks vs classical method
   - Update documentation with usage examples
   
   Closes #123"

**Commit Message Format:**

* First line: Concise summary (≤50 characters)
* Blank line
* Detailed description if needed
* Reference related issues

Step 8: Push and Create Pull Request
------------------------------------

.. code-block:: bash

   # Push your branch to your fork
   git push origin feature/your-feature-name

Then create a pull request through GitHub's web interface.

Pull Request Guidelines
=======================

When creating your pull request:

**Title and Description**
-------------------------

* Use a clear, descriptive title
* Reference any related issues (e.g., "Fixes #123")
* Explain what changes you made and why
* Include any breaking changes or migration notes

**Checklist**
-------------

Before submitting, ensure your PR includes:

* [ ] Code follows our style guidelines
* [ ] All tests pass
* [ ] New code has ≥90% test coverage
* [ ] Documentation is updated
* [ ] CHANGELOG.md is updated (for significant changes)
* [ ] Type hints are included
* [ ] Performance impact is considered

**Example PR Description:**

.. code-block:: markdown

   ## Description
   This PR implements quantum amplitude estimation for Value at Risk calculations, 
   providing a quadratic speedup over classical Monte Carlo methods for certain 
   problem structures.
   
   ## Changes Made
   - Added QAE algorithm implementation in `quantum.py`
   - Created comprehensive test suite with 95% coverage
   - Added performance benchmarks comparing quantum vs classical
   - Updated documentation with usage examples
   - Added integration with existing VaR calculation pipeline
   
   ## Testing
   - All existing tests pass
   - New tests added for QAE implementation
   - Performance tests validate speedup claims
   - Integration tests ensure compatibility
   
   ## Breaking Changes
   None
   
   ## Related Issues
   Closes #123
   Addresses #89

Review Process
==============

What to Expect
--------------

1. **Automated Checks**: Our CI will run tests and style checks
2. **Initial Review**: A maintainer will review within a few days
3. **Feedback**: You may receive requests for changes
4. **Iteration**: Work together to refine the contribution
5. **Approval**: Once ready, your changes will be merged

Tips for Smooth Reviews
-----------------------

* **Respond promptly** to reviewer feedback
* **Ask questions** if feedback isn't clear
* **Be open** to suggestions and alternative approaches
* **Keep PRs focused** - smaller changes are easier to review
* **Update tests** when making code changes

Common Issues and Solutions
===========================

**Tests Failing**

.. code-block:: bash

   # Run specific test file
   pytest tests/test_your_feature.py -v
   
   # Run with debugging
   pytest tests/test_your_feature.py -v -s --pdb

**Coverage Too Low**

.. code-block:: bash

   # See what lines aren't covered
   pytest --cov=quactuary --cov-report=term-missing
   
   # Generate HTML report for detailed view
   pytest --cov=quactuary --cov-report=html

**Style Issues**

.. code-block:: bash

   # Auto-format with black
   black .
   
   # Check for issues
   flake8 quactuary/
   
   # Type checking
   mypy quactuary/

**Merge Conflicts**

.. code-block:: bash

   # Update your branch with latest main
   git fetch upstream
   git checkout main
   git merge upstream/main
   git checkout feature/your-feature-name
   git merge main
   
   # Resolve conflicts and commit
   git add .
   git commit -m "Resolve merge conflicts"

Getting Help
============

If you need help at any point:

* **Ask on GitHub**: Comment on your PR or issue
* **Check documentation**: Review our development guides
* **Search issues**: Someone may have had the same question
* **Be patient**: Maintainers are volunteers with day jobs

We're here to help make your contribution successful!