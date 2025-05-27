.. _submitting_changes:

******************
Submitting Changes
******************

This guide walks you through the process of submitting your contributions to quactuary, from preparing your changes to getting them merged into the main repository.

.. contents:: Table of Contents
   :local:
   :depth: 2

Before You Submit
=================

Pre-submission Checklist
-------------------------

Before creating a pull request, ensure your changes meet these requirements:

**Code Quality**
  - [ ] Code follows PEP 8 style guidelines
  - [ ] All new code has comprehensive docstrings (Google format)
  - [ ] Type hints are included for all public functions
  - [ ] No lint errors from flake8 or similar tools

**Testing**
  - [ ] All existing tests pass
  - [ ] New code has ≥90% test coverage
  - [ ] Edge cases and error conditions are tested
  - [ ] Integration tests included for new features

**Documentation**
  - [ ] Docstrings follow our standards
  - [ ] User-facing changes are documented
  - [ ] Examples are included and working
  - [ ] Changelog is updated for significant changes

**Git Hygiene**
  - [ ] Commits have clear, descriptive messages
  - [ ] Branch is up to date with main
  - [ ] No merge conflicts
  - [ ] Commit history is clean (consider squashing)

Running Pre-submission Checks
------------------------------

Run these commands to verify your changes:

.. code-block:: bash

   # Code formatting
   black .
   isort .
   
   # Style checking
   flake8 quactuary/ tests/
   
   # Type checking
   mypy quactuary/
   
   # Run full test suite with coverage
   pytest --cov=quactuary --cov-report=html --cov-report=term
   
   # Check documentation builds
   cd docs/
   make html

Creating a Pull Request
=======================

Step 1: Push Your Branch
------------------------

Push your feature branch to your fork:

.. code-block:: bash

   # Ensure your branch is up to date
   git fetch upstream
   git checkout main
   git merge upstream/main
   git checkout your-feature-branch
   git rebase main  # or git merge main
   
   # Push to your fork
   git push origin your-feature-branch

Step 2: Create the Pull Request
-------------------------------

1. Navigate to your fork on GitHub
2. Click "Compare & pull request" for your branch
3. Select the base repository: ``AlexFiliakov/quactuary``
4. Select the base branch: ``main``
5. Write a clear title and description

Pull Request Title
------------------

Use a clear, descriptive title that summarizes the change:

**Good titles:**
- "Add quantum amplitude estimation for VaR calculation"
- "Fix edge case in compound distribution sampling"
- "Improve performance of JIT-compiled simulation"
- "Add comprehensive documentation for pricing module"

**Avoid:**
- "Bug fix"
- "Update code"
- "Various improvements"

Pull Request Description
------------------------

Write a comprehensive description using this template:

.. code-block:: markdown

   ## Description
   Brief summary of what this PR does and why it's needed.
   
   ## Type of Change
   - [ ] Bug fix (non-breaking change which fixes an issue)
   - [ ] New feature (non-breaking change which adds functionality)
   - [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
   - [ ] Documentation update
   - [ ] Performance improvement
   - [ ] Code refactoring
   
   ## Changes Made
   - Detailed list of changes
   - Include both what was added and what was modified
   - Mention any files that were deleted or moved
   
   ## Testing
   - Describe how you tested your changes
   - Include test coverage information
   - Mention any manual testing performed
   
   ## Performance Impact
   (If applicable)
   - Benchmarks or performance measurements
   - Memory usage considerations
   - Scalability implications
   
   ## Breaking Changes
   (If applicable)
   - What breaks and why
   - Migration guide for users
   - Version compatibility notes
   
   ## Documentation
   - [ ] Documentation has been updated
   - [ ] Docstrings added/updated
   - [ ] Examples included
   - [ ] Changelog updated
   
   ## Related Issues
   - Closes #123
   - Fixes #456
   - Related to #789
   
   ## Additional Notes
   Any other information reviewers should know.

Example Pull Request Description
--------------------------------

.. code-block:: markdown

   ## Description
   This PR implements quantum amplitude estimation (QAE) for Value at Risk 
   calculations, providing a theoretical quadratic speedup over classical 
   Monte Carlo methods for portfolios with specific structure.
   
   ## Type of Change
   - [x] New feature (non-breaking change which adds functionality)
   
   ## Changes Made
   - Added `QuantumVaRCalculator` class in `quantum.py`
   - Implemented QAE circuit construction and execution
   - Added integration with existing VaR calculation pipeline
   - Created comprehensive test suite with 95% coverage
   - Added performance benchmarks comparing quantum vs classical
   - Updated documentation with usage examples and theory
   
   ## Testing
   - All existing tests pass
   - Added 23 new tests for QAE implementation
   - Coverage: 95% for new code, 92% overall
   - Tested on both simulator and real quantum hardware
   - Performance tests validate speedup claims
   
   ## Performance Impact
   - 4x speedup for portfolios with >10,000 policies (theoretical)
   - 2x speedup observed on current quantum simulators  
   - Memory usage: ~50MB additional for circuit construction
   - No impact on classical code paths
   
   ## Breaking Changes
   None - all changes are additive and backward compatible.
   
   ## Documentation
   - [x] Documentation has been updated
   - [x] Docstrings added for all new functions
   - [x] Examples included in docstrings and user guide
   - [x] Theory section added explaining QAE algorithm
   
   ## Related Issues
   - Closes #123 (Add quantum VaR calculation)
   - Related to #89 (Quantum algorithm roadmap)
   
   ## Additional Notes
   - Requires Qiskit ≥1.0.0
   - Quantum features remain experimental
   - Classical fallback when quantum backend unavailable

Review Process
==============

What Happens After Submission
------------------------------

1. **Automated Checks**: CI will run tests and style checks
2. **Initial Review**: A maintainer will review within 2-3 business days
3. **Feedback Phase**: You may receive requests for changes
4. **Iteration**: Work with reviewers to address feedback
5. **Approval**: Once approved, changes will be merged

Types of Reviews
----------------

**Technical Review**
  - Code correctness and efficiency
  - Test coverage and quality
  - Architecture and design decisions
  - Performance implications

**Documentation Review**
  - Clarity and completeness of docstrings
  - User guide accuracy
  - Example correctness
  - Consistency with existing docs

**Domain Review**
  - Actuarial correctness
  - Mathematical accuracy
  - Industry best practices
  - Regulatory considerations (if applicable)

Responding to Feedback
======================

How to Handle Review Comments
-----------------------------

1. **Read thoroughly**: Understand each comment before responding
2. **Ask questions**: If something isn't clear, ask for clarification
3. **Be responsive**: Respond within a few days when possible
4. **Be open**: Consider suggestions even if you initially disagree
5. **Explain decisions**: If you disagree, explain your reasoning

Making Changes
--------------

When reviewers request changes:

.. code-block:: bash

   # Make your changes
   git checkout your-feature-branch
   # Edit files...
   
   # Add and commit
   git add .
   git commit -m "Address review feedback: improve error handling"
   
   # Push updates
   git push origin your-feature-branch

The pull request will automatically update with your new commits.

Resolving Conversations
-----------------------

- Mark conversations as "resolved" after addressing them
- Leave a comment explaining what you changed
- Don't resolve conversations that you disagree with - discuss instead

Common Review Feedback
======================

Code Issues
-----------

**"This function is too complex"**
  - Break into smaller functions
  - Reduce cyclomatic complexity
  - Add helper methods

**"Missing error handling"**
  - Add appropriate try/catch blocks
  - Validate input parameters
  - Provide helpful error messages

**"Performance concern"**
  - Profile the code to identify bottlenecks
  - Consider algorithmic improvements
  - Add performance tests

Testing Issues
--------------

**"Insufficient test coverage"**
  - Add tests for uncovered lines
  - Test edge cases and error conditions
  - Include integration tests

**"Tests are flaky"**
  - Fix non-deterministic behavior
  - Use proper mocking for external dependencies
  - Set random seeds for reproducibility

**"Missing performance tests"**
  - Add benchmarks for performance-critical code
  - Test scalability with large inputs
  - Set performance regression thresholds

Documentation Issues
--------------------

**"Unclear docstring"**
  - Improve parameter descriptions
  - Add more detailed examples
  - Clarify the purpose and behavior

**"Missing user documentation"**
  - Add user guide sections
  - Include tutorial notebooks
  - Update API reference

Advanced Submission Topics
==========================

Large Features
--------------

For significant features:

1. **Discuss first**: Open an issue to discuss the approach
2. **Break into phases**: Consider multiple smaller PRs
3. **Feature flags**: Use flags to enable features incrementally
4. **Documentation**: Include comprehensive docs and examples

Breaking Changes
----------------

If your change breaks backward compatibility:

1. **Discuss necessity**: Ensure the breaking change is justified
2. **Version planning**: Coordinate with maintainers on timing
3. **Migration guide**: Provide clear upgrade instructions
4. **Deprecation period**: Consider deprecating before removing

Performance Changes
-------------------

For performance-related changes:

1. **Benchmark first**: Measure current performance
2. **Profile changes**: Verify improvements with profiling
3. **Regression tests**: Add tests to prevent performance regression
4. **Document impact**: Explain performance characteristics

Release Process Integration
===========================

Version Planning
----------------

Changes are integrated into releases based on:

- **Patch releases** (x.x.1): Bug fixes, documentation
- **Minor releases** (x.1.x): New features, non-breaking changes  
- **Major releases** (1.x.x): Breaking changes, major features

Your PR will be labeled and assigned to appropriate milestones.

Changelog Updates
-----------------

For user-facing changes, update ``CHANGELOG.md``:

.. code-block:: markdown

   ## [Unreleased]
   
   ### Added
   - Quantum amplitude estimation for VaR calculations (#123)
   - New distribution: Inverse Gaussian (#456)
   
   ### Changed
   - Improved performance of JIT compilation (#789)
   
   ### Fixed
   - Edge case in compound distribution sampling (#321)
   
   ### Deprecated
   - `old_function_name` in favor of `new_function_name` (#654)

Troubleshooting Submissions
===========================

CI Failures
------------

**Tests failing**:

.. code-block:: bash

   # Run the same tests locally
   pytest tests/test_specific.py -v
   
   # Check for environment differences
   pip list  # Compare with CI environment

**Style check failures**:

.. code-block:: bash

   # Auto-fix most issues
   black .
   isort .
   
   # Check remaining issues
   flake8 quactuary/

**Documentation build failures**:

.. code-block:: bash

   cd docs/
   make clean
   make html
   # Check for syntax errors in rst files

Merge Conflicts
---------------

If your branch has conflicts with main:

.. code-block:: bash

   # Update your local main
   git checkout main
   git pull upstream main
   
   # Rebase your feature branch
   git checkout your-feature-branch
   git rebase main
   
   # Resolve conflicts manually, then:
   git add .
   git rebase --continue
   
   # Force push (since history changed)
   git push --force-with-lease origin your-feature-branch

Stale Branches
--------------

If your PR sits idle for a while:

1. **Rebase on latest main** to resolve conflicts
2. **Address any new CI failures**
3. **Ping reviewers** if needed
4. **Consider breaking into smaller PRs** if very large

Getting Help
============

If you need assistance:

- **Comment on your PR**: Ask specific questions
- **GitHub Discussions**: For broader questions
- **Issue tracker**: For bug reports or feature discussions
- **Documentation**: Check our development guides

Communication Tips
==================

Effective Communication
-----------------------

- **Be patient**: Reviews take time, especially for large changes
- **Be humble**: Everyone's goal is improving the project
- **Be collaborative**: Work with reviewers, not against them
- **Be clear**: Explain your reasoning and approach

Cultural Considerations
-----------------------

- **Assume good intent**: Reviewers want to help improve your code
- **English proficiency**: Don't worry if English isn't your first language
- **Different perspectives**: Embrace diverse viewpoints and approaches
- **Learning opportunity**: Use reviews to improve your skills

Success Metrics
===============

A successful contribution:

- **Solves a real problem** for users
- **Maintains code quality** standards
- **Includes comprehensive tests** and documentation
- **Integrates smoothly** with existing codebase
- **Follows project conventions** and best practices

Your contributions make quactuary better for everyone. Thank you for taking the time to follow these guidelines and help us maintain a high-quality project!