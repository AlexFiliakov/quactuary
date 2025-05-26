---
task_id: T014
status: completed
complexity: Medium
last_updated: 2025-05-26 13:56
---

# Task: API Documentation and Best Practices Update

## Description
Following the integration test fixes in T011, several API changes and best practices were discovered that need to be documented. This includes parameter naming conventions for distributions, optimization configuration usage, and tolerance guidelines for stochastic tests.

## Goal / Objectives
- Document all API changes discovered during integration test fixes
- Create best practices guide for writing robust tests
- Update code examples to reflect current API
- Ensure consistency across documentation

## Acceptance Criteria
- [x] All distribution parameter names are documented
- [x] OptimizationConfig usage is clearly explained
- [x] Testing best practices guide is created
- [x] Code examples in documentation are updated
- [x] API migration guide is provided for users

## Subtasks
- [x] Document distribution parameter conventions
  - Create table of all distributions and their parameters
  - Document that 'loc' is used instead of 'location' or 'threshold'
  - Add examples for each distribution type
  - Update docstrings in distribution classes
- [x] Create OptimizationConfig usage guide
  - Document all configuration options
  - Provide examples of common optimization combinations
  - Explain auto_optimize vs manual configuration
  - Show how to create custom optimization strategies
- [x] Write testing best practices guide
  - Document appropriate tolerances for stochastic methods
  - Explain when to use @pytest.mark.skip for hardware-dependent tests
  - Provide guidelines for numerical accuracy testing
  - Show how to write robust integration tests
- [x] Update existing documentation
  - Review and update all code examples in docs/
  - Update README.md with current API examples
  - Fix any outdated parameter names in notebooks
  - Ensure consistency across all documentation
- [x] Create API migration guide
  - List all breaking changes from previous versions
  - Provide before/after examples
  - Include automated migration scripts if possible
  - Add troubleshooting section

## Notes
This documentation update is critical for preventing future confusion about the API. Many of the integration test failures were due to outdated parameter names and unclear optimization configuration requirements.

## Output Log
[2025-05-26 12:25] Task created following T011 completion to document API discoveries
[2025-05-26 13:31] Task status updated to in_progress
[2025-05-26 13:35] Created comprehensive distribution parameter reference guide at docs/source/user_guide/distribution_parameters.rst
[2025-05-26 13:36] Verified distribution docstrings are already consistent with loc parameter naming
[2025-05-26 13:40] Created comprehensive OptimizationConfig usage guide at docs/source/user_guide/optimization_config.rst
[2025-05-26 13:44] Created testing best practices guide at docs/source/user_guide/testing_best_practices.rst
[2025-05-26 13:48] Updated README.md and notebooks to use explicit parameter names for distributions
[2025-05-26 13:51] Created comprehensive API migration guide at docs/source/user_guide/api_migration.rst with automated migration script