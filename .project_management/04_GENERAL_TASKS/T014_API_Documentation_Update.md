---
task_id: T014
status: open
complexity: Medium
last_updated: 2025-05-26 12:25
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
- [ ] All distribution parameter names are documented
- [ ] OptimizationConfig usage is clearly explained
- [ ] Testing best practices guide is created
- [ ] Code examples in documentation are updated
- [ ] API migration guide is provided for users

## Subtasks
- [ ] Document distribution parameter conventions
  - Create table of all distributions and their parameters
  - Document that 'loc' is used instead of 'location' or 'threshold'
  - Add examples for each distribution type
  - Update docstrings in distribution classes
- [ ] Create OptimizationConfig usage guide
  - Document all configuration options
  - Provide examples of common optimization combinations
  - Explain auto_optimize vs manual configuration
  - Show how to create custom optimization strategies
- [ ] Write testing best practices guide
  - Document appropriate tolerances for stochastic methods
  - Explain when to use @pytest.mark.skip for hardware-dependent tests
  - Provide guidelines for numerical accuracy testing
  - Show how to write robust integration tests
- [ ] Update existing documentation
  - Review and update all code examples in docs/
  - Update README.md with current API examples
  - Fix any outdated parameter names in notebooks
  - Ensure consistency across all documentation
- [ ] Create API migration guide
  - List all breaking changes from previous versions
  - Provide before/after examples
  - Include automated migration scripts if possible
  - Add troubleshooting section

## Notes
This documentation update is critical for preventing future confusion about the API. Many of the integration test failures were due to outdated parameter names and unclear optimization configuration requirements.

## Output Log
[2025-05-26 12:25] Task created following T011 completion to document API discoveries