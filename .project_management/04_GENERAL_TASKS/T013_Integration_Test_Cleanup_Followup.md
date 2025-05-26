---
task_id: T013
status: open
complexity: Medium
last_updated: 2025-05-26 12:21
---

# Task: Integration Test Cleanup Follow-up

## Description
Following the completion of T011, there are still 8 failing integration tests that need investigation and resolution. These tests are failing for various reasons including tolerance issues, API mismatches, and test design problems that weren't addressed in the initial fix.

## Goal / Objectives
- Fix remaining 8 failing integration tests
- Ensure all integration tests pass reliably
- Document any architectural decisions or API changes needed
- Clean up test suite for maintainability

## Acceptance Criteria
- [ ] All integration tests pass without failures
- [ ] Tests are robust and not overly sensitive to stochastic variations
- [ ] Test documentation is clear about what's being tested
- [ ] No flaky tests that fail intermittently

## Subtasks
- [ ] Fix test_heterogeneous_small_portfolio failure
  - Investigate tolerance issues with mixed portfolio types
  - May need to adjust assertions for heterogeneous cases
- [ ] Fix test_property_insurance_portfolio failure
  - Check if Pareto distribution parameters are causing issues
  - Verify tail risk calculations are appropriate
- [ ] Fix test_mixed_lines_portfolio failure
  - Similar to heterogeneous portfolio issues
  - May need portfolio-specific tolerances
- [ ] Fix test_speedup_targets_by_size[large] failure
  - Last remaining performance test failure
  - Consider if large portfolio speedup expectations are realistic
- [ ] Review test_moments_preservation failure
  - Statistical test may be too strict
  - Consider using more appropriate statistical tests
- [ ] Review test_correlation_structure_preservation failure
  - Complex test that may need redesign
  - Consider if this level of testing is necessary
- [ ] Review test_extreme_value_handling failure
  - Edge case test that may have unrealistic expectations
  - Ensure extreme value handling is tested appropriately

## Notes
This is a follow-up task from T011 which successfully reduced integration test failures from 50+ to 8. The remaining failures appear to be more complex issues that require deeper investigation rather than simple API fixes.

## Output Log
[2025-05-26 12:21] Task created as follow-up to T011 for remaining integration test failures