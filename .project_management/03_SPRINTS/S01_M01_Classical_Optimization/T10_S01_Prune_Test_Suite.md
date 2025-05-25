---
task_id: T10_S01
sprint: S01
sequence: 10
status: open
title: Prune and Reorganize Test Suite
assigned_to: TBD
estimated_hours: 3
actual_hours: 0
priority: medium
risk: low
dependencies: [T07_S01, T08_S01]
last_updated: 2025-01-25
---

# T10_S01: Prune and Reorganize Test Suite

## Description
Consolidate the proliferated test files and remove trivial tests to create a clean, organized test suite. This addresses the critical file organization violations identified in the project review where there are 4 pricing test files and 4 compound test files with unclear purposes.

## Acceptance Criteria
- [ ] Consolidate duplicate test files into single organized files
- [ ] Remove trivial tests (imports, basic Python functionality)
- [ ] Maintain all meaningful test coverage
- [ ] Create clear test organization strategy
- [ ] Document test file purposes and organization
- [ ] Ensure all consolidated tests pass
- [ ] Reduce total test file count by at least 50%

## Subtasks

### 1. Audit Current Test Structure
- [ ] Document all existing test files and their purposes:
  ```
  Current pricing tests:
  - test_pricing.py
  - test_pricing_comprehensive.py  
  - test_pricing_coverage.py
  - test_pricing_full_coverage.py
  
  Current compound tests:
  - test_compound.py
  - test_compound_additional.py
  - test_compound_comprehensive.py
  - test_compound_final.py
  ```
- [ ] Analyze overlap and duplication between files
- [ ] Identify unique tests that must be preserved
- [ ] Count total test cases and coverage

### 2. Identify Trivial Tests for Removal
- [ ] Find and mark for deletion:
  - test_init.py - basic import tests
  - test_setup.py - Python environment tests
  - test_imports.py - module import verification
  - Basic "assert True" style tests
  - Tests that verify Python itself works
- [ ] Document why each trivial test is being removed
- [ ] Verify no important edge cases are hidden in trivial tests

### 3. Design New Test Organization
- [ ] Create consolidated test file structure:
  ```
  tests/
  ├── test_pricing.py          # All pricing functionality
  ├── test_compound.py         # All compound distributions  
  ├── test_book.py             # Policy and portfolio tests
  ├── test_distributions/      # Distribution-specific tests
  │   ├── test_frequency.py
  │   └── test_severity.py
  └── backend/                 # Backend-specific tests
      ├── test_backend.py
      └── test_imports.py       # Keep only if truly necessary
  ```
- [ ] Define clear scope for each remaining test file
- [ ] Plan test class organization within files

### 4. Consolidate Pricing Tests
- [ ] Merge all pricing test files into single test_pricing.py:
  - Core pricing functionality from test_pricing.py
  - Edge cases from test_pricing_comprehensive.py
  - Coverage tests from test_pricing_coverage.py
  - Additional coverage from test_pricing_full_coverage.py
- [ ] Organize into logical test classes:
  ```python
  class TestPricingModelBasics:
      # Basic pricing model functionality
  
  class TestAggregateStatistics:
      # Aggregate statistics calculations
  
  class TestExcessLayerPricing:
      # Excess layer and reinsurance pricing
  
  class TestEdgeCasesAndErrors:
      # Edge cases and error handling
  ```
- [ ] Remove duplicate test cases
- [ ] Ensure comprehensive coverage is maintained

### 5. Consolidate Compound Distribution Tests
- [ ] Merge all compound test files into single test_compound.py:
  - Basic functionality from test_compound.py
  - Additional cases from test_compound_additional.py
  - Comprehensive tests from test_compound_comprehensive.py
  - Final tests from test_compound_final.py
- [ ] Organize into test classes by functionality:
  ```python
  class TestCompoundDistributionFactory:
      # Factory method and creation
  
  class TestAnalyticalSolutions:
      # Poisson-Exponential, Tweedie, etc.
  
  class TestApproximationMethods:
      # Panjer recursion, FFT, etc.
  
  class TestNumericalStability:
      # Edge cases and numerical issues
  ```
- [ ] Remove redundant test cases
- [ ] Preserve all meaningful edge case tests

### 6. Remove Trivial Test Files
- [ ] Delete test_init.py after verifying no hidden important tests
- [ ] Delete test_setup.py if only testing Python basics
- [ ] Remove test_validation.py if covered elsewhere
- [ ] Clean up any other trivial test files identified
- [ ] Update test discovery patterns if needed

### 7. Update Test Configuration
- [ ] Update pytest configuration for new file structure
- [ ] Verify test discovery still works correctly
- [ ] Update coverage configuration if needed
- [ ] Check CI/CD integration still works
- [ ] Update any test runner scripts

### 8. Documentation and Validation
- [ ] Document new test organization strategy:
  ```markdown
  # Test Organization
  
  ## File Structure
  - test_pricing.py: All pricing model functionality
  - test_compound.py: Compound distribution implementations
  - test_book.py: Policy terms and portfolio management
  
  ## Test Categories
  - Unit tests: Test individual methods and functions
  - Integration tests: Test component interactions
  - Edge case tests: Test boundary conditions and errors
  ```
- [ ] Run full test suite to ensure nothing was broken
- [ ] Verify coverage metrics are maintained
- [ ] Update development documentation

### 9. Create Test Maintenance Guidelines
- [ ] Document guidelines for future test additions:
  - When to add new test files vs adding to existing
  - Test class organization patterns
  - Naming conventions
  - What not to test (avoid trivial tests)
- [ ] Add guidelines to development documentation
- [ ] Include in run_dev.py help if relevant

## Before/After Metrics
- **Before**: ~18 test files, many with unclear purposes
- **Target**: ~8-10 test files with clear organization
- **Trivial tests removed**: test_init.py, test_setup.py, test_imports.py
- **Consolidated**: 4 pricing files → 1, 4 compound files → 1
- **Maintained**: All meaningful test coverage and edge cases

## Dependencies
- Should be done after T07 (PricingModel refactor) and T08 (compound simplification)
- May need to update tests based on architecture changes
- Should be final cleanup task before sprint completion

## Output Log
<!-- Add timestamped entries for each subtask completion -->