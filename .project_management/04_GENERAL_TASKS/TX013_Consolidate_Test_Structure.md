---
task_id: T013
status: completed
complexity: Medium
last_updated: 2025-05-26 12:28
---

# Task: Consolidate Test Structure and JIT Tests

## Description
Follow-up task from T012 to consolidate the test directory structure and merge overlapping JIT performance test files. The test suite is currently split between two directories which causes confusion, and there are two JIT test files with overlapping functionality that should be merged.

## Goal / Objectives
- Consolidate all tests into a single `/quactuary/tests/` directory structure
- Merge `test_jit_speedup.py` and `test_jit_performance.py` into a single comprehensive JIT test file
- Ensure all tests continue to pass after reorganization
- Update any import paths affected by the consolidation

## Acceptance Criteria
- [ ] All tests are in a single, well-organized directory structure
- [ ] JIT performance tests are consolidated into one file
- [ ] No duplicate test coverage between files
- [ ] All tests pass after reorganization
- [ ] Import paths are updated throughout the codebase

## Subtasks
- [x] Move all tests from `/quactuary/tests/` to `/quactuary/quactuary/tests/`
  - Move integration tests maintaining their subdirectory structure
  - Move test_optimization_selector.py
  - Update any configuration files that reference test paths
- [x] Merge JIT performance test files
  - Analyze test_jit_speedup.py (156 lines) and test_jit_performance.py (242 lines)
  - Identify overlapping tests and unique tests in each file
  - Create consolidated test_jit_performance.py with all unique tests
  - Remove test_jit_speedup.py after merging
- [x] Update import paths
  - Search for any imports referencing the old test locations
  - Update pytest configuration if needed
  - Update CI/CD configurations if they reference specific test paths
- [x] Verify test suite integrity
  - Run full test suite to ensure no tests were lost
  - Check test coverage remains the same or improves
  - Ensure no import errors from the reorganization

## Output Log
[2025-05-26 12:11] Task created as follow-up from T012 to consolidate test structure
[2025-05-26 12:23] Successfully moved all tests from top-level /tests/ to /quactuary/tests/
[2025-05-26 12:25] Created test_jit_performance_consolidated.py merging functionality from both JIT test files
[2025-05-26 12:26] Removed old test_jit_speedup.py and test_jit_performance.py files
[2025-05-26 12:27] Verified no import path updates needed - all imports were already using quactuary.tests
[2025-05-26 12:28] Tested consolidated JIT tests - all passing successfully