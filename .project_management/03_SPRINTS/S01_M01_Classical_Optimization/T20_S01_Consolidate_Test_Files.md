---
task_id: T20_S01
sprint_sequence_id: S01
status: open # open | in_progress | pending_review | done | failed | blocked
complexity: Low # Low | Medium | High
last_updated: 2025-05-26 09:20
---

# Task: Consolidate Test Files

## Description
Consolidate duplicate and related test files to improve test organization and reduce redundancy.

## Goal / Objectives
- Merge related test files for compound distributions
- Organize test files by feature area
- Eliminate test duplication
- Improve test discoverability

## Technical Requirements
- Preserve all existing test coverage
- Maintain clear test organization
- Use pytest best practices
- Keep tests fast and isolated

## Acceptance Criteria
- [ ] Compound distribution tests consolidated
- [ ] No duplicate test cases
- [ ] Clear test file naming convention
- [ ] All tests passing
- [ ] Test coverage maintained at 95%+
- [ ] Test execution time not increased

## Subtasks

### 1. Test Directory Consolidation
- [ ] Move all tests from quactuary/tests/ to tests/
  ```bash
  # Identify what needs to be moved
  find quactuary/tests -name "*.py" -type f
  # Create unified structure
  mkdir -p tests/unit/{distributions,backend,utils}
  mkdir -p tests/integration
  mkdir -p tests/performance
  ```
- [ ] Resolve any naming conflicts
  - Check for duplicate test file names
  - Rename files to be more specific if needed
  - Update imports in moved files
- [ ] Update import paths in all test files
  - Change relative imports to absolute
  - Fix any broken module references
- [ ] Verify pytest still discovers all tests
  ```bash
  pytest --collect-only | grep "test session starts" -A 5
  ```

### 2. Compound Distribution Test Consolidation
- [ ] Analyze test files: test_compound.py, test_compound_binomial.py, test_compound_binomial_comprehensive.py
  - Count test methods in each file
  - Identify overlap in what they test
  - Note unique test scenarios in each
- [ ] Create test coverage matrix
  | Test Scenario | test_compound | test_compound_binomial | test_comprehensive |
  |--------------|--------------|----------------------|-------------------|
  | Basic API    | ✓            | ✓                    | ✓                 |
  | Edge cases   | Partial      | ✓                    | ✓                 |
  | Performance  | ✗            | ✗                    | ✓                 |
- [ ] Design consolidated structure
  ```python
  # Proposed structure
  tests/unit/distributions/
    test_compound_basic.py       # Basic API tests
    test_compound_analytical.py  # Analytical solutions
    test_compound_binomial.py    # Binomial-specific
    test_compound_properties.py  # Statistical properties
  tests/integration/
    test_compound_scenarios.py   # Real-world scenarios
  tests/performance/
    test_compound_benchmarks.py  # Performance tests
  ```
- [ ] Merge tests preserving all coverage
  - Start with union of all test cases
  - Remove exact duplicates
  - Parameterize similar tests
  - Use fixtures to reduce setup duplication
- [ ] Remove redundant test files
  - Verify all tests moved to new structure
  - Check coverage hasn't decreased
  - Delete old files

### 3. Test Organization by Feature Area
- [ ] Review current test organization
  - List all test files and their purposes
  - Identify tests in wrong locations
  - Find tests that should be split
- [ ] Implement consistent structure
  ```
  tests/
  ├── unit/               # Isolated unit tests
  │   ├── distributions/  # Distribution tests
  │   ├── backend/       # Backend switching tests
  │   ├── book/          # Policy/portfolio tests
  │   └── utils/         # Utility function tests
  ├── integration/       # Multi-component tests
  │   ├── scenarios/     # End-to-end scenarios
  │   ├── optimization/  # Optimization combos
  │   └── api/          # API stability tests
  └── performance/       # Benchmark tests
      ├── speed/        # Execution time tests
      └── memory/       # Memory usage tests
  ```
- [ ] Add appropriate pytest markers
  ```python
  # In pytest.ini
  markers =
    unit: marks tests as unit tests (fast, isolated)
    integration: marks tests as integration tests
    performance: marks tests as performance tests
    slow: marks tests as slow (> 1 second)
    memory_intensive: marks tests requiring > 1GB RAM
    requires_numba: marks tests requiring numba
  ```
- [ ] Update conftest.py files
  - Create shared fixtures at appropriate levels
  - Remove duplicate fixture definitions
  - Add fixture documentation

### 4. Test Performance Optimization
- [ ] Profile test execution times
  ```bash
  pytest --durations=20  # Show 20 slowest tests
  ```
- [ ] Identify slow tests (>1 second)
  - Add @pytest.mark.slow decorator
  - Consider if test can be optimized
  - Move to performance suite if appropriate
- [ ] Optimize test fixtures for reuse
  ```python
  # Change from function scope to module/session scope where safe
  @pytest.fixture(scope="module")
  def large_portfolio():
      # Expensive setup, reused across module
      return create_large_portfolio()
  ```
- [ ] Reduce test data generation overhead
  - Use smaller datasets for unit tests
  - Cache generated test data
  - Use deterministic seeds for reproducibility
- [ ] Implement test parallelization
  ```bash
  # Install pytest-xdist
  pip install pytest-xdist
  # Run tests in parallel
  pytest -n auto  # Uses all CPU cores
  ```

### 5. Fix Currently Failing Tests
- [ ] Prioritize test fixes based on impact
  1. API contract tests (prevent breaking changes)
  2. Core functionality tests (ensure correctness)
  3. Integration tests (ensure components work together)
  4. Performance tests (can temporarily skip if needed)
- [ ] Create skip decorators for known issues
  ```python
  @pytest.mark.skip(reason="Waiting for API update in T009")
  def test_old_api():
      pass
  
  @pytest.mark.xfail(reason="Known issue, tracked in #123")
  def test_edge_case():
      pass
  ```
- [ ] Update tests for new APIs
  - Reference T009-T011 for specific fixes needed
  - Create compatibility shims where appropriate
  - Document any behavior changes

### 6. Documentation and Guidelines
- [ ] Create tests/README.md
  ```markdown
  # Test Organization
  
  ## Structure
  - unit/ - Fast, isolated tests of single components
  - integration/ - Tests of multiple components
  - performance/ - Benchmarks and performance tests
  
  ## Running Tests
  - All tests: `pytest`
  - Unit only: `pytest tests/unit -m unit`
  - Skip slow: `pytest -m "not slow"`
  
  ## Writing Tests
  - Use descriptive names: test_<what>_<condition>_<expected>
  - One assertion per test method
  - Use fixtures for common setup
  ```
- [ ] Update CONTRIBUTING.md testing section
  - Test naming conventions
  - When to use each test type
  - How to run different test suites
  - Coverage requirements
- [ ] Add pre-commit hooks for testing
  ```yaml
  # .pre-commit-config.yaml
  - repo: local
    hooks:
    - id: pytest-check
      name: pytest-check
      entry: pytest tests/unit -x --tb=short
      language: system
      pass_filenames: false
      always_run: true
  ```

## Implementation Notes
- Use pytest fixtures effectively to reduce duplication
- Consider parametrized tests for similar test cases
- Maintain backward compatibility for CI/CD

## Output Log

[2025-05-26 09:20]: Task created from T11 subtask extraction