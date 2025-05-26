# Test Design Guidelines

This document provides guidelines for designing and implementing tests in the quactuary project.

## Test Categories

### 1. Unit Tests
**Purpose**: Test individual functions or classes in isolation

**When to use**:
- Testing pure functions with no external dependencies
- Testing individual class methods
- Testing edge cases and error handling
- Testing data validation logic

**Characteristics**:
- Fast execution (< 0.1 seconds)
- No file I/O or network calls
- Deterministic results
- Heavy use of mocks for dependencies

**Example**:
```python
def test_lognormal_mean():
    """Unit test for Lognormal distribution mean calculation."""
    dist = Lognormal(shape=1.0, loc=0, scale=1.0)
    expected_mean = np.exp(0.5)  # Analytical result
    assert abs(dist.mean() - expected_mean) < 1e-10
```

### 2. Integration Tests
**Purpose**: Test interactions between multiple components

**When to use**:
- Testing pricing model with actual distributions
- Testing optimization combinations
- Testing data flow through multiple layers
- Testing configuration and setup

**Characteristics**:
- Moderate execution time (0.1 - 60 seconds)
- May use real data structures
- Tests actual integration points
- Limited use of mocks

**Example**:
```python
@pytest.mark.integration
def test_portfolio_pricing_integration():
    """Integration test for portfolio pricing workflow."""
    portfolio = generate_portfolio()
    pm = PricingModel(portfolio)
    result = pm.simulate(n_sims=1000)
    assert result.estimates['mean'] > 0
```

### 3. End-to-End Tests
**Purpose**: Test complete business scenarios

**When to use**:
- Testing complete workflows from start to finish
- Testing realistic business scenarios
- Validating system behavior under production-like conditions
- Testing performance characteristics

**Characteristics**:
- Longer execution time (1-300 seconds)
- Uses realistic data volumes
- Tests full stack integration
- No mocks except for external services

**Example**:
```python
@pytest.mark.integration
@pytest.mark.slow
def test_insurance_portfolio_workflow():
    """E2E test for complete insurance portfolio analysis."""
    # Create realistic portfolio
    # Run full pricing workflow
    # Validate business results
```

## Choosing Test Types

### Decision Matrix

| Scenario | Unit Test | Integration Test | E2E Test |
|----------|-----------|------------------|-----------|
| Algorithm correctness | ✓ | | |
| Distribution parameters | ✓ | | |
| Error handling | ✓ | | |
| Component interaction | | ✓ | |
| Optimization effectiveness | | ✓ | |
| Performance validation | | ✓ | ✓ |
| Business logic validation | | | ✓ |
| User workflow | | | ✓ |

## Best Practices

### 1. Test Isolation
```python
@pytest.fixture
def clean_environment():
    """Ensure test isolation."""
    # Setup
    original_backend = get_backend()
    yield
    # Teardown
    set_backend(original_backend)
```

### 2. Appropriate Use of Mocks

**DO Mock**:
- External services (databases, APIs)
- File system operations in unit tests
- Time-dependent functions
- Random number generation (when testing logic, not statistics)

**DON'T Mock**:
- Core business logic
- Mathematical operations
- Data structures you're testing
- Integration points in integration tests

### 3. Fixtures Best Practices

**Scope Management**:
```python
@pytest.fixture(scope="module")  # Expensive setup
def large_portfolio():
    return generate_portfolio(n_policies=10000)

@pytest.fixture(scope="function")  # Cheap setup
def small_portfolio():
    return generate_portfolio(n_policies=10)
```

**Parameterized Fixtures**:
```python
@pytest.fixture(params=["small", "medium", "large"])
def portfolio_size(request):
    sizes = {"small": 10, "medium": 100, "large": 1000}
    return generate_portfolio(n_policies=sizes[request.param])
```

### 4. Performance Testing Patterns

**Relative Comparisons**:
```python
def test_optimization_speedup():
    # Compare relative performance, not absolute times
    baseline_time = measure_baseline()
    optimized_time = measure_optimized()
    speedup = baseline_time / optimized_time
    
    # Use profile-based expectations
    min_speedup = get_test_config()['expectations']['min_speedup']
    assert speedup >= min_speedup
```

**Resource Monitoring**:
```python
def test_memory_efficiency(memory_monitor):
    memory_monitor.record("start")
    # Run operation
    memory_monitor.record("peak")
    
    peak_usage = memory_monitor.get_peak_usage_mb()
    assert_memory_efficiency(peak_usage, expected_limit)
```

### 5. Statistical Testing Patterns

**Multiple Runs for Stability**:
```python
def test_statistical_property():
    results = []
    for i in range(10):  # Multiple runs
        result = run_simulation(seed=42 + i)
        results.append(result)
    
    # Use statistical tests, not exact equality
    validator = EnhancedStatisticalValidator()
    test = validator.confidence_interval_test(
        np.array(results),
        expected_value
    )
    assert test['passes_test']
```

**Adaptive Tolerances**:
```python
def test_with_adaptive_tolerance():
    # Use different tolerances based on value magnitude
    validator = EnhancedStatisticalValidator()
    test = validator.adaptive_tolerance_test(
        actual_value,
        expected_value,
        base_tolerance=0.01  # 1% for large values
    )
    assert test['passes_test']
```

## Test Organization

### Directory Structure
```
tests/
├── unit/              # Fast, isolated tests
│   ├── test_distributions.py
│   ├── test_utils.py
│   └── test_validators.py
├── integration/       # Component interaction tests
│   ├── test_pricing_integration.py
│   ├── test_optimization_combinations.py
│   └── test_accuracy_validation.py
└── e2e/              # Full workflow tests
    ├── test_insurance_scenarios.py
    └── test_performance_benchmarks.py
```

### Naming Conventions
- Test files: `test_<module_name>.py`
- Test classes: `Test<ComponentName>`
- Test methods: `test_<specific_behavior>`
- Fixtures: `<noun>` or `<adjective>_<noun>`

### Documentation Standards
Every test should have:
1. Clear docstring explaining what is tested
2. Comments for non-obvious logic
3. Assertion messages explaining failures
4. TODO/NOTE comments for known issues

Example:
```python
def test_qmc_convergence_rate():
    """Test that QMC achieves better convergence than standard MC.
    
    QMC should achieve O(1/n) convergence vs O(1/sqrt(n)) for MC.
    This test may be sensitive to dimension and smoothness of integrand.
    
    TODO: Investigate dimension-dependent convergence rates
    """
    # Test implementation
    assert qmc_rate < mc_rate, (
        f"QMC convergence rate {qmc_rate:.3f} not better than "
        f"MC rate {mc_rate:.3f}"
    )
```

## Common Pitfalls to Avoid

1. **Hard-coded values**: Use fixtures and configuration
2. **Timing-based assertions**: Use relative comparisons
3. **Exact floating-point equality**: Use appropriate tolerances
4. **Resource assumptions**: Adapt to available resources
5. **Order dependencies**: Each test should be independent
6. **Missing cleanup**: Always clean up resources
7. **Overmocking**: Don't mock what you're testing
8. **Undermocking**: Don't test external dependencies

## Test Maintenance

### Regular Review
- Review flaky tests monthly
- Update tolerances based on empirical data
- Remove obsolete tests
- Add tests for new features

### Performance Baseline Management
- Update baselines quarterly
- Document reason for baseline changes
- Track performance trends
- Alert on significant regressions

### Documentation Updates
- Keep test docs in sync with code
- Document test environment requirements
- Maintain troubleshooting guides
- Share test patterns and learnings