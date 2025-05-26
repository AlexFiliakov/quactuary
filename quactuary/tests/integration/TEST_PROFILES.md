# Test Environment Profiles

This document describes the test environment profiles available for running the quactuary integration test suite.

## Overview

The test suite uses environment profiles to adapt test execution based on available system resources. This ensures tests run reliably across different hardware configurations while maximizing test coverage when resources permit.

## Available Profiles

### 1. Minimal Profile
- **Purpose**: CI/CD environments and resource-constrained systems
- **Requirements**: 
  - CPUs: 1-2 cores
  - Memory: < 4GB RAM
- **Characteristics**:
  - Excludes slow, memory-intensive, and hardware-dependent tests
  - Reduced simulation counts
  - Relaxed performance expectations
  - Test timeout: 5 minutes
- **Use Case**: Quick validation, CI pipelines, development on laptops

### 2. Standard Profile
- **Purpose**: Typical development machines
- **Requirements**:
  - CPUs: 4+ cores
  - Memory: 8GB+ RAM
- **Characteristics**:
  - Includes most tests except extreme hardware-dependent ones
  - Moderate simulation counts
  - Balanced performance expectations
  - Test timeout: 3 minutes
- **Use Case**: Daily development, pre-commit testing

### 3. Performance Profile
- **Purpose**: Dedicated testing hardware or high-end workstations
- **Requirements**:
  - CPUs: 8+ cores
  - Memory: 16GB+ RAM
- **Characteristics**:
  - Runs all tests including performance benchmarks
  - Full simulation counts
  - Strict performance expectations
  - Test timeout: 2 minutes
- **Use Case**: Performance validation, release testing

### 4. CI Profile
- **Purpose**: Continuous Integration environments
- **Requirements**: Same as minimal
- **Characteristics**:
  - Excludes flaky tests
  - Includes code coverage reporting
  - Strict failure handling (stops on first failure)
  - Generates XML reports
- **Use Case**: Automated CI/CD pipelines

### 5. Quick Profile
- **Purpose**: Rapid feedback during development
- **Requirements**: Minimal
- **Characteristics**:
  - Only unit tests (no integration tests)
  - Stops on first failure
  - Minimal output
- **Use Case**: TDD, rapid iteration

## Usage

### Automatic Profile Detection

```bash
# Auto-detect best profile based on system resources
python quactuary/tests/run_tests_with_profile.py
```

### Manual Profile Selection

```bash
# Run with specific profile
python quactuary/tests/run_tests_with_profile.py minimal
python quactuary/tests/run_tests_with_profile.py standard
python quactuary/tests/run_tests_with_profile.py performance

# Run specific tests with profile
python quactuary/tests/run_tests_with_profile.py standard -k test_accuracy

# Run with additional pytest options
python quactuary/tests/run_tests_with_profile.py minimal --verbose --pdb
```

### Environment Variables

Each profile sets the following environment variables:

- `QUACTUARY_TEST_PROFILE`: Profile name (minimal/standard/performance)
- `QUACTUARY_MAX_WORKERS`: Maximum parallel workers
- `QUACTUARY_MAX_MEMORY_MB`: Maximum memory usage in MB
- `QUACTUARY_MAX_TEST_DURATION`: Maximum test duration in seconds

### Direct pytest Usage

You can also set the profile environment variables manually:

```bash
# Linux/Mac
export QUACTUARY_TEST_PROFILE=minimal
pytest quactuary/tests/integration/

# Windows
set QUACTUARY_TEST_PROFILE=minimal
pytest quactuary/tests/integration/
```

## Test Markers and Profiles

Tests are categorized using pytest markers:

- `@pytest.mark.hardware_dependent`: Tests requiring specific hardware
- `@pytest.mark.slow`: Tests taking > 1 minute
- `@pytest.mark.memory_intensive`: Tests requiring > 4GB RAM
- `@pytest.mark.performance`: Performance validation tests
- `@pytest.mark.statistical`: Tests using statistical validation
- `@pytest.mark.flaky`: Tests that may fail intermittently

Profile configurations automatically include/exclude tests based on these markers.

## Adaptive Test Behavior

Tests can adapt their behavior based on the active profile:

```python
from quactuary.tests.integration.test_config import get_test_config, adapt_test_parameters

def test_example():
    # Get current test configuration
    config = get_test_config()
    profile = config['environment']['profile']
    
    # Adapt parameters based on profile
    base_params = {
        'n_simulations': 100_000,
        'n_policies': 1000
    }
    adapted = adapt_test_parameters(base_params)
    
    # Use adapted parameters
    run_simulation(n_sims=adapted['n_simulations'])
```

## Performance Expectations by Profile

| Metric | Minimal | Standard | Performance |
|--------|---------|----------|-------------|
| Min Speedup | 1.2x | 1.5x | 2.0x |
| Max Speedup | 2.0x | 4.0x | 8.0x |
| Memory Limit | 2GB | 4GB | 8GB |
| Test Timeout | 5 min | 3 min | 2 min |
| Parallel Workers | 2 | 4 | 8 |
| Statistical Tolerance | 20% | 10% | 5% |

## Best Practices

1. **Development**: Use `quick` profile for rapid iteration, `standard` for pre-commit
2. **CI/CD**: Always use `ci` profile for consistent results
3. **Performance Testing**: Use `performance` profile on dedicated hardware
4. **Debugging**: Use `minimal` profile to isolate issues
5. **Resource Monitoring**: Tests automatically monitor and report resource usage

## Troubleshooting

### Tests Skipped Due to Resources

If tests are being skipped:
```bash
# Check detected profile
python quactuary/tests/run_tests_with_profile.py auto

# Force a lower profile
python quactuary/tests/run_tests_with_profile.py minimal

# See skipped test reasons
pytest -rs quactuary/tests/integration/
```

### Performance Test Failures

If performance tests fail:
1. Check system load: `top` or `htop`
2. Verify profile: `echo $QUACTUARY_TEST_PROFILE`
3. Run with relaxed expectations: Use `minimal` or `standard` profile
4. Check for background processes consuming resources

### Memory Issues

If encountering memory errors:
1. Use `minimal` profile
2. Run tests in smaller batches
3. Monitor memory: `watch -n 1 free -h`
4. Close other applications

## Profile Configuration Files

Profile configurations are stored in:
- `quactuary/tests/integration/pytest_profiles.ini`: Profile definitions
- `quactuary/tests/integration/test_config.py`: Runtime configuration
- `pytest.ini`: Base pytest configuration

To customize profiles, edit these files or create local overrides.