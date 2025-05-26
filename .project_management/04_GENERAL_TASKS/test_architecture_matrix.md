# Integration Test Architecture Matrix

## Overview
This document categorizes the integration tests in the quactuary project based on their dependencies, purposes, and characteristics.

## Test File Matrix

| File | Test Count | Active | Deprecated | Hardware Deps | Performance Deps | Statistical Deps | Primary Purpose |
|------|------------|--------|------------|---------------|------------------|------------------|-----------------|
| test_accuracy_validation.py | 23 | 23 | 0 | Low | Low | High | Statistical accuracy and numerical precision |
| test_performance_validation.py | 10 | 4 | 6 | High | Very High | Medium | Performance benchmarking and optimization validation |
| test_end_to_end_scenarios.py | 12 | 11 | 1 | High | High | Medium | Real-world insurance portfolio scenarios |
| test_optimization_combinations.py | 9 | 7 | 2 | Medium | High | Low | Optimization strategy combinations |

## Dependency Analysis

### Hardware Dependencies
- **Low**: Tests that can run on minimal hardware (1-2 cores, <4GB RAM)
- **Medium**: Tests requiring moderate resources (4+ cores, 8GB RAM)
- **High**: Tests requiring significant resources (8+ cores, 16GB+ RAM)
- **Very High**: Tests with extreme requirements (parallel processing, large memory)

### Current Hardware-Dependent Tests
1. **Memory-intensive**:
   - test_memory_usage_limits (deprecated)
   - test_memory_leak_detection (deprecated)
   - test_extreme_portfolio_scenarios (10k+ policies)

2. **CPU-intensive**:
   - test_parallel_scaling_efficiency (deprecated)
   - test_optimization_scaling_efficiency (deprecated)
   - test_large_portfolio_optimization

3. **Performance-sensitive**:
   - All tests in test_performance_validation.py
   - test_complex_optimization_combinations

## Test Categories

### 1. Statistical Accuracy Tests
- **Purpose**: Validate numerical precision and statistical correctness
- **Key characteristics**:
  - Use statistical tests (KS, Chi-square, Anderson-Darling)
  - Require appropriate tolerances for stochastic methods
  - Focus on distribution convergence
- **Example tests**:
  - test_distribution_accuracy
  - test_monte_carlo_convergence
  - test_qmc_convergence_rate

### 2. Performance Validation Tests
- **Purpose**: Ensure optimization techniques provide expected speedups
- **Key characteristics**:
  - Compare baseline vs optimized performance
  - Measure memory usage and scaling
  - Highly environment-dependent
- **Example tests**:
  - test_optimization_speedup
  - test_combined_optimization_performance
  - test_qmc_variance_reduction

### 3. End-to-End Scenario Tests
- **Purpose**: Validate realistic insurance workflows
- **Key characteristics**:
  - Use industry-specific scenarios (Property, Liability, Workers Comp, Auto)
  - Test full pricing workflow
  - Combine multiple components
- **Example tests**:
  - test_property_insurance_portfolio
  - test_workers_comp_portfolio
  - test_multi_line_portfolio

### 4. Optimization Combination Tests
- **Purpose**: Test various optimization strategy combinations
- **Key characteristics**:
  - Use OptimizationConfig for strategy management
  - Test fallback mechanisms
  - Validate compatibility of different optimizations
- **Example tests**:
  - test_vectorization_only
  - test_all_optimizations_combined
  - test_optimization_fallback_mechanism

## Problematic Patterns Identified

### 1. Overly Strict Tolerances
- Many tests use unrealistic tolerances (e.g., 1e-6 for stochastic methods)
- Should use confidence intervals or statistical tests instead

### 2. Hardware Assumptions
- Fixed expectations for CPU counts and memory
- No adaptation to available resources
- Performance targets assume high-end hardware

### 3. Missing Test Markers
- No consistent use of markers for categorization
- Missing markers for:
  - @pytest.mark.hardware_dependent
  - @pytest.mark.performance
  - @pytest.mark.statistical
  - @pytest.mark.slow

### 4. Insufficient Test Isolation
- Some tests may be affected by previous test runs
- Missing proper cleanup in fixtures
- Potential for test order dependencies

## Recommendations

1. **Implement Test Markers**:
   ```python
   @pytest.mark.hardware_dependent
   @pytest.mark.performance
   @pytest.mark.statistical
   @pytest.mark.slow
   @pytest.mark.integration
   ```

2. **Create Environment Profiles**:
   - Minimal: For CI/CD with limited resources
   - Standard: For typical development machines
   - Performance: For dedicated performance testing

3. **Improve Statistical Tests**:
   - Use appropriate statistical methods for validation
   - Implement confidence intervals
   - Consider multiple test runs for stability

4. **Make Performance Configurable**:
   - Use environment variables for performance expectations
   - Implement adaptive baselines
   - Skip tests when hardware is insufficient