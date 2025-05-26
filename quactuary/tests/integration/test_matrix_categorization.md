# Integration Test Matrix Categorization

## Overview
This document categorizes the integration test files based on their purpose, dependencies, and characteristics.

## Test File Matrix

### 1. test_accuracy_validation.py
**Purpose**: Validates numerical accuracy and statistical properties across optimization combinations

**Categories**:
- **Test Type**: Accuracy & Correctness Testing
- **Key Classes**: 
  - `StatisticalValidator`: Statistical testing framework (KS test, Anderson-Darling, Chi-square, relative error)
  - `TestNumericalAccuracy`: Numerical accuracy across optimizations
  - `TestStatisticalProperties`: Distribution shape preservation
  - `TestEdgeCases`: Zero loss handling, extreme value handling
  - `TestRiskMeasures`: VaR/TVaR validation at multiple confidence levels
  - `TestDistributionCombinations`: Various frequency/severity distribution combinations

**Dependencies**:
- Hardware: Low to moderate (CPU-bound calculations)
- Performance expectations: Relaxed tolerances (1e-2 to 5e-2 for means, 0.1-0.3 for quantiles)
- Statistical assumptions: Assumes convergence in distribution, statistical test validity

**Skip Markers**: None - all tests are active

**Notable Features**:
- Parametrized tolerance testing
- Multiple statistical test methods
- Heavy focus on edge cases (zero losses, extreme values)
- Tests distribution combinations (Poisson/Gamma, NegativeBinomial/Pareto, etc.)

---

### 2. test_performance_validation.py
**Purpose**: Validates performance improvements and efficiency metrics

**Categories**:
- **Test Type**: Performance & Scalability Testing
- **Key Classes**:
  - `PerformanceBenchmark`: Comprehensive benchmarking utilities
  - `TestSpeedupValidation`: Speedup targets by portfolio size
  - `TestMemoryEfficiency`: Memory usage and leak detection
  - `TestQMCConvergence`: QMC convergence properties
  - `TestBaselineComparison`: Regression detection

**Dependencies**:
- Hardware: High (memory and CPU intensive)
- Performance expectations: Size-dependent targets (2x-4x speedup)
- Statistical assumptions: QMC convergence rates

**Skip Markers**: 
- `test_scaling_curve_analysis` - "Deprecated: Scaling curves vary by hardware"
- `test_memory_leak_detection` - "Deprecated: Use external profiling tools"
- `test_qmc_convergence_rate` - "Deprecated: QMC rates depend on problem dimensionality"
- `test_qmc_vs_mc_comparison` - "Deprecated: Comparison covered in qmc_diagnostics tests"
- `test_baseline_regression_detection` - "Deprecated: Requires pre-established baseline data"
- `test_performance_trend_analysis` - "Deprecated: Requires historical performance data"

**Notable Features**:
- Baseline result persistence (JSON)
- Memory monitoring with psutil
- Parallel scaling efficiency measurement
- Convergence rate tracking

---

### 3. test_end_to_end_scenarios.py
**Purpose**: End-to-end testing with realistic insurance portfolio scenarios

**Categories**:
- **Test Type**: Integration & Scenario Testing
- **Key Classes**:
  - `TestSmallPortfolioScenarios`: 10-100 policies
  - `TestMediumPortfolioScenarios`: 100-1000 policies
  - `TestLargePortfolioScenarios`: 1000+ policies
  - `TestExtremeScenarios`: 10k+ policies and edge cases

**Dependencies**:
- Hardware: Very high for large/extreme scenarios
- Performance expectations: Time limits (30s-1800s depending on size)
- Statistical assumptions: Industry-specific distributions

**Skip Markers**:
- `test_parallel_scaling_validation` - "Deprecated: Parallel scaling depends on hardware"

**Notable Features**:
- Realistic insurance scenarios (Property, Liability, Workers Comp, Auto)
- Industry-specific validations (e.g., tail ratios for property insurance)
- Mixed portfolio testing
- Edge case handling (zero losses, extreme values)
- Memory limits: 1GB (small), 2GB (medium), 4GB (large), 16GB (extreme)

---

### 4. test_optimization_combinations.py
**Purpose**: Tests different combinations of optimization strategies

**Categories**:
- **Test Type**: Optimization Strategy Testing
- **Key Classes**:
  - `TestOptimizationCombinations`: Main test suite for optimization combinations

**Dependencies**:
- Hardware: Moderate to high
- Performance expectations: Strategy-dependent
- Statistical assumptions: Numerical stability across optimizations

**Skip Markers**:
- `test_full_optimization_combination` - "Deprecated: Too many variables for reliable testing"
- `test_optimization_scaling_efficiency` - "Deprecated: Scaling depends on hardware"

**Notable Features**:
- Single optimization testing (JIT, QMC, parallel, vectorization, memory)
- Binary combinations (pairs of optimizations)
- Triple combinations
- Fallback mechanism testing
- Numerical stability validation
- Uses `OptimizationConfig` and `UserPreferences` classes

---

## Summary Statistics

### Test Coverage by Category
- **Accuracy Testing**: 23 tests (including parametrized)
- **Performance Testing**: 10 tests (4 active, 6 deprecated)
- **Scenario Testing**: 12 tests (11 active, 1 deprecated)
- **Optimization Testing**: 9 tests (7 active, 2 deprecated)

### Hardware Dependencies
- **Low**: Accuracy tests
- **Moderate**: Small/medium portfolio scenarios, optimization combinations
- **High**: Performance tests, large portfolio scenarios
- **Very High**: Extreme scenarios

### Deprecated Tests
- **Total**: 9 deprecated tests
- **Reasons**: 
  - Hardware-specific behavior (4)
  - Requires external tools/data (3)
  - Too complex/unreliable (1)
  - Covered elsewhere (1)

### Key Markers Used
- `@pytest.mark.integration`: All tests
- `@pytest.mark.slow`: Large/extreme scenarios
- `@pytest.mark.memory_intensive`: Memory-heavy tests
- `@pytest.mark.performance`: Performance validation tests
- `@pytest.mark.accuracy`: Numerical accuracy tests
- `@pytest.mark.parametrize`: Extensive use for multiple test scenarios

### Common Infrastructure (from conftest.py)
- **Fixtures**: 
  - Portfolio sizes (tiny, small, medium, large, extreme)
  - Performance profiler
  - Memory monitor
  - Optimization configurations
- **Utilities**:
  - `assert_numerical_accuracy()`
  - `assert_performance_improvement()`
  - `assert_memory_efficiency()`
  - `generate_deterministic_portfolio()`