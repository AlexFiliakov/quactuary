# Removed Tests Log

This document tracks tests that were removed from the quActuary test suite due to testing non-existent functionality.

## Date: 2025-05-26

### Removed Files

1. **test_benchmarks.py**
   - Reason: Tests entirely based on mocked BenchmarkResult and PerformanceBenchmark classes that don't exist in the actual codebase
   - The file mocked these classes internally and tested the mocks, not actual functionality
   - No actual implementation exists in quactuary.benchmarks for these classes

2. **test_performance_regression.py**
   - Reason: Also tests mocked BenchmarkResult and PerformanceBenchmark classes
   - Similar to test_benchmarks.py, it creates internal mocks of non-existent classes
   - Performance testing should be done through actual implementation, not mocks

### Tests with Missing Functions

3. **test_zero_inflated_comprehensive.py** (Modified, not removed)
   - Removed test methods that reference non-existent functions:
     - `test_vuong_test_zi_vs_standard` - references `vuong_test` function
     - `test_vuong_test_no_zero_inflation` - references `vuong_test` function
     - `test_vuong_test_compound_distributions` - references `vuong_test` function
     - `test_score_test_significant_zi` - references `score_test_zi` function
     - `test_score_test_no_zi` - references `score_test_zi` function
     - `test_score_test_boundary_case` - references `score_test_zi` function
   - These functions (`vuong_test` and `score_test_zi`) are not imported or defined anywhere

### Test Organization Issues Found

4. **Split Test Directories**
   - Tests are split between two directories:
     - `/quactuary/tests/` (top-level, contains integration tests and a few others)
     - `/quactuary/quactuary/tests/` (nested, contains most unit tests)
   - This split causes confusion and should be consolidated into a single test directory

5. **Overlapping Test Files Identified**
   - **JIT Performance Tests**: 
     - `test_jit_speedup.py` (156 lines) - Tests JIT speedup vs baseline
     - `test_jit_performance.py` (242 lines) - Tests JIT compilation performance
     - Both test similar functionality and should be consolidated
   - **Compound Distribution Tests**:
     - `test_compound.py` - Already consolidated, comprehensive tests
     - `test_compound_binomial.py` - Specific binomial compound tests
     - `test_compound_binomial_comprehensive.py` - More comprehensive binomial tests
     - These appear to be appropriately separated by scope

### Summary

The removed tests were not providing value as they tested mock implementations rather than actual code. This cleanup reduces noise in the test suite and focuses testing efforts on actual functionality.

### Recommendations for Further Cleanup

1. Consolidate all tests into a single `/quactuary/tests/` directory structure
2. Merge `test_jit_speedup.py` and `test_jit_performance.py` into a single comprehensive JIT test file
3. Review property-based tests for redundancy (many files have `_comprehensive` suffix suggesting possible duplication)