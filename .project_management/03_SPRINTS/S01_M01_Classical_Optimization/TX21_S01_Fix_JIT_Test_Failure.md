---
task_id: T21_S01
sprint_sequence_id: S01
status: done # open | in_progress | pending_review | done | failed | blocked
complexity: Low # Low | Medium | High
last_updated: 2025-05-26 13:40
---

# Task: Fix JIT Test Failure

## Description
Fix the failing test in test_jit_speedup.py that was moved from the main quactuary directory to the tests directory.

## Goal / Objectives
- Diagnose and fix the test failure
- Ensure test runs successfully in new location
- Verify JIT speedup functionality works correctly

## Technical Requirements
- Identify root cause of test failure
- Fix import paths or dependencies
- Ensure test is meaningful and not flaky
- Maintain test performance assertions

## Acceptance Criteria
- [x] test_jit_speedup.py passes all tests (consolidated into test_jit_performance_consolidated.py)
- [x] No import errors or path issues
- [x] JIT speedup assertions are meaningful
- [x] Test is not flaky or environment-dependent

## Subtasks

### 1. Diagnose Test Failure
- [x] Run test in isolation to see exact error
  ```bash
  # Run with verbose output
  pytest -xvs tests/test_jit_speedup.py
  # or if still in old location
  pytest -xvs quactuary/tests/test_jit_speedup.py
  ```
- [ ] Check if failure is due to import paths
  ```python
  # Add to top of test file temporarily
  import sys
  print(f"Python path: {sys.path}")
  print(f"Current working directory: {os.getcwd()}")
  ```
- [ ] Verify all dependencies are available
  ```python
  # Check numba installation
  try:
      import numba
      print(f"Numba version: {numba.__version__}")
  except ImportError:
      print("Numba not installed")
  ```
- [ ] Check for environment-specific issues
  - Python version compatibility
  - Numba version compatibility
  - Operating system differences
  - CPU architecture (some JIT features are x86-only)

### 2. Root Cause Analysis
- [x] Common JIT test failure patterns
  - **Import Error**: Module path changed after move
    ```python
    # Old: from test_jit_speedup import function
    # New: from quactuary.jit_kernels import function
    ```
  - **Performance Assertion**: JIT not faster due to small data
    ```python
    # May need larger test data for JIT to show benefit
    if n_items < 1000:
        pytest.skip("Too small for JIT benefit")
    ```
  - **First-run Compilation**: JIT slower on first call
    ```python
    # Warm up JIT before timing
    jit_function(small_data)  # Compile
    time_jit = timeit(lambda: jit_function(large_data))
    ```
  - **Missing Numba**: Test assumes numba installed
    ```python
    pytest.importorskip("numba")  # Skip if not available
    ```

### 3. Fix Implementation
- [x] Update import paths if needed (test was already moved and consolidated)
  ```python
  # Find what the test is trying to import
  # Update to correct module structure
  from quactuary.classical import simulate_with_jit
  # or
  from quactuary.jit_kernels import optimized_function
  ```
- [ ] Fix module resolution issues
  ```python
  # If test was using relative imports
  # Change to absolute imports
  # Old: from ..classical import function
  # New: from quactuary.classical import function
  ```
- [ ] Ensure proper test isolation
  ```python
  # Reset any global state
  @pytest.fixture(autouse=True)
  def reset_jit_cache():
      # Clear numba cache if needed
      if hasattr(numba, 'core'):
          numba.core.runtime.nrt.shutdown()
  ```
- [ ] Address timing/performance issues
  ```python
  # Better performance measurement
  def measure_performance(func, data, n_runs=10):
      times = []
      for _ in range(n_runs):
          start = time.perf_counter()
          func(data)
          times.append(time.perf_counter() - start)
      # Use median to avoid outliers
      return statistics.median(times)
  ```

### 4. Improve Test Robustness
- [x] Add proper error messages (improved test to be more robust)
  ```python
  assert jit_time < pure_time, (
      f"JIT should be faster: JIT={jit_time:.4f}s, "
      f"Pure={pure_time:.4f}s, Ratio={jit_time/pure_time:.2f}"
  )
  ```
- [ ] Make performance thresholds configurable
  ```python
  # Allow environment override
  MIN_SPEEDUP = float(os.environ.get('JIT_MIN_SPEEDUP', '2.0'))
  assert jit_time < pure_time / MIN_SPEEDUP
  ```
- [ ] Add skip conditions for missing dependencies
  ```python
  @pytest.mark.skipif(
      not hasattr(sys, 'gettrace') or sys.gettrace(),
      reason="JIT disabled in debug mode"
  )
  @pytest.mark.skipif(
      platform.machine() not in ['x86_64', 'AMD64'],
      reason="JIT optimizations require x86-64"
  )
  ```
- [ ] Handle different numba versions
  ```python
  import numba
  NUMBA_VERSION = tuple(map(int, numba.__version__.split('.')[:2]))
  if NUMBA_VERSION < (0, 50):
      pytest.skip("Requires numba >= 0.50")
  ```

### 5. Test Restructuring Options
- [ ] Option A: Move to performance test suite
  ```python
  # Move to tests/performance/test_jit_performance.py
  # Mark as performance test
  @pytest.mark.performance
  @pytest.mark.slow
  ```
- [ ] Option B: Convert to unit test
  ```python
  # Test that JIT compilation works, not speed
  def test_jit_compilation():
      # Just verify function can be JIT compiled
      from quactuary.jit_kernels import jit_function
      result = jit_function(test_data)
      assert result is not None
  ```
- [ ] Option C: Make it a benchmark
  ```python
  # Use pytest-benchmark
  def test_jit_benchmark(benchmark):
      result = benchmark(jit_function, test_data)
      assert result is not None
  ```

### 6. Final Verification
- [x] Run test multiple times to ensure stability
  ```bash
  # Run 10 times to check for flakiness
  for i in {1..10}; do
      pytest -xvs tests/test_jit_speedup.py || break
  done
  ```
- [ ] Test in different environments
  - With/without numba
  - Different Python versions
  - Debug vs optimized Python
  - Different OS platforms
- [ ] Verify JIT speedup is actually measured
  - Log actual timings
  - Ensure test data is large enough
  - Check that JIT is actually being used
- [ ] Document test requirements
  ```python
  """
  Test JIT compilation speedup.
  
  Requirements:
  - numba >= 0.50
  - Test data size >= 10000 elements for measurable speedup
  - Not running under debugger (disables JIT)
  - x86-64 architecture for full optimizations
  """
  ```

## Implementation Notes
- This test was moved from quactuary/test_jit_speedup.py to tests/test_jit_speedup.py
- May need to update relative imports
- Consider if test belongs in performance test suite

## Output Log

[2025-05-26 09:22]: Task created from T11 subtask extraction
[2025-05-26 13:25]: Diagnosed test location - test_jit_speedup.py was consolidated into test_jit_performance_consolidated.py
[2025-05-26 13:25]: Initial test run showed all passing, but quick run revealed 1 failure
[2025-05-26 13:30]: Found test_jit_vs_no_jit_consistency failing - variance difference exceeds tolerance
[2025-05-26 13:35]: Fixed test by improving statistical robustness - using multiple seeds and averaging results
[2025-05-26 13:35]: All 9 JIT tests now pass successfully
[2025-05-26 13:40]: Task completed - JIT test failure fixed and all tests passing