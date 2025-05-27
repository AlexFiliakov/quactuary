---
task_id: TX19_S01
sprint_sequence_id: S01
status: deferred # open | in_progress | pending_review | done | failed | blocked | deferred
complexity: Medium # Low | Medium | High
last_updated: 2025-05-26 23:20
---

# Task: Merge JIT Implementations

## Description
Consolidate JIT-compiled versions into their base modules using feature flags instead of separate files.

## Goal / Objectives
- Merge classical_jit.py into classical.py with JIT as optional feature
- Merge sobol_optimized.py into sobol.py with optimization flag
- Reduce file proliferation and maintenance burden
- Improve code organization

## Technical Requirements
- Implement feature flags for JIT compilation
- Use conditional imports for numba dependencies
- Preserve performance benefits of JIT compilation
- Maintain backward compatibility

## Acceptance Criteria
- [ ] classical_jit.py merged into classical.py
- [ ] sobol_optimized.py merged into sobol.py
- [ ] JIT functionality controlled by feature flags
- [ ] No performance regression when JIT enabled
- [ ] All tests passing
- [ ] Duplicate files removed

## Subtasks

### 1. Classical JIT Merge
- [ ] Analyze differences between classical.py and classical_jit.py
  - Document all functions that have JIT versions
  - Compare implementations for correctness
  - Note any algorithm differences beyond JIT decoration
  - Identify shared utility functions
- [ ] Design unified API with JIT as optional feature
  ```python
  # Recommended implementation pattern
  def simulate(self, n_sims: int, use_jit: bool = None):
      """
      Simulate aggregate losses.
      
      Args:
          n_sims: Number of simulations
          use_jit: Use JIT compilation. If None, auto-detect based on size.
      """
      if use_jit is None:
          use_jit = self._should_use_jit(n_sims)
      
      if use_jit and HAS_NUMBA:
          return self._simulate_jit(n_sims)
      else:
          return self._simulate_pure(n_sims)
  ```
- [ ] Add use_jit parameter to classical simulation functions
  - Update method signatures
  - Add auto-detection logic based on problem size
  - Document when JIT is beneficial
- [ ] Implement conditional JIT compilation
  - Lazy import of numba to improve import time
  - Graceful fallback if numba not available
  - Clear error messages if JIT requested but unavailable
- [ ] Update all imports from classical_jit
  - Find all imports: `grep -r "classical_jit" .`
  - Update to use classical with use_jit parameter
  - Verify functionality preserved
- [ ] Remove classical_jit.py after verification
  - Run full test suite
  - Benchmark to ensure no performance regression
  - Archive file in case rollback needed

### 2. Sobol Optimization Merge
- [ ] Compare sobol.py and sobol_optimized.py implementations
  - Identify optimization techniques used
  - Check if optimizations are always beneficial
  - Document performance characteristics
- [ ] Add optimization flags to sobol.py
  ```python
  class SobolEngine:
      def __init__(self, dimension: int, scramble: bool = False, 
                   optimize: bool = None):
          """
          Args:
              optimize: Use optimized algorithms. If None, auto-detect.
          """
          if optimize is None:
              optimize = dimension > 10  # Example heuristic
  ```
- [ ] Merge optimized algorithms with feature detection
  - Integrate optimized algorithms as methods
  - Add runtime selection based on parameters
  - Preserve both implementations during transition
- [ ] Update imports throughout codebase
  - Replace sobol_optimized imports
  - Update any configuration that selected optimization
- [ ] Remove sobol_optimized.py
  - Final verification of functionality
  - Performance benchmarking
  - Clean removal with git

### 3. JIT Kernel Consolidation
- [ ] Review jit_kernels.py usage
  - List all functions in jit_kernels.py
  - Find where each is used
  - Determine if they belong in specific modules
- [ ] Consider merging into relevant modules
  - Option A: Keep as shared JIT utilities module
  - Option B: Move each kernel to its primary usage module
  - Option C: Create utils/jit_helpers.py for shared kernels
  - **Recommendation:** Option A if >3 modules use kernels
- [ ] Update documentation for JIT usage
  - Add JIT section to user guide
  - Document performance benefits
  - Provide installation instructions for numba
- [ ] Create consistent JIT patterns
  ```python
  # Standard pattern for JIT-optional functions
  try:
      from numba import jit, prange
      HAS_NUMBA = True
  except ImportError:
      HAS_NUMBA = False
      # Define dummy decorators
      def jit(*args, **kwargs):
          def decorator(func):
              return func
          return decorator
      prange = range
  ```

### 4. Testing and Performance Verification
- [ ] Test both JIT-enabled and disabled paths
  - Parameterize tests with use_jit=[True, False, None]
  - Verify results match between implementations
  - Test with and without numba installed
- [ ] Benchmark performance to ensure no regression
  - Create benchmark script comparing:
    - Old separate files approach
    - New unified approach with JIT enabled
    - New unified approach with JIT disabled
  - Measure:
    - Function call overhead
    - Compilation time (first call)
    - Execution time (subsequent calls)
    - Memory usage
- [ ] Update test_jit_speedup.py for new structure
  - Adapt imports to new module structure
  - Update performance expectations if needed
  - Add tests for auto-detection logic
- [ ] Verify all optimization tests pass
  - Run: `pytest -xvs -k "jit or optim"`
  - Fix any failing tests
  - Add new tests for feature flags

### 5. Configuration and Documentation
- [ ] Add JIT configuration options
  ```python
  # In a config module or environment variables
  JIT_CONFIG = {
      'auto_threshold': 1000,  # Use JIT for n > threshold
      'force_disable': False,  # Global JIT disable
      'compilation_target': 'cpu',  # Future: 'gpu'
  }
  ```
- [ ] Create migration guide
  - Document changed imports
  - Show before/after usage examples
  - Explain new parameters
- [ ] Update performance tuning guide
  - When to use JIT
  - How to configure for best performance
  - Troubleshooting JIT issues

## Implementation Notes
- Use environment variables or config for default JIT settings
- Consider lazy loading of numba to improve import times
- Document JIT requirements and installation

## Output Log

[2025-05-26 09:18]: Task created from T11 subtask extraction
[2025-05-26 23:20]: Task deferred - Sprint closed. Migrated to General Task T020