---
task_id: T19_S01
sprint_sequence_id: S01
status: open # open | in_progress | pending_review | done | failed | blocked
complexity: Medium # Low | Medium | High
last_updated: 2025-05-26 09:18
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
- [ ] Add use_jit parameter to classical simulation functions
- [ ] Implement conditional JIT compilation
- [ ] Update all imports from classical_jit
- [ ] Remove classical_jit.py after verification

### 2. Sobol Optimization Merge
- [ ] Compare sobol.py and sobol_optimized.py implementations
- [ ] Add optimization flags to sobol.py
- [ ] Merge optimized algorithms with feature detection
- [ ] Update imports throughout codebase
- [ ] Remove sobol_optimized.py

### 3. JIT Kernel Consolidation
- [ ] Review jit_kernels.py usage
- [ ] Consider merging into relevant modules
- [ ] Update documentation for JIT usage

### 4. Testing and Performance Verification
- [ ] Test both JIT-enabled and disabled paths
- [ ] Benchmark performance to ensure no regression
- [ ] Update test_jit_speedup.py for new structure
- [ ] Verify all optimization tests pass

## Implementation Notes
- Use environment variables or config for default JIT settings
- Consider lazy loading of numba to improve import times
- Document JIT requirements and installation

## Output Log

[2025-05-26 09:18]: Task created from T11 subtask extraction