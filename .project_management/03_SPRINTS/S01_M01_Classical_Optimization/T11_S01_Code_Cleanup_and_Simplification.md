---
task_id: T11_S01
sprint_id: S01
title: Code Cleanup and Simplification
status: open
priority: critical
created: 2025-05-25 19:17
updated: 2025-05-25 19:17
---

# Task: Code Cleanup and Simplification

## 🎯 Objective
Clean up the codebase by removing experimental/duplicate files, consolidating compound distribution implementations, organizing test files properly, and simplifying over-engineered abstractions.

## 📋 Context
The project review identified significant technical debt including:
- 6+ versions of compound distribution files
- Test files outside the tests/ directory
- Experimental/profiling scripts scattered throughout
- Over-engineered abstractions (strategy pattern, quantum placeholders)
- Build artifacts in version control

## ✅ Acceptance Criteria
- [ ] All compound distribution variants consolidated into single compound.py
- [ ] All test files moved to appropriate locations in tests/ directory
- [ ] Experimental/duplicate files removed
- [ ] Build artifacts added to .gitignore and removed
- [ ] Over-engineered abstractions simplified
- [ ] No test failures after cleanup
- [ ] Code coverage maintained at 95%+

## 📝 Subtasks

### 1. Pre-Cleanup Analysis and Strategy
- [ ] Create cleanup branch: `git checkout -b cleanup/consolidate-technical-debt`
- [ ] Run full test suite to establish baseline: `pytest --cov --cov-branch`
- [ ] Document current coverage percentage and failing tests
- [ ] Create file dependency graph using `grep -r "import.*compound" --include="*.py"`
- [ ] Identify all external dependencies on files to be removed
- [ ] Create rollback plan in case cleanup introduces issues

### 2. Consolidate Compound Distribution Files
- [ ] Analyze all compound distribution variants:
  - compound.py (current main file - 712 lines)
  - compound_binomial.py (check for unique binomial features)
  - compound_final.py (review for production-ready features)
  - compound_simplified.py (intermediate simplification)
  - compound_simplified_final.py (703 lines - T08 result)
  - compound_v2.py (693 lines - check for improvements)
- [ ] Create feature comparison matrix:
  - Analytical methods (FFT, recursive, exact)
  - Simulation methods (Monte Carlo, vectorized)
  - Distribution support (Poisson, Binomial, NegBin)
  - Performance optimizations (JIT, vectorization)
  - API compatibility
- [ ] Merge strategy:
  - Use compound_simplified_final.py as base (from T08)
  - Port any missing analytical methods from other variants
  - Preserve factory pattern from current compound.py
  - Add deprecation warnings for removed features
- [ ] Test each feature migration individually
- [ ] Update all imports systematically:
  ```bash
  # Find all imports
  grep -r "from.*compound" --include="*.py" | cut -d: -f1 | sort -u
  # Update each file carefully
  ```
- [ ] Delete redundant files only after all tests pass

### 3. Organize Test Files with Best Practices
- [ ] Move misplaced test files:
  - quactuary/test_jit_speedup.py → tests/test_jit_speedup.py
  - tests/test_benchmarks.py → Keep in tests/
  - tests/test_memory_management.py → Keep in tests/
  - tests/test_vectorized_simulation.py → Keep in tests/
- [ ] Consolidate duplicate test files:
  - test_compound*.py files → single test_compound.py
  - test_pricing*.py files → organized test_pricing.py
  - test_parallel*.py files → single test_parallel.py
- [ ] Ensure all tests still pass after reorganization
- [ ] Update any import paths as needed

### 4. Remove Experimental/Duplicate Files Strategically
- [ ] Delete experimental files:
  - profile_baseline.py
  - benchmarks.py (move useful parts to tests/)
  - code_review_T04.md
  - All compound_*.py variants except the chosen one
- [ ] Handle multiple parallel processing files:
  - Keep parallel_processing_stable.py as main
  - Delete parallel_processing.py
  - Update imports
- [ ] Handle JIT files:
  - Review classical_jit.py vs classical.py
  - Consider merging JIT as optional flag in classical.py
  - Delete redundant files

### 5. Clean Build Artifacts and Improve .gitignore
- [ ] Add to .gitignore:
  - *.egg-info/
  - __pycache__/
  - .pytest_cache/
  - *.pyc
  - build/
  - dist/
- [ ] Remove quactuary.egg-info/ directory
- [ ] Commit .gitignore updates

### 6. Simplify Over-Engineered Abstractions with Care
- [ ] Quantum module refactoring:
  - Archive quantum.py content for future reference
  - Create quantum_future.md documenting planned features
  - Remove module but preserve interface stubs if needed
  - Add NotImplementedError with helpful messages
- [ ] Pricing strategy pattern simplification:
  - Analyze current usage patterns in codebase
  - Create migration path from strategy pattern to direct methods
  - Implement feature flags for gradual migration:
    ```python
    # In pricing.py
    USE_LEGACY_STRATEGY = os.getenv('QUACTUARY_LEGACY_PRICING', 'false').lower() == 'true'
    ```
  - Merge concrete strategies into PricingModel as methods
  - Keep backend enum but simplify dispatch mechanism
  - Add deprecation warnings for strategy pattern usage
  - Document migration path for external users
- [ ] Backend abstraction review:
  - Evaluate if backend switching is actually used
  - Consider keeping minimal abstraction for numpy/jax
  - Remove unused backend implementations
- [ ] API compatibility layer:
  - Create compatibility.py for deprecated interfaces
  - Use __getattr__ for backward compatibility
  - Plan deprecation timeline (2-3 releases)

### 7. Update Documentation and Imports Systematically
- [ ] Update all import statements affected by file moves
- [ ] Update __init__.py files to reflect new structure
- [ ] Update any documentation referencing moved/deleted files
- [ ] Ensure all doctests still work

### 8. Comprehensive Verification and Quality Assurance
- [ ] Run full test suite with detailed reporting:
  ```bash
  pytest --cov --cov-branch --cov-report=html --cov-report=term-missing
  ```
- [ ] Verify coverage metrics:
  - Overall coverage ≥95%
  - No file below 90% coverage
  - Critical paths 100% covered
- [ ] Performance regression testing:
  - Run benchmarks before and after cleanup
  - Document any performance changes
  - Ensure no degradation > 5%
- [ ] Integration testing:
  - Test all example notebooks
  - Verify run_dev.py commands
  - Test pip install in fresh virtualenv
  - Check import times haven't increased
- [ ] Static analysis:
  ```bash
  # Type checking
  mypy quactuary --ignore-missing-imports
  # Code quality
  flake8 quactuary --max-line-length=100
  # Security scanning
  bandit -r quactuary
  ```
- [ ] Documentation validation:
  - All docstrings still accurate
  - Examples in docs still work
  - No broken internal links

### 9. Post-Cleanup Optimization
- [ ] Profile import times and optimize if needed
- [ ] Review and optimize __init__.py exports
- [ ] Consider lazy imports for heavy dependencies
- [ ] Document new project structure in architecture docs

## 🔗 Dependencies
- Should be done after T10_S01_Prune_Test_Suite to avoid conflicts
- May affect T11-T14 if they reference files being moved

## 💡 Technical Notes & Best Practices

### Version Control Strategy
- Create feature branch for all cleanup work
- Make atomic commits for each major change
- Use descriptive commit messages: "refactor: consolidate compound distributions into single module"
- Tag repository before major deletions: `git tag pre-cleanup-backup`

### Risk Mitigation
- Keep backup of all deleted files in `.archive/` directory temporarily
- Document all removed features in CHANGELOG.md
- Create migration guide for any breaking changes
- Consider phased approach: deprecate first, remove in next release

### Code Quality Standards
- Follow PEP 8 and existing project conventions
- Ensure all public APIs have docstrings
- Add type hints where missing
- Use consistent naming conventions

### Testing Strategy
- Run tests after each subtask completion
- Use git bisect if issues arise
- Keep test coverage report for comparison
- Consider property-based testing for critical paths

### Communication
- Document all decisions in cleanup_decisions.md
- Note any contentious choices for team review
- Keep running list of follow-up tasks
- Update project documentation immediately

## 📊 Success Metrics

### Quantitative Metrics
- File count reduced by 20-30 files (current: ~50 Python files)
- Lines of code reduced by 15-25% through deduplication
- Import time improved by >10%
- Test execution time maintained or improved
- Coverage maintained at ≥95% (currently 95.2%)
- Zero new warnings from static analysis tools

### Qualitative Metrics
- Improved code discoverability (one place for each feature)
- Clearer separation of concerns
- Reduced cognitive load for new developers
- Simplified debugging and maintenance
- Better adherence to SOLID principles

### Technical Debt Metrics
- Cyclomatic complexity reduced in key modules
- Duplicate code detection shows <5% duplication
- All TODOs and FIXMEs addressed or documented
- No circular dependencies
- Clear module hierarchy established

## 🗓️ Time Estimate
4-6 hours for complete cleanup and verification

## 📌 Output Log

### 2025-05-25 19:17
- Task created based on project review findings
- Identified critical need for cleanup before progressing to M02
- File proliferation and organization violations require immediate attention