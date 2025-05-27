---
project_name: quActuary
current_milestone_id: M01
highest_sprint_in_milestone: S01
current_sprint_id: S01
status: active
last_general_task_id: T022
last_updated: 2025-05-26 23:19
---

# Project Manifest: quActuary

This manifest serves as the central reference point for the project. It tracks the current focus and links to key documentation.

## 1. Project Vision & Overview

The quActuary package provides quantum-accelerated actuarial computations with a seamless classical/quantum backend switch. The package enables actuaries to leverage quantum computing for complex risk calculations while maintaining compatibility with classical approaches.

Key features:
- Backend System: Toggle between classical and quantum execution
- Policy management and portfolio construction
- Frequency and severity distributions with quantum state preparation
- Risk measures (VaR, TVaR) with quantum optimization

This project follows a milestone-based development approach.

## 2. Current Focus

- **Milestone:** M01 - Classical Optimization Foundation
- **Sprint:** None - S01 completed on 2025-05-26
- **Note:** Sprint S01_M01_Classical_Optimization has been successfully completed. Open tasks migrated to General Tasks.

## 3. Sprints in Current Milestone

- [S01_M01_Classical_Optimization](./03_SPRINTS/S01_M01_Classical_Optimization/) - **Completed**
  - Goal: Implement policy logic and optimize classical simulation
  - Status: Sprint closed on 2025-05-26. Remaining open tasks migrated to General Tasks (T019-T022)
  - Completed Tasks: TX01_S01_Policy_Logic (✓), TX02_S01_Compound_Distributions (✓), TX03_S01_Sobol_Sequences (✓), TX04_S01_Optimize_Classical_Simulation (✓), TX05_S01_Numerical_Stability (✓), TX06_S01_Extended_Distributions (✓), TX07_S01_Simplify_PricingModel (✓), TX08_S01_Simplify_Compounds (✓), TX09_S01_RunDev (✓), TX10_S01_Prune_Test_Suite (✓), TX11_S01_Code_Cleanup_and_Simplification (✓), TX12_S01_Parallel_Processing_Stability (✓), TX13_S01_Performance_Testing_Suite (✓), TX14_S01_Integration_Testing (✓), TX15_S01_Optimization_Documentation (✓), TX16_S01_Extended_Distribution_Testing (✓), TX17_S01_Performance_Benchmarks_Notebook (✓), TX21_S01_Fix_JIT_Test_Failure (✓)
  - Migrated to General: T18_S01 → T019, T19_S01 → T020, T20_S01 → T021, T22_S01 → T022

## 4. Planned Milestones Roadmap

### M02 - Quantum Implementation (Planned)

**Goal:** Integrate quantum algorithms with classical features to enhance runtime and accuracy

**Planned Sprints:**
- **S02_M02_Quantum_Foundation** - Set up quantum infrastructure with qiskit 1.4.2
  - Install and configure quantum framework
  - Create quantum module structure
  - Implement basic utilities
  
- **S03_M02_Excess_Loss_Algorithm** - Implement Excess Loss quantum algorithm
  - Port algorithm from reference notebook
  - Create quantum circuit implementation
  - Build classical interface
  
- **S04_M02_Quantum_Pricing_Algorithms** - Additional quantum algorithms
  - Implement algorithms from research papers
  - Create unified quantum API
  - Algorithm-specific optimizations
  
- **S05_M02_Classical_Quantum_Integration** - Decision logic framework
  - Design algorithm selection criteria
  - Implement performance profiling
  - Create seamless integration layer
  
- **S06_M02_Testing_And_Benchmarks** - Comprehensive validation
  - Performance benchmarking
  - Integration testing
  - Documentation and examples

**Key Resources:**
- Excess Loss: https://github.com/AlexFiliakov/knowledge-pricing/blob/main/Quantum%20Excess%20Evaluation%20Algorithm.ipynb
- Research: https://arxiv.org/html/2410.20841v1#S7.SS1
- Additional: https://arxiv.org/pdf/1411.5949

## 5. Key Documentation

- [Architecture Documentation](./01_PROJECT_DOCS/ARCHITECTURE.md)
- [Current Milestone Requirements](./02_REQUIREMENTS/M01_Backend_Setup/)
- [Next Milestone Requirements](./02_REQUIREMENTS/S01_M02_Quantum_Implementation/)
- [General Tasks](./04_GENERAL_TASKS/)
  - T001_QMC_Enhancement_Testing - Follow-up testing and optimization for Sobol implementation
  - TX002_RunDev_Enhancements - Optional enhancements for run_dev.py script (✓)
  - TX003_Fix_Parameter_Boundaries_Test - Quick win: Fix single test failure (✓)
  - TX004_Fix_Benchmark_Infrastructure - Infrastructure: Fix 13 benchmark test failures (✓)
  - TX005_Fix_Compound_Distribution_Failures - Core: Fix 47+ compound distribution test failures (✓)
  - TX006_Fix_Pricing_Optimization_Tests - Integration: Fix pricing and optimization tests (✓)
  - TX006_Automatic_Optimization_Selection - Intelligent automatic optimization selection (✓)
  - TX007_Fix_Utility_Performance_Tests - Cleanup: Fix remaining utility and performance tests (✓)
  - T008_Optimization_Documentation_Extended - Extended optimization documentation tasks
  - TX011_Fix_Integration_Tests - Fixed integration test failures, reduced from 50+ to 8 (✓)
  - TX013_Integration_Test_Cleanup_Followup - Follow-up task for remaining 8 integration test failures (✓)
  - TX014_API_Documentation_Update - Document API changes and best practices discovered in T011 (✓)
  - TX015_Test_Architecture_Review - Comprehensive test architecture review and refactoring (✓)
  - TX016_Performance_Baseline_Establishment - Create adaptive performance testing infrastructure (✓)
  - T017_MCP_Implementation - Implement Model-Context-Protocol server for Claude Code integration
  - T018_Fix_Remaining_Test_Failures - Fix all remaining test failures across distributions, performance, and integration

## 6. Quick Links

- **Current Sprint:** [S01 Sprint Folder](./03_SPRINTS/S01_M01_Classical_Optimization/)
- **Active Tasks:** Check sprint folder for T##_S01_*.md files
- **Project Reviews:** [Latest Review](./10_STATE_OF_PROJECT/)