---
project_name: quActuary
current_milestone_id: M01
highest_sprint_in_milestone: S01
current_sprint_id: S01
status: active
last_general_task_id: T008
last_updated: 2025-05-26 11:04
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
- **Sprint:** S01 - Optimize Classical Simulations

## 3. Sprints in Current Milestone

- [S01_M01_Classical_Optimization](./03_SPRINTS/S01_M01_Classical_Optimization/) - **Active**
  - Goal: Implement policy logic and optimize classical simulation
  - Completed Tasks: TX01_S01_Policy_Logic (✓), TX02_S01_Compound_Distributions (✓), TX03_S01_Sobol_Sequences (✓), TX04_S01_Optimize_Classical_Simulation (✓), TX05_S01_Numerical_Stability (✓), TX06_S01_Extended_Distributions (✓), TX07_S01_Simplify_PricingModel (✓), TX08_S01_Simplify_Compounds (✓), TX09_S01_RunDev (✓), TX10_S01_Prune_Test_Suite (✓), TX11_S01_Code_Cleanup_and_Simplification (✓ Partial), TX12_S01_Parallel_Processing_Stability (✓), TX13_S01_Performance_Testing_Suite (✓), TX15_S01_Optimization_Documentation (✓), TX16_S01_Extended_Distribution_Testing (✓)
  - In Progress: None
  - Core Tasks: All completed
  - Added Critical: All completed
  - Added Enhancement: TX06_S01_Extended_Distributions (✓)
  - Remaining Tasks: T14_S01_Integration_Testing, T17_S01_Performance_Benchmarks_Notebook, T18_S01_Consolidate_Compound_Distributions, T19_S01_Merge_JIT_Implementations, T20_S01_Consolidate_Test_Files, T21_S01_Fix_JIT_Test_Failure

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

## 6. Quick Links

- **Current Sprint:** [S01 Sprint Folder](./03_SPRINTS/S01_M01_Classical_Optimization/)
- **Active Tasks:** Check sprint folder for T##_S01_*.md files
- **Project Reviews:** [Latest Review](./10_STATE_OF_PROJECT/)