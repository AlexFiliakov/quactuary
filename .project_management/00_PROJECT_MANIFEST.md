---
project_name: quActuary
current_milestone_id: M01
highest_sprint_in_milestone: S01
current_sprint_id: S01
status: active
last_updated: 2025-05-25 11:35
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
  - Completed Tasks: TX01_S01_Policy_Logic (✓), TX02_S01_Compound_Distributions (✓), TX03_S01_Sobol_Sequences (✓), TX04_S01_Optimize_Classical_Simulation (✓), TX07_S01_Simplify_PricingModel (✓)
  - In Progress: None
  - Core Tasks: All completed
  - Added Critical: T08_S01_Simplify_Compounds, T09_S01_RunDev, T10_S01_Prune_Tests
  - Added Enhancement: T05_S01_Numerical_Stability, T06_S01_Extended_Distributions
  - Performance Follow-Up: T11_S01_Testing_Suite, T12_S01_Parallel_Stability, T13_S01_Integration_Testing, T14_S01_Documentation

## 4. Key Documentation

- [Architecture Documentation](./01_PROJECT_DOCS/ARCHITECTURE.md)
- [Current Milestone Requirements](./02_REQUIREMENTS/M01_Backend_Setup/)
- [General Tasks](./04_GENERAL_TASKS/)
  - New: T001_QMC_Enhancement_Testing - Follow-up testing and optimization for Sobol implementation

## 5. Quick Links

- **Current Sprint:** [S01 Sprint Folder](./03_SPRINTS/S01_M01_Classical_Optimization/)
- **Active Tasks:** Check sprint folder for T##_S01_*.md files
- **Project Reviews:** [Latest Review](./10_STATE_OF_PROJECT/)