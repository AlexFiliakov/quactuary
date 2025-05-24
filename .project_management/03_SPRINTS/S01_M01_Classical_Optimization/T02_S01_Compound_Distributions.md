---
task_id: T02_S01
sprint_sequence_id: S01
status: in_progress # open | in_progress | pending_review | done | failed | blocked
complexity: Medium # Low | Medium | High
last_updated: 2025-05-24
---

# Task: Implement Compound Distributions

## Description
Implement compound distribution simplifications for all cases wherever possible for Frequency-Severity selections. For example:
- Aggregate loss S = ∑(i=1 to N) Xi where N ~ Poisson(λ) and Xi ~ Gamma(α, β)
- S follows a Tweedie distribution with:
    - Mean: E[S] = λ × α/β
    - Variance: Var[S] = λ × α/β² × (1 + α)
    - For integer α, can use negative binomial convolution
- Research and implement all simplification cases.
- The simplifications should never introduce error or reduce accuracy.

## Goal / Objectives
- Implement all compound distributions simplifications.

## Acceptance Criteria
- [ ] Implement the subtasks in this file and describe reasons for any skipped implementations in section `## Output Log of this file`
- [ ] Design automated tests
- [ ] Initial automated tests pass with 95% `book.py` coverage

## Subtasks
- [ ] Log all identified compound distributions in this file's `## Output Log` section.
- [ ] Implement all identified compound distributions in a sensible way within the Frequency-Severity framework, list in the Output Log when you can't complete the implementation.
- [ ] Create automated tests to cover 95% of the frequency.py and severity.py and simplification files.

## Output Log

