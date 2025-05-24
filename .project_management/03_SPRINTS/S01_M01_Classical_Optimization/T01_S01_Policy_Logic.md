---
task_id: T01_S01
sprint_sequence_id: S01
status: in_progress # open | in_progress | pending_review | done | failed | blocked
complexity: Low # Low | Medium | High
last_updated: 2025-05-24
---

# Task: Implement Policy Logic

## Description
Implement all policy logic. The specific amounts are passed as variables, but all the features are listed in `book.py` class `PolicyTerms`

## Goal / Objectives
- Implement all policy logic.

## Acceptance Criteria
- [ ] Implement the subtasks in this file and describe reasons for any skipped implementations in section `## Output Log of this file`
- [ ] Design automated tests
- [ ] Initial automated tests pass with 95% `book.py` coverage

## Subtasks
- [ ] effective_date:-  date  # Effective date of the policy
- [ ] expiration_date:- date  # Expiration date of the policy
- [ ] lob:- - - - Optional[LOB] = None  # Line of business (optional)
- [ ] exposure_base:-   Optional[ExposureBase] = None  # Exposure base (Payroll, Sales, Square Footage, Vehicles, Replacement Value etc.)
- [ ] exposure_amount:- float = 0.0  # Exposure amount (e.g., limit, sum insured)
- [ ] retention_type:-  str = "deductible"  # deductible / SIR
- [ ] per_occ_retention:  float = 0.0  # retention per occurrence
- [ ] agg_retention:-   Optional[float] = None  # aggregate retention
- [ ] corridor_retention: Optional[float] = None  # corridor retention
- [ ] coinsurance:- - Optional[float] = None  # 0 → 0% insured's share, 1.0 → 100% insured's share
- [ ] per_occ_limit:-   Optional[float] = None  # per-occurrence limit (None if unlimited)
- [ ] agg_limit:- -   Optional[float] = None  # aggregate limit (Optional)
- [ ] attachment:- -  Optional[float] = None  # for XoL layers
    - (if attachment is confusing, you can leave out the implementation)
- [ ] Design automated tests to cover 95% of `book.py`

## Output Log

