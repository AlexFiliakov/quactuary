---
task_id: T01_S01
sprint_sequence_id: S01
status: in_progress # open | in_progress | pending_review | done | failed | blocked
complexity: Medium # Low | Medium | High
last_updated: 2025-05-24
---

# Task: Implement Policy Logic

## Description
Implement comprehensive policy logic for insurance contracts in the `PolicyTerms` class within `book.py`. This includes handling various retention types, limits, coinsurance, and excess-of-loss (XoL) attachments. The implementation should support both per-occurrence and aggregate applications of policy terms.

## Goal / Objectives
- Implement methods to apply policy terms to loss amounts
- Support all standard insurance policy features (deductibles, limits, coinsurance, etc.)
- Ensure calculations handle edge cases and invalid inputs gracefully
- Maintain compatibility with both classical and quantum simulation backends

## Technical Requirements
- All calculations must be vectorizable for efficient numpy operations
- Support both scalar and array inputs for loss amounts
- Maintain numerical precision for large loss amounts
- Implement proper validation for all policy parameters

## Acceptance Criteria
- [ ] All policy term calculations produce correct results for standard test cases
- [ ] Edge cases are handled appropriately (e.g., None values, negative amounts)
- [ ] Methods are properly documented with examples
- [ ] Unit tests achieve 95% coverage of policy-related code in `book.py`
- [ ] Performance benchmarks show <1ms for single policy application

## Subtasks

### 1. Core Policy Date Logic
- [ ] Implement `is_policy_active(evaluation_date)` method
- [ ] Add validation for effective_date < expiration_date
- [ ] Handle timezone considerations if dates include time components

### 2. Retention Logic Implementation
- [ ] Implement `apply_retention(loss_amount)` method supporting:
  - Deductible: Insured pays first X amount
  - SIR (Self-Insured Retention): Similar to deductible but affects limits differently
- [ ] Implement per-occurrence retention application
- [ ] Implement aggregate retention tracking and application
- [ ] Implement corridor retention (applies after primary limit)

### 3. Limit Logic Implementation  
- [ ] Implement `apply_limits(loss_amount_after_retention)` method
- [ ] Support per-occurrence limits
- [ ] Support aggregate limit tracking and exhaustion
- [ ] Handle interaction between per-occurrence and aggregate limits

### 4. Coinsurance Logic
- [ ] Implement `apply_coinsurance(loss_amount)` method
- [ ] Validate coinsurance is between 0.0 and 1.0
- [ ] Document that 0.0 = insurer pays all, 1.0 = insured pays all

### 5. Excess-of-Loss (XoL) Attachment
- [ ] Implement `apply_xol_attachment(ground_up_loss)` method
- [ ] Only respond to losses exceeding attachment point
- [ ] Apply limits to the excess amount only
- [ ] Document interaction with retentions and limits

### 6. Combined Policy Application
- [ ] Implement main `apply_policy_terms(loss_amount)` method that:
  1. Validates loss amount
  2. Applies retentions (per-occ, then aggregate if applicable)
  3. Applies XoL attachment if present
  4. Applies limits (per-occ, then aggregate if applicable)  
  5. Applies corridor retention if present
  6. Applies coinsurance
- [ ] Return detailed breakdown object showing each step

### 7. Exposure and LOB Handling
- [ ] Add methods to work with exposure_base and exposure_amount
- [ ] Implement LOB-specific logic hooks (for future enhancement)

### 8. Testing and Documentation
- [ ] Create comprehensive test suite covering:
  - Individual policy features
  - Combined feature interactions
  - Edge cases and error conditions
  - Vectorized operations
- [ ] Add docstring examples for all public methods
- [ ] Create integration tests with PricingModel class

## Implementation Notes
- Deductible vs SIR: Deductible erodes the limit, SIR does not
- Aggregate tracking requires stateful operations - consider using a PolicyState class
- XoL layers may stack - ensure the design supports this future enhancement

## Output Log

### 2025-05-24 17:30 - Task Review
- Reviewed and updated task description for clarity
- Added comprehensive technical requirements
- Reorganized subtasks with clear implementation methods
- Added implementation notes for complex interactions
- Status: Ready for implementation