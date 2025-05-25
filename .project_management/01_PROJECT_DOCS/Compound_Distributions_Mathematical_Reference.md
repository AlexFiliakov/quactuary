# Compound Distributions Mathematical Reference

## Overview

This document provides mathematical derivations and theoretical foundations for compound distributions implemented in the quActuary package. Compound distributions model aggregate losses as S = X₁ + X₂ + ... + X_N where N is the claim count (frequency) and X_i are individual claim amounts (severity).

## General Theory

### Definition
A compound distribution combines:
- **Frequency Distribution**: N ~ F(θ) for claim count  
- **Severity Distribution**: X_i ~ G(φ) for individual claim amounts
- **Aggregate Loss**: S = Σ(i=1 to N) X_i

### Fundamental Properties

**Mean**: E[S] = E[N] × E[X]

**Variance**: Var[S] = E[N] × Var[X] + Var[N] × E[X]²

**Characteristic Function**: φ_S(t) = φ_N(ln φ_X(t))

## Implemented Analytical Solutions

### 1. Poisson-Exponential Compound

**Distribution**: N ~ Poisson(λ), X_i ~ Exponential(1/θ)

**Mathematical Properties**:
- Atom at zero: P(S = 0) = e^(-λ)
- For S > 0: Follows Gamma distribution

**Derivation**:
The probability generating function of Poisson is:
G_N(z) = exp[λ(z - 1)]

For exponential severity with rate 1/θ, the characteristic function is:
φ_X(t) = θ/(θ - it)

The aggregate characteristic function becomes:
φ_S(t) = exp[λ(θ/(θ - it) - 1)]
       = exp[λit/(θ - it)]

**Moments**:
- Mean: λθ
- Variance: 2λθ²
- Skewness: 2/√λ

### 2. Poisson-Gamma Compound (Tweedie Distribution)

**Distribution**: N ~ Poisson(λ), X_i ~ Gamma(α, β)

**Mathematical Properties**:
This is a special case of the Tweedie distribution family.

**Tweedie Parameters**:
- Power parameter: p ∈ (1, 2)
- Mean: μ = λα/β  
- Dispersion: φ = β^(p-2)/[α^(p-1)λ^(2-p)]

**Conversion Formulas**:
- λ = μ^(2-p)/[φ(2-p)]
- α = (2-p)/(p-1)
- β = φ(p-1)μ^(p-1)

**PDF**: No closed form, but series expansion:
f(s) = e^(-λ) δ(s) + Σ(k=1 to ∞) [λ^k/k!] × Gamma(kα, β).pdf(s)

### 3. Geometric-Exponential Compound

**Distribution**: N ~ Geometric(p), X_i ~ Exponential(1/θ)

**Analytical Solution**: S ~ Exponential((1-p)/θ)

**Derivation**:
The probability generating function of Geometric is:
G_N(z) = pz/(1 - (1-p)z)

Combined with exponential severity:
φ_S(t) = pθ/[θ - (1-p)θ + it] = p(1-p)/[1-p - it/θ]

This simplifies to an exponential distribution with rate (1-p)/θ.

**Moments**:
- Mean: θ/(1-p)
- Variance: 2θ²/(1-p)²

### 4. Negative Binomial-Gamma Compound

**Distribution**: N ~ NegativeBinomial(r, p), X_i ~ Gamma(α, β)

**Approach**: Series expansion with Beta-Gamma mixture representation

**Mathematical Foundation**:
The negative binomial can be represented as a Gamma-Poisson mixture:
N|Λ ~ Poisson(Λ), Λ ~ Gamma(r, p/(1-p))

This leads to a compound Gamma distribution that can be expressed using special functions.

**Series Representation**:
f(s) = Σ(k=0 to ∞) NB(k; r, p) × Gamma(kα, β).pdf(s)

Where NB(k; r, p) = C(k+r-1, k) × p^r × (1-p)^k

### 5. Binomial-Lognormal Approximation

**Distribution**: N ~ Binomial(n, p), X_i ~ Lognormal(μ, σ)

**Approach**: Fenton-Wilkinson method for lognormal sums

**Mathematical Foundation**:
For sum of k lognormals, the aggregate is approximately lognormal with:
- μ_agg ≈ ln(k) + μ + ln(1 + σ²/μ²)/2
- σ²_agg ≈ ln(1 + σ²/μ²)

**Implementation**:
Uses moment matching to approximate the compound distribution as a mixture of lognormals weighted by binomial probabilities.

## Numerical Methods

### Panjer Recursion

For compound distributions where severity has support on non-negative integers:

**Recursion Formula**:
f_S(x) = (1/x) × Σ(y=1 to x) y × f_X(y) × f_S(x-y)

**Initial Condition**: f_S(0) = P_N(0)

**Application**: Exact calculation for discrete severities or discretized continuous severities.

### Fast Fourier Transform Method

For continuous severities with known characteristic functions:

**Steps**:
1. Discretize the support using FFT grid
2. Compute characteristic function values
3. Apply inverse FFT to obtain PDF
4. Integrate for CDF

**Complexity**: O(n log n) vs O(n²) for direct convolution

## Special Functions Used

### Gamma Function
Γ(z) = ∫₀^∞ t^(z-1) e^(-t) dt

**Properties**:
- Γ(n) = (n-1)! for positive integers
- Γ(z+1) = z Γ(z)

### Beta Function  
B(α, β) = Γ(α)Γ(β)/Γ(α+β)

### Modified Bessel Functions
Used in Poisson-Inverse Gaussian and related compounds:
K_ν(z) = ∫₀^∞ e^(-z cosh t) cosh(νt) dt

## Implementation Notes

### Numerical Stability
- Use logarithmic calculations for large parameters
- Implement series truncation with error bounds  
- Cache intermediate calculations for performance

### Parameter Validation
- Ensure frequency parameters are non-negative
- Validate severity parameters within distribution support
- Check for degenerate cases (λ=0, p=0 or p=1)

### Performance Optimizations
- Vectorized special function evaluations using scipy.special
- Caching for repeated parameter combinations
- JIT compilation for inner loops (when implemented)

## References

1. Jørgensen, B. (1997). *The Theory of Dispersion Models*. Chapman & Hall.
2. Klugman, S., Panjer, H., & Willmot, G. (2019). *Loss Models: From Data to Decisions* (5th ed.). Wiley.
3. Panjer, H. (1981). Recursive evaluation of a family of compound distributions. *ASTIN Bulletin*, 12(1), 22-26.
4. Dufresne, D. (2004). The log-normal approximation in financial and other computations. *Advances in Applied Probability*, 36(3), 747-773.
5. Willmot, G. E. (1986). Mixed compound Poisson distributions. *ASTIN Bulletin*, 16(1), 59-79.
6. Fenton, L. (1960). The sum of log-normal probability distributions in scatter transmission systems. *IRE Transactions on Communications Systems*, 8(1), 57-67.

## Validation Methods

### Analytical Verification
- Compare moments with theoretical formulas
- Verify special cases (e.g., λ→0, p→1)
- Check limiting distributions

### Numerical Testing
- Monte Carlo simulation comparison
- Cross-validation with alternative implementations
- Convergence testing for series expansions

### Edge Case Handling
- Zero frequency parameters
- Extreme severity parameters
- Numerical overflow/underflow prevention