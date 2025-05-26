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

**Moment Generating Function**: M_S(t) = M_N(ln M_X(t))

**Probability Generating Function**: G_S(z) = G_N(G_X(z))

### Classification of Compound Distributions

1. **Standard Compound Distributions**: Fixed frequency and severity parameters
2. **Mixed Compound Distributions**: Random frequency parameter (e.g., Mixed Poisson)
3. **Zero-Inflated Compounds**: Excess probability mass at zero
4. **Time-Varying Compounds**: Parameters changing over time

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

**Analytical Solution**: S ~ Exponential(θ/(1-p))

**Note on Parameterization**: The exponential distribution can be parameterized by rate λ = 1/θ or scale θ. Our implementation uses scale parameterization.

**Derivation**:
The probability generating function of Geometric is:
G_N(z) = pz/(1 - (1-p)z)

Combined with exponential severity:
φ_S(t) = pθ/[θ - it(1-p)]

This simplifies to an exponential distribution with scale θ/(1-p).

**Moments**:
- Mean: θ/(1-p)
- Variance: θ²/(1-p)²

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

### 5. Binomial-Exponential Compound

**Distribution**: N ~ Binomial(n, p), X_i ~ Exponential(1/θ)

**Mathematical Properties**:
- Support: [0, ∞)
- Atom at zero: P(S = 0) = (1-p)^n

**Analytical Approach**: MGF-based series expansion

**Derivation**:
The probability generating function of Binomial is:
G_N(z) = (1 - p + pz)^n

The characteristic function becomes:
φ_S(t) = (1 - p + pθ/(θ - it))^n

**PDF Expansion**:
f(s) = (1-p)^n δ(s) + Σ(k=1 to n) [C(n,k) p^k (1-p)^(n-k)] × Erlang(k, 1/θ).pdf(s)

Where Erlang(k, λ) is the sum of k exponential(λ) random variables.

**Moments**:
- Mean: npθ
- Variance: npθ²(1 + p)
- Skewness: 2(1 + 2p)/√(np(1 + p))

### 6. Binomial-Gamma Compound

**Distribution**: N ~ Binomial(n, p), X_i ~ Gamma(α, β)

**Mathematical Properties**:
Similar structure to Binomial-Exponential, but with gamma severity.

**Series Representation**:
f(s) = (1-p)^n δ(s) + Σ(k=1 to n) [C(n,k) p^k (1-p)^(n-k)] × Gamma(kα, β).pdf(s)

**Special Functions**:
For k gamma random variables, the sum follows Gamma(kα, β).
Uses incomplete gamma function for CDF calculation.

**Moments**:
- Mean: npα/β
- Variance: np(α/β²)[α + p(α + 1)]

### 7. Binomial-Lognormal Approximation

**Distribution**: N ~ Binomial(n, p), X_i ~ Lognormal(μ, σ)

**Approach**: Fenton-Wilkinson method for lognormal sums

**Mathematical Foundation**:
For sum of k independent lognormals, approximate as lognormal using moment matching:

For k lognormals:
- Mean of sum: k × exp(μ + σ²/2)
- Variance of sum: k × exp(2μ + σ²) × (exp(σ²) - 1)

Matching lognormal parameters:
- μ_k = ln(mean_sum) - σ_k²/2
- σ_k² = ln(1 + variance_sum/mean_sum²)

**Implementation**:
f(s) ≈ Σ(k=0 to n) [C(n,k) p^k (1-p)^(n-k)] × Lognormal(μ_k, σ_k).pdf(s)

**Accuracy Notes**:
- Exact for k=1
- Good approximation for moderate σ
- Less accurate for heavy-tailed lognormals

## Mixed Poisson Processes

Mixed Poisson processes model heterogeneity in frequency by allowing the Poisson rate parameter to be random.

### 1. Poisson-Gamma Mixture (Negative Binomial)

**Distribution**: N|Λ ~ Poisson(Λ), Λ ~ Gamma(r, θ)

**Marginal Distribution**: N ~ NegativeBinomial(r, p) where p = θ/(1+θ)

**Mathematical Properties**:
- Overdispersion: Var[N] = E[N](1 + E[N]/r)
- Clustering: Models contagion effects

**Moments**:
- Mean: rθ
- Variance: rθ(1 + θ)
- Index of dispersion: 1 + θ

### 2. Poisson-Inverse Gaussian Mixture

**Distribution**: N|Λ ~ Poisson(Λ), Λ ~ InverseGaussian(μ, λ)

**Probability Mass Function**:
P(N = k) = (2α/π)^(1/2) × (β/α)^k × exp(λ) × K_(k-1/2)(λ) / k!

Where K_ν is the modified Bessel function of the second kind.

**Properties**:
- Heavy-tailed frequency distribution
- Useful for modeling rare but severe events

### 3. Hierarchical Poisson Mixture

**Distribution**: Multi-level random effects
- Level 1: N_ij|λ_ij ~ Poisson(λ_ij)
- Level 2: log(λ_ij) = μ + u_i + ε_ij
- u_i ~ Normal(0, σ²_between)
- ε_ij ~ Normal(0, σ²_within)

**Application**: Portfolio segmentation with group effects

### 4. Time-Varying Poisson Process

**Distribution**: Non-homogeneous Poisson with intensity λ(t)

**Count Distribution**: N(s,t) ~ Poisson(∫_s^t λ(u)du)

**Common Intensity Functions**:
- Linear: λ(t) = a + bt
- Exponential: λ(t) = λ₀ exp(βt)
- Periodic: λ(t) = λ₀(1 + A sin(ωt))

**Simulation**: Thinning algorithm or time-change method

## Numerical Methods

### Panjer Recursion

For compound distributions where severity has support on non-negative integers:

**General Recursion Formula**:
f_S(x) = [p₀ f_X(x) + (1/x) × Σ(y=1 to x) y × f_X(y) × (a + b×y/x) × f_S(x-y)] / (1 - a×f_X(0))

**Special Cases**:
- Poisson: a = 0, b = λ
- Binomial: a = -p/(1-p), b = (n+1)p/(1-p)
- Negative Binomial: a = (1-p), b = (r-1)(1-p)

**Initial Condition**: f_S(0) = G_N(f_X(0))

**Application**: Exact calculation for discrete severities or discretized continuous severities.

### Fast Fourier Transform Method

For continuous severities with known characteristic functions:

**Steps**:
1. Discretize the support using FFT grid
2. Compute characteristic function values
3. Apply inverse FFT to obtain PDF
4. Integrate for CDF

**Complexity**: O(n log n) vs O(n²) for direct convolution

## Zero-Inflated Compound Distributions

Zero-inflated models handle excess zeros beyond what the base distribution predicts.

### General Form

**Distribution**: S = 0 with probability π, S ~ CompoundDist with probability (1-π)

**Probability Mass/Density**:
- P(S = 0) = π + (1-π)P₀
- f(s) = (1-π)f_base(s) for s > 0

Where P₀ is the probability of zero from the base compound distribution.

### Parameter Estimation

**EM Algorithm**:
1. **E-step**: Calculate posterior probability of zero-inflation
   w_i = π / (π + (1-π)f_base(0)) for s_i = 0
   
2. **M-step**: Update parameters
   π = Σw_i / n
   Update base distribution parameters using weighted likelihood

### Statistical Tests for Zero-Inflation

**Vuong Test**: Compare zero-inflated vs standard model
V = (1/√n) × Σ[log(f_zi(s_i)/f_base(s_i))] / σ_v

**Score Test**: Test H₀: π = 0
Score = Σ[(I(s_i=0) - P₀)² / P₀(1-P₀)]

## Additional Relevant Compound Distributions

### 1. Poisson-Pareto Compound

**Distribution**: N ~ Poisson(λ), X_i ~ Pareto(α, θ)

**Application**: Extreme value modeling, large loss events

**Tail Behavior**: 
- Heavy-tailed aggregate distribution
- Tail index min(α, 1) for α > 1

**Moments**: Exist only for α > k for k-th moment

### 2. Poisson-Weibull Compound

**Distribution**: N ~ Poisson(λ), X_i ~ Weibull(k, λ)

**Application**: Reliability analysis, warranty claims

**Special Cases**:
- k = 1: Reduces to Poisson-Exponential
- k = 2: Rayleigh severity

### 3. Zero-Modified Logarithmic Compound

**Distribution**: N ~ ZeroModifiedLogarithmic(p, π), X_i ~ G(φ)

**Properties**:
- No probability mass at N = 0
- Useful for "at least one claim" scenarios

### 4. Conway-Maxwell-Poisson Compound

**Distribution**: N ~ CMP(λ, ν), X_i ~ G(φ)

**Properties**:
- Flexible dispersion: ν > 1 (underdispersion), ν < 1 (overdispersion)
- Includes Poisson (ν = 1), Geometric (ν = 0), Bernoulli (ν → ∞)

## Special Functions Used

### Gamma Function
Γ(z) = ∫₀^∞ t^(z-1) e^(-t) dt

**Properties**:
- Γ(n) = (n-1)! for positive integers
- Γ(z+1) = z Γ(z)
- Log-gamma for numerical stability: log Γ(z)

### Beta Function  
B(α, β) = Γ(α)Γ(β)/Γ(α+β)

**Log-beta**: log B(α, β) = log Γ(α) + log Γ(β) - log Γ(α+β)

### Modified Bessel Functions
Used in Poisson-Inverse Gaussian and related compounds:
K_ν(z) = ∫₀^∞ e^(-z cosh t) cosh(νt) dt

**Asymptotic Approximations**:
- Large z: K_ν(z) ≈ √(π/2z) e^(-z)
- Small z: K_ν(z) ≈ Γ(ν)/2 × (2/z)^ν for ν > 0

### Incomplete Gamma Function
γ(s, x) = ∫₀^x t^(s-1) e^(-t) dt

**Regularized**: P(s, x) = γ(s, x)/Γ(s)

## Implementation Notes

### Numerical Stability

**Log-Space Calculations**:
- Use log-probabilities for products: log(ab) = log(a) + log(b)
- Log-sum-exp trick: log(Σe^x_i) = max(x) + log(Σe^(x_i - max(x)))
- Stable exponential: exp(x) with overflow/underflow protection

**Series Convergence**:
- Absolute tolerance: |term_n| < ε (default: 1e-10)
- Relative tolerance: |term_n/sum| < ε_rel (default: 1e-12)
- Maximum terms: Prevent infinite loops (default: 100-1000)

**Error Bounds**:
- Truncation error: E_trunc ≤ Σ(k=n+1 to ∞) |term_k|
- Round-off error: Consider machine epsilon in accumulation
- Total error: E_total ≤ E_trunc + E_round

### Parameter Validation

**Range Constraints**:
- Frequency parameters: λ > 0, 0 < p < 1, n ≥ 1, r > 0
- Severity parameters: Within distribution support
- Zero-inflation: 0 ≤ π < 1

**Degenerate Cases**:
- λ = 0: Point mass at S = 0
- p = 0 or p = 1: Boundary behavior
- n = 0: Empty sum

**Edge Case Handling**:
- Very large parameters: Switch to asymptotic approximations
- Very small parameters: Use Taylor expansions
- Near-zero probabilities: Maintain numerical precision

### Performance Optimizations

**Caching Strategy**:
- LRU cache for expensive calculations (Bessel functions, factorials)
- Memoization of series terms in recursions
- Pre-computed lookup tables for common parameter values

**Vectorization**:
- Batch special function calls: scipy.special.gammaln(array)
- NumPy broadcasting for array operations
- Parallel processing for Monte Carlo simulations

**Algorithm Selection**:
- Small n: Direct enumeration
- Large n: Series approximation or FFT
- Heavy tails: Importance sampling

### Convergence Criteria

**Series Expansions**:
```python
CONVERGENCE_TOLERANCE = 1e-10  # Absolute tolerance
SERIES_MAX_TERMS = 100         # Maximum iterations
RELATIVE_TOLERANCE = 1e-12     # For relative convergence
```

**Adaptive Selection**:
- n < 50: Use exact enumeration
- 50 ≤ n < 1000: Series expansion
- n ≥ 1000: Approximation methods

**Quality Metrics**:
- KL divergence for approximation quality
- Moment matching validation
- Quantile comparison for tail accuracy

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