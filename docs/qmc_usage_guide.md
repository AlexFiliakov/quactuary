# Quasi-Monte Carlo (QMC) Usage Guide for quActuary

## Overview

Quasi-Monte Carlo (QMC) methods use low-discrepancy sequences instead of pseudo-random numbers to achieve faster convergence rates for numerical integration. This guide covers optimal usage of QMC methods in actuarial applications.

## When to Use QMC

### Ideal Use Cases

1. **Tail Risk Estimation**: QMC excels at estimating high quantiles (VaR, TVaR)
   - 5-10x faster convergence for 95%+ quantiles
   - Better stability for extreme value estimation

2. **Large Portfolios**: Benefits increase with portfolio size
   - 100+ policies show significant improvements
   - Dimension allocation handles thousands of policies efficiently

3. **Smooth Integrands**: QMC works best with smooth loss distributions
   - Continuous severity distributions (Exponential, Gamma, Lognormal)
   - Avoid discrete or highly irregular distributions

4. **Moderate Dimensions**: Optimal for 10-1000 dimensions
   - Each policy typically uses 10-50 dimensions
   - Performance degrades beyond 10,000 dimensions

### When NOT to Use QMC

1. **Small Samples**: QMC needs minimum sample size for benefits
   - Use standard MC for < 500 simulations
   - QMC setup overhead not justified for small samples

2. **Discontinuous Problems**: Avoid for:
   - Binary outcomes
   - Highly discrete severity distributions
   - Jump processes

3. **Very High Dimensions**: Beyond 10,000 dimensions
   - Curse of dimensionality affects QMC too
   - Consider dimension reduction techniques first

## Scrambling Methods

### Owen Scrambling (Recommended)

```python
set_qmc_simulator(method='sobol', scramble=True, seed=42)
```

**When to use:**
- Default choice for most applications
- Provides unbiased estimates
- Enables error estimation via multiple seeds
- Best for production use

**Benefits:**
- Preserves low-discrepancy properties
- Randomization allows confidence intervals
- Robust to integrand properties

### Digital Shift Scrambling

```python
set_qmc_simulator(method='sobol', scramble=True, scramble_method='digital_shift')
```

**When to use:**
- Faster than Owen scrambling
- Good for initial exploration
- When computational budget is tight

**Limitations:**
- Less randomization than Owen
- May miss some convergence benefits

### No Scrambling

```python
set_qmc_simulator(method='sobol', scramble=False)
```

**When to use:**
- Deterministic results needed
- Debugging QMC implementation
- Benchmarking pure QMC performance

**Warning:** Cannot estimate Monte Carlo error without scrambling!

## Skip Values and Burn-in

### Optimal Skip Values

The skip parameter controls how many initial points to discard:

```python
# Skip first 2^10 points (recommended default)
engine = SobolEngine(dimension=100, scramble=True, skip=1024)

# For high-precision applications
engine = SobolEngine(dimension=100, scramble=True, skip=4096)

# For quick estimates
engine = SobolEngine(dimension=100, scramble=True, skip=256)
```

**Guidelines:**
- **skip = 2^10 (1024)**: Good default for most applications
- **skip = 2^12 (4096)**: High-precision tail estimation
- **skip = 2^8 (256)**: Quick estimates, dimension < 50
- **skip = 2^14 (16384)**: Ultra-high precision, dimension > 1000

### Choosing Skip Values

Consider these factors:

1. **Dimension**: Higher dimensions need larger skip
   ```
   skip = 2^(8 + log2(dimension/10))
   ```

2. **Precision Requirements**:
   - 1% relative error: skip = 1024
   - 0.1% relative error: skip = 4096
   - 0.01% relative error: skip = 16384

3. **Sample Size**: Larger samples can use smaller skip
   - If n_samples > 100,000: can reduce skip by factor of 2
   - If n_samples < 1,000: increase skip by factor of 2

## Best Practices for Actuarial Applications

### 1. Portfolio Structuring

```python
from quactuary.book import LOB, PolicyTerms, Inforce, Portfolio
from datetime import date

# Good: Group similar risks
# Create policy terms for a group of similar risks
terms = PolicyTerms(
    effective_date=date(2026, 1, 1),
    expiration_date=date(2027, 1, 1),
    lob=LOB.GLPL,
    exposure_base="SALES",
    exposure_amount=10_000_000,
    retention_type="deductible",
    per_occ_retention=100_000,
    coverage="occ"
)

# Create inforce buckets for similar risks
inforce_buckets = []
for i, (freq_dist, sev_dist, n_policies) in enumerate(similar_risk_groups):
    inforce = Inforce(
        n_policies=n_policies,
        terms=terms,
        frequency=freq_dist,
        severity=sev_dist,
        name=f"Risk Group {i+1}"
    )
    inforce_buckets.append(inforce)

# Create portfolio
portfolio = sum(inforce_buckets[1:], inforce_buckets[0])

# Better: Order by expected claim count
inforce_sorted = sorted(inforce_buckets, 
                       key=lambda inf: inf.frequency.mean() * inf.n_policies)
portfolio = sum(inforce_sorted[1:], inforce_sorted[0])
```

### 2. Dimension Allocation Strategy

```python
# Configure dimension allocation
allocator = DimensionAllocator(total_dims=5000)

# Allocate more dimensions to volatile inforce buckets
for inforce in portfolio.inforces:
    expected_claims = inforce.frequency.mean() * inforce.n_policies
    claim_volatility = inforce.frequency.std() * np.sqrt(inforce.n_policies)
    
    # Adjust max claims based on volatility
    max_claims = int(expected_claims + 3 * claim_volatility)
    max_claims = min(max_claims, 100 * inforce.n_policies)  # Cap for efficiency
    
    freq_dim, sev_dims = allocator.allocate_for_inforce(
        max_claims=max_claims,
        inforce_id=inforce.name
    )
```

### 3. Convergence Diagnostics

```python
from quactuary.backend import set_backend
from quactuary.sobol import set_qmc_simulator

# Run multiple scrambled sequences
n_scrambles = 10
estimates = []

for seed in range(n_scrambles):
    set_qmc_simulator(method='sobol', scramble=True, seed=seed)
    set_backend('classical')
    model = PricingModel(portfolio)
    result = model.simulate(n_sims=10000)
    estimates.append(result.estimates['VaR'])

# Check convergence
mean_estimate = np.mean(estimates)
std_estimate = np.std(estimates)
coef_variation = std_estimate / mean_estimate

if coef_variation > 0.05:
    print("Warning: High variability, increase sample size")
```

### 4. Adaptive Sampling

```python
def adaptive_qmc_pricing(portfolio, target_precision=0.01, max_samples=100000):
    """Adaptively increase samples until target precision reached."""
    n_samples = 5000
    estimates = []
    
    while n_samples <= max_samples:
        # Run with different scrambles
        batch_estimates = []
        for seed in range(5):
            set_qmc_simulator(method='sobol', scramble=True, seed=seed)
            set_backend('classical')
            model = PricingModel(portfolio)
            result = model.simulate(n_sims=n_samples, tail_alpha=0.01)
            batch_estimates.append(result.estimates['VaR'])
        
        estimates.extend(batch_estimates)
        
        # Check precision
        rel_error = np.std(estimates) / np.mean(estimates)
        if rel_error < target_precision:
            break
            
        n_samples *= 2
    
    return np.mean(estimates), rel_error, n_samples
```

## Performance Optimization Tips

### 1. Batch Processing

```python
# Good: Process inforce buckets in batches
batch_size = 10
inforce_list = portfolio.inforces
for i in range(0, len(inforce_list), batch_size):
    batch = inforce_list[i:i+batch_size]
    batch_portfolio = sum(batch[1:], batch[0]) if batch else Portfolio()
    process_batch(batch_portfolio)
```

### 2. Dimension Reuse

```python
# Cache dimension allocations for repeated calculations
dimension_cache = {}

def get_cached_dimensions(inforce_name, max_claims):
    key = (inforce_name, max_claims)
    if key not in dimension_cache:
        dimension_cache[key] = allocator.allocate_for_inforce(
            max_claims=max_claims,
            inforce_id=inforce_name
        )
    return dimension_cache[key]
```

### 3. Memory Management

```python
import gc

# For very large portfolios, process in chunks
def process_large_portfolio(inforce_list, chunk_size=10):
    results = []
    
    for i in range(0, len(inforce_list), chunk_size):
        chunk = inforce_list[i:i+chunk_size]
        if chunk:
            portfolio_chunk = sum(chunk[1:], chunk[0])
        else:
            continue
        
        set_backend('classical')
        model = PricingModel(portfolio_chunk)
        result = model.simulate(n_sims=10000)
        results.append(result)
        
        # Clean up
        del model, portfolio_chunk
        gc.collect()
    
    return aggregate_results(results)
```

## Troubleshooting Common Issues

### Issue: QMC slower than standard MC

**Causes & Solutions:**
1. Sample size too small → Use at least 1000 samples
2. Dimension too high → Check if dimension > 10000
3. Setup overhead → Reuse QMC simulator for multiple runs

### Issue: Results vary between runs

**Causes & Solutions:**
1. Different scrambling seeds → This is expected! Use average of multiple runs
2. Insufficient samples → Increase sample size
3. Poor dimension allocation → Check for dimension wrapping

### Issue: Memory errors with large portfolios

**Causes & Solutions:**
1. Too many dimensions allocated → Cap max_claims per policy
2. Large sample arrays → Process in batches
3. Memory leaks → Explicitly delete large objects and call gc.collect()

## Example: Complete QMC Workflow

```python
import numpy as np
from quactuary.sobol import set_qmc_simulator, reset_qmc_simulator
from quactuary.backend import set_backend
from quactuary.pricing import PricingModel
from quactuary.book import LOB, PolicyTerms, Inforce, Portfolio
from quactuary.distributions.frequency import Poisson, NegativeBinomial
from quactuary.distributions.severity import Lognormal, Exponential
from datetime import date

def price_with_qmc(portfolio, confidence_level=0.99, target_cv=0.01):
    """Complete QMC pricing workflow with best practices."""
    
    # 1. Determine optimal parameters
    n_policies = sum(inf.n_policies for inf in portfolio.inforces)
    dimension = estimate_dimension(portfolio)
    skip = 2**(10 + int(np.log2(dimension/100)))
    n_simulations = max(5000, 100 * len(portfolio.inforces))
    
    # 2. Run multiple scrambled sequences
    estimates = []
    n_scrambles = 10
    
    for seed in range(n_scrambles):
        # Configure QMC
        set_qmc_simulator(
            method='sobol',
            scramble=True,
            seed=seed,
            skip=skip
        )
        
        # Run pricing
        set_backend('classical')
        model = PricingModel(portfolio)
        result = model.simulate(n_sims=n_simulations, 
                              tail_alpha=1-confidence_level)
        
        # Collect estimate
        var = result.estimates['VaR']
        estimates.append(var)
    
    # 3. Compute final estimate with uncertainty
    final_estimate = np.mean(estimates)
    std_error = np.std(estimates)
    cv = std_error / final_estimate
    
    # 4. Check if more samples needed
    if cv > target_cv:
        print(f"Warning: CV={cv:.3f} exceeds target {target_cv}")
        print(f"Recommend increasing samples to {int(n_simulations * (cv/target_cv)**2)}")
    
    # 5. Clean up
    reset_qmc_simulator()
    
    return {
        'estimate': final_estimate,
        'std_error': std_error,
        'cv': cv,
        'n_simulations': n_simulations,
        'confidence_interval': (
            final_estimate - 1.96 * std_error,
            final_estimate + 1.96 * std_error
        )
    }

def estimate_dimension(portfolio):
    """Estimate total dimensions needed for portfolio."""
    total_dim = 0
    for inforce in portfolio.inforces:
        expected_claims = inforce.frequency.mean() * inforce.n_policies
        std_claims = inforce.frequency.std() * np.sqrt(inforce.n_policies)
        max_claims = int(expected_claims + 3 * std_claims)
        total_dim += 1 + min(max_claims, 50 * inforce.n_policies)
    return total_dim

# Example usage
if __name__ == "__main__":
    # Create example portfolio
    terms = PolicyTerms(
        effective_date=date(2026, 1, 1),
        expiration_date=date(2027, 1, 1),
        lob=LOB.GLPL,
        exposure_base="SALES",
        exposure_amount=10_000_000,
        retention_type="deductible",
        per_occ_retention=100_000,
        coverage="occ"
    )
    
    inforce1 = Inforce(
        n_policies=100,
        terms=terms,
        frequency=Poisson(mu=3.5),
        severity=Exponential(scale=50000),
        name="GLPL Small Accounts"
    )
    
    inforce2 = Inforce(
        n_policies=50,
        terms=terms,
        frequency=NegativeBinomial(n=10, p=0.3),
        severity=Lognormal(mu=11, sigma=1.5),
        name="GLPL Large Accounts"
    )
    
    portfolio = inforce1 + inforce2
    
    # Run QMC pricing
    results = price_with_qmc(portfolio)
    print(f"VaR estimate: ${results['estimate']:,.2f}")
    print(f"Coefficient of variation: {results['cv']:.3f}")
```

## References

1. Owen, A. B. (1998). "Scrambling Sobol' and Niederreiter-Xing points"
2. L'Ecuyer, P. & Lemieux, C. (2002). "Recent advances in randomized quasi-Monte Carlo methods"
3. Dick, J. & Pillichshammer, F. (2010). "Digital nets and sequences"