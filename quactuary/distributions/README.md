# Distributions Module

The quActuary distributions module provides a comprehensive set of probability distributions for actuarial modeling.

## Overview

The module is organized into several components:

### Core Components

1. **Frequency Distributions** (`frequency.py`)
   - Models for claim counts: Poisson, Binomial, NegativeBinomial, Geometric, Logarithmic
   - Support for empirical distributions

2. **Severity Distributions** (`severity.py`)
   - Models for claim amounts: LogNormal, Gamma, Exponential, Pareto, Weibull
   - Support for truncation and censoring

3. **Compound Distributions** (`compound.py`)
   - Aggregate loss models combining frequency and severity
   - Analytical solutions for special cases
   - Monte Carlo simulation for general cases

### Extended Distribution Support

4. **Compound Binomial** (`compound_binomial.py`)
   - `BinomialExponentialCompound`: Analytical solution using Gamma mixtures
   - `BinomialGammaCompound`: Series expansion method
   - `BinomialLognormalCompound`: Fenton-Wilkinson approximation
   - `PanjerBinomialRecursion`: Exact recursion for discrete severities

5. **Mixed Poisson Processes** (`mixed_poisson.py`)
   - `PoissonGammaMixture`: Negative Binomial distribution for overdispersion
   - `PoissonInverseGaussianMixture`: Heavier-tailed mixing
   - `HierarchicalPoissonMixture`: Multi-level portfolio modeling
   - `TimeVaryingPoissonMixture`: Seasonal and temporal patterns

6. **Zero-Inflated Models** (`zero_inflated.py`)
   - `ZeroInflatedCompound`: Base class with EM algorithm
   - `ZIPoissonCompound`, `ZINegativeBinomialCompound`, `ZIBinomialCompound`
   - `detect_zero_inflation()`: Statistical test for excess zeros

7. **Edgeworth Expansion** (`edgeworth.py`)
   - `EdgeworthExpansion`: Higher-order corrections to normal approximation
   - `CompoundDistributionEdgeworth`: Specialized for compound distributions
   - `automatic_order_selection()`: Data-driven order selection

8. **Integration and Extensions** (`compound_extensions.py`)
   - `create_extended_compound_distribution()`: Enhanced factory with all features
   - `distribution_selection_guide()`: Automated distribution recommendation
   - Performance optimizations: caching, parallel processing

## Usage Examples

### Basic Compound Distribution
```python
from quactuary.distributions import Poisson, Exponential
from quactuary.distributions import create_compound_distribution

# Standard compound
freq = Poisson(mu=10)
sev = Exponential(scale=1000)
compound = create_compound_distribution(freq, sev)

mean_loss = compound.mean()
var_95 = compound.ppf(0.95)
```

### Zero-Inflated Model
```python
from quactuary.distributions.compound_extensions import create_extended_compound_distribution

# 20% of policies never claim
zi_compound = create_extended_compound_distribution(
    frequency='poisson',
    severity='gamma',
    zero_inflated=True,
    zero_prob=0.2,
    mu=3.0, a=2.0, scale=1000
)
```

### Mixed Poisson for Heterogeneous Risk
```python
from quactuary.distributions.mixed_poisson import PoissonGammaMixture

# Risk parameter varies across portfolio
mixed = PoissonGammaMixture(alpha=3.0, beta=0.5)
# Results in overdispersion: Var > Mean
```

### Edgeworth Approximation
```python
from quactuary.distributions.edgeworth import EdgeworthExpansion

# For moderately skewed distributions
edge = EdgeworthExpansion(
    mean=10000,
    variance=2500000,
    skewness=0.8,
    excess_kurtosis=0.5
)
# More accurate tail estimates than normal approximation
```

## API Reference

All distributions follow the scipy.stats API convention:

- `pdf(x)`: Probability density/mass function
- `cdf(x)`: Cumulative distribution function  
- `ppf(q)`: Percent point function (quantile)
- `rvs(size)`: Random variate generation
- `mean()`: Expected value
- `var()`: Variance
- `std()`: Standard deviation

## Performance Considerations

- Analytical solutions are used when available (e.g., Poisson-Exponential)
- Series expansions for semi-analytical cases
- Optimized Monte Carlo with caching for general cases
- Parallel processing support for large-scale simulations
- QMC wrappers available for variance reduction

## See Also

- [Usage Examples Notebook](../../usage/Extended_Distributions_Examples.ipynb)
- [API Documentation](https://quactuary.readthedocs.io)