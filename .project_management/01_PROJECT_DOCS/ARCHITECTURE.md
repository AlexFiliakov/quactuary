# `quactuary` package architecture

## Architecture Overview

### quactuary Python Package

The package provides quantum-accelerated actuarial computations with a seamless classical/quantum backend switch:

- **Backend System**: Toggle between classical and quantum execution via `set_backend()` or `use_backend()` context manager
- **Core Modules**:
  - `book.py`: Policy terms, portfolio management (PolicyTerms, Inforce, Portfolio classes)
  - `distributions/`: Frequency (Poisson, NegativeBinomial, etc.) and severity (Pareto, Lognormal, etc.) distributions
  - `pricing.py`: PricingModel for simulations and risk measures (VaR, TVaR)
  - `quantum.py`: Quantum circuit implementations using Qiskit
  - `classical.py`: Classical Monte Carlo implementations

### Key Design Patterns

1. **Backend Abstraction**: All models support both classical and quantum backends through a unified API
2. **Pandas Integration**: Data structures designed to work seamlessly with pandas DataFrames
3. **Distribution Hierarchy**: Common base classes for frequency/severity distributions with quantum state preparation methods
4. **Portfolio Composition**: Inforce objects can be combined using `+` operator to build portfolios

### Testing Approach

- Tests compare quantum vs classical results for validation
- Mock quantum backends available for unit testing
- Coverage tracked via pytest-cov and GitHub Actions

### Version Management

- Version managed by setuptools_scm from git tags
- Coverage and lines of code badges auto-updated via GitHub Actions

## Extended Distribution Architecture

### Overview

The distribution system has been extended with advanced modeling capabilities while maintaining compatibility with the existing backend abstraction. All new distributions support both classical and quantum backends through the unified API.

### Core Distribution Extensions

#### 1. Compound Binomial Distributions (`distributions/compound_binomial.py`)

Implements compound distributions where frequency follows a binomial distribution:

- **BinomialExponentialCompound**: Analytical solution using MGF approach
  - Mean: `E[S] = n * p * θ`
  - Variance: `Var[S] = n * p * (θ² * (1 + p))`
  - PDF computed via series expansion with exponential terms

- **BinomialGammaCompound**: Semi-analytical with Bessel function representation
  - Uses scipy's special functions for numerical stability
  - Supports both shape and scale parameterizations

- **BinomialLognormalCompound**: Fenton-Wilkinson approximation for sums
  - Matches first two moments of the aggregate distribution
  - Efficient for large portfolios with lognormal severities

- **PanjerBinomialRecursion**: Exact recursive calculation
  - Supports any discrete severity distribution
  - Memory-efficient implementation with probability mass caching
  - Convergence monitoring for numerical stability

#### 2. Mixed Poisson Processes (`distributions/mixed_poisson.py`)

Captures heterogeneity in frequency distributions:

- **PoissonGammaMixture** (Negative Binomial): 
  - Models overdispersion through gamma-distributed rates
  - Closed-form expressions for all moments
  - Efficient sampling via compound generation

- **PoissonInverseGaussianMixture**:
  - Heavy-tailed frequency distribution
  - Bessel function representation for density
  - Laplace transform inversion for CDF

- **HierarchicalPoissonMixture**:
  - Multi-level random effects model
  - Between-group and within-group variance components
  - MCMC or variational inference for parameter estimation

- **TimeVaryingPoissonMixture**:
  - Non-homogeneous Poisson process
  - Flexible intensity function specification
  - Thinning algorithm for efficient sampling

#### 3. Zero-Inflated Models (`distributions/zero_inflated.py`)

Handles excess zeros in claims data:

- **Base Class Architecture**:
  - Mixture of point mass at zero and base distribution
  - EM algorithm for parameter estimation
  - Vuong test for model comparison

- **Statistical Testing**:
  - Zero-inflation detection via score test
  - Likelihood ratio test implementation
  - AIC/BIC for model selection

- **Implemented Models**:
  - ZeroInflatedPoisson
  - ZeroInflatedNegativeBinomial  
  - ZeroInflatedBinomial
  - ZeroInflatedCompound (general framework)

#### 4. Edgeworth Expansion (`distributions/edgeworth.py`)

Higher-order distribution approximations:

- **Series Expansion Framework**:
  - Hermite polynomial basis functions
  - Cumulant-based coefficient calculation
  - Automatic order selection based on sample size

- **Cornish-Fisher Expansion**:
  - Quantile function approximation
  - Monotonicity correction algorithms
  - Bootstrap confidence intervals

- **Validation and Diagnostics**:
  - Convergence monitoring
  - Remainder term estimation
  - Visual diagnostics for approximation quality

### Integration Architecture

#### Enhanced Factory Pattern (`distributions/compound_extensions.py`)

The `create_extended_compound_distribution` function provides:

1. **Intelligent Distribution Selection**:
   - Analyzes data characteristics (zero-inflation, overdispersion)
   - Recommends appropriate distribution family
   - Automatic parameter initialization

2. **Performance Optimizations**:
   - LRU caching for repeated calculations
   - Parallel processing for Monte Carlo simulations
   - Vectorized operations throughout

3. **Unified Interface**:
   ```python
   dist = create_extended_compound_distribution(
       dist_type="auto",  # Automatic selection
       frequency_params={"n": 100, "p": 0.3},
       severity_dist=lognormal(s=0.5, scale=1000),
       optimization_level="high"
   )
   ```

### Design Patterns and Best Practices

#### 1. Mixin Architecture

- **SeriesExpansionMixin**: Shared functionality for series-based calculations
- **StatisticalTestMixin**: Common statistical testing procedures
- **CachingMixin**: Transparent result caching

#### 2. Parameter Validation

- Type checking with runtime validation
- Range constraints for statistical validity
- Informative error messages for debugging

#### 3. Numerical Stability

- Log-space computations where appropriate
- Scaled calculations to prevent overflow
- Fallback algorithms for edge cases

#### 4. Quantum Readiness

All distributions implement quantum state preparation methods:
- Amplitude encoding for continuous distributions
- Basis encoding for discrete distributions
- Efficient circuit depth optimization

### Testing Strategy

1. **Unit Tests**: Each distribution class has comprehensive tests
2. **Property Tests**: Statistical properties verified (moments, bounds)
3. **Integration Tests**: Backend switching, factory function
4. **Performance Benchmarks**: Comparison with scipy/numpy baselines

### Future Extension Points

The architecture supports easy addition of:
- New compound distribution families
- Additional mixing distributions
- Custom basis functions for expansions
- Alternative parameter estimation methods
