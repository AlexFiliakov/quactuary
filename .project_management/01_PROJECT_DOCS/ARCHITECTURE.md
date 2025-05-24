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
