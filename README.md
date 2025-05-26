![QuActuary header image](images/qc-header.jpg)
# *qu*Actuary: Quantum advantage for the actuarial profession

A high-level, **pandas**-integrated actuarial framework that delivers optional quantum acceleration without requiring quantum programming expertise.

## 🚀 Launching Soon

_**Under active development**_ - We're crafting powerful solutions for Property & Casualty professionals that will transform your workflow.

Sign up for our launch notification at [quactuary.com](https://quactuary.com/) and get early access to revolutionary actuarial tools.

[![version](https://img.shields.io/github/v/tag/AlexFiliakov/quactuary?label=version&sort=semver)](https://github.com/AlexFiliakov/quactuary/releases)
[![Lines of Code](https://img.shields.io/badge/dynamic/json?label=lines%20of%20code&url=https://raw.githubusercontent.com/AlexFiliakov/quactuary/main/loc.json&query=$.SUM.code&color=blue)](https://github.com/AlexFiliakov/quactuary)
[![Tests](https://img.shields.io/github/actions/workflow/status/AlexFiliakov/quactuary/.github/workflows/python-tests.yml?branch=main)](https://github.com/AlexFiliakov/quactuary/actions)
[![Coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/AlexFiliakov/quactuary/main/coverage.json)](https://github.com/AlexFiliakov/quactuary/actions)
[![GitHub License](https://img.shields.io/github/license/AlexFiliakov/quactuary)](https://github.com/AlexFiliakov/quactuary/blob/main/LICENSE)

## Why *qu*Actuary?

Actuarial work demands intensive simulations and complex models for accurate pricing and reserving. **quActuary** bridges the gap between traditional actuarial science and quantum computing, offering potential quadratic speedups in Monte Carlo simulations without requiring quantum expertise. 

Built on IBM's Qiskit v1.4.2 framework, quActuary abstracts away the complexity of quantum circuit design while integrating seamlessly with familiar actuarial Python libraries like **chainladder-python**, **aggregate**, and **gemact**.

Each model class features quantum computing as a pluggable backend, allowing you to toggle between classical and quantum execution with simple parameter changes. The API is designed to feel intuitive to actuaries, using **pandas data structures** and familiar terminology, while keeping quantum mechanics details under the hood.

### Quantum Ready, Not Quantum Required

Get the best of both worlds: leverage quantum speedups when needed, or use the same features classically. This dual capability also serves as a validation mechanism, allowing you to verify quantum results against classical computations as you build confidence in the quantum approach.

![Duck resting on a panda](images/panda-duck.png)

## Key Goals

- **Actuarial-First Design:** Work with familiar terminology and data structures that flatten the learning curve for actuarial professionals.
- **Intuitive Actuarial Interface:** Perform pricing, reserving, and risk analysis using high-level, Pythonic pandas-like interfaces with minimal code.
- **Flexible Quantum Access:** Run simulations with default settings or dive deeper to inspect and customize quantum circuits and algorithms.
- **Seamless Environment Switching:** Develop on local simulators and deploy to IBM Quantum hardware with minimal configuration changes.

## Documentation

The official documentation is hosted on [docs.quactuary.com](https://docs.quactuary.com/).

## Roadmap
- Phase 1: Core Simulation Functions for Risk Pricing
  - Excess Loss Calculations
  - VaR and TVaR
  - Quantum Monte Carlo
  - v0.1.0 Release
- Phase 2: Reserving
  - IBNR Estimation
  - Basic Reserving Methods
  - Mack Method
  - Quantum Copula
  - Tail Risk, Risk Margin and Risk-of-Ruin Models
- Phase 3: Advanced Models
  - Generalized Linear Models
  - Portfolio Optimization
- Phase 4: Beyond P&C
  - Reinsurance
  - Life
  - Health
- Phase 5: Value-Add
  - Quantum Data Privacy & Security

## Usage Example

### Example: Expected Loss

Calculate the expected loss for a portfolio of insurance policies:

```python
import quactuary as qa
import quactuary.book as book

from datetime import date
from quactuary.backend import set_backend
from quactuary.book import (
    ExposureBase, LOB, PolicyTerms, Inforce, Portfolio)
from quactuary.distributions.frequency import Poisson, NegativeBinomial
from quactuary.distributions.severity import Pareto, Lognormal
from quactuary.pricing import PricingModel


# Workers’ Comp Bucket
wc_policy = PolicyTerms(
    effective_date=date(2026, 1, 1),
    expiration_date=date(2027, 1, 1),
    lob=LOB.WC,
    exposure_base=book.PAYROLL,
    exposure_amount=100_000_000,
    retention_type="deductible",
    per_occ_retention=500_000,
    coverage="occ"
)

# General Liability Bucket
glpl_policy = PolicyTerms(
    effective_date=date(2026, 1, 1),
    expiration_date=date(2027, 1, 1),
    lob=LOB.GLPL,
    exposure_base=book.SALES,
    exposure_amount=10_000_000_000,
    retention_type="deductible",
    per_occ_retention=1_000_000,
    coverage="occ"
)

# Frequency-Severity Distributions
wc_freq = Poisson(100)
wc_sev = Pareto(1, 0, 40_000)

glpl_freq = NegativeBinomial(50, 0.5)
glpl_sev = Lognormal(2, 0, 100_000)

# Book of Business
wc_inforce = Inforce(
    n_policies=1000,
    terms=wc_policy,
    frequency=wc_freq,
    severity=wc_sev,
    name = "WC 2026 Bucket"
)

glpl_inforce = Inforce(
    n_policies=700,
    terms=glpl_policy,
    frequency=glpl_freq,
    severity=glpl_sev,
    name = "GLPL 2026 Bucket"
)

portfolio = wc_inforce + glpl_inforce

pm = PricingModel(portfolio)

# Test using Classical Monte Carlo
set_backend("classical")
classical_result = pm.simulate(n_sims=1_000)
classical_mean = classical_result.estimates["mean"]
print(f"Classical portfolio expected loss: {classical_mean}")

# When ready, run a quantum session
set_backend("quantum", provider="ibmq")
quantum_result = pm.simulate()
quantum_mean = quantum_result.estimates["mean"]
print(f"Quantum portfolio expected loss: {quantum_mean}")
```

In this example, `quactuary` loads the specified distributions into a quantum state (using an n-qubit discrete approximation) and builds the circuit needed for the excess loss algorithm on a book of business. The user is not expected to know anything about quantum circuit design.

The Portfolio can be built up using approximate Inforce buckets, or down to policy-level granularity with individual PolicyTerms tailored to each client from your policy administration system.

### Example: Risk Measures

Extend the portfolio above and calculate risk measures:

```python
from quactuary.backend import use_backend
from quactuary.distributions.frequency import Geometric
from quactuary.distributions.severity import ContinuousUniformSeverity


# Commercial Auto Bucket
cauto_policy = PolicyTerms(
    effective_date=date(2026, 1, 1),
    expiration_date=date(2027, 1, 1),
    lob=LOB.CAuto,
    exposure_base=book.VEHICLES,
    exposure_amount=50,
    retention_type="deductible",
    per_occ_retention=100_000,
    coverage="occ"
)

# Frequency-Severity Distributions
cauto_freq = Geometric(1/8)
cauto_sev = ContinuousUniformSeverity(5_000, 95_000)

# Commercial Auto Inforce
cauto_inforce = Inforce(
    n_policies=400,
    terms=cauto_policy,
    frequency=cauto_freq,
    severity=cauto_sev,
    name = "CAuto 2026 Bucket"
)

# Add to Existing Portfolio
portfolio += cauto_inforce
pm2 = PricingModel(portfolio)

# Test using Classical Monte Carlo
with use_backend("classical", num_simulations=1_000):
    classical_result = pm2.simulate(tail_alpha=0.05, n_sims=1_000)
    classical_VaR = classical_result.estimates["VaR"]
    classical_TVaR = classical_result.estimates["TVaR"]
    print(f"Classical portfolio VaR: {classical_VaR}")
    print(f"Classical portfolio TVaR: {classical_TVaR}")

# Evaluate using the Quantum session established earlier
quantum_result = pm2.simulate(tail_alpha=0.05)
quantum_VaR = quantum_result.estimates["VaR"]
quantum_TVaR = quantum_result.estimates["TVaR"]
print(f"Quantum portfolio VaR: {quantum_VaR}")
print(f"Quantum portfolio TVaR: {quantum_TVaR}")
```

Backends can be called as `ContextManager`s to be used across multiple statements. Again, all quantum circuits are taken care of behind the scenes.

## Development

### Development Environment Setup

quActuary uses a centralized development script `run_dev.py` to streamline all development operations:

```bash
# Install development environment
python run_dev.py install

# Run tests
python run_dev.py test
python run_dev.py test --coverage
python run_dev.py test --file test_pricing.py

# Code quality
python run_dev.py lint          # Run all linters
python run_dev.py format         # Auto-format code

# Documentation
python run_dev.py docs           # Build documentation
python run_dev.py docs --serve   # Build and serve locally

# Utilities
python run_dev.py coverage       # Generate coverage report
python run_dev.py clean          # Clean build artifacts
python run_dev.py profile        # Performance profiling
python run_dev.py setup          # Initial environment setup
python run_dev.py version        # Show version info
python run_dev.py completion     # Tab completion setup
```

### Available Commands

**Core Commands:**
- **test**: Run pytest with options for coverage, specific files, or patterns
- **lint**: Run code quality tools (ruff, mypy, black)
- **format**: Auto-format code using black
- **install/build**: Install package in development mode with dependencies
- **coverage**: Generate detailed coverage reports with branch coverage options
- **clean**: Remove build artifacts and caches
- **docs**: Build Sphinx documentation with optional local server

**Utility Commands:**
- **profile**: Run performance profiling on modules or test suites
- **setup**: Set up initial development environment from scratch
- **version**: Display package version and environment information
- **completion**: Generate shell tab completion scripts for bash/zsh

### Features

- **Progress Indicators**: Visual feedback for long-running commands (tests, coverage, docs)
- **Tab Completion**: Enable shell completion with `python run_dev.py completion`
- **IDE Integration**: Supports VS Code, PyCharm, Vim, Sublime, and Emacs
- **Environment Detection**: Automatically detects and validates virtual environments

The script automatically detects virtual environments and provides helpful error messages. Use `python run_dev.py --help` or `python run_dev.py <command> --help` for detailed options.

### Example Development Workflows

**Initial Setup:**
```bash
# Clone and set up from scratch
git clone https://github.com/AlexFiliakov/quactuary.git
cd quactuary
python run_dev.py setup  # Creates venv and installs dependencies
```

**Daily Development:**
```bash
# Start your day
python run_dev.py version       # Check environment
python run_dev.py test --file test_pricing.py  # Test specific module

# Before committing
python run_dev.py format        # Auto-format code
python run_dev.py lint          # Check code quality
python run_dev.py test --coverage  # Run full test suite
```

**Performance Optimization:**
```bash
# Profile a slow test
python run_dev.py profile "pytest tests/test_quantum.py -v"

# Profile specific function
python run_dev.py profile quactuary.pricing
```

**Enable Tab Completion:**
```bash
# One-time setup
python run_dev.py completion    # Shows setup instructions
# Follow the instructions for your shell (bash/zsh)
```
