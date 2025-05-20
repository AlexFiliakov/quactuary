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
from quactuary.book import (
    ExposureBase, LOB, PolicyTerms, Inforce, Portfolio)
from quactuary.distributions.frequency import Poisson, NegativeBinomial
from quactuary.distributions.severity import Pareto, Lognormal


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
wc_sev = Pareto(0, 40_000)

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

# Test using Classical Monte Carlo
mc_expected_loss = portfolio.mean_loss(qa.backend('classical', num_simulations=1_000_000))
print(f"Classical portfolio expected loss: {mc_expected_loss}")

# When ready, run a quantum session
q_expected_loss = portfolio.mean_loss(qa.backend('quantum', confidence=0.95))
print(f"Quantum portfolio expected loss: {q_expected_loss}")
```

In this example, `quactuary` loads the specified distributions into a quantum state (using an n-qubit discrete approximation) and builds the circuit needed for the excess loss algorithm on a book of business. The user is not expected to know anything about quantum circuit design.

The Portfolio can be built up using approximate Inforce buckets, or down to policy-level granularity with individual PolicyTerms tailored to each client from your policy administration system.

### Example: Risk Measures

Extend the portfolio above and calculate risk measures:

```python
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
cauto_freq = Poisson(3)
cauto_sev = Lognormal(1.5, 0, 50_000)

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

# Test using Classical Monte Carlo
with qa.use_backend('classical', num_simulations=1_000_000):
  mc_portfolio_var = portfolio.value_at_risk(0.95)
  mc_portfolio_tvar = portfolio.tail_value_at_risk(0.95)
  print(f"Classical portfolio VaR: {mc_portfolio_var}")
  print(f"Classical portfolio TVaR: {mc_portfolio_tvar}")

# Evaluate Using Quantum Amplitude Estimation
with qa.use_backend('quantum', confidence=0.95):
  q_portfolio_var = portfolio.value_at_risk(0.95)
  q_portfolio_tvar = portfolio.tail_value_at_risk(0.95)
  print(f"Quantum portfolio VaR: {q_portfolio_var}")
  print(f"Quantum portfolio TVaR: {q_portfolio_tvar}")
```

Backends can be called as `ContextManager`s to be used across multiple statements. Again, all quantum circuits are taken care of behind the scenes.
