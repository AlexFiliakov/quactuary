![QuActuary header image](images/qc-header.jpg)
# *qu*Actuary: quantum-powered actuarial tools

A high-level, **pandas**-integrated actuarial framework with optional quantum acceleration without requiring quantum programming expertise.

*This package is still under development, presently focused on Property & Casualty (non-life) techniques.*

[![Tests](https://img.shields.io/github/actions/workflow/status/AlexFiliakov/quactuary/.github/workflows/python-tests.yml?branch=main)](https://github.com/AlexFiliakov/quactuary/actions)
[![Coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/AlexFiliakov/quactuary/main/coverage.json)](https://github.com/AlexFiliakov/quactuary/actions)

## Introduction

Actuarial computations often involve heavy simulation and complex models for pricing and reserving. **quActuary** aims to empower actuaries with quantum computing speedups (e.g., quadratic Monte Carlo gains) without requiring quantum expertise. The package will wrap IBM’s Qiskit v1.4.2 (including `qiskit_algorithms` and `qiskit_aer`) to abstract away the design of quantum circuits.

The design leverages existing actuarial Python libraries (e.g. **chainladder-python** for reserving, **aggregate** for FFT-based aggregate loss distributions, **gemact** for collective risk models and copulas) to minimize new implementations. Each model class (e.g. for pricing or risk) will embed quantum computing as a pluggable backend, so users can toggle between classical and quantum execution without needing quantum expertise.

The API will feel familiar to actuaries (pandas data structures, common actuarial terminology) while hiding quantum mechanics details. Default model outputs will be provided as pandas `Series` or `DataFrame` for seamless downstream analysis.

### Quantum Ready, but Not Quantum Required
Actuaries can reap quantum speedups in tasks like Monte Carlo simulation without changing how they solve problems. By making quantum optional (opt-in via parameters, context managers, or global config), we avoid alienating users who are not ready for it; they can continue to use all features classically. This dual capability also helps in validation: results from quantum runs can be checked against classical computations for consistency, which will be important in building trust in the quantum approach.

![Duck resting on a panda](images/panda-duck.png)

## Key Goals

- **Black-Box Ease of Use:** Provide high-level, Pythonic pandas-like interfaces for common tasks (pricing, reserving, risk measures) so that actuaries can call quantum-accelerated computations with minimal code.
- **Deeper Quantum Access:** Allow advanced users to inspect and tweak quantum circuits/algorithms. For example, users can retrieve underlying Qiskit circuits or adjust algorithm parameters (e.g. change the amplitude estimation method or number of shots).
- **Seamless Backend Switching:** Enable development on local simulators (Qiskit Aer) and easily switch to IBM Quantum hardware for production runs, without requiring code changes aside from selecting a backend.
- Actuarial Familiarity: Use terminology and data structures familiar to actuaries (such as DataFrames for loss triangles or exposures) to flatten the learning curve.

By meeting these goals, quActuary will let actuaries harness quantum computing’s potential speedups in Monte Carlo simulation and optimization while focusing on actuarial logic and business solutions instead of tinkering with quantum mechanics and quantum circuit engineering.

## Roadmap
- Phase 1: Core Simulation Functions for Risk Pricing
  - Excess Loss Calculations
  - VaR and TVaR
  - Quantum Monte Carlo
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
  - Life Insurance Applications
- Phase 5: Value-Add
  - Quantum Data Privacy & Security

## Example Usage Patterns

### Excess Severity Pricing

An actuary wants to calculate the expected excess loss for a portfolio above a $1M retention with a limit of $4M. Using **quActuary**:

```python
import quactuary as qa
from quactuary.distributions.frequency import Poisson, DeterministicFreq
from quactuary.distributions.severity  import Lognormal, ConstantSev
from quactuary.entities import PolicyTerms, Inforce, Portfolio
from quactuary.pricing import ExcessLossModel

# General Liability Bucket
gl_terms = PolicyTerms(per_occ_deductible=200_000, coinsurance=0.10)
gl_freq  = Poisson(mu=0.25)
gl_sev   = Lognormal(mu=11.5, sigma=1.2)
gl_inforce = Inforce(n_policies=10_000,
                     freq=gl_freq,
                     sev=gl_sev,
                     terms=gl_terms,
                     name="General Liability")

# Workers’ Comp Bucket
wc_terms = PolicyTerms(per_occ_deductible=0, per_occ_limit=1_000_000)
wc_freq  = DeterministicFreq(k=1)  # exactly 1 claim each year
wc_sev   = ConstantSev(amount=80_000)
wc_inforce = Inforce(n_policies=5_000,
                     freq=wc_freq,
                     sev=wc_sev,
                     terms=wc_terms,
                     name="Workers Comp")

# Book of Business
portfolio = Portfolio([gl_inforce, wc_inforce])

# Excess Loss layer
layer = ExcessLossModel(portfolio, deductible=1_000_000, limit=4_000_000)

# Test using Classical Monte Carlo
mc_layer_loss = layer.compute_excess_loss(backend('classical', num_simulations=1_000_000))
print(f"Classical layer expected loss: {mc_layer_loss}")

# When ready, run a quantum session
q_layer_loss = layer.compute_excess_loss(backend('quantum', confidence=0.95))
print(f"Quantum layer expected loss: {q_layer_loss}")
```

In this example, `ExcessLossModel` loads the specified distributions into a quantum state (using an n-qubit discrete approximation) and builds the circuit needed for the excess loss algorithm on a book of business. The user is not expected to be proficient in quantum circuit design.

### Risk Measures

Continuing with the prior Portfolio, an actuary wants to calculate risk measures for this portfolio:

```python
# Risk Measures
layer_risk_metrics = RiskMeasureModel(portfolio, deductible=1_000_000, limit=4_000_000)

# Test using Classical Monte Carlo
with qa.use_backend('classical', num_simulations=1_000_000):
  mc_layer_var = layer_risk_metrics.value_at_risk(0.95)
  mc_layer_tvar = layer_risk_metrics.tail_value_at_risk(0.95)
  print(f"Classical layer VaR: {mc_layer_var}")
  print(f"Classical layer TVaR: {mc_layer_tvar}")

# Test using Classical Monte Carlo
with qa.use_backend('quantum', confidence=0.95):
  q_layer_var = layer_risk_metrics.value_at_risk(0.95)
  q_layer_tvar = layer_risk_metrics.tail_value_at_risk(0.95)
  print(f"Quantum layer VaR: {q_layer_var}")
  print(f"Quantum layer TVaR: {q_layer_tvar}")
```

In this example, `RiskMeasureModel` loads the specified distributions into a quantum state (using an n-qubit discrete approximation) and builds the circuit needed for the risk measures on the book of business. Again, no knowledge of quantum computing is needed and the interface integrates seamlessly into business workflows.
