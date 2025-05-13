![QuActuary header image](images/qc-header.jpg)
# *qu*Actuary: quantum-powered actuarial tools

Quantum-powered actuarial tools for Python that dramatically accelerate risk calculations without requiring quantum programming expertise.

*This package is still under development, presently focused on Property & Casualty (non-life) techniques.*

## Introduction

Actuarial computations often involve heavy simulation and complex models for pricing and reserving. **quActuary** aims to empower actuaries with quantum computing speedups (e.g., quadratic Monte Carlo gains) without requiring quantum expertise. The package will wrap IBM’s Qiskit v1.4.2 (including `qiskit_algorithms` and `qiskit_aer`) to abstract away the design of quantum circuits.

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

An actuary wants to calculate the expected excess loss for a portfolio, above a retention *L* with a given coinsurance  *c*. They have fitted a loss severity distribution (such as a lognormal with parameters μ, σ). Using **quActuary**:

```python
import quactuary as qa
from quactuary.pricing import ExcessLossModel


# Define loss distribution (lognormal with mean=203k, sd=364k)
loss_model = ExcessLossModel(dist='lognormal', params={'mu': 11.5, 'sigma': 1.2}, 
                              per_occ_deductible=200_000, coinsurance=0.10,
                              per_occ_limit=None, agg_limit=None)
# Compute expected excess loss using quantum amplitude estimation
quantum_excess = loss_model.compute_excess_loss(confidence=0.95)  
print("Quantum Expected Excess Loss:", quantum_excess)

# (Optional) Compare to classical Monte Carlo for verification
classical_excess = loss_model.compute_excess_loss(classical=_samples=1_000_000)
print("Classical Expected Excess Loss:", classical_excess)
```

In this example, `ExcessLossModel` loads the specified distribution into a quantum state (using an n-qubit discrete approximation) and builds the circuit needed for the excess loss algorithm. The user is not expected to be proficient in quantum circuit design.