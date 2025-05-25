"""
Distributions module for quactuary.

This module provides a comprehensive set of probability distributions for actuarial
modeling, including frequency distributions (for claim counts), severity distributions
(for claim amounts), and compound distributions (for aggregate losses).

The module is organized into three main components:

1. **Frequency Distributions** (`frequency` module):
   - Discrete distributions for modeling the number of claims/losses
   - Implementations: Poisson, Binomial, NegativeBinomial, Geometric, Logarithmic
   - Support for zero-inflation and empirical distributions
   - Quasi-Monte Carlo (QMC) integration for improved convergence

2. **Severity Distributions** (`severity` module):
   - Continuous distributions for modeling individual claim amounts
   - Implementations: LogNormal, Gamma, Exponential, Pareto, Weibull, etc.
   - Support for truncation, censoring, and discretization
   - Integration with numerical quadrature methods

3. **Compound Distributions** (`compound` module):
   - Models for aggregate losses (sum of random number of random amounts)
   - Analytical solutions for special cases (e.g., Poisson-Exponential)
   - Simulation-based approaches for general cases
   - Panjer recursion for exact calculation with discrete severities

Key Features:
    - Consistent API following scipy.stats conventions
    - Support for both analytical and simulation-based calculations
    - Quasi-Monte Carlo integration for improved convergence
    - Efficient implementations with optional JIT compilation
    - Comprehensive parameter validation and error handling

Examples:
    Basic frequency-severity modeling:
        >>> from quactuary.distributions import Poisson, LogNormal
        >>> from quactuary.distributions import CompoundDistribution
        >>> 
        >>> # Define frequency and severity
        >>> frequency = Poisson(lambda_=100)
        >>> severity = LogNormal(mu=7, sigma=1.5)
        >>> 
        >>> # Create compound distribution
        >>> compound = CompoundDistribution.create(frequency, severity)
        >>> 
        >>> # Calculate aggregate statistics
        >>> mean = compound.mean()
        >>> var = compound.var()
        >>> var_95 = compound.ppf(0.95)  # 95% VaR

    Using empirical data:
        >>> from quactuary.distributions import Empirical
        >>> 
        >>> # Historical claim counts
        >>> claim_counts = [95, 103, 98, 110, 92]
        >>> frequency = Empirical(data=claim_counts)
        >>> 
        >>> # Fit severity distribution to historical data
        >>> claim_amounts = [1200, 3400, 890, ...]
        >>> severity = LogNormal.fit(claim_amounts)

    Quasi-Monte Carlo for better convergence:
        >>> from quactuary.distributions import wrap_for_qmc
        >>> 
        >>> # Wrap distributions for QMC sampling
        >>> qmc_frequency = wrap_for_qmc(frequency)
        >>> qmc_severity = wrap_for_qmc(severity)
        >>> 
        >>> # Use in compound distribution
        >>> compound = CompoundDistribution.create(qmc_frequency, qmc_severity)

Notes:
    - All distributions follow the scipy.stats API for consistency
    - The compound module automatically selects analytical vs simulation methods
    - QMC wrappers improve convergence for smooth integrands
    - See individual module documentation for detailed usage
"""

from .frequency import FrequencyModel
from .severity import SeverityModel
from .compound import (
    CompoundDistribution,
    AnalyticalCompound,
    SimulatedCompound,
    PoissonExponentialCompound,
    PoissonGammaCompound,
    GeometricExponentialCompound,
    NegativeBinomialGammaCompound,
    BinomialLognormalApproximation,
    PanjerRecursion,
)
from .qmc_wrapper import (
    QMCFrequencyWrapper,
    QMCSeverityWrapper,
    wrap_for_qmc,
)

__all__ = [
    'FrequencyModel',
    'SeverityModel', 
    'CompoundDistribution',
    'AnalyticalCompound',
    'SimulatedCompound',
    'PoissonExponentialCompound',
    'PoissonGammaCompound',
    'GeometricExponentialCompound',
    'NegativeBinomialGammaCompound',
    'BinomialLognormalApproximation',
    'PanjerRecursion',
    'QMCFrequencyWrapper',
    'QMCSeverityWrapper',
    'wrap_for_qmc',
]