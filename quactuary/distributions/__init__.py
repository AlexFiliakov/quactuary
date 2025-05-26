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
   - Implementations: Lognormal, Gamma, Exponential, Pareto, Weibull, etc.
   - Support for truncation, censoring, and discretization
   - Integration with numerical quadrature methods

3. **Compound Distributions** (`compound` module):
   - Models for aggregate losses (sum of random number of random amounts)
   - Analytical solutions for special cases (e.g., Poisson-Exponential)
   - Simulation-based approaches for general cases
   - Panjer recursion for exact calculation with discrete severities

4. **Extended Distribution Support**:
   - **Compound Binomial** (part of `compound` module):
     - Binomial-Exponential, Binomial-Gamma, Binomial-Lognormal
     - Panjer recursion for binomial frequency
   - **Mixed Poisson** (`mixed_poisson` module):
     - Poisson-Gamma (Negative Binomial), Poisson-Inverse Gaussian
     - Hierarchical and time-varying intensity models
   - **Zero-Inflated** (`zero_inflated` module):
     - Zero-inflated compound distributions with EM algorithm
     - Statistical tests for zero-inflation detection
   - **Edgeworth Expansion** (`edgeworth` module):
     - Higher-order moment corrections to normal approximation
     - Automatic order selection and convergence diagnostics

Key Features:
    - Consistent API following scipy.stats conventions
    - Support for both analytical and simulation-based calculations
    - Quasi-Monte Carlo integration for improved convergence
    - Efficient implementations with optional JIT compilation
    - Comprehensive parameter validation and error handling

Examples:
    Basic frequency-severity modeling:
        >>> from quactuary.distributions import Poisson, Lognormal
        >>> from quactuary.distributions import CompoundDistribution
        >>> 
        >>> # Define frequency and severity
        >>> frequency = Poisson(lambda_=100)
        >>> severity = Lognormal(shape=1.5, scale=np.exp(7))
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
        >>> severity = Lognormal.fit(claim_amounts)

    Quasi-Monte Carlo for better convergence:
        >>> from quactuary.distributions import wrap_for_qmc
        >>> 
        >>> # Wrap distributions for QMC sampling
        >>> qmc_frequency = wrap_for_qmc(frequency)
        >>> qmc_severity = wrap_for_qmc(severity)
        >>> 
        >>> # Use in compound distribution
        >>> compound = CompoundDistribution.create(qmc_frequency, qmc_severity)

    Extended distributions for complex scenarios:
        >>> from quactuary.distributions.compound_extensions import create_extended_compound_distribution
        >>> 
        >>> # Zero-inflated compound with parallel processing
        >>> zi_compound = create_extended_compound_distribution(
        ...     frequency='poisson',
        ...     severity='gamma',
        ...     zero_inflated=True,
        ...     zero_prob=0.2,
        ...     parallel=True,
        ...     mu=5.0, a=2.0, scale=1000
        ... )
        >>> 
        >>> # Mixed Poisson for heterogeneous portfolios
        >>> from quactuary.distributions.mixed_poisson import PoissonGammaMixture
        >>> mixed = PoissonGammaMixture(alpha=3.0, beta=0.5)
        >>> 
        >>> # Edgeworth expansion for accurate tail approximation
        >>> from quactuary.distributions.edgeworth import EdgeworthExpansion
        >>> edge = EdgeworthExpansion(mean=10000, variance=2500000, 
        ...                          skewness=0.8, excess_kurtosis=0.5)

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
    SimulatedCompound,
    PoissonExponentialCompound,
    PoissonGammaCompound,
    GeometricExponentialCompound,
    NegativeBinomialGammaCompound,
    create_compound_distribution,
)
from .compound import (
    BinomialExponentialCompound,
    BinomialGammaCompound,
    BinomialLognormalCompound,
    PanjerBinomialRecursion,
)
from .mixed_poisson import (
    MixedPoissonDistribution,
    PoissonGammaMixture,
    PoissonInverseGaussianMixture,
    HierarchicalPoissonMixture,
    TimeVaryingPoissonMixture,
)
from .zero_inflated import (
    ZeroInflatedCompound,
    ZIPoissonCompound,
    ZINegativeBinomialCompound,
    ZIBinomialCompound,
    detect_zero_inflation,
)
from .edgeworth import (
    EdgeworthExpansion,
    CompoundDistributionEdgeworth,
    automatic_order_selection,
)
from .compound_extensions import (
    create_extended_compound_distribution,
    distribution_selection_guide,
)
from .qmc_wrapper import (
    QMCFrequencyWrapper,
    QMCSeverityWrapper,
    wrap_for_qmc,
)

__all__ = [
    # Base models
    'FrequencyModel',
    'SeverityModel', 
    'CompoundDistribution',
    'SimulatedCompound',
    # Standard compound distributions
    'PoissonExponentialCompound',
    'PoissonGammaCompound',
    'GeometricExponentialCompound',
    'NegativeBinomialGammaCompound',
    # Compound binomial distributions
    'BinomialExponentialCompound',
    'BinomialGammaCompound',
    'BinomialLognormalCompound',
    'PanjerBinomialRecursion',
    # Mixed Poisson processes
    'MixedPoissonDistribution',
    'PoissonGammaMixture',
    'PoissonInverseGaussianMixture',
    'HierarchicalPoissonMixture',
    'TimeVaryingPoissonMixture',
    # Zero-inflated models
    'ZeroInflatedCompound',
    'ZIPoissonCompound',
    'ZINegativeBinomialCompound',
    'ZIBinomialCompound',
    'detect_zero_inflation',
    # Edgeworth expansion
    'EdgeworthExpansion',
    'CompoundDistributionEdgeworth',
    'automatic_order_selection',
    # Factory functions
    'create_compound_distribution',
    'create_extended_compound_distribution',
    'distribution_selection_guide',
    # QMC support
    'QMCFrequencyWrapper',
    'QMCSeverityWrapper',
    'wrap_for_qmc',
]