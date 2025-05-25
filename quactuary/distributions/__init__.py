"""
Distributions module for quactuary.

Provides frequency, severity, and compound distributions for actuarial modeling.
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