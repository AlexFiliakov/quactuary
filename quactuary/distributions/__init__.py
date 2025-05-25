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
]