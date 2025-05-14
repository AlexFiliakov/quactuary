"""
Severity distributions for the amount of loss per claim.
This module provides a set of severity distributions that can be used to model the
severity of claims in a given period. The distributions are designed to be used with the
quActuary library, which is a Python library for actuarial science and insurance modeling.
"""

from typing import Protocol

import numpy as np


class SeverityModel(Protocol):
    def pdf(self, x: float) -> float: ...
    def cdf(self, x: float) -> float: ...
    def rvs(self, size: int = 1) -> np.ndarray: ...

# TODO: Provide convenience wrappers so an actuary can pass in either:
# - a scalar (2 → “exactly 2 claims per year”),
# - a ready‑made scipy.stats distribution,
# - or a tiny helper like qa.freq.Poisson(mu=1.8) that just forwards to scipy.stats.poisson.
# Internally we normalize everything to an adapter object that satisfies the protocol.


class Beta(SeverityModel):
    pass


class ChiSquared(SeverityModel):
    pass


class ConstantSev(SeverityModel):
    pass


class EmpiricalSev(SeverityModel):
    pass


class Exponential(SeverityModel):
    pass


class Gamma(SeverityModel):
    pass


class Lognormal(SeverityModel):
    pass


class MixSev(SeverityModel):
    pass


class Pareto(SeverityModel):
    pass


class TriangularSev(SeverityModel):
    pass


class UniformSev(SeverityModel):
    pass


class Weibull(SeverityModel):
    pass
