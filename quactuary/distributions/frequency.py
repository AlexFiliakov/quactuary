"""
Frequency distributions for the number of claims per year.
This module provides a set of frequency distributions that can be used to model the
number of claims in a given period. The distributions are designed to be used with the
quActuary library, which is a Python library for actuarial science and insurance modeling.
"""

from typing import Protocol


class FrequencyModel(Protocol):
    def pmf(self, k: int) -> float: ...          # needed for exact quantum load
    def rvs(self, size: int = 1) -> np.ndarray: ...  # classical sampling fallback

# TODO: Provide convenience wrappers so an actuary can pass in either:
# - a scalar (2 → “exactly 2 claims per year”),
# - a ready‑made scipy.stats distribution,
# - or a tiny helper like qa.freq.Poisson(mu=1.8) that just forwards to scipy.stats.poisson.
# Internally we normalize everything to an adapter object that satisfies the protocol.


class Binomial(FrequencyModel):
    pass


class FrequencyEmpirical(FrequencyModel):
    pass


class FrequencyMix(FrequencyModel):
    pass


class FrequencyTriangular(FrequencyModel):
    pass


class FrequencyUniform(FrequencyModel):
    pass


class Geometric(FrequencyModel):
    pass


class Hypergeometric(FrequencyModel):
    pass


class NegativeBinomial(FrequencyModel):
    pass


class Poisson(FrequencyModel):
    pass
