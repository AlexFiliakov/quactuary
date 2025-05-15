"""
Severity distributions for the amount of loss per claim.
This module provides a set of severity distributions that can be used to model the
severity of claims in a given period. The distributions are designed to be used with the
quActuary library, which is a Python library for actuarial science and insurance modeling.
"""

from abc import abstractmethod
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
from scipy.stats import beta as sp_beta
from scipy.stats import chi2, expon
from scipy.stats import gamma as sp_gamma
from scipy.stats import lognorm, pareto, triang, uniform, weibull_min
from scipy.stats._distn_infrastructure import rv_frozen


@runtime_checkable
class SeverityModel(Protocol):
    """
    Severity distribution protocol for the number of claims per year.
    This class provides an interface for severity distributions that can be used
    to model the loss per claim in a given period.
    """
    @abstractmethod
    def pdf(self, x: float) -> float:
        """Return the probability density function (pdf) at x."""
        raise NotImplementedError

    @abstractmethod
    def cdf(self, x: float) -> float:
        """Return the cumulative distribution function (cdf) at x."""
        raise NotImplementedError

    @abstractmethod
    def rvs(self, size: int = 1) -> np.ndarray:
        """Return random variates from the distribution."""
        raise NotImplementedError


class _ScipySevAdapter(SeverityModel):
    """
    Adapter for any scipy.stats frozen distribution.
    This class allows the use of any frozen distribution from scipy.stats
    """

    def __init__(self, dist: rv_frozen):
        self._dist = dist

    def pdf(self, x: float) -> float:
        return float(self._dist.pdf(x))

    def cdf(self, x: float) -> float:
        return float(self._dist.cdf(x))

    def rvs(self, size: int = 1) -> np.ndarray:
        return self._dist.rvs(size=size)


def to_severity_model(obj) -> SeverityModel:
    """
    Normalize an input into a SeverityModel.
    - float or int -> ConstantSev
    - list or np.ndarray of floats -> EmpiricalSev with uniform probs
    - scipy.stats frozen distribution -> Adapter
    - Already a SeverityModel -> returned as-is
    """
    if isinstance(obj, SeverityModel):
        return obj
    if isinstance(obj, (int, float, np.integer, np.floating)):
        return ConstantSev(float(obj))
    if isinstance(obj, (list, np.ndarray, pd.Series)):
        if len(obj) == 0:
            raise ValueError(
                "Empty list or array cannot be converted to SeverityModel")
        if all(isinstance(x, (int, float, np.integer, np.floating)) for x in obj):
            values = [float(x) for x in obj]
            probs = [1.0 / len(values)] * len(values)
            return EmpiricalSev(values, probs)
        raise TypeError(f"Cannot convert {obj!r} to SeverityModel")
    if isinstance(obj, rv_frozen):
        return _ScipySevAdapter(obj)
    raise TypeError(f"Cannot convert {obj!r} to SeverityModel")


class Beta(SeverityModel):
    """
    Beta distribution for the amount of loss per claim.
    """

    def __init__(self, a: float, b: float, loc: float = 0.0, scale: float = 1.0):
        self._dist = sp_beta(a, b, loc=loc, scale=scale)

    def pdf(self, x: float) -> float:
        return float(self._dist.pdf(x))

    def cdf(self, x: float) -> float:
        return float(self._dist.cdf(x))

    def rvs(self, size: int = 1) -> np.ndarray:
        return self._dist.rvs(size=size)


class ChiSquared(SeverityModel):
    """
    Chi-squared distribution for the amount of loss per claim.
    """

    def __init__(self, df: float, loc: float = 0.0, scale: float = 1.0):
        self._dist = chi2(df, loc=loc, scale=scale)

    def pdf(self, x: float) -> float:
        return float(self._dist.pdf(x))

    def cdf(self, x: float) -> float:
        return float(self._dist.cdf(x))

    def rvs(self, size: int = 1) -> np.ndarray:
        return self._dist.rvs(size=size)


class ConstantSev(SeverityModel):
    """
    Constant severity distribution for the amount of loss per claim.
    This class represents a distribution where the loss is always a fixed value.
    """

    def __init__(self, value: float):
        self.value = value

    def pdf(self, x: float) -> float:
        return 1.0 if x == self.value else 0.0

    def cdf(self, x: float) -> float:
        return 1.0 if x >= self.value else 0.0

    def rvs(self, size: int = 1) -> np.ndarray:
        return np.full(shape=size, fill_value=self.value)


class ContinuousUniformSev(SeverityModel):
    """
    Continous uniform distribution for the amount of loss per claim.
    """

    def __init__(self, loc: float = 0.0, scale: float = 1.0):
        self._dist = uniform(loc=loc, scale=scale)

    def pdf(self, x: float) -> float:
        return float(self._dist.pdf(x))

    def cdf(self, x: float) -> float:
        return float(self._dist.cdf(x))

    def rvs(self, size: int = 1) -> np.ndarray:
        return self._dist.rvs(size=size)


class EmpiricalSev(SeverityModel):
    """
    Empirical severity distribution for the amount of loss per claim.
    This class represents a distribution where the loss is defined by a set of values
    and their corresponding probabilities.
    The probabilities must sum to 1.
    """

    def __init__(self, values: list[float], probs: list[float]):
        total = sum(probs)
        self.values = list(values)
        self.probs = np.array(probs) / total

    def pdf(self, x: float) -> float:
        return float(sum(self.probs[i] for i, v in enumerate(self.values) if v == x))

    def cdf(self, x: float) -> float:
        return float(sum(self.probs[i] for i, v in enumerate(self.values) if v <= x))

    def rvs(self, size: int = 1) -> np.ndarray:
        return np.random.choice(self.values, size=size, p=self.probs)


class Exponential(SeverityModel):
    """
    Exponential distribution for the amount of loss per claim.
    """

    def __init__(self, scale: float = 1.0, loc: float = 0.0):
        self._dist = expon(loc=loc, scale=scale)

    def pdf(self, x: float) -> float:
        return float(self._dist.pdf(x))

    def cdf(self, x: float) -> float:
        return float(self._dist.cdf(x))

    def rvs(self, size: int = 1) -> np.ndarray:
        return self._dist.rvs(size=size)


class Gamma(SeverityModel):
    """
    Gamma distribution for the amount of loss per claim.
    """

    def __init__(self, shape: float, loc: float = 0.0, scale: float = 1.0):
        self._dist = sp_gamma(shape, loc=loc, scale=scale)

    def pdf(self, x: float) -> float:
        return float(self._dist.pdf(x))

    def cdf(self, x: float) -> float:
        return float(self._dist.cdf(x))

    def rvs(self, size: int = 1) -> np.ndarray:
        return self._dist.rvs(size=size)


class Lognormal(SeverityModel):
    """
    Lognormal distribution for the amount of loss per claim.
    """

    def __init__(self, s: float, loc: float = 0.0, scale: float = 1.0):
        self._dist = lognorm(s, loc=loc, scale=scale)

    def pdf(self, x: float) -> float:
        return float(self._dist.pdf(x))

    def cdf(self, x: float) -> float:
        return float(self._dist.cdf(x))

    def rvs(self, size: int = 1) -> np.ndarray:
        return self._dist.rvs(size=size)


class MixSev(SeverityModel):
    """
    Mixture of severity distributions for the amount of loss per claim.
    """

    def __init__(self, components: list[SeverityModel], weights: list[float]):
        self.components = components
        self.weights = weights

    def pdf(self, x: float) -> float:
        return sum(w * comp.pdf(x) for comp, w in zip(self.components, self.weights))

    def cdf(self, x: float) -> float:
        return sum(w * comp.cdf(x) for comp, w in zip(self.components, self.weights))

    def rvs(self, size: int = 1) -> np.ndarray:
        choices = np.random.choice(
            len(self.components), size=size, p=self.weights)
        return np.array([self.components[i].rvs(1)[0] for i in choices])


class Pareto(SeverityModel):
    """
    Pareto distribution for the amount of loss per claim.
    """

    def __init__(self, b: float, loc: float = 0.0, scale: float = 1.0):
        self._dist = pareto(b, loc=loc, scale=scale)

    def pdf(self, x: float) -> float:
        return float(self._dist.pdf(x))

    def cdf(self, x: float) -> float:
        return float(self._dist.cdf(x))

    def rvs(self, size: int = 1) -> np.ndarray:
        return self._dist.rvs(size=size)


class TriangularSev(SeverityModel):
    """
    Triangular distribution for the amount of loss per claim.
    """

    def __init__(self, c: float, loc: float = 0.0, scale: float = 1.0):
        self._dist = triang(c, loc=loc, scale=scale)

    def pdf(self, x: float) -> float:
        return float(self._dist.pdf(x))

    def cdf(self, x: float) -> float:
        return float(self._dist.cdf(x))

    def rvs(self, size: int = 1) -> np.ndarray:
        return self._dist.rvs(size=size)


class Weibull(SeverityModel):
    """
    Weibull distribution for the amount of loss per claim.
    """

    def __init__(self, c: float, loc: float = 0.0, scale: float = 1.0):
        self._dist = weibull_min(c, loc=loc, scale=scale)

    def pdf(self, x: float) -> float:
        return float(self._dist.pdf(x))

    def cdf(self, x: float) -> float:
        return float(self._dist.cdf(x))

    def rvs(self, size: int = 1) -> np.ndarray:
        return self._dist.rvs(size=size)
