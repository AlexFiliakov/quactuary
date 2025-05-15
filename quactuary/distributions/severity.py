"""
Severity distributions for the amount of loss per claim.

This module provides severity distribution models to represent claim losses in property
and casualty (P&C) insurance applications.

Examples:
    >>> from quactuary.distributions.severity import Exponential
    >>> model = Exponential(scale=1000.0)
    >>> pdf_500 = model.pdf(500.0)
    >>> samples = model.rvs(size=5)
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
    Protocol for severity distributions representing loss amounts per claim.

    Defines methods for probability density, cumulative distribution, and sampling.
    """
    @abstractmethod
    def pdf(self, x: float) -> float:
        """
        Compute probability density at a given loss amount.

        Args:
            x (float): Loss amount.

        Returns:
            float: Probability density at x.
        """
        raise NotImplementedError

    @abstractmethod
    def cdf(self, x: float) -> float:
        """
        Compute cumulative probability up to a given loss amount.

        Args:
            x (float): Loss amount.

        Returns:
            float: Cumulative probability at x.
        """
        raise NotImplementedError

    @abstractmethod
    def rvs(self, size: int = 1) -> np.ndarray:
        """
        Draw random samples of claim losses.

        Args:
            size (int, optional): Number of samples. Defaults to 1.

        Returns:
            np.ndarray: Sampled loss values.
        """
        raise NotImplementedError


class _ScipySevAdapter(SeverityModel):
    """
    Adapter for a frozen SciPy distribution implementing SeverityModel.

    Wraps scipy.stats.rv_frozen to conform to the SeverityModel interface.
    """

    def __init__(self, dist: rv_frozen):
        """
        Initialize adapter with a frozen SciPy distribution.

        Args:
            dist (rv_frozen): Frozen scipy.stats distribution.
        """
        self._dist = dist

    def pdf(self, x: float) -> float:
        """
        Compute PDF using the wrapped distribution.

        Args:
            x (float): Loss amount.

        Returns:
            float: Probability density at x.
        """
        return float(self._dist.pdf(x))

    def cdf(self, x: float) -> float:
        """
        Compute CDF using the wrapped distribution.

        Args:
            x (float): Loss amount.

        Returns:
            float: Cumulative probability at x.
        """
        return float(self._dist.cdf(x))

    def rvs(self, size: int = 1) -> np.ndarray:
        """
        Generate random variates using the wrapped distribution.

        Args:
            size (int, optional): Number of samples. Defaults to 1.

        Returns:
            np.ndarray: Sampled loss values.
        """
        return self._dist.rvs(size=size)


def to_severity_model(obj) -> SeverityModel:
    """
    Normalize diverse inputs into a SeverityModel.

    Args:
        obj: Input to convert. Supported types:
            - float or int: returns ConstantSev
            - list, np.ndarray, pd.Series of numeric: returns EmpiricalSev
            - scipy.stats.rv_frozen: returns _ScipySevAdapter
            - SeverityModel: returned unchanged

    Returns:
        SeverityModel: Corresponding severity model.

    Raises:
        ValueError: If sequence input is empty.
        TypeError: On unsupported input types.
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
    Beta distribution modeling loss proportions.

    Args:
        a (float): Alpha (shape) parameter.
        b (float): Beta (shape) parameter.
        loc (float, optional): Location parameter. Defaults to 0.0.
        scale (float, optional): Scale parameter. Defaults to 1.0.

    Examples:
        >>> Beta(a=2, b=5, scale=1000.0).pdf(250.0)
    """

    def __init__(self, a: float, b: float, loc: float = 0.0, scale: float = 1.0):
        """
        Initialize a Beta severity distribution.

        Args:
            a (float): Alpha shape parameter.
            b (float): Beta shape parameter.
            loc (float): Lower bound of distribution.
            scale (float): Scale of distribution.
        """
        self._dist = sp_beta(a, b, loc=loc, scale=scale)

    def pdf(self, x: float) -> float:
        """
        Compute PDF at a given loss amount.

        Args:
            x (float): Loss amount.

        Returns:
            float: Density at x.
        """
        return float(self._dist.pdf(x))

    def cdf(self, x: float) -> float:
        """
        Compute CDF at a given loss amount.

        Args:
            x (float): Loss amount.

        Returns:
            float: Cumulative probability at x.
        """
        return float(self._dist.cdf(x))

    def rvs(self, size: int = 1) -> np.ndarray:
        """
        Generate random loss samples.

        Args:
            size (int, optional): Number of samples. Defaults to 1.

        Returns:
            np.ndarray: Sampled values.
        """
        return self._dist.rvs(size=size)


class ChiSquared(SeverityModel):
    """
    Chi-squared distribution for claim severity.

    Args:
        df (float): Degrees of freedom.
        loc (float, optional): Location parameter. Defaults to 0.0.
        scale (float, optional): Scale parameter. Defaults to 1.0.

    Examples:
        >>> ChiSquared(df=4, scale=500.0).cdf(200.0)
    """

    def __init__(self, df: float, loc: float = 0.0, scale: float = 1.0):
        """
        Initialize a Chi-squared severity distribution.

        Args:
            df (float): Degrees of freedom.
            loc (float): Location parameter.
            scale (float): Scale parameter.
        """
        self._dist = chi2(df, loc=loc, scale=scale)

    def pdf(self, x: float) -> float:
        return float(self._dist.pdf(x))

    def cdf(self, x: float) -> float:
        return float(self._dist.cdf(x))

    def rvs(self, size: int = 1) -> np.ndarray:
        return self._dist.rvs(size=size)


class ConstantSev(SeverityModel):
    """
    Constant severity distribution producing a fixed loss amount.

    Args:
        value (float): Fixed loss amount per claim.

    Examples:
        >>> ConstantSev(100.0).pdf(100.0)
        1.0
    """

    def __init__(self, value: float):
        """
        Initialize a constant severity model.

        Args:
            value (float): Fixed loss amount.
        """
        self.value = value

    def pdf(self, x: float) -> float:
        return 1.0 if x == self.value else 0.0

    def cdf(self, x: float) -> float:
        return 1.0 if x >= self.value else 0.0

    def rvs(self, size: int = 1) -> np.ndarray:
        return np.full(shape=size, fill_value=self.value)


class ContinuousUniformSev(SeverityModel):
    """
    Continuous uniform distribution for claim severity.

    Args:
        loc (float, optional): Lower bound of losses. Defaults to 0.0.
        scale (float, optional): Width of distribution. Defaults to 1.0.

    Examples:
        >>> ContinuousUniformSev(loc=100.0, scale=900.0).rvs(size=3)
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
    Empirical severity distribution defined by discrete support and probabilities.

    Args:
        values (list[float]): Observable loss amounts.
        probs (list[float]): Corresponding probabilities (must sum to 1).

    Raises:
        ValueError: If sum(probs) is zero.

    Examples:
        >>> EmpiricalSev([100, 500], [0.3, 0.7]).cdf(250)
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
    Exponential distribution modeling loss severities.

    Args:
        scale (float, optional): Scale (mean) of distribution. Defaults to 1.0.
        loc (float, optional): Location parameter. Defaults to 0.0.

    Examples:
        >>> Exponential(scale=1000.0).pdf(200.0)
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
    Gamma distribution modeling claim losses.

    Args:
        shape (float): Shape parameter (k).
        loc (float, optional): Location. Defaults to 0.0.
        scale (float, optional): Scale (θ). Defaults to 1.0.

    Examples:
        >>> Gamma(shape=2.0, scale=500.0).rvs(size=4)
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
    Lognormal distribution modeling multiplicative loss processes.

    Args:
        s (float): Shape parameter (sigma of underlying normal).
        loc (float, optional): Location (shift). Defaults to 0.0.
        scale (float, optional): Scale (exp(mu)). Defaults to 1.0.

    Examples:
        >>> Lognormal(s=0.5, scale=200.0).cdf(150.0)
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
    Mixture model combining multiple severity distributions.

    Args:
        components (list[SeverityModel]): List of severity models.
        weights (list[float]): Mixing weights summing to 1.

    Examples:
        >>> mix = MixSev([Exponential(1000), Gamma(2)], [0.6, 0.4])
        >>> mix.pdf(500.0)
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
    Pareto distribution modeling heavy-tailed losses.

    Args:
        b (float): Shape parameter (tail index).
        loc (float, optional): Location. Defaults to 0.0.
        scale (float, optional): Scale (xmin). Defaults to 1.0.

    Examples:
        >>> Pareto(b=3.0, scale=100.0).rvs(size=5)
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
    Triangular distribution for modeling bounded losses.

    Args:
        c (float): Mode parameter between 0 and 1.
        loc (float, optional): Lower limit. Defaults to 0.0.
        scale (float, optional): Width. Defaults to 1.0.

    Examples:
        >>> TriangularSev(c=0.5, loc=100, scale=900).cdf(500)
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
    Weibull distribution modeling failure-time-like loss processes.

    Args:
        c (float): Shape parameter.
        loc (float, optional): Location. Defaults to 0.0.
        scale (float, optional): Scale. Defaults to 1.0.

    Examples:
        >>> Weibull(c=1.5, scale=200.0).pdf(150.0)
    """

    def __init__(self, c: float, loc: float = 0.0, scale: float = 1.0):
        self._dist = weibull_min(c, loc=loc, scale=scale)

    def pdf(self, x: float) -> float:
        return float(self._dist.pdf(x))

    def cdf(self, x: float) -> float:
        return float(self._dist.cdf(x))

    def rvs(self, size: int = 1) -> np.ndarray:
        return self._dist.rvs(size=size)
