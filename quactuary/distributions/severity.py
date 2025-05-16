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
from scipy.integrate import quad
from scipy.stats import beta as sp_beta
from scipy.stats import chi2, expon
from scipy.stats import gamma as sp_gamma
from scipy.stats import (invgamma, invgauss, invweibull, lognorm, pareto, t,
                         triang, uniform, weibull_min)
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



class DiscretizedSeverity():
    """
    Discretized severity distributions. Needed for Quantum Circuits.

    Defines methods for probability mass, cumulative distribution, and sampling.

    We compute the density in each bin directly via:
    ∫ x f(x) dx

    Variables:
    sev_dist (SeverityModel): underlying severity distribution.
    step: (float): distance between bin midpoints.
    mid_x_vals: (np.ndarray): midpoint of each bin, serving as the key for `_probs`.
    bin_mean: (np.ndarray): mean value of each bin.
    _probs: (dict): dictionary of probabilities for each bin midpoint.

    Examples:
        >>> Χ²_dist = ChiSquared(df=4, scale=500.0)
        >>> discretized_Χ² = DiscretizedSeverityModel(Χ²_dist, min_val=0, max_val=8000, bins=100)
        >>> cdf_values = discretized_Χ².cdf(2000.0)
        0.5
    """

    def __init__(self, sev_dist, min_val: float, max_val: float, bins: int = 1000):
        """
        Initialize the discretized severity model, converting it into bins.
        This is required for quantum circuit modeling.

        Args:
            sev_dist (object): Underlying severity distribution.
            min_val (float): Minimum value for discretization.
            max_val (float): Maximum value for discretization.
            bins (int): Number of bins for discretization.
        """
        self.sev_dist = to_severity_model(sev_dist)

        # 1) Compute the bins and midpoints
        # build N equal‐width bins on [domain_min, domain_max]
        bin_edges = np.linspace(min_val, max_val, bins + 1)
        self.step = (max_val - min_val) / bins
        self.mid_x_vals = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # 2) Compute the bin probabilities using the analytic function
        pmf, self.bin_mean = _get_binned_moments(bin_edges, self.sev_dist)

        pmf /= np.sum(pmf)  # Normalize to sum to 1

        self._probs = {x: pmf[i] for i, x in enumerate(self.mid_x_vals)}

    def pmf(self, x: float) -> float:
        """
        Compute the probability mass function (PMF) at a given count.

        Args:
            k (int): Number of claims.

        Returns:
            float: Probability of exactly k claims.
        """
        pmf = self._probs[x] if x in self._probs.keys() else 0.0
        return pmf

    def cdf(self, x: float) -> float:
        """
        Compute the cumulative distribution function (CDF) at a given count.

        Args:
            k (int): Number of claims.

        Returns:
            float: Probability of at most k claims.
        """
        return sum(self._probs[i] for i in self._probs.keys() if i <= x)

    def rvs(self, size: int = 1) -> np.ndarray:
        """
        Draw random samples from the frequency distribution.

        Args:
            size (int, optional): Number of samples to generate. Defaults to 1.

        Returns:
            np.ndarray: Array of sampled claim counts.
        """
        return np.random.choice(self._probs.keys(), p=self._probs.values(), size=size)


def _get_binned_moments(bin_edges, sev_dist: SeverityModel):
    """
    This helper function computes for each bin defined by bin_edges:
    - pmf: Probability mass in the bin
    - bin_mean: First moment in the bin

    Args:
        bin_edges (array-like): The edges of the bins.
        sev_dist (SeverityModel): The underlying severity distribution.

    Returns:
        pmf (ndarray): Probability mass in each bin.
        bin_mean (ndarray): Exact first moment (∫ x f(x) dx) over each bin.
    """
    edges = np.asarray(bin_edges)
    a = edges[:-1]
    b = edges[1:]

    pmf = np.zeros_like(a)
    bin_mean = np.zeros_like(a)
    epsilon = 1e-12  # Small shift for discrete distributions

    for i in range(len(a)):
        pmf[i] = sev_dist.cdf(b[i]) - sev_dist.cdf(a[i] - epsilon)
        # Numerically integrate x * pdf(x) over [a[i], b[i]]
        bin_mean[i], _ = quad(lambda x: x * sev_dist.pdf(x), a[i], b[i])

    return pmf, bin_mean


"""
 ██████  ██████  ███    ███ ███    ███  ██████  ███    ██                                      
██      ██    ██ ████  ████ ████  ████ ██    ██ ████   ██                                      
██      ██    ██ ██ ████ ██ ██ ████ ██ ██    ██ ██ ██  ██                                      
██      ██    ██ ██  ██  ██ ██  ██  ██ ██    ██ ██  ██ ██                                      
 ██████  ██████  ██      ██ ██      ██  ██████  ██   ████                                      

██████  ██ ███████ ████████ ██████  ██ ██████  ██    ██ ████████ ██  ██████  ███    ██ ███████ 
██   ██ ██ ██         ██    ██   ██ ██ ██   ██ ██    ██    ██    ██ ██    ██ ████   ██ ██      
██   ██ ██ ███████    ██    ██████  ██ ██████  ██    ██    ██    ██ ██    ██ ██ ██  ██ ███████ 
██   ██ ██      ██    ██    ██   ██ ██ ██   ██ ██    ██    ██    ██ ██    ██ ██  ██ ██      ██ 
██████  ██ ███████    ██    ██   ██ ██ ██████   ██████     ██    ██  ██████  ██   ████ ███████ 
"""


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


class InverseGamma(SeverityModel):
    """
    Inverse gamma distribution modeling claim losses.

    Args:
        shape (float): Shape parameter (α).
        loc (float, optional): Location. Defaults to 0.0.
        scale (float, optional): Scale (β). Defaults to 1.0.

    Examples:
        >>> InverseGamma(a=2.0, scale=500.0).pdf(200.0)
    """

    def __init__(self, shape: float, loc: float = 0.0, scale: float = 1.0):
        self._dist = invgamma(shape, loc=loc, scale=scale)

    def pdf(self, x: float) -> float:
        return float(self._dist.pdf(x))

    def cdf(self, x: float) -> float:
        return float(self._dist.cdf(x))

    def rvs(self, size: int = 1) -> np.ndarray:
        return self._dist.rvs(size=size)


class InverseGaussian(SeverityModel):
    """
    Inverse Gaussian distribution for modeling claim losses.

    Args:
        mu (float): Mean of the distribution.
        lam (float): Shape parameter.

    Examples:
        >>> InverseGaussian(mu=500.0, lam=2).pmf(300.0)
    """

    def __init__(self, shape: float, loc: float, scale: float):
        """Inverse Gaussian(mu, lam)."""
        self._dist = invgauss(shape, loc=loc, scale=scale)

    def pdf(self, x: float) -> float:
        return float(self._dist.pdf(x))

    def cdf(self, x: float) -> float:
        return float(self._dist.cdf(x))

    def rvs(self, size: int = 1) -> np.ndarray:
        return self._dist.rvs(size=size)


class InverseWeibull(SeverityModel):
    """
    Inverse Weibull distribution modeling claim losses.

    Args:
        shape (float): Shape parameter.
        loc (float, optional): Location. Defaults to 0.0.
        scale (float, optional): Scale. Defaults to 1.0.

    Examples:
        >>> InverseWeibull(c=1.5, scale=200.0).pdf(150.0)
    """

    def __init__(self, shape: float, loc: float = 0.0, scale: float = 1.0):
        self._dist = invweibull(shape, loc=loc, scale=scale)

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


class StudentsT(SeverityModel):
    """
    Student's t-distribution for modeling heavy-tailed losses.

    Args:
        df (float): Degrees of freedom.
        loc (float, optional): Location. Defaults to 0.0.
        scale (float, optional): Scale. Defaults to 1.0.

    Examples:
        >>> StudentsT(df=5, scale=1000.0).pdf(200.0)
    """

    def __init__(self, df: float, loc: float = 0.0, scale: float = 1.0):
        self._dist = t(df, loc=loc, scale=scale)

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
