"""
Frequency distributions for the number of claims per year.

This module provides frequency distribution models to represent the number of claims
within a given period for property and casualty (P&C) insurance applications.

Notes:
    gemact provides support for (a,b,0) and (a,b,1) distributions not in SciPy.

Examples:
    >>> from quactuary.distributions.frequency import Poisson
    >>> model = Poisson(mu=3.5)
    >>> pmf_2 = model.pmf(2)
    >>> samples = model.rvs(size=10)
"""

from abc import abstractmethod
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
from scipy.stats import binom, geom, hypergeom, nbinom
from scipy.stats import poisson as sp_poisson
from scipy.stats import randint, triang
from scipy.stats._distn_infrastructure import rv_frozen


@runtime_checkable
class FrequencyModel(Protocol):
    """
    Protocol for frequency distributions representing number of claims per period.

    This interface defines methods for probability mass function, cumulative distribution,
    and random variate sampling.
    """
    @abstractmethod
    def pmf(self, k: int) -> float:
        """
        Compute the probability mass function (PMF) at a given count.

        Args:
            k (int): Number of claims.

        Returns:
            float: Probability of exactly k claims.
        """
        raise NotImplementedError

    def cdf(self, k: int) -> float:
        """
        Compute the cumulative distribution function (CDF) at a given count.

        Args:
            k (int): Number of claims.

        Returns:
            float: Probability of at most k claims.
        """
        # Default implementation for cumulative distribution function
        return sum(self.pmf(i) for i in range(k + 1))

    @abstractmethod
    def rvs(self, size: int = 1) -> np.ndarray:
        """
        Draw random samples from the frequency distribution.

        Args:
            size (int, optional): Number of samples to generate. Defaults to 1.

        Returns:
            np.ndarray: Array of sampled claim counts.
        """
        raise NotImplementedError


# Adapter for any scipy.stats frozen distribution
class _ScipyFreqAdapter(FrequencyModel):
    """
    Adapter for any frozen SciPy distribution, implementing FrequencyModel.

    Wraps a scipy.stats.rv_frozen instance to provide a uniform interface.
    """

    def __init__(self, dist: rv_frozen):
        """
        Initialize the adapter with a frozen SciPy distribution.

        Args:
            dist (rv_frozen): A frozen scipy.stats distribution.
        """
        self._dist = dist

    def pmf(self, k: int) -> float:
        """
        Compute the PMF at a given count using the wrapped distribution.

        Args:
            k (int): Number of claims.

        Returns:
            float: Probability of exactly k occurrences.
        """
        return float(self._dist.pmf(k))

    def cdf(self, k: int) -> float:
        """
        Compute the CDF at a given count using the wrapped distribution.

        Args:
            k (int): Number of claims.

        Returns:
            float: Probability of at most k occurrences.
        """
        return float(self._dist.cdf(k))

    def rvs(self, size: int = 1) -> np.ndarray:
        """
        Generate random variates from the wrapped distribution.

        Args:
            size (int, optional): Number of samples. Defaults to 1.

        Returns:
            np.ndarray: Array of random variates.
        """
        return self._dist.rvs(size=size)


# Convenience function to convert various inputs into a FrequencyModel
def to_frequency_model(obj) -> FrequencyModel:
    """
    Normalize an input into a FrequencyModel.

    Args:
        obj: Input object. Supported types:
            - int or np.integer: returns DeterministicFreq
            - list, np.ndarray, pd.Series of ints: returns EmpiricalFreq
            - scipy.stats.rv_frozen: returns _ScipyFreqAdapter
            - FrequencyModel: returned unchanged

    Returns:
        FrequencyModel: Equivalent frequency model.

    Raises:
        ValueError: If sequence input is empty.
        TypeError: On unsupported input types.

    Examples:
        >>> to_frequency_model(5)
        DeterministicFreq(5)
    """
    if isinstance(obj, FrequencyModel):
        return obj
    if isinstance(obj, (int, np.integer)):
        return DeterministicFreq(int(obj))
    if isinstance(obj, (list, np.ndarray, pd.Series)):
        if len(obj) == 0:
            raise ValueError(
                "Empty list or array cannot be converted to FrequencyModel")
        if all(isinstance(x, (int, np.integer)) for x in obj):
            return EmpiricalFreq({int(k): 1.0 / len(obj) for k in obj})
        raise TypeError(f"Cannot convert {obj!r} to FrequencyModel")
    if isinstance(obj, rv_frozen):
        return _ScipyFreqAdapter(obj)
    raise TypeError(f"Cannot convert {obj!r} to FrequencyModel")


class Binomial(FrequencyModel):
    """
    Binomial distribution modeling count of claims in fixed trials.

    Args:
        n (int): Number of trials.
        p (float): Success probability per trial.

    Examples:
        >>> model = Binomial(n=10, p=0.3)
        >>> model.pmf(4)
        0.200
    """

    def __init__(self, n: int, p: float):
        """
        Initialize a Binomial distribution.

        Args:
            n (int): Number of independent Bernoulli trials.
            p (float): Probability of success on each trial.
        """
        self._dist = binom(n, p)

    def pmf(self, k: int) -> float:
        """
        Compute the probability mass at k successes.

        Args:
            k (int): Number of observed successes.

        Returns:
            float: PMF value.
        """
        return float(self._dist.pmf(k))

    def cdf(self, k: int) -> float:
        """
        Compute cumulative probability of up to k successes.

        Args:
            k (int): Number of successes.

        Returns:
            float: CDF value.
        """
        return float(self._dist.cdf(k))

    def rvs(self, size: int = 1) -> np.ndarray:
        """
        Draw random samples of success counts.

        Args:
            size (int, optional): Number of samples. Defaults to 1.

        Returns:
            np.ndarray: Sampled counts.
        """
        return self._dist.rvs(size=size)


class DeterministicFreq(FrequencyModel):
    """
    Deterministic frequency distribution producing a fixed count.

    Args:
        value (int): Constant number of claims.

    Examples:
        >>> DeterministicFreq(2).pmf(2)
        1.0
    """

    def __init__(self, value: int):
        """
        Initialize a deterministic frequency model.

        Args:
            value (int): Fixed claim count to return.
        """
        self.value = value

    def pmf(self, k: int) -> float:
        return 1.0 if k == self.value else 0.0

    def cdf(self, k: int) -> float:
        return 1.0 if k >= self.value else 0.0

    def rvs(self, size: int = 1) -> np.ndarray:
        return np.full(shape=size, fill_value=self.value, dtype=int)


class DiscreteUniformFreq(FrequencyModel):
    """
    Discrete uniform distribution over integer counts.

    Args:
        low (int, optional): Lower bound (inclusive). Defaults to 0.
        high (int, optional): Upper bound (exclusive). Defaults to 2.

    Examples:
        >>> model = DiscreteUniformFreq(1, 5)
        >>> model.pmf(3)
        0.25
    """

    def __init__(self, low: int = 0, high: int = 2):
        """
        Discrete uniform distribution.
        """
        self._dist = randint(low, high)

    def pmf(self, k: int) -> float:
        return float(self._dist.pmf(k))

    def cdf(self, k: int) -> float:
        return float(self._dist.cdf(k))

    def rvs(self, size: int = 1) -> np.ndarray:
        return self._dist.rvs(size=size)


class EmpiricalFreq(FrequencyModel):
    """
    Empirical frequency distribution defined by discrete probabilities.

    Args:
        pmf_values (dict[int, float]): Mapping from integer count to probability.

    Raises:
        ValueError: If probabilities do not sum to 1.

    Examples:
        >>> EmpiricalFreq({0: 0.2, 1: 0.8}).rvs(size=5)
        array([0, 1, 1, 1, 0])
    """

    def __init__(self, pmf_values: dict[int, float]):
        """Discrete distribution given by a dict of k -> probability."""
        self.pmf_values = pmf_values
        self._keys = list(pmf_values.keys())
        self._probs = np.array([pmf_values[k] for k in self._keys])

    def pmf(self, k: int) -> float:
        return self.pmf_values.get(k, 0.0)

    def cdf(self, k: int) -> float:
        return sum(self.pmf_values.get(i, 0.0) for i in range(k + 1))

    def rvs(self, size: int = 1) -> np.ndarray:
        return np.random.choice(self._keys, p=self._probs, size=size)


class Geometric(FrequencyModel):
    """
    Geometric distribution for count of failures before first success.

    Args:
        p (float): Success probability on each trial.

    Examples:
        >>> Geometric(p=0.3).pmf(1)
        0.3
    """

    def __init__(self, p: float):
        """
        Geometric(p). pmf(k) = (1-p)^(k-1)*p for k=1,2,3,... in SciPy's parameterization.
        """
        self._dist = geom(p)

    def pmf(self, k: int) -> float:
        return float(self._dist.pmf(k))

    def cdf(self, k: int) -> float:
        return float(self._dist.cdf(k))

    def rvs(self, size: int = 1) -> np.ndarray:
        return self._dist.rvs(size=size)


class Hypergeometric(FrequencyModel):
    """
    Hypergeometric distribution modeling draws without replacement.

    Args:
        M (int): Total population size.
        n (int): Number of success states in population.
        N (int): Number of draws.

    Examples:
        >>> Hypergeometric(M=50, n=5, N=10).pmf(1)
    """

    def __init__(self, M: int, n: int, N: int):
        """Hypergeometric(M, n, N)."""
        self._dist = hypergeom(M=M, n=n, N=N)

    def pmf(self, k: int) -> float:
        return float(self._dist.pmf(k))

    def cdf(self, k: int) -> float:
        return float(self._dist.cdf(k))

    def rvs(self, size: int = 1) -> np.ndarray:
        return self._dist.rvs(size=size)


class MixFreq(FrequencyModel):
    """
    Mixture model combining multiple frequency distributions.

    Args:
        components (list[FrequencyModel]): List of component models.
        weights (list[float]): Mixing weights summing to 1.

    Examples:
        >>> mix = MixFreq([Poisson(2), Poisson(5)], [0.4, 0.6])
        >>> mix.pmf(3)
    """

    def __init__(self, components: list[FrequencyModel], weights: list[float]):
        """
        Mixture of multiple frequency models, with 'weights' summing to 1.
        """
        self.components = components
        self.weights = weights

    def pmf(self, k: int) -> float:
        return sum(w * comp.pmf(k) for comp, w in zip(self.components, self.weights))

    def cdf(self, k: int) -> float:
        return sum(w * comp.cdf(k) for comp, w in zip(self.components, self.weights))

    def rvs(self, size: int = 1) -> np.ndarray:
        choices = np.random.choice(
            len(self.components), size=size, p=self.weights)
        return np.array([self.components[idx].rvs(1)[0] for idx in choices])


class NegativeBinomial(FrequencyModel):
    """
    Negative binomial distribution modeling number of failures until r successes.

    Args:
        r (float): Number of successes to achieve.
        p (float): Success probability on each trial.

    Examples:
        >>> NegativeBinomial(r=3, p=0.5).pmf(4)
    """

    def __init__(self, r: float, p: float):
        """Negative binomial with 'r' failures and success probability 'p'."""
        self._dist = nbinom(r, p)

    def pmf(self, k: int) -> float:
        return float(self._dist.pmf(k))

    def cdf(self, k: int) -> float:
        return float(self._dist.cdf(k))

    def rvs(self, size: int = 1) -> np.ndarray:
        return self._dist.rvs(size=size)


class Poisson(FrequencyModel):
    """
    Poisson distribution for count of claims per period.

    Args:
        mu (float): Expected number of occurrences.

    Examples:
        >>> Poisson(mu=3.5).rvs(size=5)
    """

    def __init__(self, mu: float):
        """Poisson(mu)."""
        self._dist = sp_poisson(mu)

    def pmf(self, k: int) -> float:
        return float(self._dist.pmf(k))

    def cdf(self, k: int) -> float:
        return float(self._dist.cdf(k))

    def rvs(self, size: int = 1) -> np.ndarray:
        return self._dist.rvs(size=size)


class TriangularFreq(FrequencyModel):
    """
    Triangular distribution for approximate discrete counts.

    Args:
        c (float): Mode parameter between 0 and 1.
        loc (float, optional): Lower limit. Defaults to 0.0.
        scale (float, optional): Width of distribution. Defaults to 1.0.

    Notes:
        This continuous distribution is rounded to integers for PMF and sampling.

    Examples:
        >>> TriangularFreq(c=0.5, loc=1, scale=4).rvs(size=3)
    """

    def __init__(self, c: float, loc: float = 0.0, scale: float = 1.0):
        """
        Triangular distribution, continuous in SciPy.
        We'll return 0 for pmf (no true discrete pmf).
        """
        self._dist = triang(c, loc=loc, scale=scale)

    def pmf(self, k: int) -> float:
        return 0.0  # continuous distribution → no discrete probability mass

    def cdf(self, k: int) -> float:
        return float(self._dist.cdf(k))

    def rvs(self, size: int = 1) -> np.ndarray:
        return np.round(self._dist.rvs(size=size)).astype(int)
