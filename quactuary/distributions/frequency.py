"""
Frequency distributions for the number of claims per year.
This module provides a set of frequency distributions that can be used to model the
number of claims in a given period. The distributions are designed to be used with the
quActuary library, which is a Python library for actuarial science and insurance modeling.

Notes:
-------
gemact provides support for (a,b,0) and (a,b,1) distributions not in SciPy.
"""

from abc import abstractmethod
from typing import Protocol, runtime_checkable

import numpy as np
from scipy.stats import binom, geom, hypergeom, nbinom
from scipy.stats import poisson as sp_poisson
from scipy.stats import randint, triang
from scipy.stats._distn_infrastructure import rv_frozen


@runtime_checkable
class FrequencyModel(Protocol):
    """
    Frequency distribution protocol for the number of claims per year.
    This class provides an interface for frequency distributions that can be used
    to model the number of claims in a given period.
    """
    @abstractmethod
    def pmf(self, k: int) -> float:
        """Return the probability mass function (pmf) at k."""
        raise NotImplementedError

    def cdf(self, k: int) -> float:
        """Return the cumulative distribution function (cdf) at k."""
        # Default implementation for cumulative distribution function
        return sum(self.pmf(i) for i in range(k + 1))

    @abstractmethod
    def rvs(self, size: int = 1) -> np.ndarray:
        """Return random variates from the distribution."""
        raise NotImplementedError


# Adapter for any scipy.stats frozen distribution
class _ScipyFreqAdapter(FrequencyModel):
    """
    Adapter for any scipy.stats frozen distribution.
    This class allows the use of any frozen distribution from scipy.stats
    """

    def __init__(self, dist: rv_frozen):
        self._dist = dist

    def pmf(self, k: int) -> float:
        return float(self._dist.pmf(k))

    def cdf(self, k: int) -> float:
        return float(self._dist.cdf(k))

    def rvs(self, size: int = 1) -> np.ndarray:
        return self._dist.rvs(size=size)


# Convenience function to convert various inputs into a FrequencyModel
def to_frequency_model(obj) -> FrequencyModel:
    """
    Normalize an input into a FrequencyModel.
    - int or np.integer -> DeterministicFreq
    - scipy.stats frozen distribution -> Adapter
    - Already a FrequencyModel -> returned as-is
    """
    if isinstance(obj, FrequencyModel):
        return obj
    if isinstance(obj, (int, np.integer)):
        return DeterministicFreq(int(obj))
    if isinstance(obj, (list, np.array)):
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
    Binomial distribution with parameters n and p.
    """

    def __init__(self, n: int, p: float):
        """Binomial(n, p) with pmf(k) = C(n,k)*p^k*(1-p)^(n-k)."""
        self._dist = binom(n, p)

    def pmf(self, k: int) -> float:
        return float(self._dist.pmf(k))

    def cdf(self, k: int) -> float:
        return float(self._dist.cdf(k))

    def rvs(self, size: int = 1) -> np.ndarray:
        return self._dist.rvs(size=size)


class DeterministicFreq(FrequencyModel):
    """
    Deterministic frequency distribution. (single value)
    """

    def __init__(self, value: int):
        """Always returns 'value'."""
        self.value = value

    def pmf(self, k: int) -> float:
        return 1.0 if k == self.value else 0.0

    def cdf(self, k: int) -> float:
        return 1.0 if k >= self.value else 0.0

    def rvs(self, size: int = 1) -> np.ndarray:
        return np.full(shape=size, fill_value=self.value, dtype=int)


class DiscreteUniformFreq(FrequencyModel):
    """
    Discrete uniform distribution with parameters loc and scale.
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
    Empirical frequency distribution given by a dict of k -> probability.
    This class allows the use of a dictionary to define the probability mass function (pmf)
    for a discrete distribution.
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
    Geometric distribution with parameter p.
    """

    def __init__(self, p: float):
        """Geometric(p). pmf(k) = (1-p)^(k-1)*p for k=1,2,3,... in SciPy's parameterization."""
        self._dist = geom(p)

    def pmf(self, k: int) -> float:
        return float(self._dist.pmf(k))

    def cdf(self, k: int) -> float:
        return float(self._dist.cdf(k))

    def rvs(self, size: int = 1) -> np.ndarray:
        return self._dist.rvs(size=size)


class Hypergeometric(FrequencyModel):
    """
    Hypergeometric distribution with parameters M, n, N.
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
    Mixture of multiple frequency models.
    This class allows the combination of multiple frequency models with specified weights.
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
    Negative binomial distribution with parameters r and p.
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
    Poisson distribution with parameter mu.
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
    Triangular distribution with parameters c, loc, and scale.
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
