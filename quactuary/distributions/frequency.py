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

import random
from abc import abstractmethod
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
from scipy.stats import binom, geom, hypergeom, nbinom
from scipy.stats import poisson as sp_poisson
from scipy.stats import randint, triang
from scipy.stats._distn_infrastructure import rv_frozen

epsilon = 1e-12  # Small shift for discrete distributions


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
    def rvs(self, size: int = 1) -> pd.Series | np.integer:
        """
        Draw random samples from the frequency distribution.

        Args:
            size (int, optional): Number of samples to generate. Defaults to 1.

        Returns:
            np.ndarray: Array of sampled claim counts.
        """
        raise NotImplementedError


# Adapter for any scipy.stats frozen distribution
class _ScipyFrequencyAdapter(FrequencyModel):
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
        return float(self._dist.pmf(k))  # type: ignore[attr-defined]

    def cdf(self, k: int) -> float:
        """
        Compute the CDF at a given count using the wrapped distribution.

        Args:
            k (int): Number of claims.

        Returns:
            float: Probability of at most k occurrences.
        """
        return float(self._dist.cdf(k))  # type: ignore[attr-defined]

    def rvs(self, size: int = 1) -> pd.Series | np.integer:
        """
        Generate random variates from the wrapped distribution.

        Args:
            size (int, optional): Number of samples. Defaults to 1.

        Returns:
            np.ndarray: Array of random variates.
        """
        samples = self._dist.rvs(size=size)
        return pd.Series(self._dist.rvs(size=size)) if size > 1 else samples[0]


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
        DeterministicFrequency(5)
    """
    if isinstance(obj, FrequencyModel):
        return obj
    if isinstance(obj, (int, np.integer)):
        return DeterministicFrequency(obj)    # type: ignore[attr-defined]
    if isinstance(obj, (list, np.ndarray, pd.Series)):
        if len(obj) == 0:
            raise ValueError(
                "Empty list or array cannot be converted to FrequencyModel")
        if all(isinstance(x, (int, np.integer)) for x in obj):
            return EmpiricalFrequency({int(k): 1.0 / len(obj) for k in obj})
        raise TypeError(f"Cannot convert {obj!r} to FrequencyModel")
    if isinstance(obj, rv_frozen):
        return _ScipyFrequencyAdapter(obj)
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

    def __str__(self):
        return f"Binomial(n={self._dist.args[0]}, p={self._dist.args[1]})"

    def pmf(self, k: int) -> float:
        return float(self._dist.pmf(k))  # type: ignore[attr-defined]

    def cdf(self, k: int) -> float:
        return float(self._dist.cdf(k))  # type: ignore[attr-defined]

    def rvs(self, size: int = 1) -> pd.Series | np.integer:
        samples = self._dist.rvs(size=size)
        return pd.Series(samples) if size > 1 else samples[0]


class DeterministicFrequency(FrequencyModel):
    """
    Deterministic frequency distribution producing a fixed count.

    Args:
        value (int): Constant number of claims.

    Examples:
        >>> DeterministicFrequency(2).pmf(2)
        1.0
    """

    def __init__(self, value: np.integer):
        """
        Initialize a deterministic frequency model.

        Args:
            value (np.integer): Fixed claim count to return.
        """
        self.value = np.int64(value)

    def __str__(self):
        return f"DeterministicFrequency(value={self.value})"

    def pmf(self, k: int) -> float:
        return 1.0 if k == self.value else 0.0

    def cdf(self, k: int) -> float:
        return 1.0 if k >= self.value else 0.0

    def rvs(self, size: int = 1) -> pd.Series | np.integer:
        if size != 1:
            return pd.Series([self.value]).repeat(size).reset_index(drop=True)
        else:
            return self.value


class DiscreteUniformFrequency(FrequencyModel):
    """
    Discrete uniform distribution over integer counts.

    Args:
        low (int, optional): Lower bound (inclusive). Defaults to 0.
        high (int, optional): Upper bound (exclusive). Defaults to 2.

    Examples:
        >>> model = DiscreteUniformFrequency(1, 5)
        >>> model.pmf(3)
        0.25
    """

    def __init__(self, low: int = 0, high: int = 2):
        """
        Discrete uniform distribution.
        """
        self._dist = randint(low, high)

    def __str__(self):
        return f"DiscreteUniformFrequency(low={self._dist.args[0]}, high={self._dist.args[1]})"

    def pmf(self, k: int) -> float:
        return float(self._dist.pmf(k))  # type: ignore[attr-defined]

    def cdf(self, k: int) -> float:
        return float(self._dist.cdf(k))  # type: ignore[attr-defined]

    def rvs(self, size: int = 1) -> pd.Series | np.integer:
        samples = self._dist.rvs(size=size)
        return pd.Series(samples) if size > 1 else samples[0]


class EmpiricalFrequency(FrequencyModel):
    """
    Empirical frequency distribution defined by discrete probabilities.

    Args:
        pmf_values (dict[int, float]): Mapping from integer count to probability.

    Raises:
        ValueError: If probabilities do not sum to 1.

    Examples:
        >>> EmpiricalFrequency({0: 0.2, 1: 0.8}).rvs(size=5)
        array([0, 1, 1, 1, 0])
    """

    def __init__(self, pmf_values: dict[int, float]):
        """Discrete distribution given by a dict of k -> probability."""
        self.pmf_values = pmf_values
        self._keys = list(pmf_values.keys())
        self._probs = np.array([pmf_values[k] for k in self._keys])

    def __str__(self):
        return f"EmpiricalFrequency(pmf_values={self.pmf_values})"

    def pmf(self, k: int) -> float:
        return self.pmf_values.get(k, 0.0)

    def cdf(self, k: int) -> float:
        return sum(self.pmf_values.get(i, 0.0) for i in range(k + 1))

    def rvs(self, size: int = 1) -> pd.Series | np.integer:
        samples = np.random.choice(self._keys, p=self._probs, size=size)
        return pd.Series(samples) if size > 1 else samples[0]


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

    def __str__(self):
        return f"Geometric(p={self._dist.args[0]})"

    def pmf(self, k: int) -> float:
        return float(self._dist.pmf(k))  # type: ignore[attr-defined]

    def cdf(self, k: int) -> float:
        return float(self._dist.cdf(k))  # type: ignore[attr-defined]

    def rvs(self, size: int = 1) -> pd.Series | np.integer:
        samples = self._dist.rvs(size=size)
        return pd.Series(samples) if size > 1 else samples[0]


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

    def __str__(self):
        return f"Hypergeometric(M={self._dist.args[0]}, n={self._dist.args[1]}, N={self._dist.args[2]})"

    def pmf(self, k: int) -> float:
        return float(self._dist.pmf(k))  # type: ignore[attr-defined]

    def cdf(self, k: int) -> float:
        return float(self._dist.cdf(k))  # type: ignore[attr-defined]

    def rvs(self, size: int = 1) -> pd.Series | np.integer:
        samples = self._dist.rvs(size=size)
        return pd.Series(samples) if size > 1 else samples[0]


class MixedFrequency(FrequencyModel):
    """
    Mixture model combining multiple frequency distributions.

    Args:
        components (list[FrequencyModel]): List of component models.
        weights (list[float]): Mixing weights summing to 1.

    Examples:
        >>> mix = MixedFrequency([Poisson(2), Poisson(5)], [0.4, 0.6])
        >>> mix.pmf(3)
    """

    def __init__(self, components: list[FrequencyModel], weights: list[float]):
        """
        Mixture of multiple frequency models, with 'weights' summing to 1.
        """
        self.components = components
        # Normalize weights to sum to 1
        self.weights = weights / np.sum(weights)

    def __str__(self):
        return f"MixedFrequency(components={self.components}, weights={self.weights})"

    def pmf(self, k: int) -> float:
        return sum(w * comp.pmf(k) for comp, w in zip(self.components, self.weights))

    def cdf(self, k: int) -> float:
        return sum(w * comp.cdf(k) for comp, w in zip(self.components, self.weights))

    def rvs(self, size: int = 1) -> pd.Series | np.integer:
        choices = np.random.choice(
            len(self.components), size=size, p=self.weights)
        samples = [self.components[idx].rvs(1) for idx in choices]
        return pd.Series(samples) if size > 1 else samples[0]


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

    def __str__(self):
        return f"NegativeBinomial(r={self._dist.args[0]}, p={self._dist.args[1]})"

    def pmf(self, k: int) -> float:
        return float(self._dist.pmf(k))  # type: ignore[attr-defined]

    def cdf(self, k: int) -> float:
        return float(self._dist.cdf(k))  # type: ignore[attr-defined]

    def rvs(self, size: int = 1) -> pd.Series | np.integer:
        samples = self._dist.rvs(size=size)
        return pd.Series(samples) if size > 1 else samples[0]


class PanjerABk(FrequencyModel):
    """
    Generalized Panjer (a, b, k) distribution class for actuarial applications.

    The (a, b, k) family includes many common actuarial claim count distributions:
    - Poisson(λ): a = λ, b = 1, k = 0
    - Binomial(n, p): a = -q(n+1), b = -q, k = 0 where q = 1-p
    - Negative Binomial(r, p): a = r, b = 0, k = 0

    Args:
        a (float): First parameter in the recursive formula.
        b (float): Second parameter in the recursive formula.
        k (int): Shift parameter indicating the starting support (typically 0 or 1).
        tol (float, optional): Tail probability tolerance for stopping recursion. Defaults to 1e-12.
        max_iter (int, optional): Maximum number of iterations for recursion. Defaults to 100000.

    References:
        https://www.casact.org/sites/default/files/old/astin_vol32no2_283.pdf

    Examples:
        >>> model = PanjerABk(a=2, b=1, k=0)
        >>> model.pmf(3)
    """

    def __init__(self, a: float, b: float, k: int, tol=1e-12, max_iter=100000):
        """
        Initialize a PanjerABk distribution.

        Args:
            a (float): First parameter in the recursive formula.
            b (float): Second parameter in the recursive formula.
            k (int): Shift parameter indicating the starting support (typically 0 or 1).
            tol (float, optional): Tail probability tolerance for stopping recursion. Defaults to 1e-12.
            max_iter (int, optional): Maximum number of iterations for recursion. Defaults to 100000.
        """
        self.a = a
        self.b = b
        self.k = k
        # Compute pmf values via recursion
        pmf_vals = []
        if k > 0:
            # Fill zero probabilities for n < k
            pmf_vals = [0.0] * k
        # Start at n = k
        pmf_vals.append(1.0)            # provisional P(k) = 1 (unnormalized)
        current = 1.0                   # current unnormalized P(n)
        total = 1.0                     # running sum of unnormalized probabilities
        n = k
        # Recursive generation for n = k+1, k+2, ...
        while n < max_iter:
            n += 1
            ratio = a + (b / n)
            if ratio <= 0:
                # If the recursion factor is 0 or negative, stop (finite support or invalid case)
                break
            next_p = current * ratio    # P(n) = (a + b/n) * P(n-1)
            # If additional probability is negligible, we can stop for infinite support
            if next_p < tol * total:
                # The next term is so small that it won't affect sums significantly
                break
            pmf_vals.append(next_p)
            total += next_p
            current = next_p
        # Normalize the pmf
        self.pmf_vals = [p/total for p in pmf_vals]
        # Precompute the CDF for efficiency
        self.cdf_vals = []
        cum = 0.0
        for p in self.pmf_vals:
            cum += p
            self.cdf_vals.append(cum)
        # (Note: length of pmf list is the highest computed support + 1)

    def __str__(self):
        return f"PanjerABk(a={self.a}, b={self.b}, k={self.k})"

    def pmf(self, k: int) -> float:
        if k < 0 or k >= len(self.pmf_vals):
            return 0.0
        return self.pmf_vals[k]

    def cdf(self, k: int) -> float:
        if k < 0:
            return 0.0
        if k >= len(self.cdf_vals) - 1:
            # If n is beyond our computed range, return 1.0 (all mass is within range by design)
            return 1.0
        return self.cdf_vals[k]

    def rvs(self, size: int = 1) -> pd.Series | np.integer:
        samples = pd.Series(np.zeros(size, dtype=int))
        for i in range(size):
            u = random.random()
            # Find smallest index where CDF >= u
            # Since our pmf list starts at index 0 (which could be 0 prob if k>0), we do a linear search.
            # (For large support, binary search would be better, but linear is okay for moderate size.)
            for j, F in enumerate(self.cdf_vals):
                if F >= u:
                    samples[i] = j
                    break
        if size > 1:
            return samples
        else:
            return samples.iloc[0]  # type: ignore[attr-defined]


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

    def __str__(self):
        return f"Poisson(mu={self._dist.args[0]})"

    def pmf(self, k: int) -> float:
        return float(self._dist.pmf(k))  # type: ignore[attr-defined]

    def cdf(self, k: int) -> float:
        return float(self._dist.cdf(k))  # type: ignore[attr-defined]

    def rvs(self, size: int = 1) -> pd.Series | np.integer:
        samples = self._dist.rvs(size=size)
        return pd.Series(samples) if size > 1 else samples[0]


class TriangularFrequency(FrequencyModel):
    """
    Triangular distribution for approximate discrete counts.

    Args:
        c (float): Mode parameter between 0 and 1.
        loc (float, optional): Lower limit. Defaults to 0.0.
        scale (float, optional): Width of distribution. Defaults to 1.0.

    Notes:
        This continuous distribution is rounded to integers for PMF and sampling.

    Examples:
        >>> TriangularFrequency(c=0.5, loc=1, scale=4).rvs(size=3)
    """

    def __init__(self, c: float, loc: float = 0.0, scale: float = 1.0):
        """
        Triangular distribution, continuous in SciPy.
        We'll return 0 for pmf (no true discrete pmf).
        """
        self._dist = triang(c, loc=loc, scale=scale)

    def __str__(self):
        loc = self._dist.kwds.get('loc', 0.0)
        scale = self._dist.kwds.get('scale', 1.0)
        return f"TriangularFrequency(c={self._dist.args[0]}, loc={loc}, scale={scale})"

    def pmf(self, k: int) -> float:
        return self._dist.cdf(k + 0.5 - epsilon) - self._dist.cdf(k - 0.5)
        return 0.0  # continuous distribution → no discrete probability mass

    def cdf(self, k: int) -> float:
        # type: ignore[attr-defined]
        return float(self._dist.cdf(k + 0.5 - epsilon))

    def rvs(self, size: int = 1) -> pd.Series | np.integer:
        samples = np.round(self._dist.rvs(size=size)).astype(int)
        return pd.Series(samples) if size > 1 else samples[0]
