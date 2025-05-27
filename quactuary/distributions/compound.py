"""
Compound distributions for aggregate loss modeling.

This module provides analytical and simulated compound distributions that combine
frequency and severity distributions for aggregate loss calculations. Where possible,
analytical solutions are used for improved performance and accuracy.

Examples:
    >>> from quactuary.distributions.frequency import Poisson
    >>> from quactuary.distributions.severity import Exponential
    >>> from quactuary.distributions.compound import create_compound_distribution
    >>> 
    >>> freq = Poisson(mu=5.0)
    >>> sev = Exponential(scale=1000.0)
    >>> compound = create_compound_distribution(freq, sev)
    >>> 
    >>> # Automatically uses analytical solution if available
    >>> mean_loss = compound.mean()
    >>> samples = compound.rvs(size=1000)
"""

from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
from scipy import stats
from scipy.optimize import brentq

from quactuary.distributions.frequency import FrequencyModel
from quactuary.distributions.severity import SeverityModel
from quactuary.utils.numerical import stable_exp, stable_log

# Configuration constants
MAX_EXACT_CLAIMS = 30  # Threshold for using approximations vs exact calculations
CONVERGENCE_TOLERANCE = 1e-10  # For series expansions
SERIES_MAX_TERMS = 100  # Default maximum terms in series expansions
CACHE_SIZE = 10000  # Default cache size for simulated distributions


def create_compound_distribution(frequency: FrequencyModel, severity: SeverityModel) -> 'CompoundDistribution':
    """
    Factory function to create appropriate compound distribution.

    Automatically selects analytical solution if available, otherwise
    returns simulated compound distribution.
    """
    # Get class names for dispatch
    freq_name = type(frequency).__name__
    sev_name = type(severity).__name__

    # Simple if-elif chain for known analytical solutions
    if freq_name == 'Poisson' and sev_name == 'Exponential':
        return PoissonExponentialCompound(frequency, severity)
    elif freq_name == 'Poisson' and sev_name == 'Gamma':
        return PoissonGammaCompound(frequency, severity)
    elif freq_name == 'Geometric' and sev_name == 'Exponential':
        return GeometricExponentialCompound(frequency, severity)
    elif freq_name == 'NegativeBinomial' and sev_name == 'Gamma':
        return NegativeBinomialGammaCompound(frequency, severity)
    elif freq_name == 'Binomial' and sev_name == 'Exponential':
        return BinomialExponentialCompound(frequency, severity)
    elif freq_name == 'Binomial' and sev_name == 'Gamma':
        return BinomialGammaCompound(frequency, severity)
    elif freq_name == 'Binomial' and sev_name == 'Lognormal':
        return BinomialLognormalCompound(frequency, severity)
    else:
        # Default to Monte Carlo simulation
        return SimulatedCompound(frequency, severity)


class CompoundDistribution(ABC):
    """
    Base class for compound distributions combining frequency and severity.
    """

    def __init__(self, frequency: FrequencyModel, severity: SeverityModel):
        self.frequency = frequency
        self.severity = severity

    @abstractmethod
    def mean(self) -> float:
        """Calculate mean of compound distribution."""
        pass

    @abstractmethod
    def var(self) -> float:
        """Calculate variance of compound distribution."""
        pass

    def std(self) -> float:
        """Calculate standard deviation of compound distribution."""
        return np.sqrt(self.var())

    @abstractmethod
    def pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Probability density function."""
        pass

    @abstractmethod
    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Cumulative distribution function."""
        pass

    @abstractmethod
    def ppf(self, q: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Percent point function (inverse CDF)."""
        pass

    @abstractmethod
    def rvs(self, size: int = 1, random_state: Optional[int] = None) -> Union[float, np.ndarray]:
        """Random variate sampling."""
        pass

    def has_analytical_solution(self) -> bool:
        """Check if this compound distribution has an analytical solution."""
        return not isinstance(self, SimulatedCompound)

    # For backward compatibility
    @classmethod
    def create(cls, frequency: FrequencyModel, severity: SeverityModel) -> 'CompoundDistribution':
        """Factory method for backward compatibility."""
        return create_compound_distribution(frequency, severity)


class SimulatedCompound(CompoundDistribution):
    """
    Compound distribution using Monte Carlo simulation.

    This is the fallback when no analytical solution is available.
    """

    def __init__(self, frequency: FrequencyModel, severity: SeverityModel):
        super().__init__(frequency, severity)
        self._cache = None

    def _ensure_cache(self):
        """Generate cache of simulated values if needed."""
        if self._cache is None:
            samples = []
            for _ in range(CACHE_SIZE):
                n_claims = self.frequency.rvs()
                if n_claims > 0:
                    losses = self.severity.rvs(size=n_claims)
                    total_loss = np.sum(losses)
                else:
                    total_loss = 0.0
                samples.append(total_loss)
            self._cache = np.array(samples)

    def mean(self) -> float:
        """Mean = E[N] * E[X]."""
        # Get mean using scipy's stats method
        freq_mean = self.frequency._dist.mean() if hasattr(
            self.frequency, '_dist') else 0.0
        sev_mean = self.severity._dist.mean() if hasattr(self.severity, '_dist') else 0.0
        return freq_mean * sev_mean

    def var(self) -> float:
        """Var = E[N] * Var[X] + Var[N] * E[X]²."""
        # Get mean and variance using scipy's stats methods
        freq_mean = self.frequency._dist.mean() if hasattr(
            self.frequency, '_dist') else 0.0
        freq_var = self.frequency._dist.var() if hasattr(self.frequency, '_dist') else 0.0
        sev_mean = self.severity._dist.mean() if hasattr(self.severity, '_dist') else 0.0
        sev_var = self.severity._dist.var() if hasattr(self.severity, '_dist') else 0.0
        return freq_mean * sev_var + freq_var * sev_mean**2

    def pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Estimate PDF using kernel density estimation on simulated data."""
        self._ensure_cache()
        kde = stats.gaussian_kde(self._cache)
        result = kde(x)
        return float(result[0]) if np.isscalar(x) else result

    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Estimate CDF using empirical distribution from simulated data."""
        self._ensure_cache()
        x_array = np.atleast_1d(x)
        result = np.array([np.mean(self._cache <= xi) for xi in x_array])
        return result[0] if np.isscalar(x) else result

    def ppf(self, q: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Estimate quantiles using empirical distribution."""
        self._ensure_cache()
        result = np.percentile(self._cache, np.array(q) * 100)
        return float(result) if np.isscalar(q) else result

    def rvs(self, size: int = 1, random_state: Optional[int] = None) -> Union[float, np.ndarray]:
        """Generate random samples via simulation."""
        if random_state is not None:
            np.random.seed(random_state)

        samples = []
        for _ in range(size):
            n_claims = self.frequency.rvs()
            if n_claims > 0:
                losses = self.severity.rvs(size=n_claims)
                total_loss = np.sum(losses)
            else:
                total_loss = 0.0
            samples.append(total_loss)

        result = np.array(samples)
        return result[0] if size == 1 else result


class SeriesExpansionMixin:
    """Mixin for common series expansion computations."""

    def _series_cdf(self, x_array, p_zero, pmf_func, cdf_func, k_max=None):
        """Common series expansion for CDF calculation."""
        result = np.zeros_like(x_array, dtype=float)
        result[x_array >= 0] = p_zero

        pos_mask = x_array > 0
        if np.any(pos_mask):
            x_pos = x_array[pos_mask]
            k_max = k_max or SERIES_MAX_TERMS

            for k in range(1, k_max + 1):
                p_k = pmf_func(k)
                cdf_k = cdf_func(x_pos, k)
                result[pos_mask] += p_k * cdf_k

                if p_k < CONVERGENCE_TOLERANCE:
                    break

        return result

    def _numerical_ppf(self, q_array, p_zero, mean_func, std_func):
        """Common numerical quantile inversion."""
        result = np.zeros_like(q_array, dtype=float)

        for i, qi in enumerate(q_array):
            if qi <= p_zero:
                result[i] = 0.0
            else:
                mean = mean_func()
                std = std_func()
                x0 = max(0.01, mean + std * stats.norm.ppf(qi))

                try:
                    result[i] = brentq(
                        lambda x: self.cdf(x) - qi,
                        0.01, x0 * 10,
                        xtol=1e-6
                    )
                except:
                    result[i] = brentq(
                        lambda x: self.cdf(x) - qi,
                        0.01, x0 * 100,
                        xtol=1e-6
                    )

        return result


class PoissonExponentialCompound(CompoundDistribution, SeriesExpansionMixin):
    """
    Compound Poisson-Exponential distribution.

    When N ~ Poisson(λ) and Xi ~ Exponential(θ), the aggregate loss
    has an atom at 0 with probability e^(-λ), and for S > 0,
    it follows a mixture of Gamma distributions.
    """

    def __init__(self, frequency, severity):
        super().__init__(frequency, severity)
        # Poisson stores mu as first argument
        self.lam = frequency._dist.args[0]
        # Exponential stores scale in kwds
        self.theta = severity._dist.kwds.get('scale', 1.0)

    def mean(self) -> float:
        """E[S] = λ * θ"""
        return self.lam * self.theta

    def var(self) -> float:
        """Var[S] = λ * θ² * 2"""
        return self.lam * self.theta**2 * 2

    def pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """PDF of compound Poisson-Exponential."""
        x_array = np.atleast_1d(x)
        result = np.zeros_like(x_array, dtype=float)

        p_zero = stable_exp(-self.lam)
        result[x_array == 0] = p_zero

        pos_mask = x_array > 0
        if np.any(pos_mask):
            x_pos = x_array[pos_mask]
            k_max = max(50, int(self.lam * 3))

            for k in range(1, k_max + 1):
                p_k = stats.poisson.pmf(k, self.lam)
                gamma_pdf = stats.gamma.pdf(x_pos, a=k, scale=self.theta)
                result[pos_mask] += p_k * gamma_pdf

                if np.max(p_k * gamma_pdf) < CONVERGENCE_TOLERANCE:
                    break

        return result[0] if np.isscalar(x) else result

    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """CDF of compound Poisson-Exponential."""
        x_array = np.atleast_1d(x)
        p_zero = stable_exp(-self.lam)

        result = self._series_cdf(
            x_array, p_zero,
            lambda k: stats.poisson.pmf(k, self.lam),
            lambda x_pos, k: stats.gamma.cdf(x_pos, a=k, scale=self.theta),
            k_max=max(50, int(self.lam * 3))
        )

        return result[0] if np.isscalar(x) else result

    def ppf(self, q: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Quantile function using numerical inversion."""
        q_array = np.atleast_1d(q)
        p_zero = stable_exp(-self.lam)
        result = self._numerical_ppf(q_array, p_zero, self.mean, self.std)
        return result[0] if np.isscalar(q) else result

    def rvs(self, size: int = 1, random_state: Optional[int] = None) -> Union[float, np.ndarray]:
        """Generate random variates."""
        if random_state is not None:
            np.random.seed(random_state)

        n_claims = stats.poisson.rvs(self.lam, size=size)
        result = np.zeros(size)

        for i in range(size):
            if n_claims[i] > 0:
                result[i] = stats.expon.rvs(
                    scale=self.theta, size=n_claims[i]).sum()

        return result[0] if size == 1 else result


class PoissonGammaCompound(CompoundDistribution, SeriesExpansionMixin):
    """
    Compound Poisson-Gamma distribution (Tweedie distribution).

    When N ~ Poisson(λ) and Xi ~ Gamma(α, β), the aggregate loss
    follows a Tweedie distribution.
    """

    def __init__(self, frequency, severity):
        super().__init__(frequency, severity)
        # Poisson stores mu as first argument
        self.lam = frequency._dist.args[0]
        # Gamma stores shape as first argument, scale in kwds
        self.alpha = severity._dist.args[0]
        self.beta = 1.0 / severity._dist.kwds.get('scale', 1.0)
        
        # Calculate Tweedie parameters
        self.p = (self.alpha + 2) / (self.alpha + 1)  # Tweedie power parameter
        self.mu = self.mean()  # Tweedie mean parameter
        self.phi = self.lam * self.alpha / (self.beta ** (2 - self.p))  # Dispersion parameter

    def mean(self) -> float:
        """E[S] = λ * α / β"""
        return self.lam * self.alpha / self.beta

    def var(self) -> float:
        """Var[S] = λ * α * (α + 1) / β²"""
        return self.lam * self.alpha * (self.alpha + 1) / (self.beta ** 2)

    def pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """PDF using series expansion for Poisson-Gamma compound."""
        x_array = np.atleast_1d(x)
        result = np.zeros_like(x_array, dtype=float)

        result[x_array == 0] = stable_exp(-self.lam)

        pos_mask = x_array > 0
        if np.any(pos_mask):
            x_pos = x_array[pos_mask]
            k_max = max(SERIES_MAX_TERMS, int(self.lam * 4))

            for k in range(1, k_max + 1):
                p_k = stats.poisson.pmf(k, self.lam)
                gamma_pdf = stats.gamma.pdf(
                    x_pos, a=k * self.alpha, scale=1/self.beta)
                result[pos_mask] += p_k * gamma_pdf

                if np.max(p_k * gamma_pdf) < CONVERGENCE_TOLERANCE:
                    break

        return result[0] if np.isscalar(x) else result

    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """CDF of compound Poisson-Gamma."""
        x_array = np.atleast_1d(x)
        p_zero = stable_exp(-self.lam)

        result = self._series_cdf(
            x_array, p_zero,
            lambda k: stats.poisson.pmf(k, self.lam),
            lambda x_pos, k: stats.gamma.cdf(
                x_pos, a=k * self.alpha, scale=1/self.beta),
            k_max=max(SERIES_MAX_TERMS, int(self.lam * 4))
        )

        return result[0] if np.isscalar(x) else result

    def ppf(self, q: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Quantile function."""
        q_array = np.atleast_1d(q)
        p_zero = stable_exp(-self.lam)
        result = self._numerical_ppf(q_array, p_zero, self.mean, self.std)
        return result[0] if np.isscalar(q) else result

    def rvs(self, size: int = 1, random_state: Optional[int] = None) -> Union[float, np.ndarray]:
        """Generate random variates."""
        if random_state is not None:
            np.random.seed(random_state)

        n_claims = stats.poisson.rvs(self.lam, size=size)
        result = np.zeros(size)

        for i in range(size):
            if n_claims[i] > 0:
                result[i] = stats.gamma.rvs(
                    a=self.alpha * n_claims[i],
                    scale=1/self.beta
                )

        return result[0] if size == 1 else result


class GeometricExponentialCompound(CompoundDistribution):
    """
    Compound Geometric-Exponential distribution.

    When N ~ Geometric(p) and Xi ~ Exponential(θ), the aggregate loss
    S follows an Exponential distribution with scale θ/(1-p).
    """

    def __init__(self, frequency, severity):
        super().__init__(frequency, severity)
        # Geometric stores p as first argument
        self.p = frequency._dist.args[0]
        # Exponential stores scale in kwds
        self.theta = severity._dist.kwds.get('scale', 1.0)
        self.aggregate_scale = self.theta / \
            (1 - self.p) if self.p < 1 else float('inf')

    def mean(self) -> float:
        """E[S] = θ / p"""
        return self.theta / self.p if self.p > 0 else float('inf')

    def var(self) -> float:
        """Var[S] = θ² * (2 - p) / p²"""
        return self.theta**2 * (2 - self.p) / (self.p**2) if self.p > 0 else float('inf')

    def pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """PDF of compound Geometric-Exponential."""
        x_array = np.atleast_1d(x)

        if self.p >= 1:
            result = np.zeros_like(x_array)
            result[x_array == 0] = 1.0
        else:
            result = stats.expon.pdf(x_array, scale=self.aggregate_scale)

        return result[0] if np.isscalar(x) else result

    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """CDF of compound Geometric-Exponential."""
        x_array = np.atleast_1d(x)

        if self.p >= 1:
            result = np.ones_like(x_array, dtype=float)
            result[x_array < 0] = 0.0
        else:
            result = stats.expon.cdf(x_array, scale=self.aggregate_scale)

        return result[0] if np.isscalar(x) else result

    def ppf(self, q: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Quantile function."""
        if self.p >= 1:
            return np.zeros_like(q)
        else:
            return stats.expon.ppf(q, scale=self.aggregate_scale)

    def rvs(self, size: int = 1, random_state: Optional[int] = None) -> Union[float, np.ndarray]:
        """Generate random variates."""
        if random_state is not None:
            np.random.seed(random_state)

        if self.p >= 1:
            return np.zeros(size) if size > 1 else 0.0
        else:
            return stats.expon.rvs(scale=self.aggregate_scale, size=size)


class NegativeBinomialGammaCompound(CompoundDistribution, SeriesExpansionMixin):
    """
    Compound NegativeBinomial-Gamma distribution.

    Uses series expansion for computation.
    """

    def __init__(self, frequency, severity):
        super().__init__(frequency, severity)
        # NegativeBinomial stores r and p as arguments
        self.r = frequency._dist.args[0]
        self.p = frequency._dist.args[1]
        # Gamma stores shape as first argument, scale in kwds
        self.alpha = severity._dist.args[0]
        self.beta = 1.0 / severity._dist.kwds.get('scale', 1.0)

    def mean(self) -> float:
        """E[S] = r * (1-p) / p * α / β"""
        freq_mean = self.r * (1 - self.p) / self.p
        sev_mean = self.alpha / self.beta
        return freq_mean * sev_mean

    def var(self) -> float:
        """Var[S] = E[N] * Var[X] + Var[N] * E[X]²"""
        freq_mean = self.r * (1 - self.p) / self.p
        freq_var = self.r * (1 - self.p) / (self.p ** 2)
        sev_mean = self.alpha / self.beta
        sev_var = self.alpha / (self.beta ** 2)
        return freq_mean * sev_var + freq_var * sev_mean**2

    def pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """PDF using series expansion."""
        x_array = np.atleast_1d(x)
        result = np.zeros_like(x_array, dtype=float)

        result[x_array == 0] = self.p ** self.r

        pos_mask = x_array > 0
        if np.any(pos_mask):
            x_pos = x_array[pos_mask]
            k_max = max(SERIES_MAX_TERMS, int(self.mean() / self.alpha * 3))

            for k in range(1, k_max + 1):
                p_k = stats.nbinom.pmf(k, self.r, self.p)
                gamma_pdf = stats.gamma.pdf(
                    x_pos, a=k * self.alpha, scale=1/self.beta)
                result[pos_mask] += p_k * gamma_pdf

                if np.max(p_k * gamma_pdf) < CONVERGENCE_TOLERANCE:
                    break

        return result[0] if np.isscalar(x) else result

    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """CDF using series expansion."""
        x_array = np.atleast_1d(x)
        p_zero = self.p ** self.r

        result = self._series_cdf(
            x_array, p_zero,
            lambda k: stats.nbinom.pmf(k, self.r, self.p),
            lambda x_pos, k: stats.gamma.cdf(
                x_pos, a=k * self.alpha, scale=1/self.beta),
            k_max=max(SERIES_MAX_TERMS, int(self.mean() / self.alpha * 3))
        )

        return result[0] if np.isscalar(x) else result

    def ppf(self, q: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Quantile function using numerical inversion."""
        q_array = np.atleast_1d(q)
        p_zero = self.p ** self.r
        result = self._numerical_ppf(q_array, p_zero, self.mean, self.std)
        return result[0] if np.isscalar(q) else result

    def rvs(self, size: int = 1, random_state: Optional[int] = None) -> Union[float, np.ndarray]:
        """Generate random variates via direct simulation."""
        if random_state is not None:
            np.random.seed(random_state)

        n_claims = stats.nbinom.rvs(self.r, self.p, size=size)
        result = np.zeros(size)

        for i in range(size):
            if n_claims[i] > 0:
                result[i] = stats.gamma.rvs(
                    a=self.alpha * n_claims[i],
                    scale=1/self.beta
                )

        return result[0] if size == 1 else result


class BinomialLognormalApproximation(CompoundDistribution):
    """
    Approximation for Binomial-Lognormal compound distribution.

    Uses Fenton-Wilkinson approximation for sum of lognormals.
    """

    def __init__(self, frequency, severity):
        super().__init__(frequency, severity)
        # Binomial stores n and p as arguments
        self.n = frequency._dist.args[0]
        self.p = frequency._dist.args[1]
        # Lognormal stores shape (sigma) as first argument, scale (exp(mu)) in kwds
        self.sigma_ln = severity._dist.args[0]
        # Need to extract mu from scale parameter
        scale = severity._dist.kwds.get('scale', 1.0)
        self.mu_ln = np.log(scale)
        self._approx_params = None

    def _get_approx_params(self):
        """Calculate approximate lognormal parameters."""
        if self._approx_params is None:
            m1 = np.exp(self.mu_ln + self.sigma_ln**2 / 2)
            m2 = np.exp(2 * self.mu_ln + 2 * self.sigma_ln**2)

            E_N = self.n * self.p
            Var_N = self.n * self.p * (1 - self.p)

            E_S = E_N * m1
            Var_S = E_N * (m2 - m1**2) + Var_N * m1**2

            if E_S > 0:
                cv_S = np.sqrt(Var_S) / E_S
                sigma_approx = np.sqrt(stable_log(1 + cv_S**2))
                mu_approx = stable_log(E_S) - sigma_approx**2 / 2
            else:
                mu_approx = -np.inf
                sigma_approx = 0

            self._approx_params = (mu_approx, sigma_approx)

        return self._approx_params

    def mean(self) -> float:
        """E[S] = n * p * E[X]"""
        return self.n * self.p * np.exp(self.mu_ln + self.sigma_ln**2 / 2)

    def var(self) -> float:
        """Var[S] using compound distribution formula"""
        m1 = np.exp(self.mu_ln + self.sigma_ln**2 / 2)
        m2 = np.exp(2 * self.mu_ln + 2 * self.sigma_ln**2)
        E_N = self.n * self.p
        Var_N = self.n * self.p * (1 - self.p)
        return E_N * (m2 - m1**2) + Var_N * m1**2

    def pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Approximate PDF using lognormal approximation."""
        x_array = np.atleast_1d(x)
        result = np.zeros_like(x_array, dtype=float)

        p_zero = (1 - self.p) ** self.n
        result[x_array == 0] = p_zero

        pos_mask = x_array > 0
        if np.any(pos_mask):
            mu_approx, sigma_approx = self._get_approx_params()
            if sigma_approx > 0:
                p_positive = 1 - p_zero
                result[pos_mask] = p_positive * stats.lognorm.pdf(
                    x_array[pos_mask],
                    s=sigma_approx,
                    scale=np.exp(mu_approx)
                )

        return result[0] if np.isscalar(x) else result

    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Approximate CDF using lognormal approximation."""
        x_array = np.atleast_1d(x)
        result = np.zeros_like(x_array, dtype=float)

        p_zero = (1 - self.p) ** self.n
        result[x_array >= 0] = p_zero

        pos_mask = x_array > 0
        if np.any(pos_mask):
            mu_approx, sigma_approx = self._get_approx_params()
            if sigma_approx > 0:
                p_positive = 1 - p_zero
                result[pos_mask] += p_positive * stats.lognorm.cdf(
                    x_array[pos_mask],
                    s=sigma_approx,
                    scale=np.exp(mu_approx)
                )

        return result[0] if np.isscalar(x) else result

    def ppf(self, q: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Approximate quantiles using lognormal approximation."""
        q_array = np.atleast_1d(q)
        result = np.zeros_like(q_array, dtype=float)

        p_zero = (1 - self.p) ** self.n

        for i, qi in enumerate(q_array):
            if qi <= p_zero:
                result[i] = 0.0
            else:
                mu_approx, sigma_approx = self._get_approx_params()
                if sigma_approx > 0:
                    q_adj = (qi - p_zero) / (1 - p_zero)
                    result[i] = stats.lognorm.ppf(
                        q_adj,
                        s=sigma_approx,
                        scale=np.exp(mu_approx)
                    )

        return result[0] if np.isscalar(q) else result

    def rvs(self, size: int = 1, random_state: Optional[int] = None) -> Union[float, np.ndarray]:
        """Generate samples using approximation."""
        if random_state is not None:
            np.random.seed(random_state)

        n_claims = stats.binom.rvs(self.n, self.p, size=size)
        result = np.zeros(size)

        for i in range(size):
            if n_claims[i] == 0:
                result[i] = 0
            elif n_claims[i] > MAX_EXACT_CLAIMS:
                # Approximate sum of lognormals
                sum_mu = self.mu_ln + stable_log(n_claims[i])
                sum_sigma = self.sigma_ln / np.sqrt(n_claims[i])
                result[i] = stats.lognorm.rvs(
                    s=sum_sigma, scale=np.exp(sum_mu))
            else:
                # Direct simulation for small counts
                result[i] = stats.lognorm.rvs(
                    s=self.sigma_ln,
                    scale=np.exp(self.mu_ln),
                    size=n_claims[i]
                ).sum()

        return result[0] if size == 1 else result


# Binomial compound distributions moved from compound_binomial.py

class BinomialExponentialCompound(CompoundDistribution, SeriesExpansionMixin):
    """
    Compound Binomial-Exponential distribution.

    When N ~ Binomial(n, p) and Xi ~ Exponential(θ), the aggregate loss
    has an atom at 0 with probability (1-p)^n, and for S > 0,
    it follows a mixture of Gamma distributions.
    """

    def __init__(self, frequency, severity):
        """Initialize Binomial-Exponential compound distribution.

        Args:
            frequency: Binomial frequency distribution
            severity: Exponential severity distribution
        """
        super().__init__(frequency, severity)
        self.n = int(frequency._dist.args[0])
        self.p = frequency._dist.args[1]
        self.theta = severity._dist.kwds.get('scale', 1.0)

    def mean(self) -> float:
        """Calculate expected value of aggregate loss.

        Returns:
            float: Expected value E[S] = n * p * θ
        """
        return self.n * self.p * self.theta

    def var(self) -> float:
        """Calculate variance of aggregate loss.

        Returns:
            float: Variance Var[S] = n * p * θ² * (2 - p)
        """
        return self.n * self.p * self.theta**2 * (2 - self.p)

    def pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Probability density function.

        Args:
            x: Value(s) at which to evaluate PDF

        Returns:
            PDF value(s) at x
        """
        x_array = np.atleast_1d(x)
        result = np.zeros_like(x_array, dtype=float)

        # Probability of zero claims
        p_zero = (1 - self.p) ** self.n
        result[x_array == 0] = p_zero

        # For positive values, use series expansion
        pos_mask = x_array > 0
        if np.any(pos_mask):
            x_pos = x_array[pos_mask]

            for k in range(1, self.n + 1):
                # Binomial PMF
                p_k = stats.binom.pmf(k, self.n, self.p)
                # Sum of k exponentials is Gamma(k, θ)
                gamma_pdf = stats.gamma.pdf(x_pos, a=k, scale=self.theta)
                result[pos_mask] += p_k * gamma_pdf

                if p_k < 1e-10:
                    break

        return result[0] if np.isscalar(x) else result

    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """CDF of compound Binomial-Exponential."""
        x_array = np.atleast_1d(x)
        p_zero = (1 - self.p) ** self.n

        result = self._series_cdf(
            x_array, p_zero,
            lambda k: stats.binom.pmf(k, self.n, self.p),
            lambda x_pos, k: stats.gamma.cdf(x_pos, a=k, scale=self.theta),
            k_max=self.n
        )

        return result[0] if np.isscalar(x) else result

    def ppf(self, q: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Quantile function using numerical inversion."""
        q_array = np.atleast_1d(q)
        p_zero = (1 - self.p) ** self.n
        result = self._numerical_ppf(q_array, p_zero, self.mean, self.std)
        return result[0] if np.isscalar(q) else result

    def rvs(self, size: int = 1, random_state: Optional[int] = None) -> Union[float, np.ndarray]:
        """Generate random variates."""
        if random_state is not None:
            np.random.seed(random_state)

        n_claims = stats.binom.rvs(self.n, self.p, size=size)
        result = np.zeros(size)

        for i in range(size):
            if n_claims[i] > 0:
                result[i] = stats.expon.rvs(
                    scale=self.theta, size=n_claims[i]).sum()

        return result[0] if size == 1 else result


class BinomialGammaCompound(CompoundDistribution, SeriesExpansionMixin):
    """
    Compound Binomial-Gamma distribution.

    When N ~ Binomial(n, p) and Xi ~ Gamma(α, β), the aggregate loss
    follows a mixture of Gamma distributions.
    """

    def __init__(self, frequency, severity):
        super().__init__(frequency, severity)
        self.n = int(frequency._dist.args[0])
        self.p = frequency._dist.args[1]
        self.alpha = severity._dist.args[0] if severity._dist.args else 1.0
        self.beta = 1.0 / severity._dist.kwds.get('scale', 1.0)

    def mean(self) -> float:
        """E[S] = n * p * α / β"""
        return self.n * self.p * self.alpha / self.beta

    def var(self) -> float:
        """Var[S] = n * p * (α/β)² * (1 - p + α)"""
        mean_sev = self.alpha / self.beta
        var_sev = self.alpha / (self.beta**2)
        return self.n * self.p * (var_sev + (1 - self.p) * mean_sev**2)

    def pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """PDF using series expansion for Binomial-Gamma compound."""
        x_array = np.atleast_1d(x)
        result = np.zeros_like(x_array, dtype=float)

        # Probability of zero claims
        p_zero = (1 - self.p) ** self.n
        result[x_array == 0] = p_zero

        # For positive values
        pos_mask = x_array > 0
        if np.any(pos_mask):
            x_pos = x_array[pos_mask]

            for k in range(1, self.n + 1):
                p_k = stats.binom.pmf(k, self.n, self.p)
                # Sum of k Gamma(α, β) is Gamma(k*α, β)
                gamma_pdf = stats.gamma.pdf(
                    x_pos, a=k * self.alpha, scale=1/self.beta)
                result[pos_mask] += p_k * gamma_pdf

                if p_k < 1e-10:
                    break

        return result[0] if np.isscalar(x) else result

    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """CDF of compound Binomial-Gamma."""
        x_array = np.atleast_1d(x)
        
        # Handle infinity explicitly
        result = np.zeros_like(x_array, dtype=float)
        result[np.isinf(x_array)] = 1.0
        
        # Handle finite values
        finite_mask = np.isfinite(x_array)
        if np.any(finite_mask):
            x_finite = x_array[finite_mask]
            p_zero = (1 - self.p) ** self.n

            finite_result = self._series_cdf(
                x_finite, p_zero,
                lambda k: stats.binom.pmf(k, self.n, self.p),
                lambda x_pos, k: stats.gamma.cdf(
                    x_pos, a=k * self.alpha, scale=1/self.beta),
                k_max=self.n
            )
            result[finite_mask] = finite_result

        return result[0] if np.isscalar(x) else result

    def ppf(self, q: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Quantile function."""
        q_array = np.atleast_1d(q)
        p_zero = (1 - self.p) ** self.n
        result = self._numerical_ppf(q_array, p_zero, self.mean, self.std)
        return result[0] if np.isscalar(q) else result

    def rvs(self, size: int = 1, random_state: Optional[int] = None) -> Union[float, np.ndarray]:
        """Generate random variates."""
        if random_state is not None:
            np.random.seed(random_state)

        n_claims = stats.binom.rvs(self.n, self.p, size=size)
        result = np.zeros(size)

        for i in range(size):
            if n_claims[i] > 0:
                result[i] = stats.gamma.rvs(
                    a=self.alpha,
                    scale=1/self.beta,
                    size=n_claims[i]
                ).sum()

        return result[0] if size == 1 else result


class BinomialLognormalCompound(CompoundDistribution):
    """
    Compound Binomial-Lognormal distribution.

    Uses the method of moments approximation for the sum of lognormals.
    """

    def __init__(self, frequency, severity):
        super().__init__(frequency, severity)
        self.n = int(frequency._dist.args[0])
        self.p = frequency._dist.args[1]

        # Extract lognormal parameters
        # scipy uses s=sigma, scale=exp(mu)
        self.sigma = severity._dist.args[0] if severity._dist.args else 1.0
        self.scale = severity._dist.kwds.get('scale', 1.0)
        self.mu = np.log(self.scale)
        self._approx_params = None
        
    def _get_approx_params(self):
        """Calculate Fenton-Wilkinson approximation parameters."""
        if self._approx_params is None:
            # For Binomial-Lognormal, use method of moments
            mean_sev = np.exp(self.mu + self.sigma**2 / 2)
            var_sev = (np.exp(self.sigma**2) - 1) * mean_sev**2
            
            # Compound moments
            mean_compound = self.n * self.p * mean_sev
            var_compound = self.n * self.p * (var_sev + (1 - self.p) * mean_sev**2)
            
            # Lognormal approximation parameters
            cv_squared = var_compound / (mean_compound**2)
            sigma_approx = np.sqrt(np.log(1 + cv_squared))
            mu_approx = np.log(mean_compound) - sigma_approx**2 / 2
            
            self._approx_params = (mu_approx, sigma_approx)
        
        return self._approx_params

    def mean(self) -> float:
        """E[S] = n * p * E[X]"""
        mean_sev = np.exp(self.mu + self.sigma**2 / 2)
        return self.n * self.p * mean_sev

    def var(self) -> float:
        """Var[S] = n * p * (Var[X] + (1-p) * E[X]²)"""
        mean_sev = np.exp(self.mu + self.sigma**2 / 2)
        var_sev = (np.exp(self.sigma**2) - 1) * mean_sev**2
        return self.n * self.p * (var_sev + (1 - self.p) * mean_sev**2)

    def _fenton_wilkinson_params(self, k):
        """Calculate Fenton-Wilkinson approximation parameters for sum of k lognormals."""
        if k == 0:
            return None, None

        # For sum of k i.i.d. lognormals
        mean_sum = k * np.exp(self.mu + self.sigma**2 / 2)
        var_sum = k * (np.exp(self.sigma**2) - 1) * \
            np.exp(2 * self.mu + self.sigma**2)

        # Fenton-Wilkinson parameters
        sigma_sum = np.sqrt(np.log(1 + var_sum / mean_sum**2))
        mu_sum = np.log(mean_sum) - sigma_sum**2 / 2

        return mu_sum, sigma_sum

    def pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """PDF using Fenton-Wilkinson approximation."""
        x_array = np.atleast_1d(x)
        result = np.zeros_like(x_array, dtype=float)

        # Probability of zero claims
        p_zero = (1 - self.p) ** self.n
        result[x_array == 0] = p_zero

        # For positive values
        pos_mask = x_array > 0
        if np.any(pos_mask):
            x_pos = x_array[pos_mask]

            for k in range(1, self.n + 1):
                p_k = stats.binom.pmf(k, self.n, self.p)
                mu_k, sigma_k = self._fenton_wilkinson_params(k)

                if mu_k is not None:
                    # Approximate sum as lognormal
                    lognorm_pdf = stats.lognorm.pdf(
                        x_pos, s=sigma_k, scale=np.exp(mu_k))
                    result[pos_mask] += p_k * lognorm_pdf

                if p_k < 1e-10:
                    break

        return result[0] if np.isscalar(x) else result

    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """CDF using Fenton-Wilkinson approximation."""
        x_array = np.atleast_1d(x)
        result = np.zeros_like(x_array, dtype=float)

        # Handle special cases
        result[x_array < 0] = 0.0
        result[np.isinf(x_array)] = 1.0

        # Handle finite non-negative values
        finite_mask = np.isfinite(x_array) & (x_array >= 0)
        if np.any(finite_mask):
            x_finite = x_array[finite_mask]
            
            # Probability of zero claims
            p_zero = (1 - self.p) ** self.n
            result[finite_mask] = p_zero

            # For positive values
            pos_mask = x_finite > 0
            if np.any(pos_mask):
                x_pos = x_finite[pos_mask]
                pos_result = np.zeros_like(x_pos)

                for k in range(1, self.n + 1):
                    p_k = stats.binom.pmf(k, self.n, self.p)
                    mu_k, sigma_k = self._fenton_wilkinson_params(k)

                    if mu_k is not None:
                        lognorm_cdf = stats.lognorm.cdf(
                            x_pos, s=sigma_k, scale=np.exp(mu_k))
                        pos_result += p_k * lognorm_cdf

                    if p_k < 1e-10:
                        break
                
                result[finite_mask & (x_array > 0)] = p_zero + pos_result

        return result[0] if np.isscalar(x) else result

    def ppf(self, q: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Quantile function using numerical inversion."""
        q_array = np.atleast_1d(q)
        p_zero = (1 - self.p) ** self.n

        from scipy.optimize import brentq
        result = np.zeros_like(q_array, dtype=float)

        for i, qi in enumerate(q_array):
            if qi <= p_zero:
                result[i] = 0.0
            else:
                # Initial guess based on mean and std
                mean = self.mean()
                std = self.std()
                x0 = max(0.01, mean + std * stats.norm.ppf(qi))

                try:
                    result[i] = brentq(
                        lambda x: self.cdf(x) - qi,
                        0.01, x0 * 10,
                        xtol=1e-6
                    )
                except:
                    result[i] = brentq(
                        lambda x: self.cdf(x) - qi,
                        0.01, x0 * 100,
                        xtol=1e-6
                    )

        return result[0] if np.isscalar(q) else result

    def rvs(self, size: int = 1, random_state: Optional[int] = None) -> Union[float, np.ndarray]:
        """Generate random variates."""
        if random_state is not None:
            np.random.seed(random_state)

        n_claims = stats.binom.rvs(self.n, self.p, size=size)
        result = np.zeros(size)

        for i in range(size):
            if n_claims[i] > 0:
                result[i] = stats.lognorm.rvs(
                    s=self.sigma,
                    scale=self.scale,
                    size=n_claims[i]
                ).sum()

        return result[0] if size == 1 else result


class PanjerBinomialRecursion:
    """
    Panjer recursion for compound distributions with binomial frequency.

    Provides exact computation of the aggregate loss distribution
    when severity distribution has finite support.
    """

    def __init__(self, n: int, p: float, severity_pmf: dict):
        """
        Initialize Panjer recursion for binomial frequency.

        Args:
            n: Number of trials in binomial distribution
            p: Success probability
            severity_pmf: Dictionary mapping severity values to probabilities
        """
        self.n = n
        self.p = p
        self.severity_pmf = severity_pmf
        self.severity_values = sorted(severity_pmf.keys())

    def compute_aggregate_pmf(self, max_value: int = None) -> dict:
        """
        Compute PMF of aggregate loss using Panjer recursion.

        Args:
            max_value: Maximum aggregate value to compute

        Returns:
            Dictionary mapping aggregate values to probabilities
        """
        if max_value is None:
            max_value = self.n * max(self.severity_values)

        # Initialize
        g = {0: (1 - self.p) ** self.n}

        # Panjer recursion coefficients for binomial
        a = -self.p / (1 - self.p)
        b = (self.n + 1) * self.p / (1 - self.p)

        # Compute recursively
        for s in range(1, max_value + 1):
            g[s] = 0
            for x in self.severity_values:
                if x <= s and (s - x) in g:
                    coeff = (a + b * x / s)
                    g[s] += coeff * self.severity_pmf[x] * g[s - x]

        return g


def create_compound_distribution(frequency: FrequencyModel, severity: SeverityModel) -> CompoundDistribution:
    """
    Factory function to create appropriate compound distribution.
    
    Automatically selects the optimal implementation based on the frequency
    and severity distribution types. Returns analytical solutions when available,
    otherwise falls back to simulation.
    
    Args:
        frequency: Frequency distribution (claim counts)
        severity: Severity distribution (claim amounts)
        
    Returns:
        CompoundDistribution instance
        
    Examples:
        >>> freq = Poisson(mu=10)
        >>> sev = Exponential(scale=1000)
        >>> compound = create_compound_distribution(freq, sev)
        >>> # Automatically uses PoissonExponentialCompound (analytical)
    """
    # Map distribution types to their optimal compound implementations
    freq_type = type(frequency).__name__
    sev_type = type(severity).__name__
    
    # Check for analytical solutions first
    if freq_type == 'Poisson':
        if sev_type == 'Exponential':
            return PoissonExponentialCompound(frequency, severity)
        elif sev_type == 'Gamma':
            return PoissonGammaCompound(frequency, severity)
    elif freq_type == 'Geometric' and sev_type == 'Exponential':
        return GeometricExponentialCompound(frequency, severity)
    elif freq_type == 'NegativeBinomial' and sev_type == 'Gamma':
        return NegativeBinomialGammaCompound(frequency, severity)
    elif freq_type == 'Binomial':
        if sev_type == 'Exponential':
            return BinomialExponentialCompound(frequency, severity)
        elif sev_type == 'Gamma':
            return BinomialGammaCompound(frequency, severity)
        elif sev_type == 'Lognormal':
            return BinomialLognormalCompound(frequency, severity)
    
    # Default to simulation
    return SimulatedCompound(frequency, severity)
