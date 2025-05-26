"""
Mixed Poisson distributions for heterogeneous risk modeling.

This module provides mixed Poisson processes where the Poisson intensity
parameter follows a probability distribution, capturing population heterogeneity.
"""

import numpy as np
from scipy import stats, special, integrate
from typing import Union, Optional, Callable

from quactuary.distributions.frequency import FrequencyModel
from quactuary.utils.numerical import stable_exp, stable_log


class MixedPoissonDistribution(FrequencyModel):
    """
    Base class for mixed Poisson distributions.
    
    In a mixed Poisson process, the intensity λ is itself a random variable,
    leading to overdispersed count distributions.
    """
    
    def __init__(self, mixing_dist_name: str, **params):
        """
        Initialize mixed Poisson distribution.
        
        Args:
            mixing_dist_name: Name of the mixing distribution
            **params: Parameters specific to the mixing distribution
        """
        self.mixing_dist_name = mixing_dist_name
        self.params = params
        self._setup_distribution()
    
    def _setup_distribution(self):
        """Setup the specific mixed distribution."""
        raise NotImplementedError("Subclasses must implement _setup_distribution")
    
    def _validate_params(self):
        """Validate distribution parameters."""
        # Implemented by subclasses
        pass


class PoissonGammaMixture(MixedPoissonDistribution):
    """
    Poisson-Gamma mixture (Negative Binomial distribution).
    
    When λ ~ Gamma(α, β) and N|λ ~ Poisson(λ), then
    N ~ NegativeBinomial(r=α, p=β/(1+β)).
    
    This is the most common mixed Poisson model, capturing
    overdispersion through gamma-distributed heterogeneity.
    """
    
    def __init__(self, alpha: float, beta: float):
        """
        Initialize Poisson-Gamma mixture.
        
        Args:
            alpha: Shape parameter of Gamma distribution
            beta: Rate parameter of Gamma distribution
        """
        self.alpha = alpha
        self.beta = beta
        super().__init__('gamma', alpha=alpha, beta=beta)
    
    def _setup_distribution(self):
        """Setup negative binomial distribution."""
        # Convert to negative binomial parameterization
        self.r = self.alpha
        self.p = self.beta / (1 + self.beta)
        
        # Use scipy's negative binomial
        self._dist = stats.nbinom(n=self.r, p=self.p)
    
    def _validate_params(self):
        """Validate parameters."""
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if self.beta <= 0:
            raise ValueError("beta must be positive")
    
    def mean(self) -> float:
        """E[N] = α/β"""
        return self.alpha / self.beta
    
    def var(self) -> float:
        """Var[N] = α/β + α/β²"""
        mean_val = self.alpha / self.beta
        return mean_val + mean_val / self.beta
    
    def mixing_density(self, lambda_val: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Gamma mixing density."""
        return stats.gamma.pdf(lambda_val, a=self.alpha, scale=1/self.beta)
    
    def conditional_mean(self, n: int) -> float:
        """E[λ|N=n] for posterior inference."""
        return (self.alpha + n) / (self.beta + 1)
    
    def conditional_var(self, n: int) -> float:
        """Var[λ|N=n] for posterior inference."""
        return (self.alpha + n) / (self.beta + 1)**2
    
    def pmf(self, k: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
        """Probability mass function."""
        return self._dist.pmf(k)
    
    def rvs(self, size: int = 1, random_state: Optional[int] = None) -> Union[int, np.ndarray]:
        """Generate random variates."""
        if random_state is not None:
            np.random.seed(random_state)
        return self._dist.rvs(size=size)


class PoissonInverseGaussianMixture(MixedPoissonDistribution):
    """
    Poisson-Inverse Gaussian mixture.
    
    When λ ~ InverseGaussian(μ, λ₀) and N|λ ~ Poisson(λ),
    the marginal distribution has heavier tails than negative binomial.
    """
    
    def __init__(self, mu: float, lambda_shape: float):
        """
        Initialize Poisson-Inverse Gaussian mixture.
        
        Args:
            mu: Mean of inverse Gaussian distribution
            lambda_shape: Shape parameter of inverse Gaussian
        """
        self.mu = mu
        self.lambda_shape = lambda_shape
        super().__init__('invgauss', mu=mu, lambda_shape=lambda_shape)
    
    def _setup_distribution(self):
        """Setup distribution using numerical methods."""
        self._validate_params()
        # Store parameters for numerical calculations
        self._mixing_dist = stats.invgauss(mu=self.mu/self.lambda_shape, scale=self.lambda_shape)
    
    def _validate_params(self):
        """Validate parameters."""
        if self.mu <= 0:
            raise ValueError("mu must be positive")
        if self.lambda_shape <= 0:
            raise ValueError("lambda_shape must be positive")
    
    def pmf(self, k: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
        """Probability mass function using numerical integration."""
        k = np.atleast_1d(k).astype(int)
        result = np.zeros_like(k, dtype=float)
        
        for i, ki in enumerate(k):
            if ki < 0:
                result[i] = 0.0
            else:
                # Integrate Poisson PMF over mixing distribution
                def integrand(lam):
                    if lam <= 0:
                        return 0.0
                    poisson_pmf = stats.poisson.pmf(ki, lam)
                    mixing_pdf = self._mixing_dist.pdf(lam)
                    return poisson_pmf * mixing_pdf
                
                # Use adaptive quadrature
                integral, _ = integrate.quad(integrand, 0, np.inf, limit=100)
                result[i] = integral
        
        return result[0] if np.isscalar(k) else result
    
    def mean(self) -> float:
        """E[N] = E[λ] = μ"""
        return self.mu
    
    def var(self) -> float:
        """Var[N] = E[λ] + Var[λ]"""
        mean_lambda = self.mu
        var_lambda = self.mu**3 / self.lambda_shape
        return mean_lambda + var_lambda
    
    def mixing_density(self, lambda_val: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Inverse Gaussian mixing density."""
        return self._mixing_dist.pdf(lambda_val)
    
    def rvs(self, size: int = 1, random_state: Optional[int] = None) -> Union[int, np.ndarray]:
        """Generate random variates."""
        if random_state is not None:
            np.random.seed(random_state)
        
        # First sample lambda values from mixing distribution
        lambda_samples = self._mixing_dist.rvs(size=size)
        
        # Then sample Poisson with those lambdas
        samples = np.zeros(size, dtype=int)
        for i in range(size):
            samples[i] = stats.poisson.rvs(lambda_samples[i])
        
        return samples[0] if size == 1 else samples


class HierarchicalPoissonMixture(MixedPoissonDistribution):
    """
    Hierarchical Poisson mixture for portfolio modeling.
    
    Supports multiple levels of mixing for complex portfolio structures:
    - Portfolio level: λ_p ~ Gamma(α_p, β_p)
    - Group level: λ_g|λ_p ~ Gamma(α_g, λ_p)
    - Individual level: N|λ_g ~ Poisson(λ_g)
    """
    
    def __init__(self, portfolio_alpha: float, portfolio_beta: float,
                 group_alpha: float, n_groups: int = 1):
        """
        Initialize hierarchical Poisson mixture.
        
        Args:
            portfolio_alpha: Shape at portfolio level
            portfolio_beta: Rate at portfolio level
            group_alpha: Shape at group level
            n_groups: Number of groups in portfolio
        """
        self.portfolio_alpha = portfolio_alpha
        self.portfolio_beta = portfolio_beta
        self.group_alpha = group_alpha
        self.n_groups = n_groups
        
        super().__init__('hierarchical',
                        portfolio_alpha=portfolio_alpha,
                        portfolio_beta=portfolio_beta,
                        group_alpha=group_alpha,
                        n_groups=n_groups)
    
    def _setup_distribution(self):
        """Setup hierarchical distribution."""
        self._validate_params()
        # No scipy distribution for this; we implement methods directly
    
    def _validate_params(self):
        """Validate parameters."""
        if self.portfolio_alpha <= 0 or self.portfolio_beta <= 0:
            raise ValueError("Portfolio parameters must be positive")
        if self.group_alpha <= 0:
            raise ValueError("Group alpha must be positive")
        if self.n_groups < 1:
            raise ValueError("Number of groups must be at least 1")
    
    def mean(self) -> float:
        """E[N] = n_groups * α_g * α_p / β_p"""
        portfolio_mean = self.portfolio_alpha / self.portfolio_beta
        return self.n_groups * self.group_alpha * portfolio_mean
    
    def var(self) -> float:
        """Variance accounting for all levels of hierarchy."""
        # E[λ_p]
        mean_portfolio = self.portfolio_alpha / self.portfolio_beta
        # Var[λ_p]
        var_portfolio = self.portfolio_alpha / (self.portfolio_beta**2)
        
        # For each group: E[N|λ_p] = α_g * λ_p
        # E[N] = n_groups * α_g * E[λ_p]
        mean_total = self.n_groups * self.group_alpha * mean_portfolio
        
        # Var[N] using law of total variance
        # Within-group variance
        var_within = self.n_groups * self.group_alpha * mean_portfolio * (1 + self.group_alpha)
        # Between-group variance
        var_between = self.n_groups * (self.group_alpha**2) * var_portfolio
        
        return var_within + var_between
    
    def simulate_portfolio(self, size: int = 1, 
                          random_state: Optional[int] = None) -> dict:
        """
        Simulate full portfolio structure.
        
        Returns:
            Dictionary with 'total', 'by_group', and 'lambda_p' arrays
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        results = {
            'total': np.zeros(size, dtype=int),
            'by_group': np.zeros((size, self.n_groups), dtype=int),
            'lambda_p': np.zeros(size)
        }
        
        for i in range(size):
            # Sample portfolio-level intensity
            lambda_p = stats.gamma.rvs(
                a=self.portfolio_alpha,
                scale=1/self.portfolio_beta
            )
            results['lambda_p'][i] = lambda_p
            
            # Sample group-level counts
            for g in range(self.n_groups):
                lambda_g = stats.gamma.rvs(
                    a=self.group_alpha,
                    scale=lambda_p/self.group_alpha
                )
                count = stats.poisson.rvs(lambda_g)
                results['by_group'][i, g] = count
                results['total'][i] += count
        
        return results
    
    def pmf(self, k: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Probability mass function using Monte Carlo approximation.
        
        For hierarchical models, exact PMF is intractable, so we use
        Monte Carlo integration.
        """
        k = np.atleast_1d(k).astype(int)
        result = np.zeros_like(k, dtype=float)
        
        # Monte Carlo samples for approximation
        n_samples = 10000
        np.random.seed(42)  # For reproducibility
        
        # Simulate portfolio counts
        sim = self.simulate_portfolio(size=n_samples)
        total_counts = sim['total']
        
        # Estimate PMF from empirical distribution
        for i, ki in enumerate(k):
            if ki < 0:
                result[i] = 0.0
            else:
                result[i] = np.mean(total_counts == ki)
        
        return result[0] if np.isscalar(k) else result
    
    def rvs(self, size: int = 1, random_state: Optional[int] = None) -> Union[int, np.ndarray]:
        """Generate random variates (total counts only)."""
        sim = self.simulate_portfolio(size, random_state)
        return sim['total'][0] if size == 1 else sim['total']


class TimeVaryingPoissonMixture(MixedPoissonDistribution):
    """
    Mixed Poisson process with time-varying intensity.
    
    Supports intensity functions λ(t) with random parameters,
    useful for modeling seasonal patterns with uncertainty.
    """
    
    def __init__(self, base_rate: float, intensity_func: Callable,
                 param_dist: dict, time_horizon: float = 1.0):
        """
        Initialize time-varying Poisson mixture.
        
        Args:
            base_rate: Base intensity rate
            intensity_func: Function λ(t, params) for time-varying intensity
            param_dist: Distribution of parameters {'name': str, 'params': dict}
            time_horizon: Time period for counting events
        """
        self.base_rate = base_rate
        self.intensity_func = intensity_func
        self.param_dist = param_dist
        self.time_horizon = time_horizon
        
        super().__init__('time_varying',
                        base_rate=base_rate,
                        param_dist=param_dist,
                        time_horizon=time_horizon)
    
    def _setup_distribution(self):
        """Setup parameter distribution."""
        self._validate_params()
        
        # Setup distribution for random parameters
        dist_name = self.param_dist['name']
        dist_params = self.param_dist['params']
        
        if dist_name == 'normal':
            self._param_dist = stats.norm(**dist_params)
        elif dist_name == 'gamma':
            self._param_dist = stats.gamma(**dist_params)
        elif dist_name == 'beta':
            self._param_dist = stats.beta(**dist_params)
        else:
            raise ValueError(f"Unsupported parameter distribution: {dist_name}")
    
    def _validate_params(self):
        """Validate parameters."""
        if self.base_rate <= 0:
            raise ValueError("Base rate must be positive")
        if self.time_horizon <= 0:
            raise ValueError("Time horizon must be positive")
        if 'name' not in self.param_dist or 'params' not in self.param_dist:
            raise ValueError("param_dist must have 'name' and 'params' keys")
    
    def integrated_intensity(self, params: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute ∫₀ᵀ λ(t, params) dt."""
        if np.isscalar(params):
            result, _ = integrate.quad(
                lambda t: self.intensity_func(t, params),
                0, self.time_horizon
            )
            return self.base_rate * result
        else:
            results = np.zeros_like(params)
            for i, p in enumerate(params):
                results[i], _ = integrate.quad(
                    lambda t: self.intensity_func(t, p),
                    0, self.time_horizon
                )
            return self.base_rate * results
    
    def mean(self) -> float:
        """E[N] = base_rate * E[∫λ(t,θ)dt]."""
        # Monte Carlo approximation
        n_samples = 10000
        param_samples = self._param_dist.rvs(size=n_samples)
        integrated = self.integrated_intensity(param_samples)
        return np.mean(integrated)
    
    def var(self) -> float:
        """Variance using law of total variance."""
        # Monte Carlo approximation
        n_samples = 10000
        param_samples = self._param_dist.rvs(size=n_samples)
        integrated = self.integrated_intensity(param_samples)
        
        # E[Var[N|θ]] + Var[E[N|θ]]
        conditional_means = integrated
        conditional_vars = integrated  # For Poisson
        
        return np.mean(conditional_vars) + np.var(conditional_means)
    
    def pmf(self, k: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Probability mass function using Monte Carlo integration.
        
        For time-varying intensity, we integrate over the parameter distribution.
        """
        k = np.atleast_1d(k).astype(int)
        result = np.zeros_like(k, dtype=float)
        
        # Monte Carlo integration
        n_samples = 5000
        param_samples = self._param_dist.rvs(size=n_samples)
        lambdas = self.integrated_intensity(param_samples)
        
        for i, ki in enumerate(k):
            if ki < 0:
                result[i] = 0.0
            else:
                # Average Poisson PMF over parameter distribution
                pmf_values = stats.poisson.pmf(ki, lambdas)
                result[i] = np.mean(pmf_values)
        
        return result[0] if np.isscalar(k) else result
    
    def rvs(self, size: int = 1, random_state: Optional[int] = None) -> Union[int, np.ndarray]:
        """Generate random variates."""
        if random_state is not None:
            np.random.seed(random_state)
        
        # Sample parameters
        params = self._param_dist.rvs(size=size)
        
        # Compute integrated intensities
        lambdas = self.integrated_intensity(params)
        
        # Sample Poisson counts
        samples = stats.poisson.rvs(lambdas, size=size)
        
        return samples[0] if size == 1 else samples
    
    def sample_process(self, params: float = None, 
                      random_state: Optional[int] = None) -> np.ndarray:
        """
        Sample actual event times from the process.
        
        Args:
            params: Fixed parameters (if None, sample from distribution)
            random_state: Random seed
            
        Returns:
            Array of event times in [0, time_horizon]
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        if params is None:
            params = self._param_dist.rvs()
        
        # Use thinning algorithm
        lambda_max = self.base_rate * 2  # Upper bound
        
        events = []
        t = 0
        
        while t < self.time_horizon:
            # Time to next potential event
            dt = stats.expon.rvs(scale=1/lambda_max)
            t += dt
            
            if t < self.time_horizon:
                # Accept/reject
                lambda_t = self.base_rate * self.intensity_func(t, params)
                if np.random.rand() < lambda_t / lambda_max:
                    events.append(t)
        
        return np.array(events)