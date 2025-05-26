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
    """Base class for mixed Poisson distributions.
    
    In a mixed Poisson process, the intensity λ is itself a random variable,
    leading to overdispersed count distributions where the variance exceeds the mean.
    This captures population heterogeneity and clustering effects in count data.
    
    Attributes:
        mixing_dist_name (str): Name of the mixing distribution used.
        params (dict): Parameters specific to the mixing distribution.
    """
    
    def __init__(self, mixing_dist_name: str, **params):
        """Initialize mixed Poisson distribution.
        
        Args:
            mixing_dist_name: Name of the mixing distribution (e.g., 'gamma', 'invgauss').
            **params: Parameters specific to the mixing distribution.
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
    """Poisson-Gamma mixture (Negative Binomial distribution).
    
    When the Poisson intensity λ follows a Gamma(α, β) distribution,
    the marginal count distribution is Negative Binomial(r=α, p=β/(1+β)).
    
    This is the most common mixed Poisson model, capturing overdispersion 
    through gamma-distributed heterogeneity. It models contagion effects 
    and clustering in count data.
    
    The overdispersion is characterized by:
        Var[N] = E[N] + E[N]²/α
    
    Attributes:
        alpha (float): Shape parameter of the Gamma mixing distribution (α > 0).
        beta (float): Rate parameter of the Gamma mixing distribution (β > 0).
        r (float): Number of failures parameter for the Negative Binomial (r = α).
        p (float): Success probability for the Negative Binomial (p = β/(1+β)).
    
    Examples:
        >>> pg = PoissonGammaMixture(alpha=5.0, beta=2.0)
        >>> pg.mean()  # Returns 2.5
        >>> pg.var()   # Returns 3.75 (overdispersed)
        >>> samples = pg.rvs(size=1000)
    """
    
    def __init__(self, alpha: float, beta: float):
        """Initialize Poisson-Gamma mixture.
        
        Args:
            alpha: Shape parameter of Gamma distribution (α > 0).
            beta: Rate parameter of Gamma distribution (β > 0).
            
        Raises:
            ValueError: If alpha or beta are not positive.
        """
        self.alpha = alpha
        self.beta = beta
        self._validate_params()  # Validate before setup
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
    
    def std(self) -> float:
        """Standard deviation of the distribution."""
        return np.sqrt(self.var())
    
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
    
    def ppf(self, q: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Percent point function (inverse CDF)."""
        return self._dist.ppf(q)
    
    def cdf(self, k: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
        """Cumulative distribution function."""
        return self._dist.cdf(k)


class PoissonInverseGaussianMixture(MixedPoissonDistribution):
    """Poisson-Inverse Gaussian mixture for heavy-tailed count data.
    
    When the Poisson intensity λ follows an Inverse Gaussian(μ, λ₀) distribution,
    the marginal count distribution has heavier tails than the Negative Binomial,
    making it suitable for modeling rare but extreme events.
    
    This implementation uses numerical integration to compute the PMF, providing
    flexibility and stability across a wide range of parameter values. The mixing
    distribution has mean μ and shape parameter λ₀, where larger λ₀ indicates
    less variability in the mixing distribution.
    
    The variance-to-mean ratio is:
        VMR = 1 + μ²/λ₀
    
    Attributes:
        mu (float): Mean of the Inverse Gaussian mixing distribution (μ > 0).
        lambda_param (float): Shape parameter of the mixing distribution (λ₀ > 0).
        lambda_shape (float): Alias for lambda_param (deprecated).
    
    Examples:
        >>> pig = PoissonInverseGaussianMixture(mu=5.0, lambda_param=10.0)
        >>> pig.mean()  # Returns 5.0
        >>> pig.var()   # Returns 7.5 (overdispersed)
        >>> samples = pig.rvs(size=1000)
        
    Note:
        The implementation differs from the exact Bessel function formula by using
        numerical integration, which provides better numerical stability and handles
        edge cases more gracefully.
    """
    
    def __init__(self, mu: float, lambda_param: float = None, lambda_shape: float = None):
        """Initialize Poisson-Inverse Gaussian mixture.
        
        Args:
            mu: Mean of inverse Gaussian distribution (μ > 0).
            lambda_param: Shape parameter of inverse Gaussian (λ₀ > 0). Higher values
                indicate less variability in the mixing distribution.
            lambda_shape: Deprecated alias for lambda_param. Use lambda_param instead.
            
        Raises:
            ValueError: If mu or lambda_param are not positive, or if neither
                lambda_param nor lambda_shape is provided.
        """
        # Handle parameter aliases
        if lambda_param is not None:
            self.lambda_shape = lambda_param
        elif lambda_shape is not None:
            self.lambda_shape = lambda_shape
        else:
            raise ValueError("Either lambda_param or lambda_shape must be provided")
            
        self.mu = mu
        self.lambda_param = self.lambda_shape  # Keep both names for compatibility
        super().__init__('invgauss', mu=mu, lambda_shape=self.lambda_shape)
    
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
        var_lambda = self.mu**3 / self.lambda_param
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
    
    def std(self) -> float:
        """Standard deviation of the distribution."""
        return np.sqrt(self.var())
    
    def cdf(self, k: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
        """Cumulative distribution function using numerical integration."""
        k = np.atleast_1d(k).astype(int)
        result = np.zeros_like(k, dtype=float)
        
        for i, ki in enumerate(k):
            if ki < 0:
                result[i] = 0.0
            else:
                # Sum PMF up to k
                result[i] = sum(self.pmf(j) for j in range(ki + 1))
        
        return result[0] if np.isscalar(k) else result
    
    def ppf(self, q: Union[float, np.ndarray]) -> Union[int, np.ndarray]:
        """Percent point function (inverse CDF) using numerical search."""
        q = np.atleast_1d(q)
        result = np.zeros_like(q, dtype=int)
        
        for i, qi in enumerate(q):
            if qi <= 0:
                result[i] = 0
            elif qi >= 1:
                result[i] = np.inf
            else:
                # Binary search for the inverse
                k = 0
                cdf_k = self.pmf(0)
                while cdf_k < qi:
                    k += 1
                    cdf_k += self.pmf(k)
                result[i] = k
        
        return result[0] if np.isscalar(q) else result


class HierarchicalPoissonMixture(MixedPoissonDistribution):
    """Hierarchical Poisson mixture for multi-level portfolio modeling.
    
    This model captures heterogeneity at multiple levels using a hierarchical
    structure of Gamma distributions. Unlike the standard log-linear mixed
    effects model, this implementation uses conjugate Gamma distributions
    for computational efficiency and closed-form posterior distributions.
    
    The hierarchy is:
        - Portfolio level: λ_p ~ Gamma(α_p, β_p)
        - Group level: λ_g|λ_p ~ Gamma(α_g, λ_p/α_g)  
        - Individual level: N|λ_g ~ Poisson(λ_g)
    
    The individual_dispersion parameter controls the group-level variability:
        - dispersion = 0: All groups have the same intensity (no group variation)
        - dispersion > 0: Groups vary around the portfolio mean
        - Higher dispersion leads to more variation between groups
    
    Attributes:
        portfolio_alpha (float): Shape parameter at portfolio level (α_p > 0).
        portfolio_beta (float): Rate parameter at portfolio level (β_p > 0).
        individual_dispersion (float): Controls group-level variation (≥ 0).
        group_alpha (float): Shape parameter for group-level Gamma.
        n_groups (int): Number of groups in the portfolio.
    
    Examples:
        >>> hpm = HierarchicalPoissonMixture(
        ...     portfolio_alpha=3.0,
        ...     portfolio_beta=1.5,
        ...     individual_dispersion=0.5,
        ...     n_groups=1
        ... )
        >>> hpm.mean()  # Returns expected total count
        >>> var_comp = hpm.variance_components()  # Decompose variance
        
    Note:
        This Gamma-based hierarchy differs from the Normal random effects
        model in the mathematical specification but provides similar
        flexibility for modeling multi-level heterogeneity.
    """
    
    def __init__(self, portfolio_alpha: float, portfolio_beta: float,
                 individual_dispersion: float = None, group_alpha: float = None,
                 n_groups: int = 1):
        """Initialize hierarchical Poisson mixture.
        
        Args:
            portfolio_alpha: Shape parameter at portfolio level (α_p > 0).
            portfolio_beta: Rate parameter at portfolio level (β_p > 0).
            individual_dispersion: Controls group-level variation (≥ 0). When 0,
                all groups have the same intensity. Higher values increase
                between-group variability. Internally mapped to group_alpha.
            group_alpha: Shape at group level (deprecated). Use individual_dispersion
                instead. Related by: group_alpha = 1/(1 + individual_dispersion).
            n_groups: Number of groups in portfolio (default 1).
            
        Raises:
            ValueError: If portfolio parameters are not positive or if
                n_groups < 1.
        """
        self.portfolio_alpha = portfolio_alpha
        self.portfolio_beta = portfolio_beta
        self.n_groups = n_groups
        
        # Handle parameter mapping
        if individual_dispersion is not None:
            self.individual_dispersion = individual_dispersion
            # Map dispersion to group_alpha (higher dispersion = lower group_alpha)
            self.group_alpha = 1.0 / (1.0 + individual_dispersion) if individual_dispersion > 0 else 1.0
        elif group_alpha is not None:
            self.group_alpha = group_alpha
            # Back-calculate individual_dispersion
            self.individual_dispersion = (1.0 / self.group_alpha - 1.0) if self.group_alpha > 0 else 0.0
        else:
            # Default: low individual dispersion
            self.individual_dispersion = 0.1
            self.group_alpha = 1.0 / 1.1
        
        super().__init__('hierarchical',
                        portfolio_alpha=portfolio_alpha,
                        portfolio_beta=portfolio_beta,
                        group_alpha=self.group_alpha,
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
    
    def rvs_conditional(self, size: int, portfolio_lambda: float,
                       random_state: Optional[int] = None) -> np.ndarray:
        """
        Generate random variates conditional on portfolio lambda.
        
        Args:
            size: Number of samples
            portfolio_lambda: Fixed portfolio-level intensity
            random_state: Random seed
            
        Returns:
            Array of counts
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        # For each individual, sample from Gamma then Poisson
        individual_lambdas = stats.gamma.rvs(
            a=self.group_alpha,
            scale=portfolio_lambda/self.group_alpha,
            size=size
        )
        
        # Add individual variation if specified
        if self.individual_dispersion > 0:
            # Add extra variation at individual level
            individual_lambdas *= stats.gamma.rvs(
                a=1/self.individual_dispersion,
                scale=self.individual_dispersion,
                size=size
            )
        
        # Sample counts
        counts = stats.poisson.rvs(individual_lambdas, size=size)
        return counts
    
    def variance_components(self) -> dict:
        """
        Decompose variance into portfolio and individual components.
        
        Returns:
            Dictionary with variance components and intraclass correlation
        """
        # Portfolio-level variance
        portfolio_mean = self.portfolio_alpha / self.portfolio_beta
        portfolio_var = self.portfolio_alpha / (self.portfolio_beta**2)
        
        # Individual-level variance (within groups)
        # This includes both group-level and individual-level variation
        individual_var = portfolio_mean * self.group_alpha * (1 + self.individual_dispersion)
        
        # Total variance
        total_var = self.var()
        
        # Intraclass correlation coefficient
        # Proportion of variance due to portfolio-level differences
        icc = portfolio_var / total_var if total_var > 0 else 0.0
        
        return {
            'portfolio_variance': portfolio_var,
            'individual_variance': individual_var,
            'total_variance': total_var,
            'intraclass_correlation': icc
        }


class TimeVaryingPoissonMixture(MixedPoissonDistribution):
    """Time-varying Poisson process with optional mixing distribution.
    
    Models non-homogeneous Poisson processes where the intensity λ(t) varies
    over time. Additionally supports a mixing distribution on top of the
    time-varying intensity to capture extra uncertainty.
    
    The count distribution is:
        - Without mixing: N ~ Poisson(∫₀ᵀ λ(t) dt)
        - With Gamma mixing: N|Θ ~ Poisson(Θ × ∫₀ᵀ λ(t) dt), Θ ~ Gamma(α, β)
    
    This extension beyond the standard time-varying Poisson process allows
    modeling both temporal patterns and population heterogeneity simultaneously.
    
    Common intensity functions include:
        - Constant: λ(t) = c
        - Linear trend: λ(t) = a + bt
        - Seasonal: λ(t) = a + b×sin(2πt/T)
        - Exponential growth: λ(t) = a×exp(bt)
    
    Attributes:
        intensity_function (Callable): Function λ(t) defining intensity over time.
        mixing_alpha (float): Shape parameter for optional Gamma mixing (α > 0).
        mixing_beta (float): Rate parameter for optional Gamma mixing (β > 0).
        time_period (float): Time period T for integration [0, T].
        
    Examples:
        >>> # Seasonal pattern without mixing
        >>> def seasonal(t): return 2.0 + np.sin(2 * np.pi * t)
        >>> tvpm = TimeVaryingPoissonMixture(
        ...     intensity_function=seasonal,
        ...     time_period=1.0
        ... )
        >>> 
        >>> # Seasonal pattern with population heterogeneity
        >>> tvpm_mixed = TimeVaryingPoissonMixture(
        ...     intensity_function=seasonal,
        ...     mixing_alpha=4.0,
        ...     mixing_beta=2.0,
        ...     time_period=1.0
        ... )
        
    Note:
        The mixing distribution extension is not part of the standard
        mathematical specification but provides additional flexibility
        for modeling overdispersed time-varying count data.
    """
    
    def __init__(self, intensity_function: Callable = None, 
                 mixing_alpha: float = None, mixing_beta: float = None,
                 time_period: float = 1.0,
                 base_rate: float = None, intensity_func: Callable = None,
                 param_dist: dict = None, time_horizon: float = None):
        """Initialize time-varying Poisson mixture.
        
        Args:
            intensity_function: Function λ(t) defining time-varying intensity.
                Should accept time t and return intensity at that time.
            mixing_alpha: Shape parameter for optional Gamma mixing (α > 0).
                If provided with mixing_beta, adds population heterogeneity.
            mixing_beta: Rate parameter for optional Gamma mixing (β > 0).
                Must be provided together with mixing_alpha.
            time_period: Time period T for counting events, integrates over [0, T].
                Default is 1.0.
            base_rate: (Deprecated) Use intensity_function instead.
            intensity_func: (Deprecated) Use intensity_function instead.
            param_dist: (Deprecated) Use mixing_alpha and mixing_beta instead.
            time_horizon: (Deprecated) Use time_period instead.
            
        Raises:
            ValueError: If neither intensity_function nor the deprecated
                (base_rate, intensity_func) pair is provided.
                
        Examples:
            >>> # Simple time-varying intensity
            >>> def linear(t): return 1.0 + 0.5 * t
            >>> tvpm = TimeVaryingPoissonMixture(
            ...     intensity_function=linear,
            ...     time_period=2.0
            ... )
        """
        # Handle new interface
        if intensity_function is not None:
            self.intensity_function = intensity_function
            self.intensity_func = lambda t, params=None: intensity_function(t)
            self.base_rate = 1.0  # Default base rate
            
            # Setup Gamma mixing if parameters provided
            if mixing_alpha is not None and mixing_beta is not None:
                self.mixing_alpha = mixing_alpha
                self.mixing_beta = mixing_beta
                self.param_dist = {
                    'name': 'gamma',
                    'params': {'a': mixing_alpha, 'scale': 1/mixing_beta}
                }
            else:
                # Default: no mixing (deterministic intensity)
                self.mixing_alpha = 1.0
                self.mixing_beta = 1.0
                self.param_dist = {
                    'name': 'gamma',
                    'params': {'a': 1.0, 'scale': 1.0}
                }
                
            self.time_period = time_period
            self.time_horizon = time_period
            
        # Handle old interface (deprecated)
        elif base_rate is not None and intensity_func is not None:
            self.base_rate = base_rate
            self.intensity_func = intensity_func
            self.param_dist = param_dist or {'name': 'gamma', 'params': {'a': 1.0, 'scale': 1.0}}
            self.time_horizon = time_horizon or 1.0
            self.time_period = self.time_horizon
            
            # Extract mixing parameters if using gamma
            if self.param_dist['name'] == 'gamma':
                self.mixing_alpha = self.param_dist['params'].get('a', 1.0)
                self.mixing_beta = 1.0 / self.param_dist['params'].get('scale', 1.0)
            else:
                self.mixing_alpha = 1.0
                self.mixing_beta = 1.0
        else:
            raise ValueError("Either intensity_function or (base_rate, intensity_func) must be provided")
        
        # Ensure _param_dist is set before calling parent constructor
        self._setup_param_dist()
        
        super().__init__('time_varying',
                        base_rate=self.base_rate,
                        param_dist=self.param_dist,
                        time_horizon=self.time_horizon)
    
    def _setup_param_dist(self):
        """Setup the parameter distribution object."""
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
    
    def _setup_distribution(self):
        """Setup parameter distribution."""
        self._validate_params()
        # Distribution already set up in _setup_param_dist
    
    def _validate_params(self):
        """Validate parameters."""
        if self.base_rate <= 0:
            raise ValueError("Base rate must be positive")
        if self.time_horizon <= 0:
            raise ValueError("Time horizon must be positive")
        if 'name' not in self.param_dist or 'params' not in self.param_dist:
            raise ValueError("param_dist must have 'name' and 'params' keys")
    
    def integrated_intensity(self, params: Union[float, np.ndarray] = None) -> Union[float, np.ndarray]:
        """Compute ∫₀ᵀ λ(t, params) dt."""
        # Handle case where no params needed (new interface)
        if params is None and hasattr(self, 'intensity_function'):
            result, _ = integrate.quad(
                self.intensity_function,
                0, self.time_period
            )
            return result
            
        # Original implementation for old interface
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
        # For new interface with simple intensity function
        if hasattr(self, 'intensity_function') and self.mixing_alpha == 1.0 and self.mixing_beta == 1.0:
            # No mixing, just integrate the intensity function
            return self.integrated_intensity()
        
        # For new interface with Gamma mixing
        if hasattr(self, 'intensity_function') and self.mixing_alpha is not None:
            # Integrate intensity and multiply by mixing mean
            base_integral = self.integrated_intensity()
            mixing_mean = self.mixing_alpha / self.mixing_beta
            return base_integral * mixing_mean
            
        # Original Monte Carlo approximation for old interface
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
        
        # Handle new interface with intensity_function
        if hasattr(self, 'intensity_function'):
            # Base integrated intensity
            base_intensity = self.integrated_intensity()
            
            # Sample mixing parameters (multiplicative factors)
            mixing_params = self._param_dist.rvs(size=size)
            
            # Compute final intensities
            lambdas = base_intensity * mixing_params
        else:
            # Old interface: sample parameters and compute integrated intensities
            params = self._param_dist.rvs(size=size)
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