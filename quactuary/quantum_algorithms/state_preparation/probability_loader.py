"""
Probability distribution loading utilities for quantum state preparation.

This module provides functions to prepare quantum states from various
probability distributions commonly used in actuarial modeling.
"""

import numpy as np
from typing import Tuple, Optional, Union, Callable
from scipy import stats
from scipy.special import erf


def prepare_lognormal_state(
    mu: float = 0.0, 
    sigma: float = 1.0, 
    num_qubits: int = 6,
    domain_min: float = 0.1,
    domain_max: float = 10.0,
    use_analytic_binning: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare lognormal distribution for quantum encoding.
    
    Based on the quantum excess loss algorithm approach, this function
    discretizes a lognormal distribution into bins suitable for quantum
    amplitude encoding.
    
    Args:
        mu: Scale parameter (mean of underlying normal).
        sigma: Shape parameter (std dev of underlying normal).
        num_qubits: Number of qubits determining discretization (2^n bins).
        domain_min: Minimum value of the domain.
        domain_max: Maximum value of the domain.
        use_analytic_binning: If True, use exact CDF integration for bins.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (probabilities, x_values)
            - probabilities: Normalized probability for each bin
            - x_values: Center points of each bin
            
    Examples:
        >>> # Standard lognormal
        >>> probs, x_vals = prepare_lognormal_state(mu=0, sigma=1, num_qubits=4)
        >>> print(f"Number of bins: {len(probs)}")
        Number of bins: 16
        >>> print(f"Sum of probabilities: {np.sum(probs):.6f}")
        Sum of probabilities: 1.000000
        
        >>> # Higher resolution
        >>> probs, x_vals = prepare_lognormal_state(num_qubits=8)
        >>> print(f"Resolution: {len(probs)} bins")
        Resolution: 256 bins
    """
    # Number of discretization points
    n_points = 2**num_qubits
    
    # Create bin edges
    bin_edges = np.linspace(domain_min, domain_max, n_points + 1)
    
    # Calculate bin centers
    x_values = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    if use_analytic_binning:
        # Use exact integration over each bin using CDF
        # This is more accurate than point evaluation
        
        # Helper function for normal CDF
        def phi(z):
            return 0.5 * (1 + erf(z / np.sqrt(2)))
        
        # Calculate probabilities for each bin
        probabilities = np.zeros(n_points)
        
        for i in range(n_points):
            # Integrate lognormal PDF from bin_edges[i] to bin_edges[i+1]
            if bin_edges[i] <= 0:
                # Handle edge case at zero
                z_low = -np.inf
            else:
                z_low = (np.log(bin_edges[i]) - mu) / sigma
            
            z_high = (np.log(bin_edges[i+1]) - mu) / sigma
            
            # P(a < X < b) = Φ((ln(b) - μ)/σ) - Φ((ln(a) - μ)/σ)
            probabilities[i] = phi(z_high) - phi(z_low)
    else:
        # Simple point evaluation at bin centers
        dist = stats.lognorm(s=sigma, scale=np.exp(mu))
        probabilities = dist.pdf(x_values)
        
        # Normalize to account for discretization
        probabilities = probabilities / np.sum(probabilities)
    
    # Ensure normalization
    prob_sum = np.sum(probabilities)
    if prob_sum > 0:
        probabilities = probabilities / prob_sum
    else:
        # Fallback to uniform if something went wrong
        probabilities = np.ones(n_points) / n_points
    
    return probabilities, x_values


def prepare_distribution_state(
    distribution_name: str,
    params: dict,
    num_qubits: int = 6,
    domain_min: Optional[float] = None,
    domain_max: Optional[float] = None,
    use_analytic: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare quantum state from common actuarial distributions.
    
    Supports various distributions used in insurance modeling:
    - lognormal: Loss severity modeling
    - gamma: Claim amounts
    - pareto: Heavy-tailed losses
    - weibull: Failure times
    - exponential: Waiting times
    - beta: Loss ratios
    
    Args:
        distribution_name: Name of the distribution ('lognormal', 'gamma', etc.).
        params: Dictionary of distribution parameters.
        num_qubits: Number of qubits for discretization.
        domain_min: Minimum domain value (auto-determined if None).
        domain_max: Maximum domain value (auto-determined if None).
        use_analytic: Use analytic integration where available.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (probabilities, x_values)
        
    Examples:
        >>> # Lognormal for severity
        >>> probs, x = prepare_distribution_state(
        ...     'lognormal', 
        ...     {'mu': 7, 'sigma': 1.5},
        ...     num_qubits=6
        ... )
        
        >>> # Gamma for aggregate losses
        >>> probs, x = prepare_distribution_state(
        ...     'gamma',
        ...     {'alpha': 2, 'beta': 1/1000},  # Mean = 2000
        ...     num_qubits=7
        ... )
        
        >>> # Pareto for extreme losses
        >>> probs, x = prepare_distribution_state(
        ...     'pareto',
        ...     {'alpha': 2.5, 'scale': 1000},
        ...     domain_max=100000
        ... )
    """
    # Get the scipy distribution
    dist_map = {
        'lognormal': (stats.lognorm, ['s', 'scale']),
        'gamma': (stats.gamma, ['a', 'scale']),
        'pareto': (stats.pareto, ['b', 'scale']),
        'weibull': (stats.weibull_min, ['c', 'scale']),
        'exponential': (stats.expon, ['scale']),
        'beta': (stats.beta, ['a', 'b']),
        'uniform': (stats.uniform, ['loc', 'scale']),
    }
    
    if distribution_name not in dist_map:
        raise ValueError(
            f"Unknown distribution: {distribution_name}. "
            f"Supported: {list(dist_map.keys())}"
        )
    
    # Special handling for lognormal (use optimized function)
    if distribution_name == 'lognormal':
        mu = params.get('mu', 0)
        sigma = params.get('sigma', params.get('s', 1))
        return prepare_lognormal_state(
            mu, sigma, num_qubits, 
            domain_min or 0.1, 
            domain_max or 10.0,
            use_analytic
        )
    
    # General distribution handling
    dist_class, param_names = dist_map[distribution_name]
    
    # Map parameters
    scipy_params = {}
    if distribution_name == 'gamma':
        scipy_params['a'] = params.get('alpha', params.get('a', 2))
        scipy_params['scale'] = params.get('beta', params.get('scale', 1))
    elif distribution_name == 'pareto':
        scipy_params['b'] = params.get('alpha', params.get('b', 2))
        scipy_params['scale'] = params.get('scale', 1)
    elif distribution_name == 'beta':
        scipy_params['a'] = params.get('alpha', params.get('a', 2))
        scipy_params['b'] = params.get('beta', params.get('b', 2))
    else:
        scipy_params = {k: params.get(k) for k in param_names if k in params}
    
    # Create distribution
    dist = dist_class(**scipy_params)
    
    # Determine domain if not specified
    if domain_min is None:
        domain_min = dist.ppf(0.001)  # 0.1% quantile
    if domain_max is None:
        domain_max = dist.ppf(0.999)  # 99.9% quantile
    
    # Discretize
    return discretize_distribution(
        dist, num_qubits, domain_min, domain_max, use_analytic
    )


def discretize_distribution(
    distribution: stats.rv_continuous,
    num_qubits: int,
    domain_min: float,
    domain_max: float,
    use_analytic: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Discretize a continuous distribution for quantum encoding.
    
    Args:
        distribution: Scipy continuous distribution object.
        num_qubits: Number of qubits (determines resolution).
        domain_min: Minimum domain value.
        domain_max: Maximum domain value.
        use_analytic: Use CDF integration for accuracy.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (probabilities, x_values)
        
    Examples:
        >>> # Discretize a normal distribution
        >>> dist = stats.norm(loc=100, scale=15)
        >>> probs, x = discretize_distribution(dist, 6, 50, 150)
        >>> print(f"Captured probability: {np.sum(probs):.4f}")
    """
    n_bins = 2**num_qubits
    
    # Create bins
    bin_edges = np.linspace(domain_min, domain_max, n_bins + 1)
    x_values = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    if use_analytic:
        # Use CDF for exact integration
        cdf_values = distribution.cdf(bin_edges)
        probabilities = np.diff(cdf_values)
    else:
        # Use PDF evaluation at centers
        probabilities = distribution.pdf(x_values)
        # Account for bin width
        bin_width = (domain_max - domain_min) / n_bins
        probabilities = probabilities * bin_width
    
    # Normalize
    total = np.sum(probabilities)
    if total > 0:
        probabilities = probabilities / total
    else:
        probabilities = np.ones(n_bins) / n_bins
    
    return probabilities, x_values


def prepare_empirical_distribution(
    data: Union[np.ndarray, list],
    num_qubits: int,
    domain_min: Optional[float] = None,
    domain_max: Optional[float] = None,
    kde_bandwidth: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare quantum state from empirical data using kernel density estimation.
    
    Args:
        data: Empirical data samples.
        num_qubits: Number of qubits for encoding.
        domain_min: Minimum domain (uses data min if None).
        domain_max: Maximum domain (uses data max if None).
        kde_bandwidth: Bandwidth for KDE (auto if None).
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (probabilities, x_values)
        
    Examples:
        >>> # From loss data
        >>> losses = np.random.lognormal(7, 1.5, 1000)
        >>> probs, x = prepare_empirical_distribution(losses, num_qubits=6)
    """
    data = np.array(data)
    
    # Determine domain
    if domain_min is None:
        domain_min = np.min(data) * 0.9
    if domain_max is None:
        domain_max = np.max(data) * 1.1
    
    # Create KDE
    kde = stats.gaussian_kde(data, bw_method=kde_bandwidth)
    
    # Discretize
    n_bins = 2**num_qubits
    x_values = np.linspace(domain_min, domain_max, n_bins)
    probabilities = kde(x_values)
    
    # Normalize
    probabilities = probabilities / np.sum(probabilities)
    
    return probabilities, x_values