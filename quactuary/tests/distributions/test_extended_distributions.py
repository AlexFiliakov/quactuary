"""
Comprehensive tests for extended distribution support.
"""

import numpy as np
import pytest
from scipy import stats

from quactuary.distributions.frequency import Poisson, Binomial, NegativeBinomial
from quactuary.distributions.severity import Exponential, Gamma, Lognormal
from quactuary.distributions.mixed_poisson import (
    PoissonGammaMixture,
    PoissonInverseGaussianMixture,
    HierarchicalPoissonMixture,
    TimeVaryingPoissonMixture
)
from quactuary.distributions.zero_inflated import (
    ZeroInflatedCompound,
    detect_zero_inflation
)
from quactuary.distributions.edgeworth import (
    EdgeworthExpansion,
    CompoundDistributionEdgeworth,
    automatic_order_selection
)
from quactuary.distributions.compound_extensions import (
    create_extended_compound_distribution,
    distribution_selection_guide
)


class TestMixedPoissonDistributions:
    """Test mixed Poisson processes."""
    
    def test_poisson_gamma_mixture(self):
        """Test Poisson-Gamma mixture (Negative Binomial)."""
        alpha, beta = 3.0, 0.5
        mixed = PoissonGammaMixture(alpha=alpha, beta=beta)
        
        # Check mean and variance
        expected_mean = alpha / beta
        expected_var = alpha / beta + alpha / (beta**2)
        
        assert np.isclose(mixed.mean(), expected_mean)
        assert np.isclose(mixed.var(), expected_var)
        
        # Check it matches negative binomial
        r = alpha
        p = beta / (1 + beta)
        nbinom = stats.nbinom(n=r, p=p)
        
        # Compare PMF at several points
        for k in [0, 1, 5, 10, 20]:
            assert np.isclose(mixed.pmf(k), nbinom.pmf(k))
    
    def test_poisson_inverse_gaussian(self):
        """Test Poisson-Inverse Gaussian mixture."""
        mu, lambda_shape = 5.0, 2.0
        mixed = PoissonInverseGaussianMixture(mu=mu, lambda_shape=lambda_shape)
        
        # Check mean
        assert np.isclose(mixed.mean(), mu)
        
        # Check variance (should be larger than Poisson)
        poisson_var = mu
        assert mixed.var() > poisson_var
        
        # Test PMF computation
        pmf_vals = mixed.pmf([0, 1, 5, 10])
        assert np.all(pmf_vals >= 0)
        assert np.sum(pmf_vals) <= 1
        
        # Test sampling
        samples = mixed.rvs(size=1000, random_state=42)
        assert np.all(samples >= 0)
        assert np.all(samples == samples.astype(int))
    
    def test_hierarchical_poisson(self):
        """Test hierarchical Poisson mixture."""
        hier = HierarchicalPoissonMixture(
            portfolio_alpha=2.0,
            portfolio_beta=0.5,
            group_alpha=3.0,
            n_groups=5
        )
        
        # Test basic properties
        assert hier.mean() > 0
        assert hier.var() > hier.mean()  # Overdispersion
        
        # Test portfolio simulation
        sim = hier.simulate_portfolio(size=100, random_state=42)
        
        assert 'total' in sim
        assert 'by_group' in sim
        assert 'lambda_p' in sim
        
        assert sim['by_group'].shape == (100, 5)
        assert np.all(sim['total'] == np.sum(sim['by_group'], axis=1))
    
    def test_time_varying_poisson(self):
        """Test time-varying Poisson mixture."""
        # Sinusoidal intensity function
        def intensity_func(t, amplitude):
            return 1 + amplitude * np.sin(2 * np.pi * t)
        
        tv_poisson = TimeVaryingPoissonMixture(
            base_rate=10.0,
            intensity_func=intensity_func,
            param_dist={'name': 'beta', 'params': {'a': 2, 'b': 2}},
            time_horizon=1.0
        )
        
        # Test mean computation
        mean_val = tv_poisson.mean()
        assert 5 < mean_val < 15  # Reasonable range
        
        # Test event sampling
        events = tv_poisson.sample_process(params=0.5, random_state=42)
        assert np.all(events >= 0)
        assert np.all(events <= 1.0)
        assert len(events) > 0


class TestZeroInflatedDistributions:
    """Test zero-inflated compound distributions."""
    
    def test_zero_inflated_basic(self):
        """Test basic zero-inflated compound distribution."""
        freq = Poisson(mu=5.0)
        sev = Exponential(scale=1000)
        zi_compound = ZeroInflatedCompound(freq, sev, zero_prob=0.3)
        
        # Check mean and variance
        base_mean = freq._dist.mean() * sev._dist.mean()
        assert np.isclose(zi_compound.mean(), 0.7 * base_mean)
        
        # Check PDF at zero
        base_p0 = np.exp(-5)  # Poisson P(N=0)
        total_p0 = 0.3 + 0.7 * base_p0
        assert np.isclose(zi_compound.pdf(0), total_p0)
        
        # Check CDF properties
        assert zi_compound.cdf(-1) == 0
        assert zi_compound.cdf(0) >= zi_compound.pdf(0)
        assert zi_compound.cdf(np.inf) == 1
    
    def test_zero_inflation_em_algorithm(self):
        """Test EM algorithm for parameter estimation."""
        # Generate data with known zero-inflation
        np.random.seed(42)
        true_zero_prob = 0.25
        n_samples = 1000
        
        # Generate zero-inflated data
        is_zero = np.random.rand(n_samples) < true_zero_prob
        data = np.zeros(n_samples)
        
        freq = Poisson(mu=3.0)
        sev = Gamma(a=2.0, scale=500)
        
        # Generate non-zero values
        for i in range(n_samples):
            if not is_zero[i]:
                n_claims = freq.rvs()
                if n_claims > 0:
                    data[i] = np.sum(sev.rvs(size=n_claims))
        
        # Fit model
        zi_compound = ZeroInflatedCompound(freq, sev, zero_prob=0.1)
        fit_result = zi_compound.fit_em(data, verbose=False)
        
        # Check convergence
        assert fit_result['converged']
        
        # Check estimated zero_prob is reasonable
        assert 0.15 < fit_result['zero_prob'] < 0.35
    
    def test_zero_inflation_detection(self):
        """Test statistical detection of zero-inflation."""
        freq = Poisson(mu=2.0)
        sev = Exponential(scale=1000)
        
        # Generate standard compound data
        np.random.seed(42)
        standard_data = []
        for _ in range(500):
            n = freq.rvs()
            if n > 0:
                standard_data.append(np.sum(sev.rvs(size=n)))
            else:
                standard_data.append(0)
        
        # Test should not detect zero-inflation
        is_zi, diag = detect_zero_inflation(
            np.array(standard_data), freq, sev
        )
        assert not is_zi
        
        # Add extra zeros
        zi_data = np.concatenate([standard_data, np.zeros(200)])
        
        # Test should detect zero-inflation
        is_zi, diag = detect_zero_inflation(
            np.array(zi_data), freq, sev
        )
        assert is_zi
        assert diag['excess_zeros'] > 0


class TestEdgeworthExpansion:
    """Test Edgeworth expansion approximations."""
    
    def test_edgeworth_basic(self):
        """Test basic Edgeworth expansion."""
        # Create expansion for a known distribution
        mean, var = 100, 400
        skewness = 0.5
        excess_kurtosis = 0.3
        
        edgeworth = EdgeworthExpansion(
            mean=mean,
            variance=var,
            skewness=skewness,
            excess_kurtosis=excess_kurtosis
        )
        
        # Test PDF properties
        x = np.linspace(mean - 3*np.sqrt(var), mean + 3*np.sqrt(var), 100)
        pdf_vals = edgeworth.pdf(x, order=4)
        
        # Should be mostly positive
        assert np.sum(pdf_vals < 0) < 5  # Allow few negative values
        
        # Should integrate to approximately 1
        integral = np.trapz(pdf_vals, x)
        assert 0.95 < integral < 1.05
    
    def test_edgeworth_validation(self):
        """Test Edgeworth expansion validation."""
        # Good case - small skewness and kurtosis
        good_edge = EdgeworthExpansion(
            mean=50, variance=100,
            skewness=0.3, excess_kurtosis=0.2
        )
        diag = good_edge.validate_expansion()
        assert diag['valid']
        
        # Bad case - large skewness
        bad_edge = EdgeworthExpansion(
            mean=50, variance=100,
            skewness=2.5, excess_kurtosis=5.0
        )
        diag = bad_edge.validate_expansion()
        assert not diag['valid']
    
    def test_cornish_fisher_expansion(self):
        """Test Cornish-Fisher quantile approximation."""
        edgeworth = EdgeworthExpansion(
            mean=1000, variance=10000,
            skewness=0.6, excess_kurtosis=0.4
        )
        
        # Test quantiles
        quantiles = [0.05, 0.1, 0.5, 0.9, 0.95]
        cf_quantiles = edgeworth.ppf(quantiles, method='cornish-fisher')
        
        # Should be monotonic
        assert np.all(np.diff(cf_quantiles) > 0)
        
        # Median should be close to mean for moderate skewness
        assert abs(cf_quantiles[2] - 1000) < 50
    
    def test_automatic_order_selection(self):
        """Test automatic order selection."""
        # Small sample, moderate moments
        order = automatic_order_selection(
            skewness=0.4,
            excess_kurtosis=0.3,
            sample_size=30
        )
        assert order == 2
        
        # Large sample, small moments
        order = automatic_order_selection(
            skewness=0.2,
            excess_kurtosis=0.1,
            sample_size=1000
        )
        assert order == 2
        
        # Large sample, moderate moments
        order = automatic_order_selection(
            skewness=0.8,
            excess_kurtosis=1.5,
            sample_size=500
        )
        assert order == 4


class TestIntegration:
    """Test integration of all components."""
    
    def test_extended_factory_function(self):
        """Test extended factory with various options."""
        # Standard compound
        compound1 = create_extended_compound_distribution(
            'poisson', 'exponential',
            mu=5.0, scale=1000
        )
        assert compound1.mean() == 5000
        
        # Zero-inflated
        compound2 = create_extended_compound_distribution(
            'poisson', 'gamma',
            zero_inflated=True,
            zero_prob=0.2,
            mu=3.0, a=2.0, scale=500
        )
        assert isinstance(compound2, ZeroInflatedCompound)
        
        # With Edgeworth
        compound3 = create_extended_compound_distribution(
            'binomial', 'lognormal',
            use_edgeworth=True,
            n=10, p=0.4, mu=6.0, sigma=1.0
        )
        assert hasattr(compound3, 'edgeworth')
    
    def test_distribution_selection_guide(self):
        """Test distribution selection guidance."""
        # High frequency variability
        guide = distribution_selection_guide(
            mean_frequency=10,
            cv_frequency=2.0,
            mean_severity=1000,
            cv_severity=0.8,
            has_zeros=True,
            sample_size=50
        )
        
        assert "Mixed Poisson" in guide
        assert "zero-inflated" in guide
        assert "Small sample" in guide
    
    def test_parallel_simulation(self):
        """Test parallel computation support."""
        freq = Poisson(mu=10)
        sev = Lognormal(s=1.0, scale=1000)
        
        # Create optimized compound with parallel support
        from quactuary.distributions.compound_extensions import SimulatedCompoundOptimized
        
        compound = SimulatedCompoundOptimized(
            freq, sev,
            cache_size=10000,
            parallel=True
        )
        
        # Force cache generation
        compound._ensure_cache()
        
        # Check cache statistics
        assert compound._cache_stats is not None
        assert 'quantiles' in compound._cache_stats
        
        # Test methods work correctly
        mean_est = compound.mean()
        assert 5000 < mean_est < 15000  # Reasonable range


def test_parameter_boundaries():
    """Test parameter boundary conditions."""
    # Zero probability boundary
    freq = Poisson(mu=5)
    sev = Exponential(scale=100)
    
    # Zero inflation probability = 0 (no inflation)
    zi0 = ZeroInflatedCompound(freq, sev, zero_prob=0.0)
    assert zi0.mean() == freq._dist.mean() * sev._dist.mean()
    
    # Zero inflation probability close to 1
    zi1 = ZeroInflatedCompound(freq, sev, zero_prob=0.99)
    assert zi1.mean() < 10  # Very small mean
    
    # Mixed Poisson with extreme parameters
    with pytest.raises(ValueError):
        PoissonGammaMixture(alpha=-1, beta=1)
    
    with pytest.raises(ValueError):
        PoissonGammaMixture(alpha=1, beta=0)


def test_convergence_properties():
    """Test convergence of approximations."""
    # Edgeworth convergence for increasing sample size
    true_skew = 0.5
    true_kurt = 0.3
    
    # As sample size increases, higher orders should be preferred
    orders = []
    for n in [20, 50, 100, 500, 1000]:
        order = automatic_order_selection(true_skew, true_kurt, n)
        orders.append(order)
    
    # Should be non-decreasing
    assert all(orders[i] <= orders[i+1] for i in range(len(orders)-1))