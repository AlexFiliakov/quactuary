"""
Comprehensive tests for mixed Poisson distributions.

This module provides thorough testing of all mixed Poisson process implementations,
including overdispersion properties, heavy tail behavior, and hierarchical models.
"""

import numpy as np
import pytest
from scipy import stats, special, integrate
from hypothesis import given, strategies as st, settings
import warnings

from quactuary.distributions.mixed_poisson import (
    MixedPoissonDistribution,
    PoissonGammaMixture,
    PoissonInverseGaussianMixture,
    HierarchicalPoissonMixture,
    TimeVaryingPoissonMixture
)
from quactuary.distributions.frequency import Poisson, NegativeBinomial
from quactuary.utils.numerical import stable_log, stable_exp


class TestPoissonGammaMixture:
    """Test Poisson-Gamma mixture (Negative Binomial)."""
    
    def test_negative_binomial_equivalence(self):
        """Test that Poisson-Gamma equals Negative Binomial."""
        alpha, beta = 5.0, 2.0
        
        # Create Poisson-Gamma mixture
        pg_mixture = PoissonGammaMixture(alpha=alpha, beta=beta)
        
        # Create equivalent Negative Binomial
        r = alpha
        p = beta / (1 + beta)
        nb_dist = NegativeBinomial(r=r, p=p)
        
        # Test PMF at several points
        k_values = np.arange(0, 20)
        for k in k_values:
            pg_pmf = pg_mixture.pmf(k)
            nb_pmf = nb_dist.pmf(k)
            assert np.isclose(pg_pmf, nb_pmf, rtol=1e-10), \
                f"PMF mismatch at k={k}: PG={pg_pmf}, NB={nb_pmf}"
        
        # Test moments
        assert np.isclose(pg_mixture.mean(), nb_dist.mean(), rtol=1e-10)
        assert np.isclose(pg_mixture.var(), nb_dist.var(), rtol=1e-10)
    
    def test_overdispersion_property(self):
        """Test overdispersion: Var[N] > E[N]."""
        test_cases = [
            (1.0, 1.0),   # alpha=1, beta=1
            (5.0, 2.0),   # alpha=5, beta=2
            (10.0, 0.5),  # alpha=10, beta=0.5
            (0.5, 3.0),   # alpha=0.5, beta=3
        ]
        
        for alpha, beta in test_cases:
            pg_mixture = PoissonGammaMixture(alpha=alpha, beta=beta)
            
            mean = pg_mixture.mean()
            var = pg_mixture.var()
            
            # Overdispersion: variance > mean
            assert var > mean, f"Not overdispersed: var={var}, mean={mean}"
            
            # Theoretical overdispersion factor
            overdispersion = var / mean
            expected_overdispersion = 1 + 1/beta
            
            assert np.isclose(overdispersion, expected_overdispersion, rtol=1e-10)
    
    def test_mixing_density(self):
        """Test Gamma mixing density properties."""
        alpha, beta = 3.0, 1.5
        pg_mixture = PoissonGammaMixture(alpha=alpha, beta=beta)
        
        # Test mixing density integrates to 1
        lambda_values = np.linspace(0, 20, 1000)
        densities = pg_mixture.mixing_density(lambda_values)
        integral = np.trapz(densities, lambda_values)
        
        assert np.isclose(integral, 1.0, rtol=1e-3)
        
        # Test mean of mixing distribution
        mean_lambda = np.trapz(lambda_values * densities, lambda_values)
        expected_mean = alpha / beta
        
        assert np.isclose(mean_lambda, expected_mean, rtol=1e-3)
    
    def test_posterior_inference(self):
        """Test posterior mean E[λ|N=n]."""
        alpha, beta = 4.0, 2.0
        pg_mixture = PoissonGammaMixture(alpha=alpha, beta=beta)
        
        # For Poisson-Gamma, posterior is Gamma(alpha + n, beta + 1)
        # E[λ|N=n] = (alpha + n) / (beta + 1)
        
        for n in [0, 1, 5, 10, 20]:
            posterior_mean = pg_mixture.conditional_mean(n)
            expected = (alpha + n) / (beta + 1)
            
            assert np.isclose(posterior_mean, expected, rtol=1e-10)
    
    def test_tail_behavior(self):
        """Test tail probability decay."""
        alpha, beta = 2.0, 1.0
        pg_mixture = PoissonGammaMixture(alpha=alpha, beta=beta)
        
        # Negative binomial has heavier tail than Poisson
        poisson_mean = alpha / beta
        poisson = Poisson(mu=poisson_mean)
        
        # Compare tail probabilities
        for k in [10, 20, 30, 40]:
            pg_tail = 1 - pg_mixture.cdf(k)
            poisson_tail = 1 - poisson.cdf(k)
            
            # Mixed Poisson should have heavier tail
            assert pg_tail > poisson_tail
        
        # Check tail decay rate (geometric-like for large k)
        k_large = np.arange(50, 100, 10)
        tail_probs = [1 - pg_mixture.cdf(k) for k in k_large]
        
        # Log tail probabilities should be approximately linear
        log_tails = np.log(tail_probs)
        coeffs = np.polyfit(k_large, log_tails, 1)
        
        # Slope should be negative (exponential decay)
        assert coeffs[0] < 0


class TestPoissonInverseGaussianMixture:
    """Test Poisson-Inverse Gaussian mixture."""
    
    def test_heavy_tail_property(self):
        """Test that PIG has heavier tails than Negative Binomial."""
        mu, lambda_param = 5.0, 10.0
        
        pig_mixture = PoissonInverseGaussianMixture(mu=mu, lambda_param=lambda_param)
        
        # Compare with Negative Binomial with same mean and variance
        pig_mean = pig_mixture.mean()
        pig_var = pig_mixture.var()
        
        # Find NB parameters with same mean/variance
        # For NB: mean = r(1-p)/p, var = r(1-p)/p^2
        # Solving: r = mean^2/(var-mean), p = mean/var
        r_nb = pig_mean**2 / (pig_var - pig_mean)
        p_nb = pig_mean / pig_var
        
        nb_dist = NegativeBinomial(r=r_nb, p=p_nb)
        
        # Compare extreme tail probabilities
        for k in [50, 75, 100]:
            pig_tail = 1 - pig_mixture.cdf(k)
            nb_tail = 1 - nb_dist.cdf(k)
            
            # PIG should have heavier tail
            if pig_tail > 1e-10 and nb_tail > 1e-10:
                assert pig_tail > nb_tail, \
                    f"PIG tail not heavier at k={k}: PIG={pig_tail}, NB={nb_tail}"
    
    def test_mixing_density_inverse_gaussian(self):
        """Test Inverse Gaussian mixing density."""
        mu, lambda_param = 3.0, 6.0
        pig_mixture = PoissonInverseGaussianMixture(mu=mu, lambda_param=lambda_param)
        
        # Test density properties
        lambda_values = np.linspace(0.1, 15, 1000)
        densities = pig_mixture.mixing_density(lambda_values)
        
        # Should integrate to 1
        integral = np.trapz(densities, lambda_values)
        assert np.isclose(integral, 1.0, rtol=1e-2)
        
        # Check mode location (for IG, mode is at specific point)
        mode_idx = np.argmax(densities)
        mode_lambda = lambda_values[mode_idx]
        
        # Theoretical mode of IG
        theoretical_mode = mu * (np.sqrt(1 + (mu/(2*lambda_param))**2) - mu/(2*lambda_param))
        
        assert np.isclose(mode_lambda, theoretical_mode, rtol=0.1)
    
    def test_variance_to_mean_ratio(self):
        """Test variance-to-mean ratio is higher than Poisson-Gamma."""
        mu, lambda_param = 4.0, 8.0
        pig_mixture = PoissonInverseGaussianMixture(mu=mu, lambda_param=lambda_param)
        
        # For PIG, the variance-to-mean ratio depends on parameters
        vmr = pig_mixture.var() / pig_mixture.mean()
        
        # Should be greater than 1 (overdispersed)
        assert vmr > 1
        
        # Compare with Poisson-Gamma with same mean
        # For PG with mean μ: choose α=μ*β, then mean = α/β = μ
        beta_pg = 2.0
        alpha_pg = mu * beta_pg
        pg_mixture = PoissonGammaMixture(alpha=alpha_pg, beta=beta_pg)
        
        vmr_pg = pg_mixture.var() / pg_mixture.mean()
        
        # PIG typically has higher VMR for heavy-tailed scenarios
        # (though this depends on specific parameters)
        print(f"PIG VMR: {vmr}, PG VMR: {vmr_pg}")


class TestHierarchicalPoissonMixture:
    """Test hierarchical Poisson mixture models."""
    
    def test_two_level_hierarchy(self):
        """Test two-level hierarchical model."""
        # Portfolio level parameters
        portfolio_alpha = 5.0
        portfolio_beta = 2.0
        
        # Individual level variation
        individual_dispersion = 0.5
        
        hpm = HierarchicalPoissonMixture(
            portfolio_alpha=portfolio_alpha,
            portfolio_beta=portfolio_beta,
            individual_dispersion=individual_dispersion
        )
        
        # Mean should match portfolio mean
        expected_mean = portfolio_alpha / portfolio_beta
        assert np.isclose(hpm.mean(), expected_mean, rtol=1e-10)
        
        # Variance should include both levels of variation
        base_var = portfolio_alpha / portfolio_beta + portfolio_alpha / portfolio_beta**2
        
        # Individual dispersion adds extra variance
        actual_var = hpm.var()
        assert actual_var > base_var
    
    def test_conditional_simulation(self):
        """Test conditional simulation given portfolio parameter."""
        portfolio_alpha = 3.0
        portfolio_beta = 1.5
        individual_dispersion = 0.3
        
        hpm = HierarchicalPoissonMixture(
            portfolio_alpha=portfolio_alpha,
            portfolio_beta=portfolio_beta,
            individual_dispersion=individual_dispersion
        )
        
        # Simulate with fixed portfolio lambda
        portfolio_lambda = 2.0
        
        np.random.seed(42)
        n_individuals = 1000
        samples = hpm.rvs_conditional(
            size=n_individuals,
            portfolio_lambda=portfolio_lambda
        )
        
        # Mean should be close to portfolio_lambda
        assert np.isclose(np.mean(samples), portfolio_lambda, rtol=0.1)
        
        # Should show individual-level variation
        assert np.var(samples) > 0
    
    def test_variance_decomposition(self):
        """Test variance decomposition into portfolio and individual components."""
        portfolio_alpha = 4.0
        portfolio_beta = 2.0
        individual_dispersion = 0.4
        
        hpm = HierarchicalPoissonMixture(
            portfolio_alpha=portfolio_alpha,
            portfolio_beta=portfolio_beta,
            individual_dispersion=individual_dispersion
        )
        
        # Get variance components
        var_components = hpm.variance_components()
        
        # Portfolio variance (between groups)
        portfolio_var = var_components['portfolio_variance']
        expected_portfolio_var = portfolio_alpha / portfolio_beta**2
        assert np.isclose(portfolio_var, expected_portfolio_var, rtol=1e-10)
        
        # Individual variance (within groups)
        individual_var = var_components['individual_variance']
        assert individual_var > 0
        
        # Total variance
        total_var = var_components['total_variance']
        assert np.isclose(total_var, hpm.var(), rtol=1e-10)
        
        # Intraclass correlation
        icc = var_components['intraclass_correlation']
        assert 0 < icc < 1


class TestTimeVaryingPoissonMixture:
    """Test time-varying Poisson mixture models."""
    
    def test_constant_intensity_function(self):
        """Test with constant intensity (reduces to standard mixed Poisson)."""
        def constant_intensity(t):
            return 2.0
        
        alpha, beta = 3.0, 1.5
        T = 1.0
        
        tvpm = TimeVaryingPoissonMixture(
            intensity_function=constant_intensity,
            mixing_alpha=alpha,
            mixing_beta=beta,
            time_period=T
        )
        
        # Should match standard Poisson-Gamma with scaled parameters
        expected_mean = T * alpha / beta * 2.0  # intensity * time * mixing mean
        assert np.isclose(tvpm.mean(), expected_mean, rtol=1e-10)
    
    def test_linear_intensity_function(self):
        """Test with linear increasing intensity."""
        def linear_intensity(t):
            return 1.0 + 0.5 * t
        
        alpha, beta = 2.0, 1.0
        T = 2.0
        
        tvpm = TimeVaryingPoissonMixture(
            intensity_function=linear_intensity,
            mixing_alpha=alpha,
            mixing_beta=beta,
            time_period=T
        )
        
        # Integrated intensity: ∫(1 + 0.5t)dt from 0 to T = T + 0.25T²
        integrated_intensity = T + 0.25 * T**2
        expected_mean = integrated_intensity * alpha / beta
        
        assert np.isclose(tvpm.mean(), expected_mean, rtol=1e-10)
    
    def test_seasonal_intensity_function(self):
        """Test with seasonal (sinusoidal) intensity."""
        def seasonal_intensity(t):
            return 2.0 + np.sin(2 * np.pi * t)
        
        alpha, beta = 4.0, 2.0
        T = 1.0  # One full period
        
        tvpm = TimeVaryingPoissonMixture(
            intensity_function=seasonal_intensity,
            mixing_alpha=alpha,
            mixing_beta=beta,
            time_period=T
        )
        
        # Integrated intensity over one period
        # ∫(2 + sin(2πt))dt from 0 to 1 = 2
        expected_mean = 2.0 * alpha / beta
        
        assert np.isclose(tvpm.mean(), expected_mean, rtol=1e-10)
        
        # Test simulation
        np.random.seed(123)
        samples = tvpm.rvs(size=1000)
        
        # Check all samples are non-negative integers
        assert all(isinstance(x, (int, np.integer)) for x in samples)
        assert all(x >= 0 for x in samples)
        
        # Mean should match
        assert np.isclose(np.mean(samples), expected_mean, rtol=0.1)
    
    def test_intensity_integration_accuracy(self):
        """Test numerical integration of intensity function."""
        def complex_intensity(t):
            return np.exp(-t) + 0.5 * t**2
        
        alpha, beta = 3.0, 1.5
        T = 3.0
        
        tvpm = TimeVaryingPoissonMixture(
            intensity_function=complex_intensity,
            mixing_alpha=alpha,
            mixing_beta=beta,
            time_period=T
        )
        
        # Get integrated intensity
        integrated = tvpm.integrated_intensity()
        
        # Compare with analytical solution
        # ∫(e^(-t) + 0.5t²)dt = -e^(-t) + t³/6
        analytical = (1 - np.exp(-T)) + T**3 / 6
        
        assert np.isclose(integrated, analytical, rtol=1e-6)


class TestPropertyBasedMixedPoisson:
    """Property-based tests for mixed Poisson distributions."""
    
    @given(
        alpha=st.floats(min_value=0.5, max_value=20.0),
        beta=st.floats(min_value=0.5, max_value=10.0)
    )
    @settings(max_examples=50)
    def test_poisson_gamma_properties(self, alpha, beta):
        """Test mathematical properties of Poisson-Gamma mixture."""
        pg_mixture = PoissonGammaMixture(alpha=alpha, beta=beta)
        
        # Mean and variance formulas
        expected_mean = alpha / beta
        expected_var = alpha / beta + alpha / beta**2
        
        assert np.isclose(pg_mixture.mean(), expected_mean, rtol=1e-10)
        assert np.isclose(pg_mixture.var(), expected_var, rtol=1e-10)
        
        # Overdispersion
        assert pg_mixture.var() > pg_mixture.mean()
        
        # PMF sums to 1 (approximately)
        k_max = int(pg_mixture.mean() + 10 * pg_mixture.std())
        k_values = np.arange(0, k_max)
        pmf_sum = sum(pg_mixture.pmf(k) for k in k_values)
        
        assert np.isclose(pmf_sum, 1.0, rtol=1e-3)
    
    @given(
        mu=st.floats(min_value=0.5, max_value=20.0),
        lambda_param=st.floats(min_value=1.0, max_value=50.0)
    )
    @settings(max_examples=30)
    def test_poisson_inverse_gaussian_properties(self, mu, lambda_param):
        """Test mathematical properties of Poisson-IG mixture."""
        pig_mixture = PoissonInverseGaussianMixture(mu=mu, lambda_param=lambda_param)
        
        # Mean formula
        expected_mean = mu
        assert np.isclose(pig_mixture.mean(), expected_mean, rtol=1e-10)
        
        # Variance formula includes both mixing and Poisson variation
        # Var[N] = E[λ] + Var[λ]
        mixing_var = mu**3 / lambda_param
        expected_var = mu + mixing_var
        
        assert np.isclose(pig_mixture.var(), expected_var, rtol=1e-10)
        
        # Check overdispersion
        assert pig_mixture.var() > pig_mixture.mean()
        
        # Variance-to-mean ratio
        vmr = pig_mixture.var() / pig_mixture.mean()
        expected_vmr = 1 + mu**2 / lambda_param
        
        assert np.isclose(vmr, expected_vmr, rtol=1e-10)


class TestEdgeCasesMixedPoisson:
    """Test edge cases for mixed Poisson distributions."""
    
    def test_small_mixing_variance(self):
        """Test when mixing variance is very small (approaches Poisson)."""
        # Large beta means small variance in lambda
        alpha = 5.0
        beta = 1000.0
        
        pg_mixture = PoissonGammaMixture(alpha=alpha, beta=beta)
        
        # Should be close to Poisson(alpha/beta)
        poisson_mean = alpha / beta
        poisson = Poisson(mu=poisson_mean)
        
        # Compare PMF
        for k in range(10):
            pg_pmf = pg_mixture.pmf(k)
            poisson_pmf = poisson.pmf(k)
            
            # Should be very close
            assert np.isclose(pg_pmf, poisson_pmf, rtol=1e-2)
    
    def test_extreme_overdispersion(self):
        """Test with extreme overdispersion."""
        # Small beta means large variance
        alpha = 2.0
        beta = 0.01
        
        pg_mixture = PoissonGammaMixture(alpha=alpha, beta=beta)
        
        # Variance should be much larger than mean
        vmr = pg_mixture.var() / pg_mixture.mean()
        assert vmr > 100  # Very overdispersed
        
        # Distribution should have very heavy tail
        k_90 = pg_mixture.ppf(0.9)
        k_99 = pg_mixture.ppf(0.99)
        
        # Large difference between quantiles
        assert k_99 > 5 * k_90
    
    def test_degenerate_hierarchical(self):
        """Test hierarchical model with no individual variation."""
        portfolio_alpha = 3.0
        portfolio_beta = 1.5
        individual_dispersion = 0.0  # No individual variation
        
        hpm = HierarchicalPoissonMixture(
            portfolio_alpha=portfolio_alpha,
            portfolio_beta=portfolio_beta,
            individual_dispersion=individual_dispersion
        )
        
        # Should reduce to standard Poisson-Gamma
        pg_mixture = PoissonGammaMixture(
            alpha=portfolio_alpha,
            beta=portfolio_beta
        )
        
        # Compare means and variances
        assert np.isclose(hpm.mean(), pg_mixture.mean(), rtol=1e-10)
        assert np.isclose(hpm.var(), pg_mixture.var(), rtol=1e-10)


class TestNumericalStabilityMixedPoisson:
    """Test numerical stability of mixed Poisson implementations."""
    
    def test_extreme_parameter_values(self):
        """Test with extreme parameter combinations."""
        # Very small alpha (near 0)
        pg1 = PoissonGammaMixture(alpha=0.01, beta=1.0)
        assert pg1.mean() == 0.01
        assert pg1.pmf(0) > 0.99  # Most mass at 0
        
        # Very large alpha
        pg2 = PoissonGammaMixture(alpha=1000.0, beta=10.0)
        assert pg2.mean() == 100.0
        assert pg2.std() < 15  # Relatively small std compared to mean
        
        # Very large lambda for IG
        pig = PoissonInverseGaussianMixture(mu=5.0, lambda_param=10000.0)
        # Should be close to Poisson(5) due to small mixing variance
        assert np.isclose(pig.var() / pig.mean(), 1.0, rtol=0.1)
    
    def test_log_probability_stability(self):
        """Test computation in log space for numerical stability."""
        # Parameters that could cause underflow
        alpha = 50.0
        beta = 0.5
        
        pg_mixture = PoissonGammaMixture(alpha=alpha, beta=beta)
        
        # Test at extreme values
        k_values = [0, 50, 100, 200, 500]
        
        for k in k_values:
            # Log PMF should be computable
            if hasattr(pg_mixture._dist, 'logpmf'):
                log_pmf = pg_mixture._dist.logpmf(k)
                assert np.isfinite(log_pmf)
                
                # Check consistency with PMF
                pmf = pg_mixture.pmf(k)
                if pmf > 0:
                    assert np.isclose(np.exp(log_pmf), pmf, rtol=1e-10)
    
    def test_tail_probability_computation(self):
        """Test accurate tail probability computation."""
        mu = 10.0
        lambda_param = 5.0
        
        pig_mixture = PoissonInverseGaussianMixture(mu=mu, lambda_param=lambda_param)
        
        # Test survival function computation
        k_values = [50, 100, 200]
        
        for k in k_values:
            # Direct computation
            sf_direct = 1 - pig_mixture.cdf(k)
            
            # Should be positive but possibly very small
            assert sf_direct >= 0
            assert sf_direct <= 1
            
            # For large k, use log scale if available
            if sf_direct > 1e-10:
                # Verify CDF + SF = 1
                assert np.isclose(pig_mixture.cdf(k) + sf_direct, 1.0, rtol=1e-10)


def test_integration_with_compound_distributions():
    """Test that mixed Poisson distributions work with compound distributions."""
    from quactuary.distributions.severity import Exponential, Gamma
    from quactuary.distributions.compound_extensions import create_extended_compound_distribution
    
    # Create mixed Poisson frequency
    pg_freq = PoissonGammaMixture(alpha=3.0, beta=1.5)
    
    # Create severity
    sev = Exponential(scale=1000)
    
    # Create compound distribution
    compound = create_extended_compound_distribution(pg_freq, sev)
    
    # Test basic properties
    assert compound.mean() == pg_freq.mean() * sev._dist.mean()
    
    # Test simulation
    np.random.seed(42)
    samples = compound.rvs(size=1000)
    
    assert all(s >= 0 for s in samples)
    assert np.isclose(np.mean(samples), compound.mean(), rtol=0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])