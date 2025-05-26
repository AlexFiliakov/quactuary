"""
Property-based tests for all extended distribution implementations.

This module uses hypothesis to validate mathematical properties across
the entire parameter space for all distribution types.
"""

import numpy as np
import pytest
from scipy import stats
from hypothesis import given, strategies as st, settings, assume
import warnings

from quactuary.distributions.frequency import Poisson, NegativeBinomial, Binomial
from quactuary.distributions.severity import Exponential, Gamma, Lognormal
from quactuary.distributions.compound import (
    CompoundDistribution,
    BinomialExponentialCompound,
    BinomialGammaCompound,
    BinomialLognormalCompound,
    create_compound_distribution
)
from quactuary.distributions.mixed_poisson import (
    PoissonGammaMixture,
    PoissonInverseGaussianMixture
)
from quactuary.distributions.zero_inflated import (
    ZIPoissonCompound,
    ZINegativeBinomialCompound,
    ZIBinomialCompound
)
from quactuary.distributions.compound_extensions import create_extended_compound_distribution
from quactuary.utils.numerical import stable_log, stable_exp


class TestMomentMatching:
    """Test that theoretical and empirical moments match."""
    
    @given(
        n=st.integers(min_value=5, max_value=50),
        p=st.floats(min_value=0.1, max_value=0.9),
        scale=st.floats(min_value=10, max_value=5000)
    )
    @settings(max_examples=50, deadline=None)
    def test_binomial_exponential_moments(self, n, p, scale):
        """Test moment matching for Binomial-Exponential."""
        freq = Binomial(n=n, p=p)
        sev = Exponential(scale=scale)
        compound = BinomialExponentialCompound(freq, sev)
        
        # Theoretical moments
        theoretical_mean = n * p * scale
        theoretical_var = n * p * scale**2 * (2 - p)
        
        # Check exact match
        assert np.isclose(compound.mean(), theoretical_mean, rtol=1e-10)
        assert np.isclose(compound.var(), theoretical_var, rtol=1e-10)
        
        # Generate samples if expected count is reasonable
        if n * p >= 1:
            np.random.seed(42)
            samples = compound.rvs(size=5000)
            
            empirical_mean = np.mean(samples)
            empirical_var = np.var(samples)
            
            # Empirical should be close to theoretical
            assert np.isclose(empirical_mean, theoretical_mean, rtol=0.05)
            assert np.isclose(empirical_var, theoretical_var, rtol=0.1)
    
    @given(
        n=st.integers(min_value=5, max_value=40),
        p=st.floats(min_value=0.2, max_value=0.8),
        alpha=st.floats(min_value=0.5, max_value=5.0),
        scale=st.floats(min_value=10, max_value=1000)
    )
    @settings(max_examples=50, deadline=None)
    def test_binomial_gamma_moments(self, n, p, alpha, scale):
        """Test moment matching for Binomial-Gamma."""
        freq = Binomial(n=n, p=p)
        sev = Gamma(shape=alpha, scale=scale)
        compound = BinomialGammaCompound(freq, sev)
        
        # Theoretical moments
        mean_sev = alpha * scale
        var_sev = alpha * scale**2
        
        theoretical_mean = n * p * mean_sev
        theoretical_var = n * p * (var_sev + (1 - p) * mean_sev**2)
        
        # Check exact match
        assert np.isclose(compound.mean(), theoretical_mean, rtol=1e-10)
        assert np.isclose(compound.var(), theoretical_var, rtol=1e-10)
        
        # Third moment check (if implemented)
        if hasattr(compound, 'skewness'):
            assert compound.skewness() > 0  # Should be positively skewed
    
    @given(
        alpha=st.floats(min_value=0.5, max_value=20.0),
        beta=st.floats(min_value=0.5, max_value=10.0),
        sev_scale=st.floats(min_value=10, max_value=5000)
    )
    @settings(max_examples=50, deadline=None)
    def test_poisson_gamma_mixture_moments(self, alpha, beta, sev_scale):
        """Test moments for Poisson-Gamma mixture compounds."""
        pg_freq = PoissonGammaMixture(alpha=alpha, beta=beta)
        sev = Exponential(scale=sev_scale)
        
        # Create compound
        compound = create_extended_compound_distribution(pg_freq, sev)
        
        # Theoretical moments
        freq_mean = alpha / beta
        freq_var = alpha / beta + alpha / beta**2
        sev_mean = sev_scale
        sev_var = sev_scale**2
        
        theoretical_mean = freq_mean * sev_mean
        theoretical_var = freq_mean * sev_var + freq_var * sev_mean**2
        
        # Check
        assert np.isclose(compound.mean(), theoretical_mean, rtol=1e-10)
        
        # For simulated compounds, variance might be approximate
        if hasattr(compound, 'var'):
            assert np.isclose(compound.var(), theoretical_var, rtol=0.01)
    
    @given(
        mu=st.floats(min_value=1, max_value=20),
        zero_prob=st.floats(min_value=0.0, max_value=0.8),
        sev_shape=st.floats(min_value=0.5, max_value=3.0),
        sev_scale=st.floats(min_value=50, max_value=2000)
    )
    @settings(max_examples=50, deadline=None)
    def test_zero_inflated_moments(self, mu, zero_prob, sev_shape, sev_scale):
        """Test moment relationships for zero-inflated compounds."""
        freq = Poisson(mu=mu)
        sev = Gamma(shape=sev_shape, scale=sev_scale)
        
        # Standard compound
        base_compound = create_compound_distribution(freq, sev)
        
        # Zero-inflated version
        zi_compound = ZIPoissonCompound(freq, sev, zero_prob)
        
        # Moment relationships
        assert np.isclose(
            zi_compound.mean(),
            (1 - zero_prob) * base_compound.mean(),
            rtol=1e-10
        )
        
        # Variance relationship (law of total variance)
        base_mean = base_compound.mean()
        base_var = base_compound.var()
        expected_var = ((1 - zero_prob) * base_var + 
                       (1 - zero_prob) * zero_prob * base_mean**2)
        
        assert np.isclose(zi_compound.var(), expected_var, rtol=1e-10)


class TestDistributionBounds:
    """Test that distributions satisfy mathematical bounds."""
    
    @given(
        n=st.integers(min_value=1, max_value=100),
        p=st.floats(min_value=0.01, max_value=0.99),
        sigma=st.floats(min_value=0.1, max_value=2.0),
        scale=st.floats(min_value=100, max_value=10000)
    )
    @settings(max_examples=50, deadline=None)
    def test_cdf_bounds_and_monotonicity(self, n, p, sigma, scale):
        """Test CDF bounds and monotonicity for compound distributions."""
        freq = Binomial(n=n, p=p)
        sev = Lognormal(shape=sigma, scale=scale)
        compound = BinomialLognormalCompound(freq, sev)
        
        # Test points
        x_values = np.logspace(0, 6, 50)
        cdf_values = compound.cdf(x_values)
        
        # Bounds: 0 <= CDF <= 1
        assert all(0 <= cdf <= 1 for cdf in cdf_values)
        
        # Monotonicity: CDF should be non-decreasing
        diffs = np.diff(cdf_values)
        assert all(diff >= -1e-10 for diff in diffs), \
            "CDF is not monotonic"
        
        # Limits
        assert compound.cdf(0) >= (1 - p)**n  # At least P(N=0)
        assert compound.cdf(np.inf) == 1.0
        assert compound.cdf(-1) == 0.0
    
    @given(
        r=st.floats(min_value=0.5, max_value=20.0),
        p=st.floats(min_value=0.1, max_value=0.9),
        zero_prob=st.floats(min_value=0.0, max_value=0.9)
    )
    @settings(max_examples=50, deadline=None)
    def test_pdf_non_negative(self, r, p, zero_prob):
        """Test that PDFs are non-negative."""
        freq = NegativeBinomial(r=r, p=p)
        sev = Exponential(scale=1000)
        
        # Create zero-inflated compound
        zi_compound = ZINegativeBinomialCompound(freq, sev, zero_prob)
        
        # Test at various points
        test_points = [0, 100, 500, 1000, 5000, 10000]
        
        for x in test_points:
            pdf_val = zi_compound.pdf(x)
            assert pdf_val >= 0, f"Negative PDF at x={x}: {pdf_val}"
            assert np.isfinite(pdf_val), f"Non-finite PDF at x={x}: {pdf_val}"
    
    @given(
        alpha=st.floats(min_value=1.0, max_value=10.0),
        beta=st.floats(min_value=0.5, max_value=5.0)
    )
    @settings(max_examples=50, deadline=None)
    def test_pmf_sum_to_one(self, alpha, beta):
        """Test that PMFs sum to approximately 1."""
        pg_mixture = PoissonGammaMixture(alpha=alpha, beta=beta)
        
        # For discrete distributions, sum PMF up to a reasonable limit
        mean = pg_mixture.mean()
        std = pg_mixture.std()
        k_max = int(mean + 10 * std)
        
        pmf_sum = sum(pg_mixture.pmf(k) for k in range(k_max + 1))
        
        # Should be very close to 1
        assert np.isclose(pmf_sum, 1.0, rtol=1e-3), \
            f"PMF sum = {pmf_sum}, not close to 1"
    
    @given(
        mu=st.floats(min_value=0.5, max_value=20.0),
        lambda_param=st.floats(min_value=1.0, max_value=50.0)
    )
    @settings(max_examples=50, deadline=None)
    def test_survival_function_consistency(self, mu, lambda_param):
        """Test that S(x) = 1 - F(x) for all distributions."""
        pig_mixture = PoissonInverseGaussianMixture(mu=mu, lambda_param=lambda_param)
        
        # Test at several points
        test_points = [0, 5, 10, 20, 50, 100]
        
        for k in test_points:
            cdf = pig_mixture.cdf(k)
            sf = pig_mixture.sf(k) if hasattr(pig_mixture, 'sf') else 1 - cdf
            
            assert np.isclose(cdf + sf, 1.0, rtol=1e-10), \
                f"CDF + SF != 1 at k={k}: {cdf} + {sf} = {cdf + sf}"


class TestMonotonicityProperties:
    """Test monotonicity properties of distributions."""
    
    @given(
        n=st.integers(min_value=10, max_value=50),
        p1=st.floats(min_value=0.1, max_value=0.4),
        p2=st.floats(min_value=0.5, max_value=0.9),
        scale=st.floats(min_value=100, max_value=1000)
    )
    @settings(max_examples=30, deadline=None)
    def test_mean_monotonicity_in_parameters(self, n, p1, p2, scale):
        """Test that mean increases with frequency parameters."""
        assume(p1 < p2)
        
        # Create two compounds with different p values
        freq1 = Binomial(n=n, p=p1)
        freq2 = Binomial(n=n, p=p2)
        sev = Exponential(scale=scale)
        
        compound1 = BinomialExponentialCompound(freq1, sev)
        compound2 = BinomialExponentialCompound(freq2, sev)
        
        # Mean should increase with p
        assert compound1.mean() < compound2.mean()
        
        # Variance should also increase
        assert compound1.var() < compound2.var()
    
    @given(
        zero_prob1=st.floats(min_value=0.0, max_value=0.3),
        zero_prob2=st.floats(min_value=0.4, max_value=0.8),
        mu=st.floats(min_value=2.0, max_value=10.0)
    )
    @settings(max_examples=30, deadline=None)
    def test_zero_inflation_effect(self, zero_prob1, zero_prob2, mu):
        """Test that increasing zero inflation decreases mean."""
        assume(zero_prob1 < zero_prob2)
        
        freq = Poisson(mu=mu)
        sev = Gamma(shape=2.0, scale=500)
        
        zi_compound1 = ZIPoissonCompound(freq, sev, zero_prob1)
        zi_compound2 = ZIPoissonCompound(freq, sev, zero_prob2)
        
        # Higher zero inflation should decrease mean
        assert zi_compound1.mean() > zi_compound2.mean()
        
        # Check PDF at zero
        assert zi_compound1.pdf(0) < zi_compound2.pdf(0)


class TestNumericalStabilityProperties:
    """Test numerical stability across parameter ranges."""
    
    @given(
        n=st.integers(min_value=1, max_value=1000),
        p=st.floats(min_value=0.001, max_value=0.999),
        scale=st.floats(min_value=0.1, max_value=100000)
    )
    @settings(max_examples=50, deadline=None)
    def test_extreme_parameters_stability(self, n, p, scale):
        """Test stability with extreme parameter combinations."""
        freq = Binomial(n=n, p=p)
        sev = Exponential(scale=scale)
        
        # Should not raise errors
        compound = create_compound_distribution(freq, sev)
        
        # Basic computations should work
        mean = compound.mean()
        assert np.isfinite(mean)
        assert mean >= 0
        
        # PDF at 0 should be computable
        pdf_0 = compound.pdf(0)
        assert np.isfinite(pdf_0)
        assert 0 <= pdf_0 <= 1
        
        # CDF should work
        if mean > 0:
            cdf_mean = compound.cdf(mean)
            assert np.isfinite(cdf_mean)
            assert 0 <= cdf_mean <= 1
    
    @given(
        alpha=st.floats(min_value=0.01, max_value=100.0),
        beta=st.floats(min_value=0.01, max_value=100.0)
    )
    @settings(max_examples=50, deadline=None)
    def test_log_probability_computation(self, alpha, beta):
        """Test that log probabilities are stable."""
        pg_mixture = PoissonGammaMixture(alpha=alpha, beta=beta)
        
        # Test at various k values
        mean = pg_mixture.mean()
        k_values = [0, 1, int(mean), int(mean * 10), int(mean * 100)]
        
        for k in k_values:
            if k >= 0:
                pmf = pg_mixture.pmf(k)
                
                if pmf > 0:
                    log_pmf = stable_log(pmf)
                    assert np.isfinite(log_pmf)
                    
                    # Check consistency
                    assert np.isclose(stable_exp(log_pmf), pmf, rtol=1e-10)


class TestSpecialCases:
    """Test special cases and edge behaviors."""
    
    @given(
        n=st.integers(min_value=1, max_value=1),  # n=1 is special
        p=st.floats(min_value=0.1, max_value=0.9),
        scale=st.floats(min_value=100, max_value=1000)
    )
    @settings(max_examples=20)
    def test_binomial_n_equals_one(self, n, p, scale):
        """Test Binomial(1,p) reduces to Bernoulli."""
        freq = Binomial(n=1, p=p)
        sev = Exponential(scale=scale)
        compound = BinomialExponentialCompound(freq, sev)
        
        # P(S=0) should be 1-p
        assert np.isclose(compound.pdf(0), 1 - p, rtol=1e-10)
        
        # For x > 0, should be p * Exponential(scale)
        x = scale * 2
        expected_pdf = p * stats.expon.pdf(x, scale=scale)
        assert np.isclose(compound.pdf(x), expected_pdf, rtol=1e-10)
    
    @given(
        r=st.floats(min_value=10.0, max_value=100.0),
        p=st.floats(min_value=0.7, max_value=0.95)
    )
    @settings(max_examples=20)
    def test_negative_binomial_large_r(self, r, p):
        """Test NB approaches normal for large r."""
        # For large r, NB approaches normal
        nb = NegativeBinomial(r=r, p=p)
        
        mean = nb.mean() if hasattr(nb, 'mean') else r * (1 - p) / p
        var = nb.var() if hasattr(nb, 'var') else r * (1 - p) / p**2
        
        # Standardized third moment should be small
        if mean > 10 and var > 0:
            # Generate samples
            np.random.seed(42)
            samples = nb.rvs(size=1000)
            
            standardized = (samples - np.mean(samples)) / np.std(samples)
            skewness = np.mean(standardized**3)
            
            # Should be close to 0 for large r
            assert abs(skewness) < 1.0 / np.sqrt(r)
    
    @given(
        zero_prob=st.floats(min_value=0.99, max_value=0.999)
    )
    @settings(max_examples=20)
    def test_extreme_zero_inflation(self, zero_prob):
        """Test behavior with extreme zero inflation."""
        freq = Poisson(mu=5.0)
        sev = Exponential(scale=1000)
        
        zi_compound = ZIPoissonCompound(freq, sev, zero_prob)
        
        # Almost all mass at zero
        assert zi_compound.pdf(0) > zero_prob
        
        # Mean should be very small
        base_mean = 5.0 * 1000
        assert zi_compound.mean() < (1 - zero_prob) * base_mean * 1.1
        
        # High quantiles should still be finite
        q99 = zi_compound.ppf(0.99)
        assert np.isfinite(q99)
        
        # But most quantiles should be 0
        assert zi_compound.ppf(0.5) == 0
        assert zi_compound.ppf(0.9) == 0


class TestConsistencyAcrossImplementations:
    """Test consistency between different implementation approaches."""
    
    @given(
        mu=st.floats(min_value=1.0, max_value=20.0),
        scale=st.floats(min_value=100, max_value=5000)
    )
    @settings(max_examples=30, deadline=None)
    def test_poisson_exponential_implementations(self, mu, scale):
        """Test different ways to create Poisson-Exponential compound."""
        freq = Poisson(mu=mu)
        sev = Exponential(scale=scale)
        
        # Method 1: Direct creation
        compound1 = create_compound_distribution(freq, sev)
        
        # Method 2: Extended factory
        compound2 = create_extended_compound_distribution(freq, sev)
        
        # Method 3: String-based
        compound3 = create_extended_compound_distribution(
            'poisson', 'exponential',
            mu=mu, scale=scale
        )
        
        # All should have same mean
        mean1 = compound1.mean()
        mean2 = compound2.mean()
        
        assert np.isclose(mean1, mean2, rtol=1e-10)
        assert np.isclose(mean1, mu * scale, rtol=1e-10)
        
        # PDFs should match
        x = mu * scale
        pdf1 = compound1.pdf(x)
        pdf2 = compound2.pdf(x)
        
        assert np.isclose(pdf1, pdf2, rtol=1e-6)
    
    @given(
        n=st.integers(min_value=5, max_value=30),
        p=st.floats(min_value=0.2, max_value=0.8),
        shape=st.floats(min_value=1.0, max_value=4.0),
        scale=st.floats(min_value=50, max_value=500)
    )
    @settings(max_examples=30, deadline=None)
    def test_analytical_vs_simulation(self, n, p, shape, scale):
        """Test analytical solutions match simulation."""
        freq = Binomial(n=n, p=p)
        sev = Gamma(shape=shape, scale=scale)
        
        # Analytical compound
        compound_analytical = BinomialGammaCompound(freq, sev)
        
        # Simulation-based (if different implementation exists)
        # For now, just test that analytical gives reasonable results
        
        # Generate samples from analytical
        np.random.seed(42)
        samples = compound_analytical.rvs(size=5000)
        
        # Compare moments
        empirical_mean = np.mean(samples)
        theoretical_mean = compound_analytical.mean()
        
        assert np.isclose(empirical_mean, theoretical_mean, rtol=0.05)
        
        # Check distribution shape
        if n * p > 2:  # Need reasonable expected count
            # Should have some spread
            assert np.std(samples) > 0
            # Should have some non-zero values
            assert np.sum(samples > 0) > len(samples) * 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])