"""
Tests for compound binomial distributions.
"""

import numpy as np
import pytest
from scipy import stats

from quactuary.distributions.frequency import Binomial
from quactuary.distributions.severity import Exponential, Gamma, Lognormal
from quactuary.distributions.compound import (
    BinomialExponentialCompound,
    BinomialGammaCompound,
    BinomialLognormalCompound,
    PanjerBinomialRecursion
)


class TestBinomialExponentialCompound:
    """Test Binomial-Exponential compound distribution."""
    
    def test_mean_variance(self):
        """Test analytical mean and variance calculations."""
        n, p = 10, 0.3
        scale = 1000
        
        freq = Binomial(n=n, p=p)
        sev = Exponential(scale=scale)
        compound = BinomialExponentialCompound(freq, sev)
        
        # Expected values
        expected_mean = n * p * scale
        expected_var = n * p * scale**2 * (2 - p)
        
        assert np.isclose(compound.mean(), expected_mean)
        assert np.isclose(compound.var(), expected_var)
    
    def test_zero_probability(self):
        """Test probability mass at zero."""
        n, p = 5, 0.2
        freq = Binomial(n=n, p=p)
        sev = Exponential(scale=100)
        compound = BinomialExponentialCompound(freq, sev)
        
        # P(S=0) = (1-p)^n
        expected_p0 = (1 - p) ** n
        assert np.isclose(compound.pdf(0), expected_p0)
        assert np.isclose(compound.cdf(0), expected_p0)
    
    def test_pdf_cdf_consistency(self):
        """Test that PDF integrates to CDF."""
        freq = Binomial(n=3, p=0.4)
        sev = Exponential(scale=50)
        compound = BinomialExponentialCompound(freq, sev)
        
        # Test at several points
        x_values = np.array([0, 10, 50, 100, 200])
        for x in x_values[1:]:  # Skip 0 due to atom
            # Numerical integration of PDF should approximate CDF
            # Start integration from small positive value to exclude discrete mass at 0
            x_grid = np.linspace(0.001, x, 1000)
            pdf_integral = np.trapz(compound.pdf(x_grid), x_grid)
            p0 = compound.pdf(0)
            
            assert np.isclose(compound.cdf(x), p0 + pdf_integral, rtol=1e-2)
    
    def test_random_sampling(self):
        """Test random variate generation."""
        freq = Binomial(n=20, p=0.5)
        sev = Exponential(scale=100)
        compound = BinomialExponentialCompound(freq, sev)
        
        # Generate samples
        np.random.seed(42)
        samples = compound.rvs(size=10000)
        
        # Check sample statistics
        assert np.isclose(np.mean(samples), compound.mean(), rtol=0.05)
        assert np.isclose(np.var(samples), compound.var(), rtol=0.1)
        
        # Check non-negativity
        assert np.all(samples >= 0)
    
    def test_quantile_function(self):
        """Test percent point function (ppf)."""
        freq = Binomial(n=5, p=0.3)
        sev = Exponential(scale=200)
        compound = BinomialExponentialCompound(freq, sev)
        
        # P(S=0) = (1-p)^n
        p_zero = (1 - 0.3) ** 5
        
        # Test standard quantiles
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        for q in quantiles:
            x_q = compound.ppf(q)
            
            # For quantiles below p_zero, ppf should return 0
            # and cdf(0) = p_zero, not q
            if q <= p_zero:
                assert x_q == 0
                assert np.isclose(compound.cdf(x_q), p_zero, rtol=1e-4)
            else:
                # For quantiles above p_zero, cdf(ppf(q)) should equal q
                assert np.isclose(compound.cdf(x_q), q, rtol=1e-4)
            
            # Quantile should be non-negative
            assert x_q >= 0


class TestBinomialGammaCompound:
    """Test Binomial-Gamma compound distribution."""
    
    def test_mean_variance(self):
        """Test analytical mean and variance calculations."""
        n, p = 15, 0.4
        alpha, beta = 2.0, 0.01  # scale = 1/beta = 100
        
        freq = Binomial(n=n, p=p)
        sev = Gamma(shape=alpha, scale=1/beta)
        compound = BinomialGammaCompound(freq, sev)
        
        # Expected values
        mean_sev = alpha / beta
        var_sev = alpha / (beta**2)
        expected_mean = n * p * mean_sev
        expected_var = n * p * (var_sev + (1 - p) * mean_sev**2)
        
        assert np.isclose(compound.mean(), expected_mean)
        assert np.isclose(compound.var(), expected_var)
    
    def test_special_case_exponential(self):
        """Test that Gamma(1, β) reduces to Exponential case."""
        n, p = 8, 0.25
        beta = 0.002  # scale = 500
        
        freq = Binomial(n=n, p=p)
        sev_gamma = Gamma(shape=1.0, scale=1/beta)
        sev_exp = Exponential(scale=1/beta)
        
        compound_gamma = BinomialGammaCompound(freq, sev_gamma)
        compound_exp = BinomialExponentialCompound(freq, sev_exp)
        
        # Should have same mean and variance
        assert np.isclose(compound_gamma.mean(), compound_exp.mean())
        assert np.isclose(compound_gamma.var(), compound_exp.var())
        
        # PDF should match at several points
        x_test = np.array([0, 100, 500, 1000, 2000])
        pdf_gamma = compound_gamma.pdf(x_test)
        pdf_exp = compound_exp.pdf(x_test)
        
        np.testing.assert_allclose(pdf_gamma, pdf_exp, rtol=1e-6)
    
    def test_random_sampling(self):
        """Test random variate generation."""
        freq = Binomial(n=10, p=0.6)
        sev = Gamma(shape=3.0, scale=50)
        compound = BinomialGammaCompound(freq, sev)
        
        # Generate samples
        np.random.seed(123)
        samples = compound.rvs(size=10000)
        
        # Check sample statistics
        assert np.isclose(np.mean(samples), compound.mean(), rtol=0.05)
        assert np.isclose(np.var(samples), compound.var(), rtol=0.1)


class TestBinomialLognormalCompound:
    """Test Binomial-Lognormal compound distribution."""
    
    def test_mean_variance(self):
        """Test analytical mean and variance calculations."""
        n, p = 12, 0.35
        mu, sigma = 6.0, 1.2
        
        freq = Binomial(n=n, p=p)
        sev = Lognormal(shape=sigma, scale=np.exp(mu))
        compound = BinomialLognormalCompound(freq, sev)
        
        # Expected values
        mean_sev = np.exp(mu + sigma**2 / 2)
        var_sev = (np.exp(sigma**2) - 1) * mean_sev**2
        expected_mean = n * p * mean_sev
        expected_var = n * p * (var_sev + (1 - p) * mean_sev**2)
        
        assert np.isclose(compound.mean(), expected_mean)
        assert np.isclose(compound.var(), expected_var)
    
    def test_fenton_wilkinson_approximation(self):
        """Test Fenton-Wilkinson parameter calculation."""
        freq = Binomial(n=5, p=0.5)
        sev = Lognormal(shape=0.5, scale=1000)
        compound = BinomialLognormalCompound(freq, sev)
        
        # Test for different k values
        for k in [1, 2, 3, 4, 5]:
            mu_k, sigma_k = compound._fenton_wilkinson_params(k)
            
            # Approximate sum should have correct mean
            approx_mean = np.exp(mu_k + sigma_k**2 / 2)
            exact_mean = k * np.exp(compound.mu + compound.sigma**2 / 2)
            
            assert np.isclose(approx_mean, exact_mean, rtol=1e-6)
    
    def test_pdf_normalization(self):
        """Test that PDF integrates to 1."""
        freq = Binomial(n=3, p=0.4)
        sev = Lognormal(shape=0.8, scale=500)
        compound = BinomialLognormalCompound(freq, sev)
        
        # Create fine grid for integration
        x_max = compound.mean() + 5 * compound.std()
        x_grid = np.linspace(0, x_max, 5000)
        
        # Include atom at zero
        p0 = compound.pdf(0)
        
        # Integrate continuous part
        pdf_vals = compound.pdf(x_grid[1:])
        integral = np.trapz(pdf_vals, x_grid[1:])
        
        total = p0 + integral
        assert np.isclose(total, 1.0, rtol=1e-2)
    
    def test_random_sampling(self):
        """Test random variate generation."""
        freq = Binomial(n=8, p=0.7)
        sev = Lognormal(shape=1.0, scale=1000)
        compound = BinomialLognormalCompound(freq, sev)
        
        # Generate samples
        np.random.seed(456)
        samples = compound.rvs(size=10000)
        
        # Check sample statistics
        assert np.isclose(np.mean(samples), compound.mean(), rtol=0.05)
        assert np.isclose(np.std(samples), compound.std(), rtol=0.1)
        
        # Check that zero appears with correct frequency
        p_zero_empirical = np.mean(samples == 0)
        p_zero_theoretical = (1 - 0.7) ** 8
        # Use absolute tolerance for very small probabilities
        assert np.isclose(p_zero_empirical, p_zero_theoretical, atol=5e-5)


class TestPanjerBinomialRecursion:
    """Test Panjer recursion for binomial frequency."""
    
    def test_simple_case(self):
        """Test Panjer recursion with simple discrete severity."""
        n, p = 3, 0.5
        
        # Simple severity: P(X=1) = 0.7, P(X=2) = 0.3
        severity_pmf = {1: 0.7, 2: 0.3}
        
        recursion = PanjerBinomialRecursion(n, p, severity_pmf)
        agg_pmf = recursion.compute_aggregate_pmf(max_value=6)
        
        # Check probabilities sum to 1
        total_prob = sum(agg_pmf.values())
        assert np.isclose(total_prob, 1.0)
        
        # Check P(S=0) = (1-p)^n
        assert np.isclose(agg_pmf[0], 0.5**3)
        
        # Manual calculation for P(S=1)
        # S=1 only if N=1 and X=1
        p_s1 = stats.binom.pmf(1, n, p) * 0.7
        assert np.isclose(agg_pmf[1], p_s1)
    
    def test_recursion_coefficients(self):
        """Test that recursion produces correct results."""
        n, p = 5, 0.4
        severity_pmf = {10: 0.3, 20: 0.5, 30: 0.2}
        
        recursion = PanjerBinomialRecursion(n, p, severity_pmf)
        agg_pmf = recursion.compute_aggregate_pmf()
        
        # Compare with direct calculation for small values
        # P(S=10) = P(N=1)*P(X=10)
        p_s10_direct = stats.binom.pmf(1, n, p) * 0.3
        assert np.isclose(agg_pmf.get(10, 0), p_s10_direct)
        
        # P(S=20) = P(N=1)*P(X=20) + P(N=2)*P(X=10)^2
        p_s20_direct = (stats.binom.pmf(1, n, p) * 0.5 + 
                        stats.binom.pmf(2, n, p) * 0.3**2)
        assert np.isclose(agg_pmf.get(20, 0), p_s20_direct, rtol=1e-6)
    
    def test_comparison_with_simulation(self):
        """Compare Panjer recursion with Monte Carlo simulation."""
        n, p = 4, 0.6
        severity_pmf = {100: 0.4, 200: 0.4, 300: 0.2}
        
        # Panjer recursion
        recursion = PanjerBinomialRecursion(n, p, severity_pmf)
        agg_pmf = recursion.compute_aggregate_pmf()
        
        # Monte Carlo simulation
        np.random.seed(789)
        n_sims = 100000
        samples = []
        
        for _ in range(n_sims):
            n_claims = stats.binom.rvs(n, p)
            if n_claims == 0:
                total = 0
            else:
                severities = np.random.choice(
                    list(severity_pmf.keys()),
                    size=n_claims,
                    p=list(severity_pmf.values())
                )
                total = np.sum(severities)
            samples.append(total)
        
        # Compare probabilities
        for value, prob in agg_pmf.items():
            if prob > 0.001:  # Only check significant probabilities
                empirical_prob = np.mean(np.array(samples) == value)
                assert np.isclose(prob, empirical_prob, rtol=0.08)


def test_integration_with_factory():
    """Test that compound binomial distributions integrate with factory."""
    from quactuary.distributions.compound import create_compound_distribution
    
    # Test each binomial compound type
    freq = Binomial(n=10, p=0.4)
    
    # Exponential (should return BinomialExponentialCompound)
    sev_exp = Exponential(scale=100)
    compound_exp = create_compound_distribution(freq, sev_exp)
    assert isinstance(compound_exp, BinomialExponentialCompound)
    
    # Gamma (should return BinomialGammaCompound) 
    sev_gamma = Gamma(shape=2.0, scale=50)
    compound_gamma = create_compound_distribution(freq, sev_gamma)
    assert isinstance(compound_gamma, BinomialGammaCompound)
    
    # Lognormal (should return BinomialLognormalCompound)
    sev_lognorm = Lognormal(shape=1.0, scale=1000)
    compound_lognorm = create_compound_distribution(freq, sev_lognorm)
    assert isinstance(compound_lognorm, BinomialLognormalCompound)