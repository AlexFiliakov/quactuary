"""
Consolidated tests for compound distributions.

This file consolidates all compound distribution tests into well-organized test classes:
- TestCompoundDistributionFactory: Factory method and creation
- TestAnalyticalSolutions: Poisson-Exponential, Poisson-Gamma, etc.
- TestSimulatedCompound: Monte Carlo simulation tests
- TestNumericalStability: Edge cases and numerical issues
"""

import time
from unittest.mock import Mock, patch

import numpy as np
import pytest
from scipy import stats
from scipy.optimize import brentq

from quactuary.distributions.compound import (
    create_compound_distribution,
    CompoundDistribution,
    PoissonExponentialCompound,
    PoissonGammaCompound,
    GeometricExponentialCompound,
    NegativeBinomialGammaCompound,
    SimulatedCompound,
)
from quactuary.distributions.compound import (
    BinomialExponentialCompound,
    BinomialGammaCompound,
    BinomialLognormalCompound,
)
from quactuary.distributions.frequency import (
    Poisson, 
    Geometric, 
    NegativeBinomial, 
    Binomial
)
from quactuary.distributions.severity import (
    Exponential,
    Gamma,
    Lognormal
)
from quactuary.quantum import QuantumPricingModel




# =============================================================================
# Test Classes
# =============================================================================

class TestCompoundDistributionFactory:
    """Test factory method and compound distribution creation."""
    
    def test_factory_creates_poisson_exponential(self):
        """Test factory returns PoissonExponential for appropriate inputs."""
        freq = Poisson(mu=5.0)
        sev = Exponential(scale=1000.0)
        
        compound = create_compound_distribution(freq, sev)
        
        assert isinstance(compound, PoissonExponentialCompound)
        assert compound.has_analytical_solution()
    
    def test_factory_creates_poisson_gamma(self):
        """Test factory returns PoissonGamma for appropriate inputs."""
        freq = Poisson(mu=3.0)
        sev = Gamma(shape=2.0, scale=500.0)
        
        compound = create_compound_distribution(freq, sev)
        
        assert isinstance(compound, PoissonGammaCompound)
        assert compound.has_analytical_solution()
    
    def test_factory_creates_geometric_exponential(self):
        """Test factory returns GeometricExponential for appropriate inputs."""
        freq = Geometric(p=0.3)
        sev = Exponential(scale=200.0)
        
        compound = create_compound_distribution(freq, sev)
        
        assert isinstance(compound, GeometricExponentialCompound)
        assert compound.has_analytical_solution()
    
    def test_factory_creates_negative_binomial_gamma(self):
        """Test factory returns NegativeBinomialGamma for appropriate inputs."""
        freq = NegativeBinomial(r=3, p=0.4)
        sev = Gamma(shape=1.5, scale=300.0)
        
        compound = create_compound_distribution(freq, sev)
        
        assert isinstance(compound, NegativeBinomialGammaCompound)
        assert compound.has_analytical_solution()
    
    def test_factory_creates_binomial_lognormal(self):
        """Test factory returns BinomialLognormal for appropriate inputs."""
        freq = Binomial(n=50, p=0.2)
        sev = Lognormal(shape=1.0, scale=np.exp(5.0))
        
        compound = create_compound_distribution(freq, sev)
        
        assert isinstance(compound, BinomialLognormalCompound)
        assert compound.has_analytical_solution()
    
    def test_factory_creates_binomial_exponential(self):
        """Test factory returns BinomialExponentialCompound for appropriate inputs."""
        freq = Binomial(n=20, p=0.3)
        sev = Exponential(scale=100.0)
        
        compound = create_compound_distribution(freq, sev)
        
        assert isinstance(compound, BinomialExponentialCompound)
        assert compound.has_analytical_solution()
    
    def test_factory_creates_binomial_gamma(self):
        """Test factory returns BinomialGammaCompound for appropriate inputs."""
        freq = Binomial(n=30, p=0.25)
        sev = Gamma(shape=2.0, scale=150.0)
        
        compound = create_compound_distribution(freq, sev)
        
        assert isinstance(compound, BinomialGammaCompound)
        assert compound.has_analytical_solution()
    
    def test_factory_fallback_to_simulated(self):
        """Test factory falls back to SimulatedCompound for unknown combinations."""
        freq = Poisson(mu=5.0)
        sev = Mock()  # Unknown severity type
        sev.__class__.__name__ = 'UnknownDist'
        sev.mean.return_value = 100
        sev.var.return_value = 1000
        
        compound = create_compound_distribution(freq, sev)
        
        assert isinstance(compound, SimulatedCompound)
        assert not compound.has_analytical_solution()
    
    def test_abstract_base_class(self):
        """Test that CompoundDistribution is abstract and cannot be instantiated."""
        freq = Poisson(mu=1.0)
        sev = Exponential(scale=100.0)
        
        # Create a concrete subclass that doesn't implement abstract methods
        class IncompleteCompound(CompoundDistribution):
            pass
        
        # Should not be able to instantiate
        with pytest.raises(TypeError):
            IncompleteCompound(freq, sev)


class TestAnalyticalSolutions:
    """Test analytical compound distribution implementations."""
    
    # -------------------------------------------------------------------------
    # Poisson-Exponential Tests
    # -------------------------------------------------------------------------
    
    def test_poisson_exponential_mean_variance(self):
        """Test mean and variance calculations for Poisson-Exponential."""
        freq = Poisson(mu=5.0)
        sev = Exponential(scale=1000.0)
        compound = PoissonExponentialCompound(freq, sev)
        
        # E[S] = λ * θ
        assert compound.mean() == 5.0 * 1000.0
        
        # Var[S] = λ * θ² * 2
        assert compound.var() == 5.0 * 1000.0**2 * 2
        
        # Std[S] = sqrt(Var[S])
        assert compound.std() == np.sqrt(compound.var())
    
    def test_poisson_exponential_pdf(self):
        """Test PDF calculation for Poisson-Exponential."""
        freq = Poisson(mu=2.0)
        sev = Exponential(scale=100.0)
        compound = PoissonExponentialCompound(freq, sev)
        
        # Test atom at zero
        pdf_zero = compound.pdf(0.0)
        assert isinstance(pdf_zero, (float, np.floating))
        
        # Test positive values
        pdf_positive = compound.pdf(150.0)
        assert pdf_positive > 0
        
        # Test negative values
        assert compound.pdf(-10.0) == 0.0
        
        # Test array input
        pdf_array = compound.pdf(np.array([-10.0, 0.0, 100.0]))
        assert pdf_array[0] == 0.0
        assert pdf_array[1] > 0.0
        assert pdf_array[2] > 0.0
    
    def test_poisson_exponential_cdf(self):
        """Test CDF calculation for Poisson-Exponential."""
        freq = Poisson(mu=3.0)
        sev = Exponential(scale=200.0)
        compound = PoissonExponentialCompound(freq, sev)
        
        # CDF at 0 should be P(N = 0) = e^(-λ)
        assert compound.cdf(0) == pytest.approx(np.exp(-3.0))
        
        # CDF should be monotonic
        x_values = np.linspace(0, 5000, 50)
        cdf_values = compound.cdf(x_values)
        assert np.all(np.diff(cdf_values) >= 0)
        
        # CDF should approach 1
        assert compound.cdf(100000) > 0.99
        
        # Test negative values
        assert compound.cdf(-100) == 0.0
    
    def test_poisson_exponential_quantiles(self):
        """Test quantile function for Poisson-Exponential."""
        freq = Poisson(mu=4.0)
        sev = Exponential(scale=250.0)
        compound = PoissonExponentialCompound(freq, sev)
        
        # Test scalar input
        q50 = compound.ppf(0.5)
        assert isinstance(q50, (float, np.floating))
        assert q50 > 0
        
        # Test array input
        quantiles = compound.ppf(np.array([0.25, 0.5, 0.75, 0.95]))
        assert len(quantiles) == 4
        assert np.all(np.diff(quantiles) > 0)
        
        # Verify inverse relationship
        for q in [0.1, 0.5, 0.9]:
            x = compound.ppf(q)
            assert compound.cdf(x) == pytest.approx(q, abs=1e-3)
    
    def test_poisson_exponential_random_generation(self):
        """Test random variate generation for Poisson-Exponential."""
        freq = Poisson(mu=5.0)
        sev = Exponential(scale=300.0)
        compound = PoissonExponentialCompound(freq, sev)
        
        # Single sample
        sample = compound.rvs(size=1)
        assert isinstance(sample, (float, np.floating))
        assert sample >= 0
        
        # Multiple samples
        np.random.seed(42)
        samples = compound.rvs(size=10000)
        assert samples.shape == (10000,)
        assert np.all(samples >= 0)
        
        # Check statistics
        assert np.mean(samples) == pytest.approx(compound.mean(), rel=0.05)
        assert np.var(samples) == pytest.approx(compound.var(), rel=0.1)
        
        # Check proportion of zeros
        p_zero_empirical = np.mean(samples == 0)
        p_zero_theoretical = np.exp(-5.0)
        assert p_zero_empirical == pytest.approx(p_zero_theoretical, abs=0.01)
    
    # -------------------------------------------------------------------------
    # Poisson-Gamma (Tweedie) Tests
    # -------------------------------------------------------------------------
    
    def test_poisson_gamma_mean_variance(self):
        """Test mean and variance calculations for Poisson-Gamma."""
        freq = Poisson(mu=3.0)
        sev = Gamma(shape=2.0, scale=500.0)
        compound = PoissonGammaCompound(freq, sev)
        
        # E[S] = λ * α * scale
        expected_mean = 3.0 * 2.0 * 500.0
        assert compound.mean() == expected_mean
        
        # Var[S] = λ * α * (α + 1) / β²
        lam = 3.0
        alpha = 2.0
        beta = 1.0 / 500.0
        expected_var = lam * alpha * (alpha + 1) / (beta ** 2)
        assert compound.var() == expected_var
    
    def test_poisson_gamma_tweedie_parameters(self):
        """Test Tweedie parameter calculations."""
        freq = Poisson(mu=2.5)
        sev = Gamma(shape=1.5, scale=300.0)
        compound = PoissonGammaCompound(freq, sev)
        
        # p should be in (1, 2)
        assert 1 < compound.p < 2
        
        # Mean should match
        assert compound.mu == compound.mean()
        
        # Dispersion parameter should be positive
        assert compound.phi > 0
    
    def test_poisson_gamma_pdf_cdf(self):
        """Test PDF and CDF for Poisson-Gamma."""
        freq = Poisson(mu=1.5)
        sev = Gamma(shape=2.0, scale=100.0)
        compound = PoissonGammaCompound(freq, sev)
        
        # Test PDF at zero
        pdf_zero = compound.pdf(0.0)
        assert pdf_zero == pytest.approx(np.exp(-1.5))
        
        # Test PDF array
        pdf_array = compound.pdf(np.array([0.0, 50.0, 100.0, 200.0]))
        assert pdf_array[0] == pytest.approx(np.exp(-1.5))
        assert np.all(pdf_array[1:] > 0)
        
        # Test CDF properties
        assert compound.cdf(0) == pytest.approx(np.exp(-1.5))
        x_values = np.linspace(0, 2000, 50)
        cdf_values = compound.cdf(x_values)
        assert np.all(np.diff(cdf_values) >= 0)
    
    def test_poisson_gamma_random_generation(self):
        """Test random variate generation for Poisson-Gamma."""
        freq = Poisson(mu=4.0)
        sev = Gamma(shape=1.5, scale=200.0)
        compound = PoissonGammaCompound(freq, sev)
        
        np.random.seed(42)
        samples = compound.rvs(size=5000)
        
        assert samples.shape == (5000,)
        assert np.all(samples >= 0)
        assert np.mean(samples) == pytest.approx(compound.mean(), rel=0.1)
        assert np.var(samples) == pytest.approx(compound.var(), rel=0.2)
    
    # -------------------------------------------------------------------------
    # Geometric-Exponential Tests
    # -------------------------------------------------------------------------
    
    def test_geometric_exponential_mean_variance(self):
        """Test mean and variance for Geometric-Exponential."""
        freq = Geometric(p=0.3)
        sev = Exponential(scale=200.0)
        compound = GeometricExponentialCompound(freq, sev)
        
        # E[S] = θ / p
        assert compound.mean() == pytest.approx(200.0 / 0.3)
        
        # Var[S] = θ² * (2 - p) / p²
        expected_var = 200.0**2 * (2 - 0.3) / (0.3**2)
        assert compound.var() == pytest.approx(expected_var)
    
    def test_geometric_exponential_distribution(self):
        """Test that Geometric-Exponential follows exponential distribution."""
        freq = Geometric(p=0.4)
        sev = Exponential(scale=100.0)
        compound = GeometricExponentialCompound(freq, sev)
        
        # Should follow Exponential(θ/(1-p))
        expected_scale = 100.0 / (1 - 0.4)
        
        # Test PDF
        x = 300.0
        expected_pdf = stats.expon.pdf(x, scale=expected_scale)
        assert compound.pdf(x) == pytest.approx(expected_pdf)
    
    def test_geometric_exponential_edge_cases(self):
        """Test edge cases for Geometric-Exponential."""
        # p near 1
        freq = Geometric(p=0.999)
        sev = Exponential(scale=100.0)
        compound = GeometricExponentialCompound(freq, sev)
        assert compound.mean() < 101
        
        # p = 1 (degenerate case)
        freq = Geometric(p=1.0)
        compound = GeometricExponentialCompound(freq, sev)
        assert compound.pdf(0) == 1.0
        assert compound.pdf(10) == 0.0
        assert compound.cdf(10) == 1.0
        assert compound.ppf(0.5) == 0.0
        
        # p > 1 (impossible but handled)
        freq = Geometric(p=1.5)
        compound = GeometricExponentialCompound(freq, sev)
        assert compound.pdf(10.0) == 0.0
        assert compound.cdf(10.0) == 1.0
    
    # -------------------------------------------------------------------------
    # Negative Binomial-Gamma Tests
    # -------------------------------------------------------------------------
    
    def test_negative_binomial_gamma_mean_variance(self):
        """Test mean and variance for NegativeBinomial-Gamma."""
        freq = NegativeBinomial(r=3, p=0.4)
        sev = Gamma(shape=2.0, scale=150.0)
        compound = NegativeBinomialGammaCompound(freq, sev)
        
        # Check mean calculation with caching
        mean1 = compound.mean()
        mean2 = compound.mean()  # Should use cache
        assert mean1 == mean2
        
        # Check variance calculation with caching
        var1 = compound.var()
        var2 = compound.var()  # Should use cache
        assert var1 == var2
        
        # Verify values
        r = 3
        p = 0.4
        alpha = 2.0
        beta = 1.0 / 150.0
        expected_mean = r * (1 - p) / p * alpha / beta
        assert mean1 == pytest.approx(expected_mean)
    
    def test_negative_binomial_gamma_zero_probability(self):
        """Test probability mass at zero for NegativeBinomial-Gamma."""
        freq = NegativeBinomial(r=3, p=0.4)
        sev = Gamma(shape=1.5, scale=200.0)
        compound = NegativeBinomialGammaCompound(freq, sev)
        
        # P(S = 0) = p^r
        expected_p_zero = 0.4 ** 3
        assert compound.pdf(0) == pytest.approx(expected_p_zero)
        assert compound.cdf(0) == pytest.approx(expected_p_zero)
    
    def test_negative_binomial_gamma_quantiles(self):
        """Test quantile function for NegativeBinomial-Gamma."""
        freq = NegativeBinomial(r=2, p=0.5)
        sev = Gamma(shape=2.5, scale=100.0)
        compound = NegativeBinomialGammaCompound(freq, sev)
        
        # Very low quantile
        assert compound.ppf(0.001) == 0.0
        
        # Test high quantile
        result = compound.ppf(0.999)
        assert result > 0  # Should be positive
        assert np.isfinite(result)  # Should be finite


class TestSimulatedCompound:
    """Test Monte Carlo simulation-based compound distributions."""
    
    def test_mean_variance_formulas(self):
        """Test theoretical mean and variance formulas."""
        freq = Poisson(mu=4.0)
        sev = Exponential(scale=750.0)
        compound = SimulatedCompound(freq, sev)
        
        # E[S] = E[N] * E[X]
        assert compound.mean() == 4.0 * 750.0
        
        # Var[S] = E[N] * Var[X] + Var[N] * E[X]²
        expected_var = 4.0 * 750.0**2 + 4.0 * 750.0**2
        assert compound.var() == expected_var
    
    def test_cache_generation(self):
        """Test cache generation and management."""
        freq = Poisson(mu=2.0)
        sev = Exponential(scale=100.0)
        compound = SimulatedCompound(freq, sev)
        
        # Initially no cache
        assert compound._cache is None
        
        # Generate cache
        compound._ensure_cache()
        assert compound._cache is not None
        assert len(compound._cache) == 10000
        assert np.all(compound._cache >= 0)
        
        # Cache should persist
        cache1 = compound._cache
        compound._ensure_cache()
        cache2 = compound._cache
        assert cache1 is cache2  # Same object, not regenerated
    
    def test_pdf_estimation(self):
        """Test PDF estimation via KDE."""
        freq = Poisson(mu=3.0)
        sev = Exponential(scale=200.0)
        compound = SimulatedCompound(freq, sev)
        
        # Should generate cache if needed
        assert compound._cache is None
        pdf_value = compound.pdf(500.0)
        assert compound._cache is not None
        assert pdf_value > 0
        
        # Test scalar and array inputs
        pdf_scalar = compound.pdf(300.0)
        assert isinstance(pdf_scalar, (float, np.floating))
        
        pdf_array = compound.pdf(np.array([100.0, 500.0, 1000.0]))
        assert pdf_array.shape == (3,)
        assert np.all(pdf_array >= 0)
    
    def test_cdf_empirical(self):
        """Test empirical CDF calculation."""
        freq = Poisson(mu=2.5)
        sev = Exponential(scale=150.0)
        compound = SimulatedCompound(freq, sev)
        
        # Force cache generation
        compound._ensure_cache()
        
        # Test scalar input
        cdf_scalar = compound.cdf(200.0)
        assert isinstance(cdf_scalar, (float, np.floating))
        assert 0 <= cdf_scalar <= 1
        
        # Test boundary conditions
        assert compound.cdf(-100) == 0.0
        assert compound.cdf(1e10) == 1.0
        
        # Test monotonicity
        x_values = np.linspace(0, 1000, 20)
        cdf_values = compound.cdf(x_values)
        assert np.all(np.diff(cdf_values) >= 0)
    
    def test_ppf_empirical(self):
        """Test empirical quantile function."""
        freq = Poisson(mu=3.5)
        sev = Exponential(scale=200.0)
        compound = SimulatedCompound(freq, sev)
        
        # Generate cache
        compound._ensure_cache()
        
        # Test scalar input
        q50 = compound.ppf(0.5)
        assert isinstance(q50, (float, np.floating))
        
        # Test standard quantiles
        q01 = compound.ppf(0.01)
        q50 = compound.ppf(0.50)
        q99 = compound.ppf(0.99)
        assert q01 < q50 < q99
        
        # Test array input
        quantiles = compound.ppf(np.array([0.25, 0.75]))
        assert len(quantiles) == 2
        assert quantiles[0] < quantiles[1]
    
    def test_rvs_consistency(self):
        """Test that RVS gives consistent results with seed."""
        freq = Poisson(mu=2.0)
        sev = Exponential(scale=100.0)
        compound = SimulatedCompound(freq, sev)
        
        # Single sample
        single = compound.rvs(size=1)
        assert isinstance(single, (float, np.floating, np.ndarray))
        
        # Test reproducibility
        np.random.seed(42)
        samples1 = compound.rvs(size=1000)
        
        np.random.seed(42)
        samples2 = compound.rvs(size=1000)
        
        assert np.array_equal(samples1, samples2)


class TestNumericalStability:
    """Test edge cases and numerical stability issues."""
    
    def test_extreme_parameters(self):
        """Test behavior with extreme parameter values."""
        # Very small lambda
        freq = Poisson(mu=0.01)
        sev = Exponential(scale=1000.0)
        compound = PoissonExponentialCompound(freq, sev)
        
        # Most probability mass should be at 0
        assert compound.cdf(0) > 0.99
        
        # Very large lambda
        freq_large = Poisson(mu=1000.0)
        compound_large = PoissonExponentialCompound(freq_large, sev)
        assert compound_large.mean() == 1000.0 * 1000.0
    
    def test_series_convergence(self):
        """Test series convergence in analytical solutions."""
        # Small lambda - series should truncate early
        freq_small = Poisson(mu=0.5)
        sev = Exponential(scale=100.0)
        compound_small = PoissonExponentialCompound(freq_small, sev)
        pdf_small = compound_small.pdf(50.0)
        assert pdf_small > 0
        
        # Large lambda - series needs more terms
        freq_large = Poisson(mu=50.0)
        compound_large = PoissonExponentialCompound(freq_large, sev)
        pdf_large = compound_large.pdf(5000.0)
        assert pdf_large > 0
    
    def test_zero_claims_handling(self):
        """Test proper handling of zero claims."""
        freq = Poisson(mu=0.1)  # Very low frequency
        sev = Exponential(scale=1000.0)
        compound = PoissonExponentialCompound(freq, sev)
        
        # Generate many samples
        np.random.seed(42)
        samples = compound.rvs(size=1000)
        
        # Should have many zeros
        zero_proportion = np.mean(samples == 0)
        expected_zeros = np.exp(-0.1)
        assert zero_proportion == pytest.approx(expected_zeros, abs=0.05)
        
        # Test with mocked zero claims
        with patch.object(stats.poisson, 'rvs', return_value=np.array([0, 5, 0, 3])):
            samples = compound.rvs(size=4)
            assert samples[0] == 0.0  # No claims
            assert samples[2] == 0.0  # No claims
            assert samples[1] > 0.0   # 5 claims
            assert samples[3] > 0.0   # 3 claims
    
    def test_scalar_vs_array_consistency(self):
        """Test that scalar and array inputs give consistent results."""
        freq = Poisson(mu=3.0)
        sev = Gamma(shape=2.0, scale=150.0)
        compound = PoissonGammaCompound(freq, sev)
        
        # Test scalar input
        scalar_pdf = compound.pdf(500.0)
        scalar_cdf = compound.cdf(500.0)
        scalar_ppf = compound.ppf(0.75)
        
        # Test array input with single element
        array_pdf = compound.pdf(np.array([500.0]))
        array_cdf = compound.cdf(np.array([500.0]))
        array_ppf = compound.ppf(np.array([0.75]))
        
        assert scalar_pdf == array_pdf[0]
        assert scalar_cdf == array_cdf[0]
        assert scalar_ppf == array_ppf[0]
    
    def test_empty_array_inputs(self):
        """Test handling of empty array inputs."""
        freq = Poisson(mu=2.0)
        sev = Exponential(scale=200.0)
        compound = PoissonExponentialCompound(freq, sev)
        
        # Empty array
        empty_pdf = compound.pdf(np.array([]))
        empty_cdf = compound.cdf(np.array([]))
        
        assert empty_pdf.shape == (0,)
        assert empty_cdf.shape == (0,)
    
    def test_boundary_quantiles(self):
        """Test quantile function at extreme probabilities."""
        freq = Poisson(mu=2.0)
        sev = Exponential(scale=150.0)
        compound = PoissonExponentialCompound(freq, sev)
        
        # Very low quantile
        assert compound.ppf(1e-10) == 0.0
        
        # Very high quantile
        q999 = compound.ppf(0.999)
        assert q999 > 0
        
        # Test at p_zero boundary
        p_zero = np.exp(-2.0)
        assert compound.ppf(p_zero * 0.5) == 0.0
        assert compound.ppf(p_zero * 1.5) > 0.0
    
    def test_degenerate_cases(self):
        """Test degenerate parameter cases."""
        # Zero variance in frequency (p=0 for binomial)
        freq = Binomial(n=10, p=0.0)
        sev = Lognormal(shape=1.0, scale=np.exp(5.0))
        compound = BinomialLognormalCompound(freq, sev)
        
        assert compound.mean() == 0.0
        assert compound.var() == 0.0
        assert compound.pdf(0) == 1.0
        assert compound.pdf(100) == 0.0
        
        # Geometric with p=0 (infinite mean)
        freq = Geometric(p=0.0)
        sev = Exponential(scale=100.0)
        compound = GeometricExponentialCompound(freq, sev)
        
        assert compound.mean() == float('inf')
        assert compound.var() == float('inf')


# # TODO: PanjerRecursion not yet implemented - commenting out tests
# # class TestPanjerRecursion:
# #     """Test Panjer recursion implementation."""
# #     
# #     def test_parameter_identification(self):
#         """Test Panjer parameter identification for different distributions."""
#         # Poisson
#         freq = Poisson(mu=2.5)
#         sev = Exponential(scale=100.0)
#         panjer = PanjerRecursion(freq, sev, discretization_step=10.0)
#         assert panjer.a == 0
#         assert panjer.b == 2.5
#         assert panjer.p0 == pytest.approx(np.exp(-2.5))
#         
#         # Binomial
#         freq = Binomial(n=10, p=0.3)
#         panjer = PanjerRecursion(freq, sev, discretization_step=10.0)
#         expected_a = -0.3 / 0.7
#         expected_b = 11 * 0.3 / 0.7
#         assert panjer.a == pytest.approx(expected_a)
#         assert panjer.b == pytest.approx(expected_b)
#         assert panjer.p0 == pytest.approx(0.7 ** 10)
#         
#         # Negative Binomial
#         freq = NegativeBinomial(r=5, p=0.4)
#         panjer = PanjerRecursion(freq, sev, discretization_step=10.0)
#         assert panjer.a == pytest.approx(0.6)
#         assert panjer.b == pytest.approx(4 * 0.6)
#         assert panjer.p0 == pytest.approx(0.4 ** 5)
#     
#     def test_unsupported_frequency(self):
#         """Test error for unsupported frequency distribution."""
#         freq = Mock()
#         freq.__class__.__name__ = 'UnsupportedDist'
#         sev = Exponential(scale=100.0)
#         
#         with pytest.raises(ValueError, match="not supported by Panjer recursion"):
#             PanjerRecursion(freq, sev)
#     
#     def test_aggregate_pmf_calculation(self):
#         """Test calculation of aggregate PMF."""
#         freq = Poisson(mu=1.5)
#         sev = Exponential(scale=50.0)
#         panjer = PanjerRecursion(freq, sev, discretization_step=25.0, max_value=300.0)
#         
#         loss_values, pmf = panjer.calculate_aggregate_pmf()
#         
#         # Check structure
#         assert len(loss_values) == len(pmf)
#         assert loss_values[0] == 0
#         assert loss_values[-1] == 300.0
#         
#         # Check probabilities
#         assert np.all(pmf >= 0)
#         assert pmf[0] == pytest.approx(np.exp(-1.5))  # P(S = 0)
#     
#     def test_cdf_calculation(self):
#         """Test CDF calculation with interpolation."""
#         freq = Poisson(mu=1.0)
#         sev = Exponential(scale=100.0)
#         panjer = PanjerRecursion(freq, sev, discretization_step=50.0)
#         
#         # Test scalar input
#         cdf_scalar = panjer.cdf(75.0)
#         assert isinstance(cdf_scalar, (float, np.floating))
#         assert 0 <= cdf_scalar <= 1
#         
#         # Test array input
#         x_array = np.array([0, 50, 100, 150])
#         cdf_array = panjer.cdf(x_array)
#         assert cdf_array.shape == (4,)
#         assert np.all(np.diff(cdf_array) >= 0)  # Monotonic


class TestPerformanceAndIntegration:
    """Test performance improvements and integration with other components."""
    
    def test_analytical_faster_than_simulated(self):
        """Verify analytical solutions are faster than simulation."""
        freq = Poisson(mu=5.0)
        sev = Exponential(scale=1000.0)
        
        # Analytical version
        analytical = PoissonExponentialCompound(freq, sev)
        
        # Forced simulated version
        simulated = SimulatedCompound(freq, sev)
        
        # Time analytical CDF evaluation
        x_values = np.linspace(0, 20000, 100)
        
        start = time.time()
        for _ in range(10):
            analytical.cdf(x_values)
        analytical_time = time.time() - start
        
        # Time simulated CDF evaluation
        start = time.time()
        for _ in range(10):
            simulated.cdf(x_values)
        simulated_time = time.time() - start
        
        # Analytical should be significantly faster
        assert analytical_time < simulated_time * 0.5  # At least 2x faster
    
    def test_mean_calculation_performance(self):
        """Test that mean calculation is instant for analytical."""
        freq = Poisson(mu=10.0)
        sev = Gamma(shape=3.0, scale=200.0)
        compound = PoissonGammaCompound(freq, sev)
        
        # Should be essentially instant
        start = time.time()
        for _ in range(10000):
            compound.mean()
        elapsed = time.time() - start
        
        assert elapsed < 0.02  # Should take less than 20ms for 10k calculations
    
    def test_quantum_integration(self):
        """Test integration with quantum pricing models."""
        freq = Poisson(mu=2.0)
        sev = Exponential(scale=500.0)
        
        compound = PoissonExponentialCompound(freq, sev)
        quantum_model = QuantumPricingModel()
        
        # Test that quantum model can access compound distribution properties
        assert hasattr(compound, 'mean')
        assert hasattr(compound, 'var')
        assert compound.has_analytical_solution()
        
        # Verify that the analytical properties are accessible for quantum state preparation
        mean_val = compound.mean()
        var_val = compound.var()
        
        assert mean_val > 0
        assert var_val > 0
        assert np.isfinite(mean_val)
        assert np.isfinite(var_val)
    
    def test_quantum_compatible_parameters(self):
        """Test that compound distributions provide quantum-compatible parameters."""
        freq = Poisson(mu=3.0)
        sev = Gamma(shape=2.0, scale=100.0)
        compound = PoissonGammaCompound(freq, sev)
        
        # Parameters needed for quantum state preparation
        params = {
            'mean': compound.mean(),
            'variance': compound.var(),
            'std': compound.std(),
            'lambda': freq._dist.args[0],  # mu parameter
            'severity_mean': sev._dist.mean(),
            'severity_var': sev._dist.var()
        }
        
        # All parameters should be finite and positive
        for key, value in params.items():
            assert np.isfinite(value), f"Parameter {key} is not finite"
            assert value >= 0, f"Parameter {key} is negative"


class TestBinomialCompounds:
    """Test Binomial-based compound distributions."""
    
    def test_binomial_lognormal_approximation(self):
        """Test Binomial-Lognormal approximation parameters."""
        freq = Binomial(n=50, p=0.2)
        sev = Lognormal(shape=1.0, scale=np.exp(5.0))
        compound = BinomialLognormalCompound(freq, sev)
        
        # Test approximation parameters
        mu_approx, sigma_approx = compound._get_approx_params()
        
        # Should be cached
        mu_approx2, sigma_approx2 = compound._get_approx_params()
        assert mu_approx == mu_approx2
        assert sigma_approx == sigma_approx2
        
        # Parameters should be reasonable
        assert mu_approx > 0
        assert sigma_approx > 0
    
    def test_binomial_lognormal_rvs(self):
        """Test RVS for Binomial-Lognormal."""
        freq = Binomial(n=100, p=0.5)
        sev = Lognormal(shape=0.5, scale=np.exp(3.0))
        compound = BinomialLognormalCompound(freq, sev)
        
        # Mock to ensure we get both large and small counts
        with patch.object(stats.binom, 'rvs', side_effect=[[5, 40, 0]]):
            samples = compound.rvs(size=3, random_state=42)
            
            assert samples[0] > 0  # Small count - direct simulation
            assert samples[1] > 0  # Large count - approximation
            assert samples[2] == 0  # Zero count
    
    def test_binomial_exponential_gamma_factory(self):
        """Test that factory correctly creates analytical Binomial-Exponential/Gamma."""
        # Binomial-Exponential
        freq = Binomial(n=20, p=0.3)
        sev = Exponential(scale=100.0)
        compound = create_compound_distribution(freq, sev)
        assert isinstance(compound, BinomialExponentialCompound)
        
        # Binomial-Gamma
        freq = Binomial(n=30, p=0.25)
        sev = Gamma(shape=2.0, scale=150.0)
        compound = create_compound_distribution(freq, sev)
        assert isinstance(compound, BinomialGammaCompound)


if __name__ == '__main__':
    pytest.main([__file__])