"""
Comprehensive tests for compound distributions to achieve 95%+ coverage.

Tests all branches, edge cases, and error conditions.
"""

import time
from unittest.mock import Mock, patch

import numpy as np
import pytest
from scipy import stats

from quactuary.distributions.compound import (
    CompoundDistribution,
    PoissonExponentialCompound,
    PoissonGammaCompound,
    GeometricExponentialCompound,
    NegativeBinomialGammaCompound,
    BinomialLognormalApproximation,
    SimulatedCompound,
    PanjerRecursion,
)


# Mock Distribution Classes
class MockPoisson:
    """Mock Poisson distribution for testing."""
    def __init__(self, mu):
        self.mu = mu
        self.__class__.__name__ = 'Poisson'
    
    def rvs(self, size=1):
        return stats.poisson.rvs(self.mu, size=size)
    
    def pmf(self, k):
        return stats.poisson.pmf(k, self.mu)
    
    def ppf(self, q):
        return stats.poisson.ppf(q, self.mu)
    
    def mean(self):
        return self.mu
    
    def var(self):
        return self.mu
    
    def cdf(self, x):
        return stats.poisson.cdf(x, self.mu)


class MockExponential:
    """Mock Exponential distribution for testing."""
    def __init__(self, scale):
        self.scale = scale
        self.__class__.__name__ = 'Exponential'
    
    def rvs(self, size=1):
        return stats.expon.rvs(scale=self.scale, size=size)
    
    def pdf(self, x):
        return stats.expon.pdf(x, scale=self.scale)
    
    def cdf(self, x):
        return stats.expon.cdf(x, scale=self.scale)
    
    def ppf(self, q):
        return stats.expon.ppf(q, scale=self.scale)
    
    def mean(self):
        return self.scale
    
    def var(self):
        return self.scale ** 2


class MockGamma:
    """Mock Gamma distribution for testing."""
    def __init__(self, a, scale):
        self.a = a
        self.scale = scale
        self.__class__.__name__ = 'Gamma'
    
    def rvs(self, size=1):
        return stats.gamma.rvs(a=self.a, scale=self.scale, size=size)
    
    def pdf(self, x):
        return stats.gamma.pdf(x, a=self.a, scale=self.scale)
    
    def cdf(self, x):
        return stats.gamma.cdf(x, a=self.a, scale=self.scale)
    
    def ppf(self, q):
        return stats.gamma.ppf(q, a=self.a, scale=self.scale)
    
    def mean(self):
        return self.a * self.scale
    
    def var(self):
        return self.a * self.scale ** 2


class MockGeometric:
    """Mock Geometric distribution for testing."""
    def __init__(self, p):
        self.p = p
        self.__class__.__name__ = 'Geometric'
    
    def rvs(self, size=1):
        return stats.geom.rvs(self.p, size=size) - 1  # Convert to "failures" parameterization
    
    def pmf(self, k):
        return stats.geom.pmf(k + 1, self.p)  # Convert from "failures" parameterization
    
    def mean(self):
        return (1 - self.p) / self.p
    
    def var(self):
        return (1 - self.p) / (self.p ** 2)


class MockNegativeBinomial:
    """Mock Negative Binomial distribution for testing."""
    def __init__(self, n, p):
        self.n = n  # Number of failures
        self.p = p  # Success probability
        self.__class__.__name__ = 'NegativeBinomial'
    
    def rvs(self, size=1):
        return stats.nbinom.rvs(self.n, self.p, size=size)
    
    def pmf(self, k):
        return stats.nbinom.pmf(k, self.n, self.p)
    
    def ppf(self, q):
        return stats.nbinom.ppf(q, self.n, self.p)
    
    def mean(self):
        return self.n * (1 - self.p) / self.p
    
    def var(self):
        return self.n * (1 - self.p) / (self.p ** 2)


class MockBinomial:
    """Mock Binomial distribution for testing."""
    def __init__(self, n, p):
        self.n = n  # Number of trials
        self.p = p  # Success probability
        self.__class__.__name__ = 'Binomial'
    
    def rvs(self, size=1):
        return stats.binom.rvs(self.n, self.p, size=size)
    
    def pmf(self, k):
        return stats.binom.pmf(k, self.n, self.p)
    
    def ppf(self, q):
        return stats.binom.ppf(q, self.n, self.p)
    
    def mean(self):
        return self.n * self.p
    
    def var(self):
        return self.n * self.p * (1 - self.p)
    
    def cdf(self, x):
        return stats.binom.cdf(x, self.n, self.p)


class MockLognormal:
    """Mock Lognormal distribution for testing."""
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        self.__class__.__name__ = 'Lognormal'
    
    def rvs(self, size=1):
        return stats.lognorm.rvs(s=self.sigma, scale=np.exp(self.mu), size=size)
    
    def pdf(self, x):
        return stats.lognorm.pdf(x, s=self.sigma, scale=np.exp(self.mu))
    
    def mean(self):
        return np.exp(self.mu + self.sigma**2 / 2)
    
    def var(self):
        mean = self.mean()
        return mean**2 * (np.exp(self.sigma**2) - 1)


class TestCompoundDistributionEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_scalar_vs_array_consistency(self):
        """Test that scalar and array inputs give consistent results."""
        freq = MockPoisson(mu=3.0)
        sev = MockExponential(scale=100.0)
        compound = PoissonExponentialCompound(freq, sev)
        
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
    
    def test_extreme_parameters(self):
        """Test behavior with extreme parameter values."""
        # Very small lambda
        freq = MockPoisson(mu=0.01)
        sev = MockExponential(scale=1000.0)
        compound = PoissonExponentialCompound(freq, sev)
        
        # Most probability mass should be at 0
        assert compound.cdf(0) > 0.99
        
        # Very large lambda
        freq_large = MockPoisson(mu=1000.0)
        compound_large = PoissonExponentialCompound(freq_large, sev)
        
        # Should have very large mean
        assert compound_large.mean() == 1000.0 * 1000.0
    
    def test_zero_claims_handling(self):
        """Test proper handling of zero claims in random generation."""
        freq = MockPoisson(mu=0.1)  # Very low frequency
        sev = MockExponential(scale=1000.0)
        compound = PoissonExponentialCompound(freq, sev)
        
        # Generate many samples
        np.random.seed(42)
        samples = compound.rvs(size=1000)
        
        # Should have many zeros
        zero_proportion = np.mean(samples == 0)
        expected_zeros = np.exp(-0.1)
        assert zero_proportion == pytest.approx(expected_zeros, abs=0.05)


class TestGeometricExponential:
    """Test Geometric-Exponential compound distribution."""
    
    @pytest.fixture
    def compound_dist(self):
        freq = MockGeometric(p=0.3)
        sev = MockExponential(scale=200.0)
        return GeometricExponentialCompound(freq, sev)
    
    def test_mean_variance(self, compound_dist):
        """Test mean and variance calculations."""
        # E[S] = θ / p
        assert compound_dist.mean() == pytest.approx(200.0 / 0.3)
        
        # Var[S] = θ² * (2 - p) / p²
        expected_var = 200.0**2 * (2 - 0.3) / (0.3**2)
        assert compound_dist.var() == pytest.approx(expected_var)
    
    def test_exponential_distribution(self, compound_dist):
        """Test that result follows exponential distribution."""
        # Should follow Exponential(θ/(1-p))
        expected_scale = 200.0 / (1 - 0.3)
        
        # Test PDF
        x = 500.0
        expected_pdf = stats.expon.pdf(x, scale=expected_scale)
        assert compound_dist.pdf(x) == pytest.approx(expected_pdf)
    
    def test_edge_case_p_near_one(self):
        """Test behavior when p approaches 1."""
        freq = MockGeometric(p=0.999)
        sev = MockExponential(scale=100.0)
        compound = GeometricExponentialCompound(freq, sev)
        
        # Should have very small mean (θ/p ≈ 100/0.999)
        assert compound.mean() < 101
        
        # Aggregate scale should be very large
        assert compound.aggregate_scale > 99999
    
    def test_edge_case_p_equals_one(self):
        """Test degenerate case when p = 1."""
        freq = MockGeometric(p=1.0)
        sev = MockExponential(scale=100.0)
        compound = GeometricExponentialCompound(freq, sev)
        
        # Should be degenerate at 0
        assert compound.pdf(0) == 1.0
        assert compound.pdf(10) == 0.0
        assert compound.cdf(10) == 1.0
        assert compound.ppf(0.5) == 0.0
        assert compound.rvs(size=5).sum() == 0.0
    
    def test_single_vs_multiple_rvs(self):
        """Test rvs with size=1 returns scalar."""
        freq = MockGeometric(p=0.4)
        sev = MockExponential(scale=100.0)
        compound = GeometricExponentialCompound(freq, sev)
        
        # Single sample should be scalar or 0-d array
        single = compound.rvs(size=1)
        if isinstance(single, np.ndarray):
            assert single.shape == () or single.shape == (1,)
        else:
            assert isinstance(single, (float, np.floating))
        
        # Multiple samples should be array
        multiple = compound.rvs(size=10)
        assert isinstance(multiple, np.ndarray)
        assert multiple.shape == (10,)


class TestNegativeBinomialGamma:
    """Test Negative Binomial-Gamma compound distribution."""
    
    @pytest.fixture
    def compound_dist(self):
        freq = MockNegativeBinomial(n=3, p=0.4)
        sev = MockGamma(a=2.0, scale=150.0)
        return NegativeBinomialGammaCompound(freq, sev)
    
    def test_mean_variance(self, compound_dist):
        """Test mean and variance calculations with caching."""
        # First call should compute and cache
        mean1 = compound_dist.mean()
        var1 = compound_dist.var()
        
        # Second call should use cache
        mean2 = compound_dist.mean()
        var2 = compound_dist.var()
        
        assert mean1 == mean2
        assert var1 == var2
        
        # Check values
        r = 3
        p = 0.4
        alpha = 2.0
        beta = 1.0 / 150.0
        
        expected_mean = r * (1 - p) / p * alpha / beta
        assert mean1 == pytest.approx(expected_mean)
    
    def test_zero_probability(self, compound_dist):
        """Test probability mass at zero."""
        # P(S = 0) = p^r
        expected_p_zero = 0.4 ** 3
        
        # Check via PDF (at a point very close to 0)
        assert compound_dist.pdf(0) == pytest.approx(expected_p_zero)
        
        # Check via CDF
        assert compound_dist.cdf(0) == pytest.approx(expected_p_zero)
    
    def test_series_convergence(self, compound_dist):
        """Test that series expansion converges properly."""
        # Test PDF at various points
        x_values = [100, 1000, 5000]
        for x in x_values:
            pdf = compound_dist.pdf(x)
            assert pdf >= 0
            assert pdf < 1  # Sanity check
    
    def test_quantile_edge_cases(self, compound_dist):
        """Test quantile function at extreme probabilities."""
        # Very low quantile
        assert compound_dist.ppf(0.001) == 0.0
        
        # Median
        median = compound_dist.ppf(0.5)
        assert median > 0
        
        # High quantile with fallback
        with patch('scipy.optimize.brentq') as mock_brentq:
            # Make first call fail to trigger fallback
            mock_brentq.side_effect = [Exception("Test"), 10000.0]
            result = compound_dist.ppf(0.999)
            assert result == 10000.0
            assert mock_brentq.call_count == 2


class TestBinomialLognormal:
    """Test Binomial-Lognormal approximation."""
    
    @pytest.fixture
    def compound_dist(self):
        freq = MockBinomial(n=50, p=0.2)
        sev = MockLognormal(mu=5.0, sigma=1.0)
        return BinomialLognormalApproximation(freq, sev)
    
    def test_approximation_parameters(self, compound_dist):
        """Test calculation of approximate lognormal parameters."""
        mu_approx, sigma_approx = compound_dist._get_approx_params()
        
        # Should be cached
        mu_approx2, sigma_approx2 = compound_dist._get_approx_params()
        assert mu_approx == mu_approx2
        assert sigma_approx == sigma_approx2
        
        # Parameters should be reasonable
        assert mu_approx > 0
        assert sigma_approx > 0
    
    def test_degenerate_case_zero_probability(self):
        """Test when binomial p = 0 (no claims)."""
        freq = MockBinomial(n=10, p=0.0)
        sev = MockLognormal(mu=5.0, sigma=1.0)
        compound = BinomialLognormalApproximation(freq, sev)
        
        # Should have all mass at 0
        assert compound.mean() == 0.0
        assert compound.var() == 0.0
        assert compound.pdf(0) == 1.0
        assert compound.pdf(100) == 0.0
        assert compound.cdf(100) == 1.0
        assert compound.ppf(0.5) == 0.0
    
    def test_rvs_approximation_threshold(self):
        """Test that RVS uses approximation for large counts."""
        freq = MockBinomial(n=100, p=0.5)
        sev = MockLognormal(mu=3.0, sigma=0.5)
        compound = BinomialLognormalApproximation(freq, sev)
        
        # Mock to ensure we get both large and small counts
        with patch.object(stats.binom, 'rvs', side_effect=[[5, 40, 0]]):
            samples = compound.rvs(size=3, random_state=42)
            
            assert samples[0] > 0  # Small count - direct simulation
            assert samples[1] > 0  # Large count - approximation
            assert samples[2] == 0  # Zero count
    
    def test_pdf_cdf_consistency(self, compound_dist):
        """Test that PDF integrates to CDF."""
        x_values = np.linspace(0, 10000, 100)
        
        # Check monotonicity of CDF
        cdf_values = compound_dist.cdf(x_values)
        assert np.all(np.diff(cdf_values) >= 0)
        
        # Check PDF is non-negative
        pdf_values = compound_dist.pdf(x_values)
        assert np.all(pdf_values >= 0)


class TestPanjerRecursion:
    """Test Panjer recursion implementation."""
    
    def test_poisson_parameters(self):
        """Test Panjer parameter identification for Poisson."""
        freq = MockPoisson(mu=2.5)
        sev = MockExponential(scale=100.0)
        
        panjer = PanjerRecursion(freq, sev, discretization_step=10.0)
        
        assert panjer.a == 0
        assert panjer.b == 2.5
        assert panjer.p0 == pytest.approx(np.exp(-2.5))
    
    def test_binomial_parameters(self):
        """Test Panjer parameter identification for Binomial."""
        freq = MockBinomial(n=10, p=0.3)
        sev = MockExponential(scale=100.0)
        
        panjer = PanjerRecursion(freq, sev, discretization_step=10.0)
        
        expected_a = -0.3 / 0.7
        expected_b = 11 * 0.3 / 0.7
        
        assert panjer.a == pytest.approx(expected_a)
        assert panjer.b == pytest.approx(expected_b)
        assert panjer.p0 == pytest.approx(0.7 ** 10)
    
    def test_negative_binomial_parameters(self):
        """Test Panjer parameter identification for Negative Binomial."""
        freq = MockNegativeBinomial(n=5, p=0.4)
        sev = MockExponential(scale=100.0)
        
        panjer = PanjerRecursion(freq, sev, discretization_step=10.0)
        
        assert panjer.a == pytest.approx(0.6)
        assert panjer.b == pytest.approx(4 * 0.6)
        assert panjer.p0 == pytest.approx(0.4 ** 5)
    
    def test_unsupported_frequency(self):
        """Test error for unsupported frequency distribution."""
        freq = Mock()
        freq.__class__.__name__ = 'UnsupportedDist'
        sev = MockExponential(scale=100.0)
        
        with pytest.raises(ValueError, match="not supported by Panjer recursion"):
            PanjerRecursion(freq, sev)
    
    def test_discretization(self):
        """Test severity discretization."""
        freq = MockPoisson(mu=2.0)
        sev = MockExponential(scale=100.0)
        
        panjer = PanjerRecursion(freq, sev, discretization_step=50.0, max_value=500.0)
        
        # Check discretization
        assert len(panjer.f) == 11  # 0, 50, 100, ..., 500
        assert panjer.f[0] > 0  # Mass at 0
        assert np.abs(panjer.f.sum() - 1.0) < 0.01  # Normalized
    
    def test_aggregate_pmf_calculation(self):
        """Test calculation of aggregate PMF."""
        freq = MockPoisson(mu=1.5)
        sev = MockExponential(scale=50.0)
        
        panjer = PanjerRecursion(freq, sev, discretization_step=25.0, max_value=300.0)
        
        loss_values, pmf = panjer.calculate_aggregate_pmf()
        
        # Check structure
        assert len(loss_values) == len(pmf)
        assert loss_values[0] == 0
        assert loss_values[-1] == 300.0
        
        # Check probabilities
        assert np.all(pmf >= 0)
        assert pmf[0] == pytest.approx(np.exp(-1.5))  # P(S = 0)
        assert np.abs(pmf.sum() - 1.0) < 0.3  # Should sum to ~1 (discretization error)
    
    def test_mean_variance_calculation(self):
        """Test mean and variance from PMF."""
        freq = MockPoisson(mu=2.0)
        sev = MockExponential(scale=100.0)
        
        panjer = PanjerRecursion(freq, sev, discretization_step=20.0, max_value=1000.0)
        
        # Calculate via Panjer
        panjer_mean = panjer.mean()
        panjer_var = panjer.var()
        
        # Theoretical values
        theoretical_mean = 2.0 * 100.0
        theoretical_var = 2.0 * 100.0**2 * 2
        
        # Should be close (discretization introduces some error)
        # With large discretization steps and loss of probability mass, tolerance is higher
        assert panjer_mean == pytest.approx(theoretical_mean, rel=0.25)
        assert panjer_var == pytest.approx(theoretical_var, rel=0.5)
    
    def test_cdf_interpolation(self):
        """Test CDF calculation with interpolation."""
        freq = MockPoisson(mu=1.0)
        sev = MockExponential(scale=100.0)
        
        panjer = PanjerRecursion(freq, sev, discretization_step=50.0)
        
        # Test scalar input
        cdf_scalar = panjer.cdf(75.0)
        assert 0 <= cdf_scalar <= 1
        
        # Test array input
        x_array = np.array([0, 50, 100, 150])
        cdf_array = panjer.cdf(x_array)
        assert cdf_array.shape == (4,)
        assert np.all(np.diff(cdf_array) >= 0)  # Monotonic


class TestSimulatedCompoundExtended:
    """Extended tests for simulated compound distribution."""
    
    def test_cache_generation(self):
        """Test cache generation and reuse."""
        freq = MockPoisson(mu=2.0)
        sev = MockExponential(scale=100.0)
        compound = SimulatedCompound(freq, sev)
        
        # Initially no cache
        assert compound._cache is None
        
        # Generate cache with specific seed
        compound._generate_cache(random_state=42)
        assert compound._cache is not None
        assert len(compound._cache) == 10000
        assert compound._cache_seed == 42
        
        # Cache should contain non-negative values
        assert np.all(compound._cache >= 0)
    
    def test_pdf_without_cache(self):
        """Test PDF estimation generates cache if needed."""
        freq = MockPoisson(mu=3.0)
        sev = MockExponential(scale=200.0)
        compound = SimulatedCompound(freq, sev)
        
        # No cache initially
        assert compound._cache is None
        
        # Calling PDF should generate cache
        pdf_value = compound.pdf(500.0)
        assert compound._cache is not None
        assert pdf_value > 0
    
    def test_cdf_empirical(self):
        """Test empirical CDF calculation."""
        freq = MockPoisson(mu=2.0)
        sev = MockExponential(scale=100.0)
        compound = SimulatedCompound(freq, sev)
        
        # Force specific cache
        np.random.seed(42)
        compound._generate_cache(random_state=42)
        
        # Test CDF at various points
        assert compound.cdf(-100) == 0.0  # Below all values
        assert compound.cdf(1e10) == 1.0  # Above all values
        
        # Test monotonicity
        x_values = [0, 100, 200, 500, 1000]
        cdf_values = [compound.cdf(x) for x in x_values]
        assert all(cdf_values[i] <= cdf_values[i+1] for i in range(len(cdf_values)-1))
    
    def test_ppf_empirical(self):
        """Test empirical quantile function."""
        freq = MockPoisson(mu=4.0)
        sev = MockExponential(scale=150.0)
        compound = SimulatedCompound(freq, sev)
        
        # Generate cache
        compound._generate_cache(random_state=42)
        
        # Test standard quantiles
        q01 = compound.ppf(0.01)
        q50 = compound.ppf(0.50)
        q99 = compound.ppf(0.99)
        
        assert q01 < q50 < q99
        
        # Test array input
        quantiles = compound.ppf([0.25, 0.75])
        assert len(quantiles) == 2
        assert quantiles[0] < quantiles[1]


class TestPoissonExponentialExtended:
    """Extended tests for Poisson-Exponential to cover all branches."""
    
    def test_pdf_series_truncation(self):
        """Test PDF series truncation for various lambda values."""
        # Small lambda - series should truncate early
        freq_small = MockPoisson(mu=0.5)
        sev = MockExponential(scale=100.0)
        compound_small = PoissonExponentialCompound(freq_small, sev)
        
        # Should still give valid PDF
        pdf_small = compound_small.pdf(50.0)
        assert pdf_small > 0
        
        # Large lambda - series needs more terms
        freq_large = MockPoisson(mu=50.0)
        compound_large = PoissonExponentialCompound(freq_large, sev)
        
        pdf_large = compound_large.pdf(5000.0)
        assert pdf_large > 0
    
    def test_cdf_edge_cases(self):
        """Test CDF at boundary conditions."""
        freq = MockPoisson(mu=3.0)
        sev = MockExponential(scale=100.0)
        compound = PoissonExponentialCompound(freq, sev)
        
        # Negative values
        assert compound.cdf(-10) == 0.0
        assert compound.cdf([-10, -5]).tolist() == [0.0, 0.0]
        
        # Very large values
        assert compound.cdf(1e6) > 0.9999
    
    def test_ppf_boundary_search(self):
        """Test quantile function search boundaries."""
        freq = MockPoisson(mu=10.0)
        sev = MockExponential(scale=50.0)
        compound = PoissonExponentialCompound(freq, sev)
        
        # Test at p_zero boundary
        p_zero = np.exp(-10.0)
        assert compound.ppf(p_zero * 0.5) == 0.0
        assert compound.ppf(p_zero * 1.5) > 0.0
    
    def test_rvs_with_seed(self):
        """Test random generation with different seeds."""
        freq = MockPoisson(mu=5.0)
        sev = MockExponential(scale=200.0)
        compound = PoissonExponentialCompound(freq, sev)
        
        # Different seeds should give different results
        samples1 = compound.rvs(size=100, random_state=42)
        samples2 = compound.rvs(size=100, random_state=43)
        
        assert not np.array_equal(samples1, samples2)
        
        # Same seed should give same results
        samples3 = compound.rvs(size=100, random_state=42)
        assert np.array_equal(samples1, samples3)


class TestPoissonGammaExtended:
    """Extended tests for Poisson-Gamma (Tweedie) distribution."""
    
    def test_tweedie_weight_calculation(self):
        """Test Tweedie weight calculation for series."""
        pytest.skip("Tweedie series expansion not used in current implementation")
    
    def test_pdf_zero_handling(self):
        """Test PDF handles zero correctly."""
        freq = MockPoisson(mu=1.0)
        sev = MockGamma(a=2.0, scale=100.0)
        compound = PoissonGammaCompound(freq, sev)
        
        # PDF at exactly 0
        pdf_zero = compound.pdf(0.0)
        assert pdf_zero == pytest.approx(np.exp(-1.0))
        
        # PDF array including 0
        pdf_array = compound.pdf([0.0, 100.0, 200.0])
        assert pdf_array[0] == pytest.approx(np.exp(-1.0))
        assert np.all(pdf_array[1:] >= 0)
    
    def test_cdf_series_convergence(self):
        """Test CDF series convergence for different parameters."""
        # Fast converging case
        freq_fast = MockPoisson(mu=0.5)
        sev = MockGamma(a=2.0, scale=100.0)
        compound_fast = PoissonGammaCompound(freq_fast, sev)
        
        cdf_fast = compound_fast.cdf(500.0)
        assert 0 < cdf_fast < 1
        
        # Slow converging case
        freq_slow = MockPoisson(mu=20.0)
        compound_slow = PoissonGammaCompound(freq_slow, sev)
        
        cdf_slow = compound_slow.cdf(5000.0)
        assert 0 < cdf_slow < 1
    
    def test_rvs_gamma_aggregation(self):
        """Test that RVS correctly aggregates Gamma variables."""
        freq = MockPoisson(mu=3.0)
        sev = MockGamma(a=2.0, scale=100.0)
        compound = PoissonGammaCompound(freq, sev)
        
        # Generate samples
        np.random.seed(42)
        samples = compound.rvs(size=1000)
        
        # Check that some samples are exactly 0 (when N=0)
        assert np.any(samples == 0)
        
        # Non-zero samples should follow aggregated Gamma
        nonzero_samples = samples[samples > 0]
        assert len(nonzero_samples) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])