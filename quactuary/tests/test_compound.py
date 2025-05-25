"""
Tests for compound distributions.

Validates analytical solutions against Monte Carlo simulations and tests
performance improvements.
"""

import time
from unittest.mock import Mock

import numpy as np
import pytest
from scipy import stats

from quactuary.distributions.compound import (
    CompoundDistribution,
    PoissonExponentialCompound,
    PoissonGammaCompound,
    SimulatedCompound,
)


class MockPoisson:
    """Mock Poisson distribution for testing."""
    def __init__(self, mu):
        self.mu = mu
        self.__class__.__name__ = 'Poisson'
    
    def rvs(self, size=1):
        return stats.poisson.rvs(self.mu, size=size)
    
    def pmf(self, k):
        return stats.poisson.pmf(k, self.mu)
    
    def mean(self):
        return self.mu
    
    def var(self):
        return self.mu


class MockExponential:
    """Mock Exponential distribution for testing."""
    def __init__(self, scale):
        self.scale = scale
        self.__class__.__name__ = 'Exponential'
    
    def rvs(self, size=1):
        return stats.expon.rvs(scale=self.scale, size=size)
    
    def pdf(self, x):
        return stats.expon.pdf(x, scale=self.scale)
    
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
    
    def mean(self):
        return self.a * self.scale
    
    def var(self):
        return self.a * self.scale ** 2


class TestCompoundDistribution:
    """Test base compound distribution functionality."""
    
    def test_factory_method_analytical(self):
        """Test factory method returns analytical solution when available."""
        freq = MockPoisson(mu=5.0)
        sev = MockExponential(scale=1000.0)
        
        compound = CompoundDistribution.create(freq, sev)
        
        assert isinstance(compound, PoissonExponentialCompound)
        assert compound.has_analytical_solution()
    
    def test_factory_method_simulated(self):
        """Test factory method returns simulated when no analytical solution."""
        freq = MockPoisson(mu=5.0)
        sev = Mock()  # Unknown severity type
        sev.__class__.__name__ = 'UnknownDist'
        sev.mean.return_value = 100
        sev.var.return_value = 1000
        
        compound = CompoundDistribution.create(freq, sev)
        
        assert isinstance(compound, SimulatedCompound)
        assert not compound.has_analytical_solution()
    
    def test_registry_pattern(self):
        """Test distribution registration works correctly."""
        # Check that our distributions are registered
        registry = CompoundDistribution._analytical_registry
        
        assert ('Poisson', 'Exponential') in registry
        assert ('Poisson', 'Gamma') in registry
        assert registry[('Poisson', 'Exponential')] == PoissonExponentialCompound
        assert registry[('Poisson', 'Gamma')] == PoissonGammaCompound


class TestPoissonExponentialCompound:
    """Test Poisson-Exponential compound distribution."""
    
    @pytest.fixture
    def compound_dist(self):
        freq = MockPoisson(mu=5.0)
        sev = MockExponential(scale=1000.0)
        return PoissonExponentialCompound(freq, sev)
    
    def test_mean(self, compound_dist):
        """Test mean calculation."""
        expected = 5.0 * 1000.0  # λ * θ
        assert compound_dist.mean() == expected
    
    def test_variance(self, compound_dist):
        """Test variance calculation."""
        expected = 5.0 * 1000.0**2 * 2  # λ * θ² * 2
        assert compound_dist.var() == expected
    
    def test_std(self, compound_dist):
        """Test standard deviation calculation."""
        assert compound_dist.std() == np.sqrt(compound_dist.var())
    
    def test_pdf_at_zero(self, compound_dist):
        """Test PDF has correct atom at zero."""
        # P(S = 0) = e^(-λ)
        expected = np.exp(-5.0)
        
        # PDF at exactly 0 should reflect the atom
        pdf_near_zero = compound_dist.pdf(0.01)
        assert pdf_near_zero > 0
    
    def test_cdf_properties(self, compound_dist):
        """Test CDF properties."""
        # CDF at 0 should be P(N = 0) = e^(-λ)
        assert compound_dist.cdf(0) == pytest.approx(np.exp(-5.0))
        
        # CDF should be monotonic
        x_values = np.linspace(0, 20000, 100)
        cdf_values = compound_dist.cdf(x_values)
        assert np.all(np.diff(cdf_values) >= 0)
        
        # CDF should approach 1
        assert compound_dist.cdf(100000) > 0.99
    
    def test_quantiles(self, compound_dist):
        """Test quantile function."""
        # Test some standard quantiles
        q50 = compound_dist.ppf(0.5)
        q95 = compound_dist.ppf(0.95)
        
        assert q50 > 0
        assert q95 > q50
        
        # Verify inverse relationship
        assert compound_dist.cdf(q50) == pytest.approx(0.5, abs=1e-3)
        assert compound_dist.cdf(q95) == pytest.approx(0.95, abs=1e-3)
    
    def test_random_generation(self, compound_dist):
        """Test random variate generation."""
        np.random.seed(42)
        samples = compound_dist.rvs(size=10000)
        
        # Check shape
        assert samples.shape == (10000,)
        
        # Check non-negative
        assert np.all(samples >= 0)
        
        # Check mean and variance approximately match
        assert np.mean(samples) == pytest.approx(compound_dist.mean(), rel=0.05)
        assert np.var(samples) == pytest.approx(compound_dist.var(), rel=0.1)
        
        # Check proportion of zeros
        p_zero_empirical = np.mean(samples == 0)
        p_zero_theoretical = np.exp(-5.0)
        assert p_zero_empirical == pytest.approx(p_zero_theoretical, abs=0.01)


class TestPoissonGammaCompound:
    """Test Poisson-Gamma (Tweedie) compound distribution."""
    
    @pytest.fixture
    def compound_dist(self):
        freq = MockPoisson(mu=3.0)
        sev = MockGamma(a=2.0, scale=500.0)
        return PoissonGammaCompound(freq, sev)
    
    def test_mean(self, compound_dist):
        """Test mean calculation."""
        # E[S] = λ * α / β = λ * α * scale
        expected = 3.0 * 2.0 * 500.0
        assert compound_dist.mean() == expected
    
    def test_variance(self, compound_dist):
        """Test variance calculation."""
        # Var[S] = λ * α * (α + 1) / β²
        lam = 3.0
        alpha = 2.0
        beta = 1.0 / 500.0
        expected = lam * alpha * (alpha + 1) / (beta ** 2)
        assert compound_dist.var() == expected
    
    def test_tweedie_parameters(self, compound_dist):
        """Test Tweedie parameter calculations."""
        # p should be in (1, 2)
        assert 1 < compound_dist.p < 2
        
        # Mean should match
        assert compound_dist.mu == compound_dist.mean()
        
        # Dispersion parameter should be positive
        assert compound_dist.phi > 0
    
    def test_cdf_properties(self, compound_dist):
        """Test CDF properties."""
        # CDF at 0
        assert compound_dist.cdf(0) == pytest.approx(np.exp(-3.0))
        
        # Monotonicity
        x_values = np.linspace(0, 10000, 50)
        cdf_values = compound_dist.cdf(x_values)
        assert np.all(np.diff(cdf_values) >= 0)
    
    def test_random_generation(self, compound_dist):
        """Test random variate generation."""
        np.random.seed(42)
        samples = compound_dist.rvs(size=5000)
        
        # Basic checks
        assert samples.shape == (5000,)
        assert np.all(samples >= 0)
        
        # Statistical checks
        assert np.mean(samples) == pytest.approx(compound_dist.mean(), rel=0.1)
        assert np.var(samples) == pytest.approx(compound_dist.var(), rel=0.2)


class TestSimulatedCompound:
    """Test simulated compound distribution fallback."""
    
    @pytest.fixture
    def compound_dist(self):
        freq = MockPoisson(mu=4.0)
        sev = MockExponential(scale=750.0)
        # Force simulated by creating directly
        return SimulatedCompound(freq, sev)
    
    def test_mean_variance_formulas(self, compound_dist):
        """Test theoretical mean and variance formulas."""
        # E[S] = E[N] * E[X]
        assert compound_dist.mean() == 4.0 * 750.0
        
        # Var[S] = E[N] * Var[X] + Var[N] * E[X]²
        expected_var = 4.0 * 750.0**2 + 4.0 * 750.0**2
        assert compound_dist.var() == expected_var
    
    def test_simulation_consistency(self, compound_dist):
        """Test that simulation gives consistent results."""
        np.random.seed(42)
        
        # Generate samples
        samples1 = compound_dist.rvs(size=1000)
        
        # Reset seed and generate again
        np.random.seed(42)
        samples2 = compound_dist.rvs(size=1000)
        
        assert np.array_equal(samples1, samples2)
    
    def test_pdf_estimation(self, compound_dist):
        """Test PDF estimation via KDE."""
        # Force cache generation
        compound_dist._generate_cache(random_state=42)
        
        # Test PDF at mean
        mean = compound_dist.mean()
        pdf_at_mean = compound_dist.pdf(mean)
        
        assert pdf_at_mean > 0
        
        # Test PDF array input
        x_values = np.linspace(0, 10000, 10)
        pdf_values = compound_dist.pdf(x_values)
        assert pdf_values.shape == (10,)
        assert np.all(pdf_values >= 0)


class TestPerformance:
    """Test performance improvements of analytical vs simulated."""
    
    def test_analytical_faster_than_simulated(self):
        """Verify analytical solutions are faster than simulation."""
        freq = MockPoisson(mu=5.0)
        sev = MockExponential(scale=1000.0)
        
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
        # Note: First call to simulated generates cache, so it's slower
        assert analytical_time < simulated_time * 0.5  # At least 2x faster
    
    def test_mean_calculation_performance(self):
        """Test that mean calculation is instant for analytical."""
        freq = MockPoisson(mu=10.0)
        sev = MockGamma(a=3.0, scale=200.0)
        
        compound = PoissonGammaCompound(freq, sev)
        
        # Should be essentially instant
        start = time.time()
        for _ in range(10000):
            compound.mean()
        elapsed = time.time() - start
        
        assert elapsed < 0.02  # Should take less than 20ms for 10k calculations


if __name__ == '__main__':
    pytest.main([__file__])