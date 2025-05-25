"""
Final tests to reach 95%+ coverage for compound distributions.

Covers remaining branches and edge cases.
"""

import numpy as np
import pytest
from scipy import stats
from unittest.mock import Mock, patch

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


# Simple mock classes
class MockPoisson:
    def __init__(self, mu):
        self.mu = mu
        self.__class__.__name__ = 'Poisson'
    def mean(self): return self.mu
    def var(self): return self.mu
    def rvs(self, size=1): return stats.poisson.rvs(self.mu, size=size)
    def pmf(self, k): return stats.poisson.pmf(k, self.mu)
    def ppf(self, q): return stats.poisson.ppf(q, self.mu)
    def cdf(self, x): return stats.poisson.cdf(x, self.mu)


class MockExponential:
    def __init__(self, scale):
        self.scale = scale
        self.__class__.__name__ = 'Exponential'
    def mean(self): return self.scale
    def var(self): return self.scale ** 2
    def rvs(self, size=1): return stats.expon.rvs(scale=self.scale, size=size)
    def pdf(self, x): return stats.expon.pdf(x, scale=self.scale)
    def cdf(self, x): return stats.expon.cdf(x, scale=self.scale)
    def ppf(self, q): return stats.expon.ppf(q, scale=self.scale)


class MockGamma:
    def __init__(self, a, scale):
        self.a = a
        self.scale = scale
        self.__class__.__name__ = 'Gamma'
    def rvs(self, size=1): return stats.gamma.rvs(a=self.a, scale=self.scale, size=size)
    def pdf(self, x): return stats.gamma.pdf(x, a=self.a, scale=self.scale)
    def cdf(self, x): return stats.gamma.cdf(x, a=self.a, scale=self.scale)
    def ppf(self, q): return stats.gamma.ppf(q, a=self.a, scale=self.scale)
    def mean(self): return self.a * self.scale
    def var(self): return self.a * self.scale ** 2


class TestRemainingBranches:
    """Test remaining uncovered branches."""
    
    def test_poisson_exponential_pdf_scalar_zero(self):
        """Test PDF returns scalar for scalar input at zero."""
        freq = MockPoisson(mu=2.0)
        sev = MockExponential(scale=100.0)
        compound = PoissonExponentialCompound(freq, sev)
        
        # Scalar 0 should return scalar result
        pdf = compound.pdf(0.0)
        assert isinstance(pdf, (float, np.floating))
    
    def test_poisson_exponential_cdf_all_negative(self):
        """Test CDF with all negative values."""
        freq = MockPoisson(mu=1.0)
        sev = MockExponential(scale=50.0)
        compound = PoissonExponentialCompound(freq, sev)
        
        # All negative values
        cdf = compound.cdf(np.array([-100, -50, -10]))
        assert np.all(cdf == 0.0)
    
    def test_poisson_exponential_ppf_scalar(self):
        """Test PPF returns scalar for scalar input."""
        freq = MockPoisson(mu=3.0)
        sev = MockExponential(scale=200.0)
        compound = PoissonExponentialCompound(freq, sev)
        
        # Scalar input
        q95 = compound.ppf(0.95)
        assert isinstance(q95, (float, np.floating))
        assert q95 > 0
    
    def test_poisson_exponential_cdf_early_termination(self):
        """Test CDF series terminates early for small lambda."""
        freq = MockPoisson(mu=0.01)  # Very small
        sev = MockExponential(scale=1000.0)
        compound = PoissonExponentialCompound(freq, sev)
        
        # Should terminate quickly
        cdf = compound.cdf(100.0)
        assert 0 < cdf < 1
    
    def test_poisson_gamma_pdf_scalar_positive(self):
        """Test PDF returns scalar for positive scalar input."""
        freq = MockPoisson(mu=2.0)
        sev = MockGamma(a=1.5, scale=100.0)
        compound = PoissonGammaCompound(freq, sev)
        
        pdf = compound.pdf(150.0)
        assert isinstance(pdf, (float, np.floating))
        assert pdf > 0
    
    def test_poisson_gamma_cdf_no_positive_mask(self):
        """Test CDF when no positive values."""
        freq = MockPoisson(mu=1.0)
        sev = MockGamma(a=2.0, scale=50.0)
        compound = PoissonGammaCompound(freq, sev)
        
        # Only zero and negative
        cdf = compound.cdf(np.array([-10, 0]))
        assert cdf[0] == 0.0
        assert cdf[1] == pytest.approx(np.exp(-1.0))
    
    def test_poisson_gamma_ppf_array_input(self):
        """Test PPF with array input."""
        freq = MockPoisson(mu=1.5)
        sev = MockGamma(a=2.5, scale=75.0)
        compound = PoissonGammaCompound(freq, sev)
        
        quantiles = compound.ppf(np.array([0.1, 0.5, 0.9]))
        assert len(quantiles) == 3
        assert quantiles[0] < quantiles[1] < quantiles[2]
    
    def test_poisson_gamma_rvs_scalar(self):
        """Test RVS returns scalar for size=1."""
        freq = MockPoisson(mu=2.0)
        sev = MockGamma(a=3.0, scale=100.0)
        compound = PoissonGammaCompound(freq, sev)
        
        sample = compound.rvs(size=1)
        assert isinstance(sample, (float, np.floating))
    
    def test_geometric_exponential_pdf_array(self):
        """Test Geometric-Exponential PDF with array input."""
        freq = Mock()
        freq.p = 0.3
        freq.__class__.__name__ = 'Geometric'
        sev = MockExponential(scale=100.0)
        compound = GeometricExponentialCompound(freq, sev)
        
        pdf_array = compound.pdf(np.array([0, 100, 500]))
        assert len(pdf_array) == 3
    
    def test_geometric_exponential_cdf_scalar(self):
        """Test Geometric-Exponential CDF returns scalar."""
        freq = Mock()
        freq.p = 0.4
        freq.__class__.__name__ = 'Geometric'
        sev = MockExponential(scale=150.0)
        compound = GeometricExponentialCompound(freq, sev)
        
        cdf = compound.cdf(200.0)
        assert isinstance(cdf, (float, np.floating))
    
    def test_geometric_exponential_ppf_array(self):
        """Test Geometric-Exponential PPF with array."""
        freq = Mock()
        freq.p = 0.5
        freq.__class__.__name__ = 'Geometric'
        sev = MockExponential(scale=100.0)
        compound = GeometricExponentialCompound(freq, sev)
        
        ppf_array = compound.ppf(np.array([0.25, 0.75]))
        assert len(ppf_array) == 2
    
    def test_negative_binomial_gamma_pdf_no_zeros(self):
        """Test NB-Gamma PDF with no zeros in input."""
        freq = Mock()
        freq.n = 3
        freq.p = 0.4
        freq.__class__.__name__ = 'NegativeBinomial'
        sev = MockGamma(a=2.0, scale=100.0)
        compound = NegativeBinomialGammaCompound(freq, sev)
        
        # All positive values
        pdf = compound.pdf(np.array([50, 100, 200]))
        assert np.all(pdf > 0)
    
    def test_negative_binomial_gamma_cdf_mixed(self):
        """Test NB-Gamma CDF with mixed values."""
        freq = Mock()
        freq.n = 2
        freq.p = 0.5
        freq.__class__.__name__ = 'NegativeBinomial'
        sev = MockGamma(a=1.5, scale=150.0)
        compound = NegativeBinomialGammaCompound(freq, sev)
        
        cdf = compound.cdf(np.array([-50, 0, 100]))
        assert cdf[0] == 0.0
        assert cdf[1] == pytest.approx(0.5 ** 2)
        assert cdf[2] > cdf[1]
    
    def test_binomial_lognormal_pdf_scalar_zero(self):
        """Test Binomial-Lognormal PDF at zero."""
        freq = Mock()
        freq.n = 10
        freq.p = 0.3
        freq.__class__.__name__ = 'Binomial'
        sev = Mock()
        sev.mu = 5.0
        sev.sigma = 1.0
        sev.__class__.__name__ = 'Lognormal'
        compound = BinomialLognormalApproximation(freq, sev)
        
        pdf = compound.pdf(0.0)
        assert isinstance(pdf, (float, np.floating))
    
    def test_binomial_lognormal_cdf_scalar(self):
        """Test Binomial-Lognormal CDF returns scalar."""
        freq = Mock()
        freq.n = 20
        freq.p = 0.2
        freq.__class__.__name__ = 'Binomial'
        sev = Mock()
        sev.mu = 4.0
        sev.sigma = 0.5
        sev.__class__.__name__ = 'Lognormal'
        compound = BinomialLognormalApproximation(freq, sev)
        
        cdf = compound.cdf(1000.0)
        assert isinstance(cdf, (float, np.floating))
    
    def test_binomial_lognormal_ppf_scalar(self):
        """Test Binomial-Lognormal PPF returns scalar."""
        freq = Mock()
        freq.n = 15
        freq.p = 0.4
        freq.__class__.__name__ = 'Binomial'
        sev = Mock()
        sev.mu = 6.0
        sev.sigma = 0.8
        sev.__class__.__name__ = 'Lognormal'
        compound = BinomialLognormalApproximation(freq, sev)
        
        ppf = compound.ppf(0.75)
        assert isinstance(ppf, (float, np.floating))
    
    def test_simulated_compound_pdf_scalar(self):
        """Test simulated PDF returns scalar."""
        freq = MockPoisson(mu=2.0)
        sev = MockExponential(scale=100.0)
        compound = SimulatedCompound(freq, sev)
        
        pdf = compound.pdf(150.0)
        assert isinstance(pdf, (float, np.floating))
    
    def test_simulated_compound_cdf_scalar(self):
        """Test simulated CDF returns scalar."""
        freq = MockPoisson(mu=3.0)
        sev = MockExponential(scale=50.0)
        compound = SimulatedCompound(freq, sev)
        
        cdf = compound.cdf(100.0)
        assert isinstance(cdf, (float, np.floating))
        assert 0 <= cdf <= 1
    
    def test_simulated_compound_ppf_scalar(self):
        """Test simulated PPF returns scalar."""
        freq = MockPoisson(mu=1.5)
        sev = MockExponential(scale=200.0)
        compound = SimulatedCompound(freq, sev)
        
        ppf = compound.ppf(0.9)
        assert isinstance(ppf, (float, np.floating))
    
    def test_panjer_recursion_edge_cases(self):
        """Test Panjer recursion edge cases."""
        freq = MockPoisson(mu=2.0)
        sev = MockExponential(scale=100.0)
        
        # Very fine discretization
        panjer = PanjerRecursion(freq, sev, discretization_step=1.0, max_value=100.0)
        
        # Test CDF with scalar
        cdf = panjer.cdf(50.0)
        assert isinstance(cdf, (float, np.floating))
        assert 0 <= cdf <= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])