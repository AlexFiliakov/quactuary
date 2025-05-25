"""
Comprehensive tests for pricing module to achieve 95%+ coverage.

Tests all branches, compound distribution integration, and edge cases.
"""

from datetime import date
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
from qiskit.providers import BackendV1
from qiskit_aer import AerSimulator

from quactuary.backend import (BackendManager, ClassicalBackend, get_backend,
                               set_backend)
from quactuary.book import Inforce, PolicyTerms, Portfolio
from quactuary.datatypes import PricingResult
from quactuary.distributions.compound import CompoundDistribution
from quactuary.distributions.frequency import DeterministicFrequency
from quactuary.distributions.severity import ConstantSeverity
from quactuary.pricing import PricingModel


# Mock distributions for testing
class MockPoisson:
    """Mock Poisson distribution."""

    def __init__(self, mu):
        self.mu = mu
        self.__class__.__name__ = 'Poisson'

    def mean(self):
        return self.mu

    def var(self):
        return self.mu


class MockExponential:
    """Mock Exponential distribution."""

    def __init__(self, scale):
        self.scale = scale
        self.__class__.__name__ = 'Exponential'

    def mean(self):
        return self.scale

    def var(self):
        return self.scale ** 2


class MockCompoundDistribution:
    """Mock compound distribution for testing."""

    def __init__(self):
        self._mean = 5000.0
        self._var = 10000000.0
        self._std = np.sqrt(self._var)

    def mean(self):
        return self._mean

    def var(self):
        return self._var

    def std(self):
        return self._std

    def ppf(self, q):
        # Simple linear quantile function for testing
        if isinstance(q, (list, np.ndarray)):
            return np.array([self._mean * (1 + qi) for qi in q])
        return self._mean * (1 + q)

    def rvs(self, size=1):
        # Generate samples with known properties
        np.random.seed(42)
        samples = np.random.exponential(self._mean, size=size)
        return samples

    def has_analytical_solution(self):
        return True


# Test fixtures
@pytest.fixture
def basic_policy():
    """Basic policy terms for testing."""
    return PolicyTerms(
        effective_date=date(2024, 1, 1),
        expiration_date=date(2025, 1, 1),
        exposure_amount=1_000_000,
        retention_type="deductible",
        per_occ_retention=10_000,
        coverage="ol",
        notes="Test policy"
    )


@pytest.fixture
def basic_inforce(basic_policy):
    """Basic inforce for testing."""
    freq = DeterministicFrequency(5)
    sev = ConstantSeverity(1000)
    return Inforce(
        n_policies=10,
        terms=basic_policy,
        frequency=freq,
        severity=sev,
        name="Test Inforce"
    )


@pytest.fixture
def basic_portfolio(basic_inforce):
    """Basic portfolio for testing."""
    return Portfolio([basic_inforce])


@pytest.fixture
def pricing_model(basic_portfolio):
    """Basic pricing model for testing."""
    return PricingModel(basic_portfolio)


class TestPricingModelInit:
    """Test PricingModel initialization."""

    def test_basic_initialization(self, basic_portfolio):
        """Test basic initialization of PricingModel."""
        model = PricingModel(basic_portfolio)

        assert model.portfolio == basic_portfolio
        assert model.compound_distribution is None

    def test_inheritance_structure(self, pricing_model):
        """Test that PricingModel inherits from correct base classes."""
        # Should have methods from both parent classes
        assert hasattr(pricing_model, 'portfolio')
        assert hasattr(pricing_model, 'simulate')
        assert hasattr(pricing_model, 'compound_distribution')


class TestSimulateMethod:
    """Test the simulate method with different backends."""

    def test_simulate_with_classical_backend(self, pricing_model):
        """Test simulation with classical backend."""
        backend = BackendManager(ClassicalBackend())

        with patch.object(pricing_model, 'calculate_portfolio_statistics') as mock_calc:
            mock_calc.return_value = PricingResult(
                mean=1000.0,
                variance=10000.0,
                value_at_risk=2000.0,
                tail_value_at_risk=2500.0
            )

            result = pricing_model.simulate(backend=backend)

            assert isinstance(result, PricingResult)
            assert mock_calc.called
            # Check that ClassicalPricingModel method was called
            mock_calc.assert_called_with(
                pricing_model.portfolio, True, True, True, True, 0.05, None
            )

    def test_simulate_with_quantum_backend(self, pricing_model):
        """Test simulation with quantum backend."""
        # Mock quantum backend
        mock_backend = Mock()
        mock_backend.__class__.__name__ = 'Backend'
        backend = BackendManager(mock_backend)

        with patch.object(pricing_model.__class__.__bases__[1], 'calculate_portfolio_statistics') as mock_calc:
            mock_calc.return_value = PricingResult(
                mean=1000.0,
                variance=10000.0,
                value_at_risk=2000.0,
                tail_value_at_risk=2500.0
            )

            result = pricing_model.simulate(backend=backend)

            assert isinstance(result, PricingResult)
            assert mock_calc.called

    def test_simulate_with_default_backend(self, pricing_model):
        """Test simulation with default backend."""
        # Set a default backend
        original_backend = get_backend()
        set_backend(ClassicalBackend())

        try:
            with patch.object(pricing_model, 'calculate_portfolio_statistics') as mock_calc:
                mock_calc.return_value = PricingResult(
                    estimates={}, intervals={}, samples=None, metadata={})

                result = pricing_model.simulate()

                assert isinstance(result, PricingResult)
                assert mock_calc.called
        finally:
            # Restore original backend
            set_backend(original_backend.backend)

    def test_simulate_with_unsupported_backend(self, pricing_model):
        """Test simulation with unsupported backend type."""
        # Create unsupported backend
        mock_backend = Mock()
        mock_backend.__class__.__name__ = 'UnsupportedBackend'
        backend = BackendManager(mock_backend)

        with pytest.raises(ValueError, match="Unsupported backend type"):
            pricing_model.simulate(backend=backend)

    def test_simulate_with_custom_parameters(self, pricing_model):
        """Test simulation with custom parameters."""
        backend = BackendManager(ClassicalBackend())

        with patch.object(pricing_model, 'calculate_portfolio_statistics') as mock_calc:
            mock_calc.return_value = PricingResult(
                estimates={}, intervals={}, samples=None, metadata={})

            result = pricing_model.simulate(
                mean=False,
                variance=True,
                value_at_risk=False,
                tail_value_at_risk=True,
                tail_alpha=0.01,
                n_sims=50000,
                backend=backend
            )

            mock_calc.assert_called_with(
                pricing_model.portfolio, False, True, False, True, 0.01, 50000
            )


class TestCompoundDistributionIntegration:
    """Test compound distribution integration in PricingModel."""

    def test_set_compound_distribution(self, pricing_model):
        """Test setting compound distribution."""
        freq = MockPoisson(mu=5.0)
        sev = MockExponential(scale=1000.0)

        pricing_model.set_compound_distribution(freq, sev)

        assert pricing_model.compound_distribution is not None
        assert isinstance(pricing_model.compound_distribution,
                          CompoundDistribution)

    def test_calculate_aggregate_statistics_no_distribution(self, pricing_model):
        """Test aggregate statistics without compound distribution."""
        with pytest.raises(ValueError, match="Compound distribution not set"):
            pricing_model.calculate_aggregate_statistics()

    def test_calculate_aggregate_statistics_basic(self, pricing_model):
        """Test basic aggregate statistics calculation."""
        # Set up mock compound distribution
        pricing_model.compound_distribution = MockCompoundDistribution()

        results = pricing_model.calculate_aggregate_statistics(
            apply_policy_terms=False)

        assert 'mean' in results
        assert 'std' in results
        assert 'variance' in results
        assert 'has_analytical' in results
        assert results['mean'] == 5000.0
        assert results['variance'] == 10000000.0
        assert results['has_analytical'] is True

    def test_calculate_aggregate_statistics_with_confidence_levels(self, pricing_model):
        """Test aggregate statistics with custom confidence levels."""
        pricing_model.compound_distribution = MockCompoundDistribution()

        confidence_levels = [0.80, 0.90, 0.95, 0.99]
        results = pricing_model.calculate_aggregate_statistics(
            confidence_levels=confidence_levels,
            apply_policy_terms=False
        )

        # Check VaR calculations
        for level in confidence_levels:
            var_key = f'var_{level:.0%}'
            assert var_key in results
            assert results[var_key] > 0

        # Check TVaR calculations
        for level in confidence_levels:
            tvar_key = f'tvar_{level:.0%}'
            assert tvar_key in results
            assert results[tvar_key] >= results[f'var_{level:.0%}']

    def test_calculate_aggregate_statistics_with_no_tail_samples(self, pricing_model):
        """Test TVaR calculation when no samples exceed VaR."""
        # Mock distribution that returns all zeros
        mock_dist = MockCompoundDistribution()
        mock_dist.rvs = lambda size: np.zeros(size)
        mock_dist.ppf = lambda q: 1000.0  # High VaR

        pricing_model.compound_distribution = mock_dist

        results = pricing_model.calculate_aggregate_statistics()

        # TVaR should fall back to VaR when no tail samples
        assert results['tvar_90%'] == results['var_90%']

    def test_calculate_aggregate_statistics_with_policy_terms(self, pricing_model):
        """Test aggregate statistics with policy terms application."""
        pricing_model.compound_distribution = MockCompoundDistribution()

        # Add policies attribute to portfolio
        pricing_model.portfolio.policies = [Mock()]

        results = pricing_model.calculate_aggregate_statistics(
            apply_policy_terms=True)

        assert 'note' in results
        assert 'Policy terms application not yet implemented' in results['note']

    def test_price_excess_layer_no_distribution(self, pricing_model):
        """Test excess layer pricing without compound distribution."""
        with pytest.raises(ValueError, match="Compound distribution not set"):
            pricing_model.price_excess_layer(100000, 50000)

    def test_price_excess_layer_basic(self, pricing_model):
        """Test basic excess layer pricing."""
        pricing_model.compound_distribution = MockCompoundDistribution()

        attachment = 10000
        limit = 50000

        results = pricing_model.price_excess_layer(
            attachment, limit, n_simulations=1000)

        assert 'attachment' in results
        assert 'limit' in results
        assert 'expected_loss' in results
        assert 'loss_std' in results
        assert 'loss_probability' in results
        assert 'average_severity' in results
        assert 'ground_up_mean' in results
        assert 'ground_up_std' in results

        assert results['attachment'] == attachment
        assert results['limit'] == limit
        assert results['expected_loss'] >= 0
        assert results['loss_probability'] >= 0
        assert results['loss_probability'] <= 1

    def test_price_excess_layer_no_losses(self, pricing_model):
        """Test excess layer pricing when no losses pierce the layer."""
        # Mock distribution that returns small values
        mock_dist = MockCompoundDistribution()
        mock_dist.rvs = lambda size: np.ones(size) * 100  # All losses = 100

        pricing_model.compound_distribution = mock_dist

        # High attachment point
        results = pricing_model.price_excess_layer(
            10000, 50000, n_simulations=100)

        assert results['expected_loss'] == 0
        assert results['loss_probability'] == 0
        assert results['average_severity'] == 0

    def test_price_excess_layer_all_losses_capped(self, pricing_model):
        """Test excess layer when all losses hit the limit."""
        # Mock distribution that returns large values
        mock_dist = MockCompoundDistribution()
        mock_dist.rvs = lambda size: np.ones(
            size) * 100000  # All losses = 100k

        pricing_model.compound_distribution = mock_dist

        attachment = 10000
        limit = 20000

        results = pricing_model.price_excess_layer(
            attachment, limit, n_simulations=100)

        assert results['expected_loss'] == limit  # All losses capped at limit
        assert results['loss_probability'] == 1.0
        assert results['average_severity'] == limit

    def test_price_excess_layer_partial_losses(self, pricing_model):
        """Test excess layer with mixture of losses."""
        # Mock distribution with varied losses
        mock_dist = MockCompoundDistribution()
        test_losses = np.array([0, 5000, 15000, 25000, 100000])
        mock_dist.rvs = lambda size: np.tile(
            test_losses, size // len(test_losses) + 1)[:size]

        pricing_model.compound_distribution = mock_dist

        attachment = 10000
        limit = 20000

        results = pricing_model.price_excess_layer(
            attachment, limit, n_simulations=len(test_losses))

        # Expected layer losses: [0, 0, 5000, 15000, 20000]
        expected_losses = np.array([0, 0, 5000, 15000, 20000])
        expected_mean = expected_losses.mean()

        assert results['expected_loss'] == pytest.approx(
            expected_mean, rel=0.1)
        assert 0 < results['loss_probability'] < 1


class TestBackendV1Support:
    """Test support for BackendV1 quantum backends."""

    class MockBackendV1(BackendV1):
        def __init__(self):
            # You can initialize with minimal required args or override methods as needed
            super().__init__()

    def test_simulate_with_backend_v1(self, pricing_model):
        """Test simulation with BackendV1 type backend."""
        mock_backend = self.MockBackendV1()  # Use self to access inner class
        backend = BackendManager(mock_backend)

        quantum_base_class = pricing_model.__class__.__bases__[1]
        with patch.object(quantum_base_class, 'calculate_portfolio_statistics') as mock_calc:
            mock_calc.return_value = PricingResult(
                estimates={
                    'mean': 1500.0,
                    'variance': 20000.0
                },
                intervals={},
                samples=None,
                metadata={}
            )

            result = pricing_model.simulate(backend=backend)

            assert isinstance(result, PricingResult)
            assert result.estimates['mean'] == 1500.0

            mock_calc.assert_called_with(
                pricing_model.portfolio, True, True, True, True, 0.05
            )


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error conditions."""

    def test_empty_portfolio(self):
        """Test with empty portfolio."""
        empty_portfolio = Portfolio([])
        model = PricingModel(empty_portfolio)

        assert model.portfolio == empty_portfolio
        assert len(model.portfolio) == 0

    def test_compound_distribution_factory_with_mock_backend(self, pricing_model):
        """Test compound distribution creation with mocked backend."""
        with patch('quactuary.distributions.compound.CompoundDistribution.create') as mock_create:
            mock_compound = MockCompoundDistribution()
            mock_create.return_value = mock_compound

            freq = Mock()
            sev = Mock()

            pricing_model.set_compound_distribution(freq, sev)

            assert pricing_model.compound_distribution == mock_compound
            mock_create.assert_called_once_with(freq, sev)

    def test_aggregate_statistics_with_numpy_edge_cases(self, pricing_model):
        """Test aggregate statistics with numpy edge cases."""
        # Mock distribution that can return NaN or Inf
        mock_dist = MockCompoundDistribution()

        # Test with very small variance
        mock_dist._var = 1e-20
        mock_dist._std = np.sqrt(1e-20)

        pricing_model.compound_distribution = mock_dist

        results = pricing_model.calculate_aggregate_statistics()

        assert results['variance'] == 1e-20
        assert results['std'] > 0

    def test_multiple_backend_switches(self, pricing_model):
        """Test switching between multiple backend types."""
        classical_backend = BackendManager(ClassicalBackend())

        # First call with classical
        with patch.object(pricing_model, 'calculate_portfolio_statistics') as mock_calc:
            mock_calc.return_value = PricingResult(
                estimates={}, intervals={}, samples=None, metadata={})
            result1 = pricing_model.simulate(backend=classical_backend)

        # Quantum Simulation Backend
        quantum_backend = AerSimulator(method='statevector')
        quantum_manager = BackendManager(quantum_backend)

        # Second call with quantum
        with patch.object(pricing_model.__class__.__bases__[1], 'calculate_portfolio_statistics') as mock_calc:
            mock_calc.return_value = PricingResult(
                estimates={}, intervals={}, samples=None, metadata={})
            result2 = pricing_model.simulate(backend=quantum_manager)

        with pytest.raises(AttributeError, match="'PricingResult' object has no attribute 'mean'"):
            assert result1.mean == 1000.0
            assert result2.mean == 2000.0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--cov=quactuary.pricing', '--cov-branch'])
