"""
Consolidated tests for pricing module.

Organized into 4 main test classes:
- TestPricingModelBasics: Basic pricing model functionality
- TestAggregateStatistics: Aggregate statistics calculations
- TestExcessLayerPricing: Excess layer and reinsurance pricing
- TestEdgeCasesAndErrors: Edge cases and error handling
"""

from datetime import date
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
from qiskit.providers import Backend, BackendV1
from qiskit_aer import AerSimulator

from quactuary.backend import (BackendManager, ClassicalBackend, get_backend,
                               set_backend)
from quactuary.book import Inforce, PolicyTerms, Portfolio
from quactuary.datatypes import PricingResult
from quactuary.distributions.compound import CompoundDistribution
from quactuary.distributions.frequency import (Binomial,
                                                DeterministicFrequency,
                                                Poisson)
from quactuary.distributions.severity import (ConstantSeverity,
                                               Exponential,
                                               Gamma)
from quactuary.pricing import PricingModel


# Mock distributions for testing
class MockPoisson:
    """Mock Poisson distribution."""

    def __init__(self, mu):
        self.mu = mu
        self.__class__.__name__ = 'Poisson'
        # Mock scipy distribution object
        self._dist = Mock()
        self._dist.args = [mu]
        self._dist.mean.return_value = mu
        self._dist.var.return_value = mu

    def mean(self):
        return self.mu

    def var(self):
        return self.mu


class MockExponential:
    """Mock Exponential distribution."""

    def __init__(self, scale):
        self.scale = scale
        self.__class__.__name__ = 'Exponential'
        # Mock scipy distribution object
        self._dist = Mock()
        self._dist.kwds = {'scale': scale}
        self._dist.mean.return_value = scale
        self._dist.var.return_value = scale ** 2

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


class MockBackendV1(BackendV1):
    """Mock BackendV1 for testing."""

    def __init__(self):
        # Create configuration first
        configuration = Mock()
        configuration.backend_name = 'mock_backend'
        configuration.backend_version = '1.0'
        configuration.n_qubits = 5
        configuration.basis_gates = ['u1', 'u2', 'u3', 'cx']
        configuration.gates = []
        configuration.local = True
        configuration.simulator = True
        configuration.conditional = True
        configuration.open_pulse = False
        configuration.memory = True
        configuration.max_shots = 10000
        configuration.coupling_map = None
        
        # Pass configuration to parent
        super().__init__(configuration)
        
    def _default_options(self):
        return None
        
    def run(self, circuits, **kwargs):
        return Mock()


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


class TestPricingModelBasics:
    """Test basic pricing model functionality."""

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

    def test_empty_portfolio(self):
        """Test with empty portfolio."""
        empty_portfolio = Portfolio([])
        model = PricingModel(empty_portfolio)

        assert model.portfolio == empty_portfolio
        assert len(model.portfolio) == 0

    def test_set_compound_distribution(self, pricing_model):
        """Test setting compound distribution."""
        freq = MockPoisson(mu=5.0)
        sev = MockExponential(scale=1000.0)

        pricing_model.set_compound_distribution(freq, sev)

        assert pricing_model.compound_distribution is not None
        assert isinstance(pricing_model.compound_distribution,
                          CompoundDistribution)

    def test_set_compound_distribution_different_types(self, pricing_model):
        """Test with different distribution types."""
        # Poisson-Exponential
        poisson_freq = Poisson(mu=5)
        exp_sev = Exponential(scale=1000)
        pricing_model.set_compound_distribution(poisson_freq, exp_sev)
        assert pricing_model.compound_distribution is not None

        # Binomial-Gamma
        binomial_freq = Binomial(n=10, p=0.3)
        gamma_sev = Gamma(shape=2, scale=500)
        pricing_model.set_compound_distribution(binomial_freq, gamma_sev)
        assert pricing_model.compound_distribution is not None

    def test_simulate_with_classical_backend(self, pricing_model):
        """Test simulation with classical backend."""
        backend = BackendManager(ClassicalBackend())

        # When a backend is specified, a new strategy is created
        # So we need to mock get_strategy_for_backend instead
        with patch('quactuary.pricing.get_strategy_for_backend') as mock_get_strategy:
            mock_strategy = Mock()
            mock_strategy.calculate_portfolio_statistics.return_value = PricingResult(
                estimates={
                    'mean': 1000.0,
                    'variance': 10000.0,
                    'VaR': 2000.0,
                    'TVaR': 2500.0
                },
                intervals={},
                samples=None,
                metadata={'n_sims': 10000}
            )
            mock_get_strategy.return_value = mock_strategy

            result = pricing_model.simulate(backend=backend)

            assert isinstance(result, PricingResult)
            assert result.estimates['mean'] == 1000.0
            assert result.estimates['variance'] == 10000.0
            assert result.estimates['VaR'] == 2000.0
            assert result.estimates['TVaR'] == 2500.0

    def test_simulate_with_quantum_backend(self, pricing_model):
        """Test simulation with quantum backend."""
        # Use AerSimulator as quantum backend
        from qiskit_aer import AerSimulator
        quantum_backend = AerSimulator()
        backend = BackendManager(quantum_backend)

        # Quantum strategy will raise NotImplementedError
        with pytest.raises(NotImplementedError, match="Quantum pricing strategy is not yet implemented"):
            pricing_model.simulate(backend=backend)

    def test_simulate_with_backend_v1(self, pricing_model):
        """Test simulation with BackendV1 type backend."""
        mock_backend = MockBackendV1()
        backend = BackendManager(mock_backend)

        # BackendV1 will use QuantumPricingStrategy which raises NotImplementedError
        with pytest.raises(NotImplementedError, match="Quantum pricing strategy is not yet implemented"):
            result = pricing_model.simulate(backend=backend)

    def test_simulate_with_default_backend(self, pricing_model):
        """Test simulation with default backend."""
        # Set a default backend
        original_backend = get_backend()
        set_backend('classical')

        try:
            with patch.object(pricing_model.strategy, 'calculate_portfolio_statistics') as mock_calc:
                mock_calc.return_value = PricingResult(
                    estimates={
                        'mean': 1000.0,
                        'variance': 10000.0,
                        'VaR': 2000.0,
                        'TVaR': 2500.0
                    },
                    intervals={},
                    samples=None,
                    metadata={'n_sims': 10000}
                )

                result = pricing_model.simulate()

                assert isinstance(result, PricingResult)
                assert mock_calc.called
        finally:
            # Restore original backend
            if hasattr(original_backend.backend, '__class__') and original_backend.backend.__class__.__name__ == 'ClassicalBackend':
                set_backend('classical')
            else:
                # For quantum backends, we can't easily restore, so just set classical
                set_backend('classical')

    def test_simulate_with_custom_parameters(self, pricing_model):
        """Test simulation with custom parameters."""
        backend = BackendManager(ClassicalBackend())

        with patch.object(pricing_model.strategy, 'calculate_portfolio_statistics') as mock_calc:
            mock_calc.return_value = PricingResult(
                estimates={
                    'mean': 1000.0,
                    'variance': 10000.0,
                    'VaR': 2000.0,
                    'TVaR': 2500.0
                },
                intervals={},
                samples=None,
                metadata={'n_sims': 50000}
            )

            result = pricing_model.simulate(
                mean=False,
                variance=True,
                value_at_risk=False,
                tail_value_at_risk=True,
                tail_alpha=0.01,
                n_sims=50000,
                backend=backend
            )

            # Since we're using a specific backend, it will create a new strategy
            # so we can't directly check the call on pricing_model.strategy
            assert isinstance(result, PricingResult)


class TestAggregateStatistics:
    """Test aggregate statistics calculations."""

    def test_calculate_aggregate_statistics_no_distribution(self, pricing_model):
        """Test aggregate statistics without compound distribution."""
        # Without compound distribution, it should use empirical calculation
        results = pricing_model.calculate_aggregate_statistics(n_simulations=100)
        
        assert 'mean' in results
        assert 'std' in results
        assert 'variance' in results
        assert results['has_analytical'] is False
        assert results['method'] == 'empirical'

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

    def test_calculate_aggregate_statistics_with_default_confidence_levels(self, pricing_model):
        """Test aggregate statistics with default confidence levels."""
        pricing_model.compound_distribution = MockCompoundDistribution()

        results = pricing_model.calculate_aggregate_statistics()

        # Check default VaR levels
        assert 'var_90%' in results
        assert 'var_95%' in results
        assert 'var_99%' in results

        # Check TVaR levels
        assert 'tvar_90%' in results
        assert 'tvar_95%' in results
        assert 'tvar_99%' in results

    def test_calculate_aggregate_statistics_with_custom_confidence_levels(self, pricing_model):
        """Test aggregate statistics with custom confidence levels."""
        pricing_model.compound_distribution = MockCompoundDistribution()

        confidence_levels = [0.80, 0.85, 0.90, 0.95, 0.975, 0.99]
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
            var_key = f'var_{level:.0%}'
            # TVaR should be >= VaR at same confidence level
            assert results[tvar_key] >= results[var_key]

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

        # The portfolio already has inforces which act as policies
        results = pricing_model.calculate_aggregate_statistics(
            apply_policy_terms=True)

        assert 'note' in results
        assert 'Policy terms application included in calculation' in results['note']

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

    def test_compound_distribution_with_different_models(self, pricing_model):
        """Test compound distribution with various frequency and severity models."""
        # Test Poisson-Exponential
        freq1 = Poisson(mu=5)
        sev1 = Exponential(scale=1000)
        pricing_model.set_compound_distribution(freq1, sev1)

        results1 = pricing_model.calculate_aggregate_statistics()
        assert results1['mean'] > 0
        assert results1['std'] > 0

        # Test Binomial-Gamma
        freq2 = Binomial(n=10, p=0.3)
        sev2 = Gamma(shape=2, scale=500)
        pricing_model.set_compound_distribution(freq2, sev2)

        results2 = pricing_model.calculate_aggregate_statistics()
        assert results2['mean'] > 0
        assert results2['std'] > 0


class TestExcessLayerPricing:
    """Test excess layer and reinsurance pricing."""

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

    def test_price_excess_layer_zero_attachment(self, pricing_model):
        """Test layer with zero attachment."""
        pricing_model.compound_distribution = MockCompoundDistribution()

        results = pricing_model.price_excess_layer(0, 10000, 500)
        assert results['attachment'] == 0
        assert results['expected_loss'] >= 0

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

    def test_price_excess_layer_different_parameters(self, pricing_model):
        """Test layer pricing with various parameters."""
        pricing_model.compound_distribution = MockCompoundDistribution()

        # Test multiple attachment/limit combinations
        test_cases = [
            (0, 10000),      # Ground-up
            (5000, 5000),    # Small layer
            (10000, 90000),  # Large layer
            (100000, 50000), # High attachment
        ]

        for attachment, limit in test_cases:
            results = pricing_model.price_excess_layer(
                attachment, limit, n_simulations=1000)
            
            assert results['attachment'] == attachment
            assert results['limit'] == limit
            assert results['expected_loss'] >= 0
            assert results['loss_probability'] >= 0


class TestEdgeCasesAndErrors:
    """Test edge cases and error handling."""

    def test_simulate_with_unsupported_backend(self, pricing_model):
        """Test simulation with unsupported backend type."""
        # Create unsupported backend
        mock_backend = Mock()
        # Create a truly unsupported backend type
        class UnsupportedBackend:
            pass
        
        unsupported = UnsupportedBackend()
        
        with pytest.raises(ValueError, match="Unsupported backend type"):
            backend = BackendManager(unsupported)

            # The error is raised during BackendManager creation, not simulate

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

    def test_multiple_backend_switches(self, pricing_model):
        """Test switching between multiple backend types."""
        classical_backend = BackendManager(ClassicalBackend())

        # First call with classical
        with patch('quactuary.pricing.get_strategy_for_backend') as mock_get_strategy:
            mock_strategy = Mock()
            mock_strategy.calculate_portfolio_statistics.return_value = PricingResult(
                estimates={
                    'mean': 1000.0,
                    'variance': 10000.0,
                    'VaR': 2000.0,
                    'TVaR': 2500.0
                },
                intervals={},
                samples=None,
                metadata={'n_sims': 10000}
            )
            mock_get_strategy.return_value = mock_strategy
            
            result1 = pricing_model.simulate(backend=classical_backend)
            
            assert isinstance(result1, PricingResult)
            assert result1.mean == 1000.0
        
        # Quantum backend would use QuantumPricingStrategy which raises NotImplementedError
        quantum_backend = AerSimulator(method='statevector')
        quantum_manager = BackendManager(quantum_backend)
        
        with pytest.raises(NotImplementedError, match="Quantum pricing strategy is not yet implemented"):
            result2 = pricing_model.simulate(backend=quantum_manager)

    def test_zero_frequency_distribution(self, pricing_model):
        """Test with zero frequency."""
        zero_freq = DeterministicFrequency(0)
        sev = ConstantSeverity(100)
        
        pricing_model.set_compound_distribution(zero_freq, sev)

        results = pricing_model.calculate_aggregate_statistics()
        assert results['mean'] == 0

        # Price layer with zero frequency
        layer_results = pricing_model.price_excess_layer(1000, 5000)
        assert layer_results['expected_loss'] == 0
        assert layer_results['loss_probability'] == 0

    def test_tvar_calculation_edge_case(self, pricing_model):
        """Test TVaR calculation when no samples exceed VaR."""
        freq = DeterministicFrequency(3)
        sev = ConstantSeverity(100)
        pricing_model.set_compound_distribution(freq, sev)

        # Mock the compound distribution to return controlled samples
        with patch.object(pricing_model.compound_distribution, 'rvs') as mock_rvs:
            # All samples are below 100
            mock_rvs.return_value = np.ones(10000) * 50

            with patch.object(pricing_model.compound_distribution, 'ppf') as mock_ppf:
                # VaR is high
                mock_ppf.return_value = 1000

                results = pricing_model.calculate_aggregate_statistics()

                # TVaR should fallback to VaR when no samples exceed it
                assert results['tvar_90%'] == 1000
                assert results['tvar_95%'] == 1000
                assert results['tvar_99%'] == 1000

    def test_no_losses_exceed_attachment(self, pricing_model):
        """Test when no losses exceed attachment."""
        freq = DeterministicFrequency(3)
        sev = ConstantSeverity(100)
        pricing_model.set_compound_distribution(freq, sev)

        # Mock to return small losses
        with patch.object(pricing_model.compound_distribution, 'rvs') as mock_rvs:
            mock_rvs.return_value = np.ones(1000) * 10  # All losses are 10

            results = pricing_model.price_excess_layer(
                attachment=1000,  # High attachment
                limit=5000,
                n_simulations=1000
            )

            assert results['expected_loss'] == 0
            assert results['loss_probability'] == 0
            assert results['average_severity'] == 0


# Legacy test for backwards compatibility
def test_pricing_model():
    """Legacy test from original test_pricing.py."""
    test_policy = PolicyTerms(
        effective_date=date(2027, 1, 1),
        expiration_date=date(2028, 1, 1),
        exposure_amount=5_000_000,
        retention_type="SIR",
        per_occ_retention=40_000,
        coverage="cm",
        notes="Sparse str test"
    )

    test_freq = DeterministicFrequency(3)
    test_sev = ConstantSeverity(100)

    test_inforce = Inforce(
        n_policies=5,
        terms=test_policy,
        frequency=test_freq,
        severity=test_sev,
        name="Test Inforce"
    )

    test_portfolio = Portfolio([test_inforce])

    pm = PricingModel(test_portfolio)
    assert pm.portfolio == test_portfolio
    new_manager = BackendManager(ClassicalBackend())
    sample = pm.simulate(backend=new_manager)
    assert isinstance(sample, PricingResult)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--cov=quactuary.pricing', '--cov-branch'])