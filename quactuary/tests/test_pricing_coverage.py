"""
Comprehensive test coverage for pricing module.
"""
# Mock the qiskit imports to avoid import errors
import sys
from datetime import date
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

# Create mock modules
mock_qiskit = MagicMock()
mock_qiskit.__version__ = "1.4.2"
sys.modules['qiskit'] = mock_qiskit
sys.modules['qiskit.providers'] = MagicMock()

from quactuary.book import Inforce, PolicyTerms, Portfolio
# Now we can import our modules
from quactuary.datatypes import PricingResult
from quactuary.distributions.frequency import DeterministicFrequency
from quactuary.distributions.severity import ConstantSeverity
from quactuary.pricing import PricingModel

# Test data
test_policy = PolicyTerms(
    effective_date=date(2027, 1, 1),
    expiration_date=date(2028, 1, 1),
    exposure_amount=5_000_000,
    retention_type="SIR",
    per_occ_retention=40_000,
    coverage="cm",
    notes="Test policy"
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


class TestPricingModel:
    """Test PricingModel class."""
    
    def test_init(self):
        """Test PricingModel initialization."""
        model = PricingModel(test_portfolio)
        assert model.portfolio == test_portfolio
        assert model.compound_distribution is None
    
    def test_set_compound_distribution(self):
        """Test setting compound distribution."""
        model = PricingModel(test_portfolio)
        model.set_compound_distribution(test_freq, test_sev)
        assert model.compound_distribution is not None
    
    def test_calculate_aggregate_statistics(self):
        """Test calculate_aggregate_statistics method."""
        model = PricingModel(test_portfolio)
        
        # Test without compound distribution set
        with pytest.raises(ValueError, match="Compound distribution not set"):
            model.calculate_aggregate_statistics()
        
        # Set compound distribution
        model.set_compound_distribution(test_freq, test_sev)
        
        # Test basic statistics
        results = model.calculate_aggregate_statistics()
        assert 'mean' in results
        assert 'std' in results
        assert 'variance' in results
        assert 'has_analytical' in results
        assert 'var_90%' in results
        assert 'var_95%' in results
        assert 'var_99%' in results
        assert 'tvar_90%' in results
        assert 'tvar_95%' in results
        assert 'tvar_99%' in results
        
        # Test with custom confidence levels
        custom_levels = [0.80, 0.85]
        results = model.calculate_aggregate_statistics(confidence_levels=custom_levels)
        assert 'var_80%' in results
        assert 'var_85%' in results
        assert 'tvar_80%' in results
        assert 'tvar_85%' in results
        
        # Test with policy terms
        results = model.calculate_aggregate_statistics(apply_policy_terms=True)
        assert 'note' in results
    
    def test_price_excess_layer(self):
        """Test price_excess_layer method."""
        model = PricingModel(test_portfolio)
        
        # Test without compound distribution set
        with pytest.raises(ValueError, match="Compound distribution not set"):
            model.price_excess_layer(1000, 5000)
        
        # Set compound distribution
        model.set_compound_distribution(test_freq, test_sev)
        
        # Test layer pricing
        results = model.price_excess_layer(
            attachment=1000,
            limit=5000,
            n_simulations=1000
        )
        
        assert results['attachment'] == 1000
        assert results['limit'] == 5000
        assert 'expected_loss' in results
        assert 'loss_std' in results
        assert 'loss_probability' in results
        assert 'average_severity' in results
        assert 'ground_up_mean' in results
        assert 'ground_up_std' in results
        
        # Test with different parameters
        results2 = model.price_excess_layer(
            attachment=0,
            limit=10000,
            n_simulations=500
        )
        assert results2['attachment'] == 0
        assert results2['limit'] == 10000
    
    @patch('quactuary.backend.get_backend')
    def test_simulate_with_classical_backend(self, mock_get_backend):
        """Test simulate method with classical backend."""
        from quactuary.backend import BackendManager, ClassicalBackend

        # Mock classical backend
        mock_backend = Mock()
        mock_backend.backend = ClassicalBackend()
        mock_get_backend.return_value = mock_backend
        
        model = PricingModel(test_portfolio)
        
        # Mock the classical pricing model method
        with patch.object(model, 'calculate_portfolio_statistics') as mock_calc:
            mock_calc.return_value = PricingResult(mean=100.0)
            
            result = model.simulate()
            
            mock_calc.assert_called_once_with(
                model.portfolio, True, True, True, True, 0.05, None
            )
            assert result.mean == 100.0
    
    @patch('quactuary.backend.get_backend')
    def test_simulate_with_quantum_backend(self, mock_get_backend):
        """Test simulate method with quantum backend."""
        from quactuary.backend import BackendManager

        # Mock quantum backend
        mock_backend = Mock()
        mock_backend.backend = Mock()
        mock_backend.backend.__class__.__name__ = 'Backend'
        mock_get_backend.return_value = mock_backend
        
        # Make Backend and BackendV1 available
        from qiskit.providers import Backend
        mock_backend.backend.__class__.__bases__ = (Backend,)
        
        model = PricingModel(test_portfolio)
        
        # Mock the quantum pricing model method
        with patch('quactuary.quantum.QuantumPricingModel.calculate_portfolio_statistics') as mock_calc:
            mock_calc.return_value = PricingResult(mean=200.0)
            
            result = model.simulate()
            
            assert result.mean == 200.0
    
    def test_simulate_with_invalid_backend(self):
        """Test simulate method with invalid backend."""
        from quactuary.backend import BackendManager

        # Create invalid backend
        invalid_backend = BackendManager(Mock())
        
        model = PricingModel(test_portfolio)
        
        with pytest.raises(ValueError, match="Unsupported backend type"):
            model.simulate(backend=invalid_backend)
    
    def test_simulate_with_parameters(self):
        """Test simulate method with various parameters."""
        from quactuary.backend import BackendManager, ClassicalBackend
        
        model = PricingModel(test_portfolio)
        backend = BackendManager(ClassicalBackend())
        
        # Mock the classical pricing model method
        with patch.object(model, 'calculate_portfolio_statistics') as mock_calc:
            mock_calc.return_value = PricingResult(mean=150.0, variance=100.0)
            
            result = model.simulate(
                mean=True,
                variance=True,
                value_at_risk=False,
                tail_value_at_risk=False,
                tail_alpha=0.01,
                n_sims=5000,
                backend=backend
            )
            
            mock_calc.assert_called_once_with(
                model.portfolio, True, True, False, False, 0.01, 5000
            )
            assert result.mean == 150.0
            assert result.variance == 100.0


class TestCompoundDistributionIntegration:
    """Test compound distribution integration with pricing model."""
    
    def test_compound_distribution_with_different_models(self):
        """Test compound distribution with various frequency and severity models."""
        from quactuary.distributions.frequency import (BinomialFrequency,
                                                       PoissonFrequency)
        from quactuary.distributions.severity import (ExponentialSeverity,
                                                      GammaSeverity)
        
        model = PricingModel(test_portfolio)
        
        # Test Poisson-Exponential
        freq1 = PoissonFrequency(lambda_=5)
        sev1 = ExponentialSeverity(scale=1000)
        model.set_compound_distribution(freq1, sev1)
        
        results1 = model.calculate_aggregate_statistics()
        assert results1['mean'] > 0
        assert results1['std'] > 0
        
        # Test Binomial-Gamma
        freq2 = BinomialFrequency(n=10, p=0.3)
        sev2 = GammaSeverity(shape=2, scale=500)
        model.set_compound_distribution(freq2, sev2)
        
        results2 = model.calculate_aggregate_statistics()
        assert results2['mean'] > 0
        assert results2['std'] > 0
    
    def test_edge_cases(self):
        """Test edge cases in pricing calculations."""
        model = PricingModel(test_portfolio)
        
        # Zero frequency
        zero_freq = DeterministicFrequency(0)
        model.set_compound_distribution(zero_freq, test_sev)
        
        results = model.calculate_aggregate_statistics()
        assert results['mean'] == 0
        
        # Price layer with zero attachment
        layer_results = model.price_excess_layer(0, 1000)
        assert layer_results['attachment'] == 0
        assert layer_results['expected_loss'] == 0