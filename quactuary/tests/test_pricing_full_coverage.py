"""
Full coverage tests for pricing module.
"""
from datetime import date
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from quactuary.book import Inforce, PolicyTerms, Portfolio
# Test without importing quactuary first to ensure mocks work
from quactuary.datatypes import PricingResult
from quactuary.distributions.compound import CompoundDistribution
from quactuary.distributions.frequency import DeterministicFrequency, Poisson
from quactuary.distributions.severity import ConstantSeverity, Exponential
from quactuary.pricing import PricingModel


# Test data
def create_test_portfolio():
    """Create test portfolio."""
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

    return Portfolio([test_inforce]), test_freq, test_sev


class TestPricingModelBasic:
    """Basic PricingModel tests."""
    
    def test_init(self):
        """Test initialization."""
        portfolio, _, _ = create_test_portfolio()
        model = PricingModel(portfolio)
        assert model.portfolio == portfolio
        assert model.compound_distribution is None
    
    def test_set_compound_distribution(self):
        """Test setting compound distribution."""
        portfolio, freq, sev = create_test_portfolio()
        model = PricingModel(portfolio)
        
        model.set_compound_distribution(freq, sev)
        assert model.compound_distribution is not None
    
    def test_set_compound_distribution_different_types(self):
        """Test with different distribution types."""
        portfolio, _, _ = create_test_portfolio()
        model = PricingModel(portfolio)
        
        # Poisson-Exponential
        poisson_freq = Poisson(lambda_=5)
        exp_sev = Exponential(scale=1000)
        model.set_compound_distribution(poisson_freq, exp_sev)
        assert model.compound_distribution is not None


class TestCalculateAggregateStatistics:
    """Test calculate_aggregate_statistics method."""
    
    def test_without_compound_distribution(self):
        """Test error when compound distribution not set."""
        portfolio, _, _ = create_test_portfolio()
        model = PricingModel(portfolio)
        
        with pytest.raises(ValueError, match="Compound distribution not set"):
            model.calculate_aggregate_statistics()
    
    def test_default_confidence_levels(self):
        """Test with default confidence levels."""
        portfolio, freq, sev = create_test_portfolio()
        model = PricingModel(portfolio)
        model.set_compound_distribution(freq, sev)
        
        results = model.calculate_aggregate_statistics()
        
        # Check basic statistics
        assert 'mean' in results
        assert 'std' in results
        assert 'variance' in results
        assert 'has_analytical' in results
        
        # Check default VaR levels
        assert 'var_90%' in results
        assert 'var_95%' in results
        assert 'var_99%' in results
        
        # Check TVaR levels
        assert 'tvar_90%' in results
        assert 'tvar_95%' in results
        assert 'tvar_99%' in results
    
    def test_custom_confidence_levels(self):
        """Test with custom confidence levels."""
        portfolio, freq, sev = create_test_portfolio()
        model = PricingModel(portfolio)
        model.set_compound_distribution(freq, sev)
        
        custom_levels = [0.80, 0.85, 0.975]
        results = model.calculate_aggregate_statistics(confidence_levels=custom_levels)
        
        assert 'var_80%' in results
        assert 'var_85%' in results
        assert 'var_97.5%' in results
        assert 'tvar_80%' in results
        assert 'tvar_85%' in results
        assert 'tvar_97.5%' in results
    
    def test_with_policy_terms(self):
        """Test with policy terms application."""
        portfolio, freq, sev = create_test_portfolio()
        model = PricingModel(portfolio)
        model.set_compound_distribution(freq, sev)
        
        results = model.calculate_aggregate_statistics(apply_policy_terms=True)
        assert 'note' in results
        assert results['note'] == 'Policy terms application not yet implemented'
    
    def test_tvar_calculation_edge_case(self):
        """Test TVaR calculation when no samples exceed VaR."""
        portfolio, freq, sev = create_test_portfolio()
        model = PricingModel(portfolio)
        
        # Create a compound distribution that returns small values
        model.set_compound_distribution(freq, sev)
        
        # Mock the compound distribution to return controlled samples
        with patch.object(model.compound_distribution, 'rvs') as mock_rvs:
            # All samples are below 100
            mock_rvs.return_value = np.ones(10000) * 50
            
            with patch.object(model.compound_distribution, 'ppf') as mock_ppf:
                # VaR is high
                mock_ppf.return_value = 1000
                
                results = model.calculate_aggregate_statistics()
                
                # TVaR should fallback to VaR when no samples exceed it
                assert results['tvar_90%'] == 1000
                assert results['tvar_95%'] == 1000
                assert results['tvar_99%'] == 1000


class TestPriceExcessLayer:
    """Test price_excess_layer method."""
    
    def test_without_compound_distribution(self):
        """Test error when compound distribution not set."""
        portfolio, _, _ = create_test_portfolio()
        model = PricingModel(portfolio)
        
        with pytest.raises(ValueError, match="Compound distribution not set"):
            model.price_excess_layer(1000, 5000)
    
    def test_basic_layer_pricing(self):
        """Test basic excess layer pricing."""
        portfolio, freq, sev = create_test_portfolio()
        model = PricingModel(portfolio)
        model.set_compound_distribution(freq, sev)
        
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
    
    def test_zero_attachment(self):
        """Test layer with zero attachment."""
        portfolio, freq, sev = create_test_portfolio()
        model = PricingModel(portfolio)
        model.set_compound_distribution(freq, sev)
        
        results = model.price_excess_layer(0, 10000, 500)
        assert results['attachment'] == 0
        assert results['expected_loss'] >= 0
    
    def test_no_losses_exceed_attachment(self):
        """Test when no losses exceed attachment."""
        portfolio, freq, sev = create_test_portfolio()
        model = PricingModel(portfolio)
        model.set_compound_distribution(freq, sev)
        
        # Mock to return small losses
        with patch.object(model.compound_distribution, 'rvs') as mock_rvs:
            mock_rvs.return_value = np.ones(1000) * 10  # All losses are 10
            
            results = model.price_excess_layer(
                attachment=1000,  # High attachment
                limit=5000,
                n_simulations=1000
            )
            
            assert results['expected_loss'] == 0
            assert results['loss_probability'] == 0
            assert results['average_severity'] == 0


class TestSimulateMethod:
    """Test simulate method with different backends."""
    
    @patch('quactuary.backend.get_backend')
    def test_default_classical_backend(self, mock_get_backend):
        """Test with default classical backend."""
        from quactuary.backend import BackendManager, ClassicalBackend
        
        portfolio, _, _ = create_test_portfolio()
        model = PricingModel(portfolio)
        
        # Mock classical backend
        mock_backend = Mock()
        mock_backend.backend = ClassicalBackend()
        mock_get_backend.return_value = mock_backend
        
        # Mock the classical calculation
        with patch('quactuary.classical.ClassicalPricingModel.calculate_portfolio_statistics') as mock_calc:
            mock_calc.return_value = PricingResult(mean=100.0, variance=50.0)
            
            result = model.simulate()
            
            mock_calc.assert_called_once_with(
                model, model.portfolio, True, True, True, True, 0.05, None
            )
            assert result.mean == 100.0
            assert result.variance == 50.0
    
    def test_explicit_classical_backend(self):
        """Test with explicit classical backend."""
        from quactuary.backend import BackendManager, ClassicalBackend
        
        portfolio, _, _ = create_test_portfolio()
        model = PricingModel(portfolio)
        backend = BackendManager(ClassicalBackend())
        
        with patch('quactuary.classical.ClassicalPricingModel.calculate_portfolio_statistics') as mock_calc:
            mock_calc.return_value = PricingResult(mean=150.0)
            
            result = model.simulate(
                mean=True,
                variance=False,
                value_at_risk=False,
                tail_value_at_risk=False,
                tail_alpha=0.01,
                n_sims=5000,
                backend=backend
            )
            
            mock_calc.assert_called_once_with(
                model, model.portfolio, True, False, False, False, 0.01, 5000
            )
    
    @patch('quactuary.backend.get_backend')
    def test_quantum_backend(self, mock_get_backend):
        """Test with quantum backend."""
        from qiskit.providers import Backend

        from quactuary.backend import BackendManager
        
        portfolio, _, _ = create_test_portfolio()
        model = PricingModel(portfolio)
        
        # Mock quantum backend
        mock_backend = Mock()
        mock_backend.backend = Mock(spec=Backend)
        mock_get_backend.return_value = mock_backend
        
        with patch('quactuary.quantum.QuantumPricingModel.calculate_portfolio_statistics') as mock_calc:
            mock_calc.return_value = PricingResult(mean=200.0)
            
            result = model.simulate()
            
            assert result.mean == 200.0
    
    @patch('quactuary.backend.get_backend')
    def test_quantum_backend_v1(self, mock_get_backend):
        """Test with quantum BackendV1."""
        from qiskit.providers import BackendV1

        from quactuary.backend import BackendManager
        
        portfolio, _, _ = create_test_portfolio()
        model = PricingModel(portfolio)
        
        # Mock quantum backend V1
        mock_backend = Mock()
        mock_backend.backend = Mock(spec=BackendV1)
        mock_get_backend.return_value = mock_backend
        
        with patch('quactuary.quantum.QuantumPricingModel.calculate_portfolio_statistics') as mock_calc:
            mock_calc.return_value = PricingResult(mean=250.0)
            
            result = model.simulate()
            
            assert result.mean == 250.0
    
    def test_invalid_backend(self):
        """Test with invalid backend type."""
        from quactuary.backend import BackendManager
        
        portfolio, _, _ = create_test_portfolio()
        model = PricingModel(portfolio)
        
        # Create invalid backend
        invalid_backend = BackendManager(Mock())
        
        with pytest.raises(ValueError, match="Unsupported backend type"):
            model.simulate(backend=invalid_backend)


class TestEdgeCases:
    """Test edge cases and special scenarios."""
    
    def test_zero_frequency_distribution(self):
        """Test with zero frequency."""
        portfolio, _, sev = create_test_portfolio()
        model = PricingModel(portfolio)
        
        zero_freq = DeterministicFrequency(0)
        model.set_compound_distribution(zero_freq, sev)
        
        results = model.calculate_aggregate_statistics()
        assert results['mean'] == 0
        
        layer_results = model.price_excess_layer(1000, 5000)
        assert layer_results['expected_loss'] == 0
        assert layer_results['loss_probability'] == 0