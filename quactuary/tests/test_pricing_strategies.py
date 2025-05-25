"""
Tests for pricing strategy pattern implementation.

Validates the refactored PricingModel architecture using composition
instead of multiple inheritance.
"""

import unittest
from unittest.mock import Mock, patch

import pandas as pd

from quactuary.book import Portfolio, PolicyTerms
from quactuary.pricing import PricingModel
from quactuary.pricing_strategies import (
    PricingStrategy, 
    ClassicalPricingStrategy, 
    QuantumPricingStrategy,
    get_strategy_for_backend
)
from quactuary.datatypes import PricingResult
from quactuary.backend import ClassicalBackend


class TestPricingStrategies(unittest.TestCase):
    """Test the pricing strategy pattern implementation."""
    
    def setUp(self):
        """Set up test portfolio."""
        # Create a simple test portfolio
        policy = PolicyTerms(
            effective_date=pd.Timestamp('2023-01-01'),
            expiration_date=pd.Timestamp('2023-12-31'),
            per_occ_retention=1000.0,
            per_occ_limit=10000.0
        )
        self.portfolio = Portfolio([policy])
    
    def test_pricing_model_default_strategy(self):
        """Test that PricingModel uses ClassicalPricingStrategy by default."""
        model = PricingModel(self.portfolio)
        
        # Check that default strategy is ClassicalPricingStrategy
        self.assertIsInstance(model.strategy, ClassicalPricingStrategy)
        self.assertEqual(model.portfolio, self.portfolio)
        self.assertIsNone(model.compound_distribution)
    
    def test_pricing_model_custom_strategy(self):
        """Test that PricingModel accepts custom strategy."""
        custom_strategy = ClassicalPricingStrategy()
        model = PricingModel(self.portfolio, strategy=custom_strategy)
        
        # Check that custom strategy is used
        self.assertEqual(model.strategy, custom_strategy)
    
    def test_classical_strategy_delegation(self):
        """Test that ClassicalPricingStrategy delegates to ClassicalPricingModel."""
        strategy = ClassicalPricingStrategy()
        
        # Mock the classical model to avoid actual computation
        with patch('quactuary.classical.ClassicalPricingModel') as mock_class:
            mock_instance = Mock()
            mock_class.return_value = mock_instance
            
            # Set up mock return value
            expected_result = PricingResult(
                estimates={'mean': 5000.0},
                intervals={},
                samples=pd.Series([4000, 5000, 6000]),
                metadata={'n_sims': 3}
            )
            mock_instance.calculate_portfolio_statistics.return_value = expected_result
            
            # Call the strategy
            result = strategy.calculate_portfolio_statistics(
                portfolio=self.portfolio,
                mean=True,
                n_sims=3
            )
            
            # Verify delegation occurred
            mock_instance.calculate_portfolio_statistics.assert_called_once_with(
                portfolio=self.portfolio,
                mean=True,
                variance=True,
                value_at_risk=True,
                tail_value_at_risk=True,
                tail_alpha=0.05,
                n_sims=3
            )
            self.assertEqual(result, expected_result)
    
    def test_quantum_strategy_not_implemented(self):
        """Test that QuantumPricingStrategy raises NotImplementedError."""
        strategy = QuantumPricingStrategy()
        
        with self.assertRaises(NotImplementedError) as context:
            strategy.calculate_portfolio_statistics(self.portfolio)
        
        self.assertIn("not yet implemented", str(context.exception))
        self.assertIn("ClassicalPricingStrategy", str(context.exception))
    
    def test_pricing_model_simulate_delegates_to_strategy(self):
        """Test that PricingModel.simulate() delegates to strategy."""
        # Create mock strategy
        mock_strategy = Mock(spec=PricingStrategy)
        expected_result = PricingResult(
            estimates={'mean': 7500.0},
            intervals={},
            samples=pd.Series([7000, 7500, 8000]),
            metadata={'n_sims': 3}
        )
        mock_strategy.calculate_portfolio_statistics.return_value = expected_result
        
        # Create model with mock strategy
        model = PricingModel(self.portfolio, strategy=mock_strategy)
        
        # Call simulate
        result = model.simulate(mean=True, variance=False, n_sims=3)
        
        # Verify strategy was called correctly
        mock_strategy.calculate_portfolio_statistics.assert_called_once_with(
            portfolio=self.portfolio,
            mean=True,
            variance=False,
            value_at_risk=True,
            tail_value_at_risk=True,
            tail_alpha=0.05,
            n_sims=3
        )
        self.assertEqual(result, expected_result)
    
    def test_get_strategy_for_backend(self):
        """Test the backend strategy factory function."""
        # Test with classical backend
        with patch('quactuary.pricing_strategies.get_backend') as mock_get_backend:
            mock_backend_manager = Mock()
            mock_backend_manager.backend = ClassicalBackend()
            mock_get_backend.return_value = mock_backend_manager
            
            strategy = get_strategy_for_backend()
            self.assertIsInstance(strategy, ClassicalPricingStrategy)
    
    def test_pricing_model_backend_override(self):
        """Test that simulate() can override backend temporarily."""
        model = PricingModel(self.portfolio)
        
        # Mock the get_strategy_for_backend function
        with patch('quactuary.pricing.get_strategy_for_backend') as mock_get_strategy:
            mock_strategy = Mock(spec=PricingStrategy)
            expected_result = PricingResult(
                estimates={'mean': 9000.0},
                intervals={},
                samples=pd.Series([8500, 9000, 9500]),
                metadata={'backend_override': True}
            )
            mock_strategy.calculate_portfolio_statistics.return_value = expected_result
            mock_get_strategy.return_value = mock_strategy
            
            # Create mock backend
            mock_backend = Mock()
            
            # Call simulate with backend override
            result = model.simulate(backend=mock_backend, n_sims=5)
            
            # Verify temporary strategy was used
            mock_get_strategy.assert_called_once_with(mock_backend)
            mock_strategy.calculate_portfolio_statistics.assert_called_once_with(
                portfolio=self.portfolio,
                mean=True,
                variance=True,
                value_at_risk=True,
                tail_value_at_risk=True,
                tail_alpha=0.05,
                n_sims=5
            )
            self.assertEqual(result, expected_result)
    
    def test_strategy_pattern_backwards_compatibility(self):
        """Test that existing API is preserved for backwards compatibility."""
        model = PricingModel(self.portfolio)
        
        # These method calls should work without errors
        # (even if they don't compute real results in this test)
        self.assertTrue(hasattr(model, 'simulate'))
        self.assertTrue(hasattr(model, 'set_compound_distribution'))
        self.assertTrue(hasattr(model, 'calculate_aggregate_statistics'))
        
        # Verify simulate method signature is preserved
        import inspect
        sig = inspect.signature(model.simulate)
        expected_params = ['mean', 'variance', 'value_at_risk', 'tail_value_at_risk', 'tail_alpha', 'n_sims', 'backend']
        actual_params = list(sig.parameters.keys())
        
        for param in expected_params:
            self.assertIn(param, actual_params)


if __name__ == '__main__':
    unittest.main()