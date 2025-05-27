"""Comprehensive tests for quantum module implementation."""

import pytest
import numpy as np
import pandas as pd

# Import from the renamed quantum_pricing.py module 
from quactuary.quantum_pricing import QuantumPricingModel
from quactuary.book import Portfolio


class TestQuantumPricingModel:
    """Test suite for QuantumPricingModel."""
    
    def test_initialization(self):
        """Test model initialization with various parameters."""
        # Default initialization
        model1 = QuantumPricingModel()
        assert model1.shots == 8192
        assert model1.optimization_level == 1
        
        # Custom initialization
        model2 = QuantumPricingModel(shots=16384, optimization_level=3)
        assert model2.shots == 16384
        assert model2.optimization_level == 3
    
    def test_mean_loss_estimation(self):
        """Test quantum mean loss estimation."""
        model = QuantumPricingModel()
        
        # Create a dummy portfolio
        policies_df = pd.DataFrame({
            'policy_id': ['P001'],
            'premium': [1000],
            'exposure': [1.0]
        })
        portfolio = Portfolio(policies_df)
        
        # Test mean loss calculation
        mean, error = model.mean_loss(portfolio, n_qubits=6)
        
        # Check that we get reasonable values
        assert isinstance(mean, float)
        assert isinstance(error, float)
        assert mean > 0
        assert error > 0
        assert error < mean  # Error should be smaller than estimate
    
    def test_variance_calculation(self):
        """Test quantum variance calculation."""
        model = QuantumPricingModel()
        
        # Create a dummy portfolio
        policies_df = pd.DataFrame({
            'policy_id': ['P001'],
            'premium': [1000],
            'exposure': [1.0]
        })
        portfolio = Portfolio(policies_df)
        
        # Test variance calculation
        variance, error = model.variance(portfolio, n_qubits=6)
        
        # Check that we get reasonable values
        assert isinstance(variance, float)
        assert isinstance(error, float)
        assert variance >= 0  # Variance must be non-negative
        assert error > 0
    
    def test_value_at_risk(self):
        """Test quantum VaR calculation."""
        model = QuantumPricingModel()
        
        # Create a dummy portfolio
        policies_df = pd.DataFrame({
            'policy_id': ['P001'],
            'premium': [1000],
            'exposure': [1.0]
        })
        portfolio = Portfolio(policies_df)
        
        # Test VaR at different confidence levels
        var_95, error_95 = model.value_at_risk(portfolio, alpha=0.95, num_qubits=6)
        var_99, error_99 = model.value_at_risk(portfolio, alpha=0.99, num_qubits=6)
        
        # VaR should increase with confidence level
        assert var_99 > var_95
        assert error_95 > 0
        assert error_99 > 0
    
    def test_tail_value_at_risk(self):
        """Test quantum TVaR calculation."""
        model = QuantumPricingModel()
        
        # Create a dummy portfolio
        policies_df = pd.DataFrame({
            'policy_id': ['P001'],
            'premium': [1000],
            'exposure': [1.0]
        })
        portfolio = Portfolio(policies_df)
        
        # Test TVaR calculation
        tvar, tvar_error = model.tail_value_at_risk(portfolio, alpha=0.95, num_qubits=6)
        var, var_error = model.value_at_risk(portfolio, alpha=0.95, num_qubits=6)
        
        # TVaR should be >= VaR at same confidence level
        assert tvar >= var
        assert tvar_error > 0
    
    def test_quantum_excess_evaluation_default(self):
        """Test quantum excess evaluation with default parameters."""
        model = QuantumPricingModel()
        
        payout, ci_width = model.quantum_excess_evaluation()
        
        # Check results are in expected range
        assert 0.0 <= payout <= 1.0
        assert ci_width >= 0
        
        # Check against expected value from pilot notebook
        expected_payout = 0.295570
        assert abs(payout - expected_payout) < 0.05  # Within 5% tolerance
    
    def test_quantum_excess_evaluation_parameters(self):
        """Test quantum excess evaluation with various parameters."""
        model = QuantumPricingModel()
        
        # Test with different qubit counts
        results = []
        for n_qubits in [4, 6, 8]:
            payout, ci = model.quantum_excess_evaluation(num_qubits=n_qubits)
            results.append((n_qubits, payout, ci))
            assert 0.0 <= payout <= 1.0
        
        # Results should converge as qubit count increases
        # (Though not necessarily monotonically due to discretization)
        print("\nQubit convergence test:")
        for n, p, c in results:
            print(f"  {n} qubits: payout={p:.6f}, CI={c:.6f}")
    
    def test_quantum_excess_evaluation_edge_cases(self):
        """Test quantum excess evaluation with edge cases."""
        model = QuantumPricingModel()
        
        # High deductible (should result in low payout)
        payout_high_deduct, _ = model.quantum_excess_evaluation(
            deductible=5.0,  # High deductible
            domain_max=10.0
        )
        assert payout_high_deduct < 0.1  # Should be very low
        
        # Low deductible (should result in higher payout)
        payout_low_deduct, _ = model.quantum_excess_evaluation(
            deductible=0.1,  # Low deductible
            domain_max=10.0
        )
        assert payout_low_deduct > payout_high_deduct
        
        # Full retention (coins=1.0, should result in zero payout)
        payout_full_retention, _ = model.quantum_excess_evaluation(
            coins=1.0  # 100% retained by cedent
        )
        assert abs(payout_full_retention) < 0.001  # Should be ~0
    
    def test_portfolio_statistics_integration(self):
        """Test full portfolio statistics calculation."""
        model = QuantumPricingModel()
        
        # Create a dummy portfolio
        policies_df = pd.DataFrame({
            'policy_id': ['P001', 'P002'],
            'premium': [1000, 2000],
            'exposure': [1.0, 1.5]
        })
        portfolio = Portfolio(policies_df)
        
        # Calculate all statistics
        result = model.calculate_portfolio_statistics(
            portfolio=portfolio,
            mean=True,
            variance=True,
            value_at_risk=True,
            tail_value_at_risk=True,
            tail_alpha=0.05,
            n_qubits=6
        )
        
        # Check result structure
        assert hasattr(result, 'estimates')
        assert hasattr(result, 'standard_errors')
        assert hasattr(result, 'metadata')
        
        # Check all requested statistics are present
        assert 'mean' in result.estimates
        assert 'variance' in result.estimates
        assert 'VaR_0.05' in result.estimates
        assert 'TVaR_0.05' in result.estimates
        
        # Check metadata
        assert result.metadata['computation_type'] == 'quantum'
        assert result.metadata['shots'] == 8192
    
    def test_quantum_circuit_helpers(self):
        """Test helper methods for quantum circuits."""
        model = QuantumPricingModel()
        
        # Test subtractor circuit
        from qiskit import QuantumRegister
        qr = QuantumRegister(4, 'q')
        sub_circuit = model._make_subtractor(qr, 3)
        
        # Check circuit properties
        assert sub_circuit.num_qubits == 4
        assert len(sub_circuit.data) > 0  # Should have gates
        
        # Test rotation application
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(4, 1)
        qr = qc.qregs[0]
        cr = qc.cregs[0]
        
        # Add a control qubit
        from qiskit import QuantumRegister
        control = QuantumRegister(1, 'control')
        payout = QuantumRegister(1, 'payout')
        qc.add_register(control)
        qc.add_register(payout)
        
        # Apply rotations
        model._apply_excess_rotations(
            qc, qr, control[0], payout[0],
            step=0.1, c_param=0.015
        )
        
        # Check that rotations were added
        assert len(qc.data) > 0


@pytest.mark.parametrize("n_qubits,expected_range", [
    (4, (0.15, 0.35)),   # Lower precision
    (6, (0.25, 0.35)),   # Medium precision  
    (8, (0.28, 0.32)),   # Higher precision
])
def test_quantum_excess_convergence(n_qubits, expected_range):
    """Test that quantum excess evaluation converges with more qubits."""
    model = QuantumPricingModel()
    
    payout, _ = model.quantum_excess_evaluation(num_qubits=n_qubits)
    
    assert expected_range[0] <= payout <= expected_range[1], \
        f"Payout {payout} not in expected range {expected_range} for {n_qubits} qubits"


if __name__ == "__main__":
    # Run basic tests
    test = TestQuantumPricingModel()
    
    print("Running quantum module tests...")
    print("=" * 50)
    
    try:
        test.test_initialization()
        print("✓ Initialization test passed")
    except Exception as e:
        print(f"✗ Initialization test failed: {e}")
    
    try:
        test.test_quantum_excess_evaluation_default()
        print("✓ Quantum excess evaluation test passed")
    except Exception as e:
        print(f"✗ Quantum excess evaluation test failed: {e}")
    
    try:
        test.test_quantum_excess_evaluation_parameters()
        print("✓ Parameter variation test passed")
    except Exception as e:
        print(f"✗ Parameter variation test failed: {e}")
    
    try:
        test.test_quantum_excess_evaluation_edge_cases()
        print("✓ Edge case test passed")
    except Exception as e:
        print(f"✗ Edge case test failed: {e}")
    
    print("\n" + "=" * 50)
    print("Basic tests completed!")