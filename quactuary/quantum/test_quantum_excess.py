"""Test script for quantum excess evaluation algorithm."""

import sys
sys.path.append('/mnt/c/Users/alexf/OneDrive/Documents/Projects/quActuary/quactuary')

from quactuary.quantum import QuantumPricingModel
import numpy as np

def test_quantum_excess_evaluation():
    """Test the quantum excess evaluation implementation."""
    print("Testing Quantum Excess Evaluation Algorithm")
    print("=" * 50)
    
    # Create quantum pricing model
    model = QuantumPricingModel()
    
    # Test with default parameters
    print("\nTest 1: Default parameters (6 qubits)")
    try:
        payout, ci_width = model.quantum_excess_evaluation()
        print(f"✓ Expected payout: {payout:.6f}")
        print(f"✓ Confidence interval width: {ci_width:.6f}")
        print(f"✓ 95% CI: [{payout - ci_width:.6f}, {payout + ci_width:.6f}]")
    except Exception as e:
        print(f"✗ Test failed: {e}")
    
    # Test with different parameters
    print("\nTest 2: Modified parameters (8 qubits, higher deductible)")
    try:
        payout, ci_width = model.quantum_excess_evaluation(
            num_qubits=8,
            deductible=2.0,
            coins=0.7
        )
        print(f"✓ Expected payout: {payout:.6f}")
        print(f"✓ Confidence interval width: {ci_width:.6f}")
    except Exception as e:
        print(f"✗ Test failed: {e}")
    
    # Compare with classical calculation
    print("\nTest 3: Comparison with classical calculation")
    try:
        # Quantum calculation
        quantum_payout, quantum_ci = model.quantum_excess_evaluation()
        
        # Classical calculation (from notebook)
        classical_payout = 0.295570  # Expected value from notebook
        
        print(f"✓ Quantum payout: {quantum_payout:.6f}")
        print(f"✓ Classical payout: {classical_payout:.6f}")
        print(f"✓ Difference: {abs(quantum_payout - classical_payout):.6f}")
        
        # Check if within reasonable tolerance
        tolerance = 0.05
        if abs(quantum_payout - classical_payout) < tolerance:
            print(f"✓ Results match within tolerance ({tolerance})")
        else:
            print(f"⚠ Results differ by more than tolerance ({tolerance})")
            
    except Exception as e:
        print(f"✗ Test failed: {e}")
    
    print("\n" + "=" * 50)
    print("Testing completed!")

if __name__ == "__main__":
    test_quantum_excess_evaluation()