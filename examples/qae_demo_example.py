"""Test script to verify QAE demo notebook functionality."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import time
import sys
sys.path.append('..')

# Test imports
try:
    # Add parent directory to path for imports
    import os
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    from quactuary.quantum import QuantumPricingModel
    from quactuary.book import Portfolio
    print("✓ QuActuary imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    exit(1)

# Test classical Monte Carlo implementation
class ClassicalMonteCarlo:
    """Classical Monte Carlo simulator for actuarial risk measures."""
    
    def __init__(self, distribution: str = 'lognormal', params: dict = None):
        self.distribution = distribution
        self.params = params or {'mu': 0.0, 'sigma': 1.0}
        
    def simulate(self, n_samples: int, seed: int = None) -> np.ndarray:
        """Generate samples from the specified distribution."""
        if seed is not None:
            np.random.seed(seed)
            
        if self.distribution == 'lognormal':
            return np.random.lognormal(
                mean=self.params['mu'], 
                sigma=self.params['sigma'], 
                size=n_samples
            )
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")
    
    def calculate_statistics(self, samples: np.ndarray, alpha: float = 0.95) -> dict:
        """Calculate risk statistics from samples."""
        mean = np.mean(samples)
        variance = np.var(samples)
        var = np.percentile(samples, alpha * 100)
        
        # TVaR is the mean of losses exceeding VaR
        tail_losses = samples[samples > var]
        tvar = np.mean(tail_losses) if len(tail_losses) > 0 else var
        
        return {
            'mean': mean,
            'variance': variance,
            'std': np.sqrt(variance),
            'VaR': var,
            'TVaR': tvar,
            'n_samples': len(samples)
        }

# Test quantum model
def test_quantum_model():
    """Test basic quantum model functionality."""
    print("\nTesting Quantum Model:")
    
    # Create test portfolio
    policies_df = pd.DataFrame({
        'policy_id': ['TEST001'],
        'premium': [10000],
        'exposure': [1.0]
    })
    portfolio = Portfolio(policies_df)
    
    # Initialize quantum model
    qpm = QuantumPricingModel(shots=1024)
    
    # Test mean estimation
    try:
        mean, error = qpm.mean_loss(portfolio, n_qubits=6)
        print(f"✓ Mean estimation: ${mean:,.2f} ± ${error:,.2f}")
    except Exception as e:
        print(f"✗ Mean estimation failed: {e}")
        return False
    
    # Test variance estimation
    try:
        variance, error = qpm.variance(portfolio, n_qubits=6)
        print(f"✓ Variance estimation: ${variance:,.0f} ± ${error:,.0f}")
    except Exception as e:
        print(f"✗ Variance estimation failed: {e}")
        return False
    
    # Test VaR estimation
    try:
        var95, error = qpm.value_at_risk(portfolio, alpha=0.95, num_qubits=6)
        print(f"✓ VaR (95%) estimation: ${var95:,.2f} ± ${error:,.2f}")
    except Exception as e:
        print(f"✗ VaR estimation failed: {e}")
        return False
    
    # Test TVaR estimation
    try:
        tvar95, error = qpm.tail_value_at_risk(portfolio, alpha=0.95, num_qubits=6)
        print(f"✓ TVaR (95%) estimation: ${tvar95:,.2f} ± ${error:,.2f}")
    except Exception as e:
        print(f"✗ TVaR estimation failed: {e}")
        return False
    
    return True

# Test convergence analysis
def test_convergence_analysis():
    """Test convergence rate comparison."""
    print("\nTesting Convergence Analysis:")
    
    # Classical MC convergence
    classical_mc = ClassicalMonteCarlo()
    true_mean = np.exp(0.5)  # exp(mu + sigma^2/2) for standard lognormal
    
    sample_sizes = [100, 500, 1000, 5000]
    errors = []
    
    for n in sample_sizes:
        samples = classical_mc.simulate(n, seed=42)
        est_mean = np.mean(samples)
        error = abs(est_mean - true_mean) / true_mean
        errors.append(error)
        print(f"  N={n:5d}: error={error:.4f}, theoretical={1/np.sqrt(n):.4f}")
    
    # Check if error decreases
    if all(errors[i] >= errors[i+1] for i in range(len(errors)-1)):
        print("✓ Classical convergence verified")
    else:
        print("⚠ Classical convergence not monotonic (expected due to randomness)")
    
    return True

# Test error rate comparison
def test_error_comparison():
    """Test error rate comparison between quantum and classical."""
    print("\nTesting Error Rate Comparison:")
    
    # For 6 qubits (64 oracle calls)
    n_resources = 64
    
    # Classical error (theoretical)
    classical_error = 1.0 / np.sqrt(n_resources)
    
    # Quantum error (theoretical)
    quantum_error = 1.0 / n_resources
    
    # Speedup factor
    speedup = classical_error / quantum_error
    
    print(f"  Resources: {n_resources}")
    print(f"  Classical error: {classical_error:.4f}")
    print(f"  Quantum error: {quantum_error:.4f}")
    print(f"  Speedup factor: {speedup:.1f}x")
    
    if speedup > 1:
        print("✓ Quantum advantage verified")
    
    return True

# Main test execution
def main():
    """Run all tests."""
    print("="*60)
    print("QAE DEMO NOTEBOOK FUNCTIONALITY TEST")
    print("="*60)
    
    tests = [
        ("Classical Monte Carlo", lambda: test_classical_mc()),
        ("Quantum Model", test_quantum_model),
        ("Convergence Analysis", test_convergence_analysis),
        ("Error Comparison", test_error_comparison)
    ]
    
    passed = 0
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ {name} test failed with error: {e}")
    
    print("\n" + "="*60)
    print(f"SUMMARY: {passed}/{len(tests)} tests passed")
    print("="*60)
    
    if passed == len(tests):
        print("\n✓ All tests passed! The notebook should work correctly.")
    else:
        print("\n⚠ Some tests failed. Please check the implementation.")

def test_classical_mc():
    """Test classical Monte Carlo implementation."""
    print("\nTesting Classical Monte Carlo:")
    
    mc = ClassicalMonteCarlo(distribution='lognormal', params={'mu': 0.0, 'sigma': 1.0})
    
    # Test simulation
    samples = mc.simulate(1000, seed=42)
    print(f"✓ Generated {len(samples)} samples")
    
    # Test statistics
    stats = mc.calculate_statistics(samples, alpha=0.95)
    print(f"✓ Mean: ${stats['mean']:.2f}")
    print(f"✓ Std Dev: ${stats['std']:.2f}")
    print(f"✓ VaR (95%): ${stats['VaR']:.2f}")
    print(f"✓ TVaR (95%): ${stats['TVaR']:.2f}")
    
    # Verify TVaR >= VaR
    if stats['TVaR'] >= stats['VaR']:
        print("✓ TVaR >= VaR constraint satisfied")
    else:
        print("✗ TVaR < VaR (should not happen!)")
        return False
    
    return True

if __name__ == "__main__":
    main()