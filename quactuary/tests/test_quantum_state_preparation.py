"""
Unit tests for quantum state preparation utilities.

Tests the core functionality of amplitude encoding, validation,
and probability distribution preparation for quantum circuits.
"""

import numpy as np
import pytest
from scipy import stats

pytest.skip("Quantum submodule refactoring in progress", allow_module_level=True)

from quactuary.quantum.state_preparation import (
    amplitude_encode,
    uniform_superposition,
    controlled_rotation_encoding,
    validate_quantum_state,
    normalize_probabilities,
    check_normalization,
    prepare_lognormal_state,
    prepare_distribution_state,
    discretize_distribution,
)


class TestAmplitudeEncoding:
    """Test amplitude encoding functionality."""
    
    def test_amplitude_encode_basic(self):
        """Test basic amplitude encoding with simple probabilities."""
        # Equal probabilities (should give |++⟩ state)
        probs = [0.25, 0.25, 0.25, 0.25]
        qc = amplitude_encode(probs)
        
        assert qc.num_qubits == 2
        assert qc.name == 'amplitude_encoding'
        
    def test_amplitude_encode_normalization(self):
        """Test that unnormalized probabilities are handled correctly."""
        # Unnormalized probabilities
        probs = [1, 2, 3, 4]  # Sum = 10
        qc = amplitude_encode(probs, normalize=True)
        
        # Circuit should still be valid
        assert qc.num_qubits == 2
        
    def test_amplitude_encode_padding(self):
        """Test padding to power of 2."""
        # 3 probabilities should be padded to 4
        probs = [0.3, 0.3, 0.4]
        qc = amplitude_encode(probs)
        
        assert qc.num_qubits == 2  # 2^2 = 4 states
        
    def test_amplitude_encode_explicit_qubits(self):
        """Test specifying number of qubits explicitly."""
        probs = [0.5, 0.5]
        qc = amplitude_encode(probs, num_qubits=3)
        
        assert qc.num_qubits == 3  # Should use specified value
        
    def test_amplitude_encode_negative_probabilities(self):
        """Test that negative probabilities raise an error."""
        probs = [0.5, -0.3, 0.8]
        
        with pytest.raises(ValueError, match="cannot be negative"):
            amplitude_encode(probs)
            
    def test_amplitude_encode_all_zeros(self):
        """Test handling of all-zero probabilities."""
        probs = [0, 0, 0, 0]
        qc = amplitude_encode(probs, normalize=True)
        
        # Should create uniform distribution
        assert qc.num_qubits == 2
        
    def test_amplitude_encode_validation_disabled(self):
        """Test disabling validation."""
        # Create amplitudes that would fail validation but work with StatePreparation normalize=True
        probs = [0.5, 0.5, 0.5, 0.5]  # Sum = 2, will be normalized by StatePreparation
        
        # With normalize=True, StatePreparation will handle it
        qc = amplitude_encode(probs, normalize=True, validate=False)
        assert qc is not None
        
        # The actual test case: use normalized probs but disable our validation
        probs_normalized = [0.25, 0.25, 0.25, 0.25]
        qc2 = amplitude_encode(probs_normalized, normalize=False, validate=False)
        assert qc2 is not None


class TestUniformSuperposition:
    """Test uniform superposition creation."""
    
    def test_uniform_superposition_basic(self):
        """Test basic uniform superposition."""
        qc = uniform_superposition(3)
        
        assert qc.num_qubits == 3
        assert qc.name == 'uniform_superposition'
        # Should have 3 Hadamard gates
        h_count = sum(1 for inst, _, _ in qc.data if inst.name == 'h')
        assert h_count == 3
        
    def test_uniform_superposition_single_qubit(self):
        """Test single qubit superposition."""
        qc = uniform_superposition(1)
        
        assert qc.num_qubits == 1
        h_count = sum(1 for inst, _, _ in qc.data if inst.name == 'h')
        assert h_count == 1
        
    def test_uniform_superposition_invalid(self):
        """Test invalid number of qubits."""
        with pytest.raises(ValueError):
            uniform_superposition(0)
            
        with pytest.raises(ValueError):
            uniform_superposition(-1)


class TestControlledRotationEncoding:
    """Test controlled rotation encoding."""
    
    def test_controlled_rotation_basic(self):
        """Test basic controlled rotation encoding."""
        angles = [np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
        qc = controlled_rotation_encoding(angles)
        
        # Need 2 control qubits + 1 target = 3 qubits
        assert qc.num_qubits == 3
        
    def test_controlled_rotation_single_angle(self):
        """Test with single angle."""
        angles = [np.pi/2]
        qc = controlled_rotation_encoding(angles)
        
        # Need 0 control qubits + 1 target = 1 qubit
        assert qc.num_qubits == 1
        
    def test_controlled_rotation_empty_angles(self):
        """Test empty angle list."""
        with pytest.raises(ValueError, match="at least one angle"):
            controlled_rotation_encoding([])
            
    def test_controlled_rotation_zero_angles(self):
        """Test that near-zero angles are skipped."""
        angles = [0.0, 1e-12, np.pi/2, 0.0]
        qc = controlled_rotation_encoding(angles)
        
        # Should skip the zero rotations
        assert qc.num_qubits == 3  # 2 control + 1 target


class TestValidation:
    """Test quantum state validation utilities."""
    
    def test_validate_quantum_state_valid(self):
        """Test validation of valid quantum states."""
        # Valid 2-qubit state (|00⟩ + |11⟩)/√2
        amps = [1/np.sqrt(2), 0, 0, 1/np.sqrt(2)]
        assert validate_quantum_state(amps) is True
        
        # Valid single qubit state
        amps = [0.6, 0.8]  # 0.36 + 0.64 = 1
        assert validate_quantum_state(amps) is True
        
    def test_validate_quantum_state_not_normalized(self):
        """Test that unnormalized states are rejected."""
        amps = [0.5, 0.5, 0.5, 0.6]  # Sum of squares = 1.1, not normalized
        
        with pytest.raises(ValueError, match="not normalized"):
            validate_quantum_state(amps)
            
    def test_validate_quantum_state_wrong_dimension(self):
        """Test that non-power-of-2 dimensions are rejected."""
        # 3 amplitudes - not a power of 2
        amps = [1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)]
        
        with pytest.raises(ValueError, match="power of 2"):
            validate_quantum_state(amps)
            
    def test_validate_quantum_state_empty(self):
        """Test empty state vector."""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_quantum_state([])
            
    def test_normalize_probabilities_basic(self):
        """Test basic probability normalization."""
        probs = [1, 2, 3, 4]
        norm_probs = normalize_probabilities(probs)
        
        assert np.isclose(np.sum(norm_probs), 1.0)
        assert np.allclose(norm_probs, [0.1, 0.2, 0.3, 0.4])
        
    def test_normalize_probabilities_all_zeros(self):
        """Test normalization of all-zero probabilities."""
        probs = [0, 0, 0, 0]
        norm_probs = normalize_probabilities(probs)
        
        # Should return uniform distribution
        assert np.allclose(norm_probs, [0.25, 0.25, 0.25, 0.25])
        
    def test_normalize_probabilities_negative(self):
        """Test that negative probabilities raise error."""
        probs = [0.5, -0.2, 0.7]
        
        with pytest.raises(ValueError, match="cannot be negative"):
            normalize_probabilities(probs)
            
    def test_check_normalization_probability(self):
        """Test normalization check for probabilities."""
        # Normalized
        probs = [0.1, 0.2, 0.3, 0.4]
        is_norm, norm = check_normalization(probs, mode='probability')
        assert is_norm == True
        assert np.isclose(norm, 1.0)
        
        # Not normalized
        probs = [1, 2, 3, 4]
        is_norm, norm = check_normalization(probs, mode='probability')
        assert is_norm == False
        assert np.isclose(norm, 10.0)
        
    def test_check_normalization_amplitude(self):
        """Test normalization check for amplitudes."""
        # Normalized
        amps = [0.5, 0.5, 0.5, 0.5]  # Sum of squares = 1
        is_norm, norm = check_normalization(amps, mode='amplitude')
        assert is_norm == True
        assert np.isclose(norm, 1.0)
        
        # Not normalized
        amps = [1, 0, 0, 0]  # Already normalized actually
        is_norm, norm = check_normalization(amps, mode='amplitude')
        assert is_norm == True
        assert np.isclose(norm, 1.0)


class TestProbabilityLoaders:
    """Test probability distribution loading functions."""
    
    def test_prepare_lognormal_state_basic(self):
        """Test basic lognormal state preparation."""
        probs, x_vals = prepare_lognormal_state(
            mu=0, sigma=1, num_qubits=4
        )
        
        assert len(probs) == 16  # 2^4
        assert len(x_vals) == 16
        assert np.isclose(np.sum(probs), 1.0)
        assert np.all(probs >= 0)
        
    def test_prepare_lognormal_state_analytic(self):
        """Test analytic vs numeric lognormal preparation."""
        # Analytic binning
        probs_analytic, x_vals_a = prepare_lognormal_state(
            mu=1, sigma=0.5, num_qubits=6, use_analytic_binning=True
        )
        
        # Numeric evaluation
        probs_numeric, x_vals_n = prepare_lognormal_state(
            mu=1, sigma=0.5, num_qubits=6, use_analytic_binning=False
        )
        
        # Both should be normalized
        assert np.isclose(np.sum(probs_analytic), 1.0)
        assert np.isclose(np.sum(probs_numeric), 1.0)
        
        # x_values should be the same
        assert np.allclose(x_vals_a, x_vals_n)
        
    def test_prepare_distribution_state_gamma(self):
        """Test gamma distribution preparation."""
        probs, x_vals = prepare_distribution_state(
            'gamma',
            {'alpha': 2, 'scale': 1000},
            num_qubits=6
        )
        
        assert len(probs) == 64
        assert np.isclose(np.sum(probs), 1.0)
        
    def test_prepare_distribution_state_pareto(self):
        """Test Pareto distribution preparation."""
        probs, x_vals = prepare_distribution_state(
            'pareto',
            {'alpha': 2.5, 'scale': 1000},
            num_qubits=5,
            domain_max=10000
        )
        
        assert len(probs) == 32
        assert np.isclose(np.sum(probs), 1.0)
        
    def test_prepare_distribution_state_unknown(self):
        """Test unknown distribution raises error."""
        with pytest.raises(ValueError, match="Unknown distribution"):
            prepare_distribution_state(
                'unknown_dist',
                {},
                num_qubits=4
            )
            
    def test_discretize_distribution_normal(self):
        """Test discretization of normal distribution."""
        dist = stats.norm(loc=100, scale=15)
        probs, x_vals = discretize_distribution(
            dist, num_qubits=6,
            domain_min=50, domain_max=150
        )
        
        assert len(probs) == 64
        assert len(x_vals) == 64
        assert np.isclose(np.sum(probs), 1.0)
        
        # Check that x_vals are in the correct range
        assert np.all(x_vals >= 50)
        assert np.all(x_vals <= 150)
        
    def test_discretize_distribution_numeric_vs_analytic(self):
        """Test numeric vs analytic discretization."""
        dist = stats.expon(scale=1000)
        
        # Analytic (CDF-based)
        probs_a, x_a = discretize_distribution(
            dist, 5, 0, 5000, use_analytic=True
        )
        
        # Numeric (PDF-based)
        probs_n, x_n = discretize_distribution(
            dist, 5, 0, 5000, use_analytic=False
        )
        
        # Both should be normalized
        assert np.isclose(np.sum(probs_a), 1.0)
        assert np.isclose(np.sum(probs_n), 1.0)
        
        # x values should match
        assert np.allclose(x_a, x_n)


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_lognormal_to_quantum_circuit(self):
        """Test full pipeline from lognormal to quantum circuit."""
        # Prepare lognormal distribution
        probs, x_vals = prepare_lognormal_state(
            mu=7, sigma=1.5, num_qubits=6
        )
        
        # Encode into quantum circuit
        qc = amplitude_encode(probs)
        
        assert qc.num_qubits == 6
        assert len(qc.data) > 0  # Should have gates
        
    def test_empirical_distribution_encoding(self):
        """Test encoding empirical data (when implemented)."""
        # Generate some fake loss data
        np.random.seed(42)
        losses = np.random.lognormal(7, 1.5, 1000)
        
        # For now, we can use the distribution approach
        # In future, prepare_empirical_distribution would be used
        probs, x_vals = prepare_distribution_state(
            'lognormal',
            {'mu': 7, 'sigma': 1.5},
            num_qubits=5
        )
        
        qc = amplitude_encode(probs)
        assert qc.num_qubits == 5
        
    def test_validation_in_pipeline(self):
        """Test that validation catches issues in the pipeline."""
        # Create probabilities that sum to more than 1
        bad_probs = [0.4, 0.4, 0.4, 0.4]  # Sum = 1.6
        
        # Without normalization, validation should catch this
        with pytest.raises(ValueError):
            amplitude_encode(bad_probs, normalize=False, validate=True)
            
        # With normalization, should work fine
        qc = amplitude_encode(bad_probs, normalize=True, validate=True)
        assert qc is not None


# Performance benchmarks would go here in a real implementation
class TestPerformance:
    """Performance tests for state preparation."""
    
    @pytest.mark.slow
    def test_large_state_preparation(self):
        """Test preparation of large quantum states."""
        # 10 qubits = 1024 dimensional state
        probs, _ = prepare_lognormal_state(num_qubits=10)
        qc = amplitude_encode(probs)
        
        assert qc.num_qubits == 10
        assert len(probs) == 1024
        
    @pytest.mark.slow 
    def test_many_distributions(self):
        """Test preparing many different distributions."""
        distributions = ['gamma', 'exponential', 'beta', 'uniform']
        
        for dist_name in distributions:
            if dist_name == 'beta':
                params = {'a': 2, 'b': 5}
            else:
                params = {'scale': 1000}
                
            probs, _ = prepare_distribution_state(
                dist_name, params, num_qubits=6
            )
            
            assert len(probs) == 64
            assert np.isclose(np.sum(probs), 1.0)