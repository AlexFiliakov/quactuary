import pytest
import numpy as np
from unittest.mock import Mock, patch
from numpy.testing import assert_array_almost_equal, assert_allclose

from quactuary.vectorized_simulation import (
    VectorizedSimulator,
    benchmark_vectorization
)
from quactuary.book import PolicyTerms, Inforce
from quactuary.distributions.frequency import Poisson
from quactuary.distributions.severity import Lognormal


class TestVectorizedSimulator:
    """Test the VectorizedSimulator class methods."""
    
    @pytest.fixture
    def sample_distributions(self):
        """Create sample frequency and severity distributions."""
        freq_dist = Poisson(mu=5.0)
        sev_dist = Lognormal(shape=1.0, scale=np.exp(8.0))
        return freq_dist, sev_dist
    
    @pytest.fixture
    def sample_policy_terms(self):
        """Create sample policy terms."""
        from datetime import date
        return PolicyTerms(
            effective_date=date(2024, 1, 1),
            expiration_date=date(2024, 12, 31),
            per_occ_retention=1000.0,
            agg_retention=5000.0,
            per_occ_limit=100000.0,
            agg_limit=500000.0
        )
    
    def test_simulate_inforce_vectorized(self, sample_distributions, sample_policy_terms):
        """Test the main vectorized simulation method."""
        freq_dist, sev_dist = sample_distributions
        n_simulations = 10000
        
        # Create an Inforce object
        inforce = Inforce(
            n_policies=100,
            terms=sample_policy_terms,
            frequency=freq_dist,
            severity=sev_dist,
            name="Test"
        )
        
        # Run vectorized simulation
        results = VectorizedSimulator.simulate_inforce_vectorized(
            inforce=inforce,
            n_sims=n_simulations
        )
        
        # Verify shape
        assert results.shape == (n_simulations,)
        
        # Verify all values are non-negative
        assert np.all(results >= 0)
        
        # Check basic statistics
        assert np.mean(results) > 0  # Should have some positive losses
        assert np.std(results) > 0   # Should have variability
    
    def test_simulate_inforce_vectorized_v2(self, sample_distributions, sample_policy_terms):
        """Test the alternative vectorized simulation method."""
        freq_dist, sev_dist = sample_distributions
        n_simulations = 10000
        
        # Create an Inforce object
        inforce = Inforce(
            n_policies=100,
            terms=sample_policy_terms,
            frequency=freq_dist,
            severity=sev_dist,
            name="Test"
        )
        
        # Run vectorized simulation v2
        results = VectorizedSimulator.simulate_inforce_vectorized_v2(
            inforce=inforce,
            n_sims=n_simulations
        )
        
        # Verify shape
        assert results.shape == (n_simulations,)
        
        # Verify all values are non-negative
        assert np.all(results >= 0)
        
        # Check basic statistics
        assert np.mean(results) > 0  # Should have some positive losses
        assert np.std(results) > 0   # Should have variability
    
    def test_consistency_between_v1_and_v2(self, sample_distributions, sample_policy_terms):
        """Test that both vectorized methods produce similar results."""
        freq_dist, sev_dist = sample_distributions
        n_simulations = 5000
        
        # Create an Inforce object
        inforce = Inforce(
            n_policies=100,
            terms=sample_policy_terms,
            frequency=freq_dist,
            severity=sev_dist,
            name="Test"
        )
        
        # Run both methods with same random seed
        np.random.seed(42)
        results_v1 = VectorizedSimulator.simulate_inforce_vectorized(
            inforce=inforce,
            n_sims=n_simulations
        )
        
        np.random.seed(42)
        results_v2 = VectorizedSimulator.simulate_inforce_vectorized_v2(
            inforce=inforce,
            n_sims=n_simulations
        )
        
        # Compare statistics rather than exact values
        # Mean should be within 5%
        mean_v1 = np.mean(results_v1)
        mean_v2 = np.mean(results_v2)
        assert abs(mean_v1 - mean_v2) / mean_v1 < 0.05
        
        # Standard deviation should be within 10%
        std_v1 = np.std(results_v1)
        std_v2 = np.std(results_v2)
        assert abs(std_v1 - std_v2) / std_v1 < 0.10
    
    def test_apply_policy_terms_vectorized(self, sample_policy_terms):
        """Test vectorized policy term application."""
        # Create test losses
        ground_up_losses = np.array([1000, 2000, 3000, 5000, 10000])
        
        # Apply simple deductible
        result = VectorizedSimulator.apply_policy_terms_vectorized(
            ground_up_losses,
            deductible=sample_policy_terms.per_occ_retention  # 1000
        )
        expected = np.array([0, 1000, 2000, 4000, 9000])
        assert_array_almost_equal(result, expected)
        
        # Apply deductible and limit
        large_losses = np.array([100000, 200000, 300000])
        result = VectorizedSimulator.apply_policy_terms_vectorized(
            large_losses,
            deductible=sample_policy_terms.per_occ_retention,  # 1000
            limit=sample_policy_terms.per_occ_limit  # 100000
        )
        # After deductible: [99000, 199000, 299000]
        # After limit: [99000, 100000, 100000]
        expected = np.array([99000, 100000, 100000])
        assert_array_almost_equal(result, expected)
        
        # Test with all zeros
        zero_losses = np.zeros(10)
        result = VectorizedSimulator.apply_policy_terms_vectorized(
            zero_losses,
            deductible=1000
        )
        assert np.all(result == 0)
        
        # Test with attachment and coinsurance
        losses = np.array([10000, 20000, 30000])
        result = VectorizedSimulator.apply_policy_terms_vectorized(
            losses,
            deductible=1000,
            attachment=5000,
            coinsurance=0.8
        )
        # After deductible: [9000, 19000, 29000]
        # After attachment: [4000, 14000, 24000]  
        # After coinsurance: [3200, 11200, 19200]
        expected = np.array([3200, 11200, 19200])
        assert_array_almost_equal(result, expected)
    
    def test_calculate_statistics_vectorized(self):
        """Test vectorized statistics calculation."""
        # Create sample simulation results
        np.random.seed(42)
        simulations = np.random.lognormal(8, 1, size=10000)
        
        stats = VectorizedSimulator.calculate_statistics_vectorized(
            simulations,
            confidence_levels=[0.90, 0.95, 0.99]
        )
        
        # Verify all statistics are present
        assert 'mean' in stats
        assert 'std' in stats
        assert 'variance' in stats
        assert 'var_90%' in stats
        assert 'var_95%' in stats
        assert 'var_99%' in stats
        assert 'tvar_90%' in stats
        assert 'tvar_95%' in stats
        assert 'tvar_99%' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert 'count' in stats
        
        # Verify statistics make sense
        assert stats['mean'] > 0
        assert stats['std'] > 0
        assert stats['var_95%'] > stats['mean']
        assert stats['tvar_95%'] > stats['var_95%']
        assert stats['min'] <= stats['mean'] <= stats['max']
        assert stats['count'] == 10000
        
        # Verify VaR ordering
        assert stats['var_90%'] <= stats['var_95%'] <= stats['var_99%']
        assert stats['tvar_90%'] <= stats['tvar_95%'] <= stats['tvar_99%']
    
    def test_zero_frequency_handling(self, sample_distributions, sample_policy_terms):
        """Test handling of zero frequency scenarios."""
        # Create distribution with very low lambda to get zeros
        freq_dist = Poisson(mu=0.01)
        _, sev_dist = sample_distributions
        
        # Create an Inforce object with just 1 policy to get many zeros
        inforce = Inforce(
            n_policies=1,
            terms=sample_policy_terms,
            frequency=freq_dist,
            severity=sev_dist,
            name="Test"
        )
        
        np.random.seed(42)
        results = VectorizedSimulator.simulate_inforce_vectorized(
            inforce=inforce,
            n_sims=1000
        )
        
        # With 1 policy and mu=0.01, should have ~990 zeros
        n_zeros = np.sum(results == 0)
        assert n_zeros > 950  # Allow some variance
        
        # Non-zero values should still be valid
        non_zero_mask = results > 0
        if np.any(non_zero_mask):
            assert np.all(results[non_zero_mask] > 0)
            # Should be relatively few non-zeros
            assert np.sum(non_zero_mask) < 50
    
    def test_extreme_values(self, sample_policy_terms):
        """Test handling of extreme severity values."""
        freq_dist = Poisson(mu=2.0)
        # High severity distribution
        sev_dist = Lognormal(shape=2.0, scale=np.exp(12.0))
        
        # Create an Inforce object
        inforce = Inforce(
            n_policies=10,  # Fewer policies to reduce computation
            terms=sample_policy_terms,
            frequency=freq_dist,
            severity=sev_dist,
            name="Test"
        )
        
        np.random.seed(42)
        results = VectorizedSimulator.simulate_inforce_vectorized(
            inforce=inforce,
            n_sims=1000
        )
        
        # Should handle large losses appropriately
        assert np.all(np.isfinite(results))
        assert np.all(results >= 0)
        
        # Should have some non-zero values with mu=2.0
        assert np.sum(results > 0) > 0
        
        # Check that extreme values are handled
        assert np.max(results) < np.inf


class TestBenchmarkVectorization:
    """Test the benchmark_vectorization function."""
    
    @patch('builtins.print')
    def test_benchmark_execution(self, mock_print):
        """Test that benchmark function executes correctly."""
        # Run benchmark - it should not crash
        benchmark_vectorization()
        
        # Verify output was printed
        assert mock_print.call_count > 0
        
        # Check that expected sections were printed
        printed_args = [call.args for call in mock_print.call_args_list if call.args]
        printed_text = ' '.join(str(arg[0]) for arg in printed_args if arg)
        assert 'VECTORIZATION BENCHMARK' in printed_text
        assert 'Standard:' in printed_text
        assert 'Vectorized v1:' in printed_text
        assert 'Vectorized v2:' in printed_text
        assert 'speedup:' in printed_text
    
    @patch('builtins.print')
    @patch('time.perf_counter')
    def test_benchmark_output(self, mock_time, mock_print):
        """Test that benchmark calculates speedup correctly."""
        # Mock timer to return predictable values
        # Standard: 0 -> 10 (10s)
        # Vectorized v1: 10 -> 12 (2s) 
        # Vectorized v2: 12 -> 13 (1s)
        mock_time.side_effect = [0, 10, 10, 12, 12, 13]
        
        benchmark_vectorization()
        
        # Check that speedup was calculated and printed
        printed_args = [call.args for call in mock_print.call_args_list if call.args]
        printed_text = ' '.join(str(arg[0]) for arg in printed_args if arg)
        
        # Should show speedup calculations
        assert '5.0x' in printed_text  # v1 speedup: 10/2 = 5x
        assert '10.0x' in printed_text  # v2 speedup: 10/1 = 10x