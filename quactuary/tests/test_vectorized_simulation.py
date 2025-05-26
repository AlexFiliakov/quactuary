import pytest
import numpy as np
from unittest.mock import Mock, patch
from numpy.testing import assert_array_almost_equal, assert_allclose

from quactuary.vectorized_simulation import (
    VectorizedSimulator,
    benchmark_vectorization
)
from quactuary.book import PolicyTerms
from quactuary.distributions.frequency import Poisson
from quactuary.distributions.severity import Lognormal


class TestVectorizedSimulator:
    """Test the VectorizedSimulator class methods."""
    
    @pytest.fixture
    def sample_distributions(self):
        """Create sample frequency and severity distributions."""
        freq_dist = Poisson(mu=5.0)
        sev_dist = Lognormal(s=1.0, scale=np.exp(8.0))
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
        
        # Run vectorized simulation
        retained, ceded, total = VectorizedSimulator.simulate_inforce_vectorized(
            frequency_dist=freq_dist,
            severity_dist=sev_dist,
            policy_terms=sample_policy_terms,
            n_simulations=n_simulations,
            random_state=42
        )
        
        # Verify shapes
        assert retained.shape == (n_simulations,)
        assert ceded.shape == (n_simulations,)
        assert total.shape == (n_simulations,)
        
        # Verify relationships
        assert_array_almost_equal(retained + ceded, total)
        
        # Verify all values are non-negative
        assert np.all(retained >= 0)
        assert np.all(ceded >= 0)
        assert np.all(total >= 0)
        
        # Verify policy terms are applied
        assert np.all(retained >= 0)  # Can't have negative retained
        assert np.all(ceded <= sample_policy_terms.agg_limit)  # Ceded limited by agg limit
    
    def test_simulate_inforce_vectorized_v2(self, sample_distributions, sample_policy_terms):
        """Test the alternative vectorized simulation method."""
        freq_dist, sev_dist = sample_distributions
        n_simulations = 10000
        
        # Run vectorized simulation v2
        retained, ceded, total = VectorizedSimulator.simulate_inforce_vectorized_v2(
            frequency_dist=freq_dist,
            severity_dist=sev_dist,
            policy_terms=sample_policy_terms,
            n_simulations=n_simulations,
            random_state=42
        )
        
        # Verify shapes
        assert retained.shape == (n_simulations,)
        assert ceded.shape == (n_simulations,)
        assert total.shape == (n_simulations,)
        
        # Verify relationships
        assert_array_almost_equal(retained + ceded, total)
        
        # Verify all values are non-negative
        assert np.all(retained >= 0)
        assert np.all(ceded >= 0)
        assert np.all(total >= 0)
    
    def test_consistency_between_v1_and_v2(self, sample_distributions, sample_policy_terms):
        """Test that both vectorized methods produce similar results."""
        freq_dist, sev_dist = sample_distributions
        n_simulations = 5000
        
        # Run both versions with same random state
        retained_v1, ceded_v1, total_v1 = VectorizedSimulator.simulate_inforce_vectorized(
            frequency_dist=freq_dist,
            severity_dist=sev_dist,
            policy_terms=sample_policy_terms,
            n_simulations=n_simulations,
            random_state=42
        )
        
        retained_v2, ceded_v2, total_v2 = VectorizedSimulator.simulate_inforce_vectorized_v2(
            frequency_dist=freq_dist,
            severity_dist=sev_dist,
            policy_terms=sample_policy_terms,
            n_simulations=n_simulations,
            random_state=42
        )
        
        # Compare statistics rather than exact values
        assert_allclose(np.mean(retained_v1), np.mean(retained_v2), rtol=0.05)
        assert_allclose(np.mean(ceded_v1), np.mean(ceded_v2), rtol=0.05)
        assert_allclose(np.mean(total_v1), np.mean(total_v2), rtol=0.05)
        
        # Compare variance
        assert_allclose(np.var(retained_v1), np.var(retained_v2), rtol=0.1)
        assert_allclose(np.var(ceded_v1), np.var(ceded_v2), rtol=0.1)
    
    def test_apply_policy_terms_vectorized(self, sample_policy_terms):
        """Test vectorized policy term application."""
        # Create sample losses
        frequencies = np.array([3, 0, 5, 2, 1])
        severity_samples = [
            np.array([1000, 2000, 3000]),  # 3 losses
            np.array([]),  # 0 losses
            np.array([500, 1500, 2500, 3500, 4500]),  # 5 losses
            np.array([10000, 20000]),  # 2 losses
            np.array([50000])  # 1 loss
        ]
        
        retained_list = []
        ceded_list = []
        total_list = []
        
        for freq, severities in zip(frequencies, severity_samples):
            if freq > 0:
                retained, ceded, total = VectorizedSimulator.apply_policy_terms_vectorized(
                    severities, sample_policy_terms
                )
                retained_list.append(retained)
                ceded_list.append(ceded)
                total_list.append(total)
            else:
                retained_list.append(0.0)
                ceded_list.append(0.0)
                total_list.append(0.0)
        
        retained = np.array(retained_list)
        ceded = np.array(ceded_list)
        total = np.array(total_list)
        
        # Verify shapes
        assert retained.shape == (5,)
        assert ceded.shape == (5,)
        assert total.shape == (5,)
        
        # Verify relationships
        assert_array_almost_equal(retained + ceded, total)
        
        # Check specific cases
        assert retained[1] == 0  # No losses
        assert ceded[1] == 0
        assert total[1] == 0
    
    def test_calculate_statistics_vectorized(self):
        """Test vectorized statistics calculation."""
        # Create sample simulation results
        simulations = np.random.lognormal(8, 1, size=10000)
        
        stats = VectorizedSimulator.calculate_statistics_vectorized(
            simulations,
            var_percentile=0.95,
            tvar_percentile=0.95
        )
        
        # Verify all statistics are present
        assert 'mean' in stats
        assert 'std' in stats
        assert 'var_95' in stats
        assert 'tvar_95' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert 'percentiles' in stats
        
        # Verify statistics make sense
        assert stats['mean'] > 0
        assert stats['std'] > 0
        assert stats['var_95'] > stats['mean']
        assert stats['tvar_95'] > stats['var_95']
        assert stats['min'] <= stats['mean'] <= stats['max']
        
        # Verify percentiles
        assert len(stats['percentiles']) > 0
        for p, v in stats['percentiles'].items():
            assert 0 <= p <= 100
            assert stats['min'] <= v <= stats['max']
    
    def test_zero_frequency_handling(self, sample_distributions, sample_policy_terms):
        """Test handling of zero frequency scenarios."""
        # Create distribution with very low lambda to get zeros
        freq_dist = PoissonDistribution(lambda_=0.01)
        _, sev_dist = sample_distributions
        
        retained, ceded, total = VectorizedSimulator.simulate_inforce_vectorized(
            frequency_dist=freq_dist,
            severity_dist=sev_dist,
            policy_terms=sample_policy_terms,
            n_simulations=1000,
            random_state=42
        )
        
        # Should have many zeros
        assert np.sum(total == 0) > 900  # Most should be zero with lambda=0.01
        
        # Non-zero values should still be valid
        non_zero_mask = total > 0
        if np.any(non_zero_mask):
            assert np.all(retained[non_zero_mask] >= 0)
            assert np.all(ceded[non_zero_mask] >= 0)
    
    def test_extreme_values(self, sample_policy_terms):
        """Test handling of extreme severity values."""
        freq_dist = PoissonDistribution(lambda_=2.0)
        # High severity distribution
        sev_dist = LogNormalDistribution(mu=12.0, sigma=2.0)
        
        retained, ceded, total = VectorizedSimulator.simulate_inforce_vectorized(
            frequency_dist=freq_dist,
            severity_dist=sev_dist,
            policy_terms=sample_policy_terms,
            n_simulations=1000,
            random_state=42
        )
        
        # Should handle large losses appropriately
        assert np.all(np.isfinite(retained))
        assert np.all(np.isfinite(ceded))
        assert np.all(np.isfinite(total))
        
        # Aggregate limit should be respected
        assert np.all(ceded <= sample_policy_terms.agg_limit)


class TestBenchmarkVectorization:
    """Test the benchmark_vectorization function."""
    
    @patch('quactuary.vectorized_simulation.PerformanceBenchmark')
    def test_benchmark_execution(self, mock_benchmark_class):
        """Test that benchmark function executes correctly."""
        mock_instance = Mock()
        mock_benchmark_class.return_value = mock_instance
        
        # Mock the portfolio creation
        mock_portfolio = [Mock() for _ in range(10)]
        mock_instance._create_test_portfolio.return_value = mock_portfolio
        
        # Run benchmark
        benchmark_vectorization(portfolio_size=10, n_simulations=1000)
        
        # Verify methods were called
        mock_instance._create_test_portfolio.assert_called_once_with(10)
        assert mock_instance.measure_performance.call_count >= 2  # At least v1 and v2
        mock_instance.generate_report.assert_called_once()
    
    @patch('quactuary.vectorized_simulation.PerformanceBenchmark')
    @patch('builtins.print')
    def test_benchmark_output(self, mock_print, mock_benchmark_class):
        """Test that benchmark prints results."""
        mock_instance = Mock()
        mock_instance.generate_report.return_value = "Vectorization Benchmark Report"
        mock_benchmark_class.return_value = mock_instance
        
        benchmark_vectorization()
        
        # Verify report was printed
        mock_print.assert_called_with("Vectorization Benchmark Report")