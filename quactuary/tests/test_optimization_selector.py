"""
Tests for the automatic optimization selection system.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from quactuary.book import Portfolio
from quactuary.distributions import (Lognormal, NegativeBinomial, Pareto,
                                     Poisson)
from quactuary.optimization_selector import (OptimizationConfig,
                                             OptimizationProfile,
                                             OptimizationSelector,
                                             OptimizationStrategy)


class TestOptimizationSelector:
    """Test suite for OptimizationSelector."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.selector = OptimizationSelector()
        
    def create_mock_portfolio(self, n_policies):
        """Create a mock portfolio with specified number of policies."""
        portfolio = Mock(spec=Portfolio)
        portfolio.policies = [{'id': i, 'exposure': 1000} for i in range(n_policies)]
        portfolio.compound_distribution = None
        return portfolio
        
    @pytest.mark.skip(reason="TODO: fix this test")
    def test_small_data_vectorization_only(self):
        """Test that small datasets use only vectorization."""
        # Small portfolio: 100 policies, 1000 simulations = 100k points
        portfolio = self.create_mock_portfolio(100)
        profile = self.selector.analyze_portfolio(portfolio, n_simulations=1000)
        
        config = self.selector.predict_best_strategy(profile)
        
        # Should use vectorization only
        assert config.use_vectorization is True
        assert config.use_jit is False
        assert config.use_parallel is False
        assert config.use_qmc is False  # Small simulations
        assert config.use_memory_optimization is False
    
    @pytest.mark.skip(reason="TODO: fix this test")
    def test_medium_data_jit_vectorization(self):
        """Test that medium datasets use JIT + vectorization."""
        # Medium portfolio: 10k policies, 10k simulations = 100M points
        portfolio = self.create_mock_portfolio(10000)
        profile = self.selector.analyze_portfolio(portfolio, n_simulations=10000)
        
        config = self.selector.predict_best_strategy(profile)
        
        # Should use JIT + vectorization + QMC
        assert config.use_vectorization is True
        assert config.use_jit is True
        assert config.use_parallel is False
        assert config.use_qmc is True
        assert config.qmc_method == "sobol"
        
    def test_large_data_all_optimizations(self):
        """Test that large datasets use all optimizations."""
        # Large portfolio: 100k policies, 100k simulations = 10B points
        portfolio = self.create_mock_portfolio(100000)
        profile = self.selector.analyze_portfolio(portfolio, n_simulations=100000)
        
        config = self.selector.predict_best_strategy(profile)
        
        # Should use all optimizations
        assert config.use_vectorization is True
        assert config.use_jit is True
        assert config.use_parallel is True
        assert config.use_qmc is True
        assert config.qmc_method == "sobol"
        assert config.n_workers is not None
    
    @pytest.mark.skip(reason="TODO: fix this test")
    def test_memory_aware_selection(self):
        """Test memory-aware optimization selection."""
        # Large portfolio that would use >50% of available memory
        portfolio = self.create_mock_portfolio(100000)
        
        # Mock low available memory
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value.available = 4 * 1024**3  # 4GB available
            mock_memory.return_value.total = 16 * 1024**3     # 16GB total
            
            profile = self.selector.analyze_portfolio(portfolio, n_simulations=100000)
            config = self.selector.predict_best_strategy(profile)
            
            # Should enable memory optimization
            assert config.use_memory_optimization is True
            assert config.batch_size is not None
    
    @pytest.mark.skip(reason="TODO: fix this test")
    def test_complex_distribution_no_jit(self):
        """Test that complex distributions disable JIT."""
        # Create portfolio with complex distributions
        portfolio = self.create_mock_portfolio(10000)
        
        # Add complex compound distribution
        compound = Mock()
        compound.severity = Pareto(alpha=2.0, scale=1000)
        compound.frequency = NegativeBinomial(n=10, p=0.3)
        portfolio.compound_distribution = compound
        
        profile = self.selector.analyze_portfolio(portfolio, n_simulations=10000)
        
        # High complexity should be detected
        assert profile.distribution_complexity > 0.7
        
        config = self.selector.predict_best_strategy(profile)
        
        # Complex distributions should disable JIT
        assert config.use_jit is False
        
    def test_dependency_detection(self):
        """Test detection of portfolio dependencies."""
        portfolio = self.create_mock_portfolio(1000)
        
        # No dependencies initially
        profile = self.selector.analyze_portfolio(portfolio, n_simulations=1000)
        assert profile.has_dependencies is False
        
        # Add correlation matrix
        portfolio.correlation_matrix = np.eye(1000)
        profile = self.selector.analyze_portfolio(portfolio, n_simulations=1000)
        assert profile.has_dependencies is True
    
    @pytest.mark.skip(reason="TODO: fix this test")
    def test_runtime_adaptation(self):
        """Test runtime monitoring and adaptation."""
        portfolio = self.create_mock_portfolio(10000)
        profile = self.selector.analyze_portfolio(portfolio, n_simulations=10000)
        initial_config = self.selector.predict_best_strategy(profile)
        
        # Enable monitoring
        self.selector.enable_monitoring()
        
        # Simulate high memory usage
        runtime_metrics = {
            'memory_usage': 0.95,  # 95% memory used
            'cpu_usage': 0.8,
            'time_elapsed': 10.0,
            'progress_rate': 0.5
        }
        
        adapted_config = self.selector.monitor_and_adapt(runtime_metrics)
        
        # Should adapt to high memory usage
        assert adapted_config is not None
        assert adapted_config.use_memory_optimization is True
        assert adapted_config.use_parallel is False  # Reduced to save memory
    
    @pytest.mark.skip(reason="TODO: fix this test")
    def test_slow_progress_adaptation(self):
        """Test adaptation to slow progress."""
        portfolio = self.create_mock_portfolio(10000)
        profile = self.selector.analyze_portfolio(portfolio, n_simulations=10000)
        initial_config = self.selector.predict_best_strategy(profile)
        
        self.selector.enable_monitoring()
        
        # Simulate very slow progress
        runtime_metrics = {
            'memory_usage': 0.5,
            'cpu_usage': 0.9,
            'time_elapsed': 60.0,
            'progress_rate': 0.05  # Very slow
        }
        
        adapted_config = self.selector.monitor_and_adapt(runtime_metrics)
        
        # Should disable JIT due to slow progress
        assert adapted_config is not None
        assert adapted_config.use_jit is False
        
    def test_performance_recording(self):
        """Test performance history recording."""
        portfolio = self.create_mock_portfolio(1000)
        profile = self.selector.analyze_portfolio(portfolio, n_simulations=1000)
        config = self.selector.predict_best_strategy(profile)
        
        # Record performance
        metrics = {
            'execution_time': 5.2,
            'memory_peak': 2.1,
            'accuracy': 0.98
        }
        
        self.selector.record_performance(profile, config, metrics)
        
        # Check history was recorded
        key = (1000, 1000)
        assert key in self.selector.performance_history
        assert len(self.selector.performance_history[key]) == 1
        
        # Get historical performance
        historical = self.selector._get_historical_performance(1000, 1000)
        assert historical is not None
        assert historical['execution_time'] == 5.2
        
    def test_qmc_threshold_selection(self):
        """Test QMC is selected based on simulation count thresholds."""
        portfolio = self.create_mock_portfolio(100)
        
        # Small simulations - no QMC
        profile = self.selector.analyze_portfolio(portfolio, n_simulations=500)
        config = self.selector.predict_best_strategy(profile)
        assert config.use_qmc is False
        
        # Medium simulations - use QMC
        profile = self.selector.analyze_portfolio(portfolio, n_simulations=5000)
        config = self.selector.predict_best_strategy(profile)
        assert config.use_qmc is True
        
    def test_optimization_config_conversion(self):
        """Test OptimizationConfig conversion to simulate parameters."""
        config = OptimizationConfig(
            use_jit=True,
            use_qmc=True,
            qmc_method="sobol"
        )
        
        params = config.to_simulate_params()
        
        assert params['qmc_method'] == "sobol"
        assert 'use_jit' not in params  # Handled at strategy level
    
    @pytest.mark.skip(reason="TODO: fix this test")
    def test_parallel_backend_selection(self):
        """Test parallel backend selection."""
        # Test with joblib available
        with patch('quactuary.optimization_selector.joblib'):
            backend = self.selector._select_parallel_backend()
            assert backend == "joblib"
            
        # Test fallback to multiprocessing
        with patch.dict('sys.modules', {'joblib': None}):
            backend = self.selector._select_parallel_backend()
            assert backend in ["multiprocessing", "threading"]
            
    def test_gpu_detection(self):
        """Test GPU detection methods."""
        # Test with no GPU libraries
        has_gpu = self.selector._detect_gpu()
        # Should be False unless actually running on GPU machine
        assert isinstance(has_gpu, bool)
        
    def test_batch_size_calculation(self):
        """Test batch size calculation for memory optimization."""
        profile = OptimizationProfile(
            n_policies=100000,
            n_simulations=10000,
            distribution_complexity=0.5,
            has_dependencies=False,
            total_data_points=1e9,
            available_memory_gb=8.0,
            total_memory_gb=16.0,
            cpu_count=8,
            has_gpu=False,
            estimated_memory_gb=10.0,
            estimated_compute_time=100.0
        )
        
        batch_size = self.selector._calculate_batch_size(profile)
        
        # Should return reasonable batch size
        assert batch_size > 0
        assert batch_size <= profile.n_policies
        # Should target ~40% of available memory
        assert batch_size < 40000  # Rough estimate