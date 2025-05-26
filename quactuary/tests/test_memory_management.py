import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from dataclasses import asdict

from quactuary.memory_management import (
    MemoryConfig,
    MemoryManager,
    StreamingSimulator,
    demonstrate_memory_management
)
from quactuary.book import PolicyTerms
from quactuary.distributions.frequency import Poisson
from quactuary.distributions.severity import Lognormal


class TestMemoryConfig:
    """Test the MemoryConfig dataclass."""
    
    def test_memory_config_defaults(self):
        """Test MemoryConfig default values."""
        config = MemoryConfig()
        
        assert config.max_memory_gb == 4.0
        assert config.target_memory_usage == 0.75
        assert config.min_batch_size == 1000
        assert config.max_batch_size == 100000
        assert config.enable_memory_mapping is False
        assert config.enable_gc_optimization is True
        assert config.compression_level == 0
    
    def test_memory_config_custom(self):
        """Test MemoryConfig with custom values."""
        config = MemoryConfig(
            max_memory_gb=8.0,
            target_memory_usage=0.5,
            min_batch_size=500,
            enable_memory_mapping=True
        )
        
        assert config.max_memory_gb == 8.0
        assert config.target_memory_usage == 0.5
        assert config.min_batch_size == 500
        assert config.enable_memory_mapping is True


class TestMemoryManager:
    """Test the MemoryManager class."""
    
    @pytest.fixture
    def mock_psutil(self):
        """Mock psutil for consistent memory measurements."""
        with patch('quactuary.memory_management.psutil') as mock:
            mock_memory = Mock()
            mock_memory.total = 16 * 1024 * 1024 * 1024  # 16 GB
            mock_memory.available = 8 * 1024 * 1024 * 1024  # 8 GB available
            mock.virtual_memory.return_value = mock_memory
            yield mock
    
    @pytest.fixture
    def memory_manager(self, mock_psutil):
        """Create a MemoryManager instance."""
        config = MemoryConfig(max_memory_gb=4.0)
        return MemoryManager(config)
    
    def test_initialization(self, memory_manager, mock_psutil):
        """Test MemoryManager initialization."""
        assert memory_manager.config.max_memory_gb == 4.0
        assert memory_manager.memory_pool == {}
        mock_psutil.virtual_memory.assert_called()
    
    def test_estimate_memory_usage(self, memory_manager):
        """Test memory usage estimation."""
        portfolio_size = 1000
        n_simulations = 10000
        
        estimate = memory_manager.estimate_memory_usage(portfolio_size, n_simulations)
        
        # Check structure
        assert 'portfolio_memory_mb' in estimate
        assert 'simulation_memory_mb' in estimate
        assert 'total_memory_mb' in estimate
        assert 'can_fit_in_memory' in estimate
        assert 'recommended_batch_size' in estimate
        
        # Check values make sense
        assert estimate['portfolio_memory_mb'] > 0
        assert estimate['simulation_memory_mb'] > 0
        assert estimate['total_memory_mb'] == (
            estimate['portfolio_memory_mb'] + estimate['simulation_memory_mb']
        )
        assert isinstance(estimate['can_fit_in_memory'], bool)
        assert estimate['recommended_batch_size'] > 0
    
    def test_calculate_optimal_batch_size(self, memory_manager):
        """Test optimal batch size calculation."""
        # Small portfolio
        batch_size = memory_manager.calculate_optimal_batch_size(
            portfolio_size=100,
            target_memory_mb=1000
        )
        assert batch_size >= memory_manager.config.min_batch_size
        assert batch_size <= memory_manager.config.max_batch_size
        
        # Large portfolio
        batch_size_large = memory_manager.calculate_optimal_batch_size(
            portfolio_size=10000,
            target_memory_mb=100
        )
        assert batch_size_large >= memory_manager.config.min_batch_size
        assert batch_size_large < batch_size  # Should be smaller for larger portfolio
    
    def test_allocate_memory_array(self, memory_manager):
        """Test memory array allocation."""
        shape = (1000, 100)
        dtype = np.float64
        
        array = memory_manager.allocate_memory_array(shape, dtype)
        
        assert array.shape == shape
        assert array.dtype == dtype
        assert len(memory_manager.memory_pool) == 1
        
        # Test reuse
        array2 = memory_manager.allocate_memory_array(shape, dtype)
        assert len(memory_manager.memory_pool) == 1  # Should reuse existing
    
    @patch('quactuary.memory_management.tempfile.NamedTemporaryFile')
    @patch('quactuary.memory_management.np.memmap')
    def test_create_memory_mapped_array(self, mock_memmap, mock_tempfile, memory_manager):
        """Test memory-mapped array creation."""
        # Enable memory mapping
        memory_manager.config.enable_memory_mapping = True
        
        # Mock tempfile
        mock_file = Mock()
        mock_file.name = '/tmp/test.dat'
        mock_tempfile.return_value.__enter__.return_value = mock_file
        
        # Mock memmap
        mock_array = Mock(spec=np.ndarray)
        mock_array.shape = (1000, 100)
        mock_memmap.return_value = mock_array
        
        shape = (1000, 100)
        dtype = np.float64
        
        array = memory_manager.create_memory_mapped_array(shape, dtype)
        
        assert array == mock_array
        mock_memmap.assert_called_once_with(
            mock_file.name,
            dtype=dtype,
            mode='w+',
            shape=shape
        )
    
    def test_optimize_gc(self, memory_manager):
        """Test garbage collection optimization."""
        with patch('gc.collect') as mock_collect:
            with patch('gc.set_threshold') as mock_threshold:
                memory_manager.optimize_gc()
                
                if memory_manager.config.enable_gc_optimization:
                    mock_collect.assert_called_once()
                    mock_threshold.assert_called_once()
    
    def test_clear_memory_pool(self, memory_manager):
        """Test clearing memory pool."""
        # Add some arrays to pool
        memory_manager.allocate_memory_array((100, 100), np.float64)
        memory_manager.allocate_memory_array((200, 200), np.float32)
        
        assert len(memory_manager.memory_pool) == 2
        
        with patch('gc.collect') as mock_collect:
            memory_manager.clear_memory_pool()
            
            assert len(memory_manager.memory_pool) == 0
            mock_collect.assert_called_once()


class TestStreamingSimulator:
    """Test the StreamingSimulator class."""
    
    @pytest.fixture
    def sample_portfolio(self):
        """Create a sample portfolio for testing."""
        from datetime import date
        portfolio = []
        for i in range(10):
            policy = Mock()
            policy.policy_id = f"P{i:03d}"
            policy.frequency_dist = Poisson(mu=5.0)
            policy.severity_dist = Lognormal(shape=1.0, scale=np.exp(8.0))
            policy.policy_terms = PolicyTerms(
                effective_date=date(2024, 1, 1),
                expiration_date=date(2024, 12, 31),
                per_occ_retention=1000.0,
                per_occ_limit=100000.0
            )
            portfolio.append(policy)
        return portfolio
    
    @pytest.fixture
    def streaming_simulator(self):
        """Create a StreamingSimulator instance."""
        config = MemoryConfig(max_memory_gb=1.0)
        return StreamingSimulator(config)
    
    def test_initialization(self, streaming_simulator):
        """Test StreamingSimulator initialization."""
        assert isinstance(streaming_simulator.memory_manager, MemoryManager)
        assert streaming_simulator.online_stats == {}
    
    def test_simulate_streaming(self, streaming_simulator, sample_portfolio):
        """Test streaming simulation."""
        n_simulations = 5000
        
        results = streaming_simulator.simulate_streaming(
            portfolio=sample_portfolio,
            n_simulations=n_simulations,
            batch_size=1000
        )
        
        # Check results structure
        assert 'statistics' in results
        assert 'memory_usage' in results
        assert 'n_batches' in results
        
        # Check statistics
        stats = results['statistics']
        assert 'mean' in stats
        assert 'variance' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert 'n' in stats
        
        # Verify counts
        assert stats['n'] == n_simulations
        assert results['n_batches'] == 5  # 5000 / 1000
        
        # Check values make sense
        assert stats['mean'] > 0
        assert stats['variance'] > 0
        assert stats['std'] > 0
        assert stats['min'] >= 0
        assert stats['max'] > stats['mean']
    
    def test_simulate_batch(self, streaming_simulator, sample_portfolio):
        """Test batch simulation."""
        batch_results = streaming_simulator._simulate_batch(
            portfolio=sample_portfolio,
            n_simulations=100,
            random_state=42
        )
        
        assert batch_results.shape == (100,)
        assert np.all(batch_results >= 0)
        assert np.all(np.isfinite(batch_results))
    
    def test_update_online_statistics(self, streaming_simulator):
        """Test online statistics updates."""
        policy_id = "P001"
        
        # First batch
        batch1 = np.array([100, 200, 300, 400, 500])
        streaming_simulator._update_online_statistics(policy_id, batch1)
        
        stats1 = streaming_simulator.online_stats[policy_id]
        assert stats1['n'] == 5
        assert stats1['mean'] == 300
        assert stats1['min'] == 100
        assert stats1['max'] == 500
        
        # Second batch
        batch2 = np.array([600, 700, 800, 900, 1000])
        streaming_simulator._update_online_statistics(policy_id, batch2)
        
        stats2 = streaming_simulator.online_stats[policy_id]
        assert stats2['n'] == 10
        assert stats2['mean'] == 550  # Average of all 10 values
        assert stats2['min'] == 100
        assert stats2['max'] == 1000
    
    def test_finalize_statistics(self, streaming_simulator):
        """Test statistics finalization."""
        # Add some statistics
        streaming_simulator.online_stats = {
            'aggregate': {
                'n': 1000,
                'mean': 50000,
                'variance': 1e8,
                'min': 0,
                'max': 200000
            }
        }
        
        final_stats = streaming_simulator._finalize_statistics()
        
        assert final_stats['n'] == 1000
        assert final_stats['mean'] == 50000
        assert final_stats['variance'] == 1e8
        assert final_stats['std'] == 10000  # sqrt(1e8)
        assert final_stats['min'] == 0
        assert final_stats['max'] == 200000
    
    def test_empty_portfolio(self, streaming_simulator):
        """Test handling of empty portfolio."""
        results = streaming_simulator.simulate_streaming(
            portfolio=[],
            n_simulations=1000,
            batch_size=100
        )
        
        assert results['statistics']['n'] == 0
        assert results['n_batches'] == 0


class TestDemonstrateMemoryManagement:
    """Test the demonstrate_memory_management function."""
    
    @patch('quactuary.memory_management.MemoryManager')
    @patch('quactuary.memory_management.StreamingSimulator')
    @patch('builtins.print')
    def test_demonstration(self, mock_print, mock_streaming, mock_manager):
        """Test the demonstration function."""
        # Mock memory manager
        mock_manager_instance = Mock()
        mock_manager_instance.estimate_memory_usage.return_value = {
            'total_memory_mb': 1000,
            'can_fit_in_memory': False,
            'recommended_batch_size': 5000
        }
        mock_manager.return_value = mock_manager_instance
        
        # Mock streaming simulator
        mock_streaming_instance = Mock()
        mock_streaming_instance.simulate_streaming.return_value = {
            'statistics': {
                'mean': 50000,
                'std': 10000,
                'n': 100000
            },
            'memory_usage': {'peak_mb': 800},
            'n_batches': 20
        }
        mock_streaming.return_value = mock_streaming_instance
        
        # Run demonstration
        demonstrate_memory_management(
            portfolio_size=1000,
            n_simulations=100000
        )
        
        # Verify components were used
        mock_manager_instance.estimate_memory_usage.assert_called_once()
        mock_streaming_instance.simulate_streaming.assert_called_once()
        
        # Verify output was printed
        assert mock_print.call_count > 0