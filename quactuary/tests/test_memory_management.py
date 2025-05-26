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
        
        assert config.max_memory_gb is None  # Auto-detection
        assert config.safety_factor == 0.8
        assert config.min_batch_size == 100
        assert config.max_batch_size == 100000
        assert config.use_disk_cache is True
        assert config.temp_dir is None
    
    def test_memory_config_custom(self):
        """Test MemoryConfig with custom values."""
        config = MemoryConfig(
            max_memory_gb=8.0,
            safety_factor=0.5,
            min_batch_size=500,
            use_disk_cache=False
        )
        
        assert config.max_memory_gb == 8.0
        assert config.safety_factor == 0.5
        assert config.min_batch_size == 500
        assert config.use_disk_cache is False


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
        # Need to create the manager within the patched context
        config = MemoryConfig(max_memory_gb=4.0)
        with patch('quactuary.memory_management.psutil.Process') as mock_process:
            mock_process.return_value = Mock()
            return MemoryManager(config)
    
    def test_initialization(self):
        """Test MemoryManager initialization."""
        # Test with explicit max_memory_gb
        with patch('quactuary.memory_management.psutil') as mock_psutil:
            mock_psutil.Process.return_value = Mock()
            
            config = MemoryConfig(max_memory_gb=4.0)
            memory_manager = MemoryManager(config)
            
            assert memory_manager.config.max_memory_gb == 4.0
            assert hasattr(memory_manager, 'process')
            mock_psutil.Process.assert_called()
            
        # Test with auto-detection (max_memory_gb=None)
        with patch('quactuary.memory_management.psutil') as mock_psutil:
            # Mock virtual memory
            mock_memory = Mock()
            mock_memory.total = 16 * 1024 * 1024 * 1024  # 16 GB
            mock_psutil.virtual_memory.return_value = mock_memory
            mock_psutil.Process.return_value = Mock()
            
            config = MemoryConfig()  # max_memory_gb defaults to None
            memory_manager = MemoryManager(config)
            
            # Should auto-calculate based on total memory
            expected_max = 16 * 0.8  # 16GB * safety_factor (0.8)
            assert memory_manager.config.max_memory_gb == expected_max
            mock_psutil.virtual_memory.assert_called()
    
    def test_estimate_memory_usage(self, memory_manager):
        """Test memory usage estimation."""
        n_policies = 1000
        n_simulations = 10000
        
        estimate_gb = memory_manager.estimate_memory_usage(n_policies, n_simulations)
        
        # Check return value is a float (GB)
        assert isinstance(estimate_gb, float)
        assert estimate_gb > 0
        
        # Test with different bytes per element
        estimate_gb_float32 = memory_manager.estimate_memory_usage(
            n_policies, n_simulations, bytes_per_element=4
        )
        assert estimate_gb_float32 < estimate_gb  # Should be smaller with float32
    
    def test_calculate_optimal_batch_size(self, memory_manager, mock_psutil):
        """Test optimal batch size calculation."""
        # Mock available memory for consistent testing
        memory_manager.get_available_memory = Mock(return_value=8.0)
        memory_manager.get_used_memory = Mock(return_value=1.0)
        
        # Small portfolio
        batch_size = memory_manager.calculate_optimal_batch_size(
            n_policies=100,
            n_simulations=10000
        )
        assert batch_size >= memory_manager.config.min_batch_size
        assert batch_size <= memory_manager.config.max_batch_size
        assert batch_size <= 10000  # Can't exceed total simulations
        
        # Large portfolio with target memory
        batch_size_large = memory_manager.calculate_optimal_batch_size(
            n_policies=10000,
            n_simulations=100000,
            target_memory_gb=1.0  # 1GB limit
        )
        assert batch_size_large >= memory_manager.config.min_batch_size
        assert batch_size_large <= memory_manager.config.max_batch_size
    
    def test_get_available_memory(self, memory_manager, mock_psutil):
        """Test getting available memory."""
        available_gb = memory_manager.get_available_memory()
        assert available_gb == 8.0  # 8GB as set in mock
        
    def test_get_used_memory(self, memory_manager):
        """Test getting used memory."""
        # Mock the process memory info
        memory_manager.process.memory_info = Mock(
            return_value=Mock(rss=2 * 1024 * 1024 * 1024)  # 2GB
        )
        
        used_gb = memory_manager.get_used_memory()
        assert used_gb == 2.0
    
    @patch('quactuary.memory_management.tempfile.NamedTemporaryFile')
    @patch('quactuary.memory_management.np.memmap')
    def test_create_memory_map(self, mock_memmap, mock_tempfile, memory_manager):
        """Test memory-mapped array creation."""
        # Mock tempfile
        mock_file = Mock()
        mock_file.name = '/tmp/test.dat'
        mock_tempfile.return_value = mock_file
        
        # Mock memmap - use np.ndarray as spec instead of np.memmap
        mock_array = Mock(spec=np.ndarray)
        mock_array.shape = (1000, 100)
        mock_memmap.return_value = mock_array
        
        shape = (1000, 100)
        dtype = np.float64
        
        array = memory_manager.create_memory_map(shape, dtype)
        
        assert array == mock_array
        mock_memmap.assert_called_once_with(
            mock_file.name,
            dtype=dtype,
            mode='w+',
            shape=shape
        )
        assert hasattr(memory_manager, '_temp_files')
        assert mock_file.name in memory_manager._temp_files
    
    def test_optimize_gc(self, memory_manager):
        """Test garbage collection optimization."""
        with patch('gc.collect') as mock_collect:
            with patch('gc.disable') as mock_disable:
                memory_manager.optimize_gc()
                
                mock_collect.assert_called_once()
                mock_disable.assert_called_once()
    
    def test_restore_gc(self, memory_manager):
        """Test restoring garbage collection."""
        with patch('gc.enable') as mock_enable:
            with patch('gc.collect') as mock_collect:
                memory_manager.restore_gc()
                
                mock_enable.assert_called_once()
                mock_collect.assert_called_once()
    
    def test_cleanup_temp_files(self, memory_manager):
        """Test cleanup of temporary files."""
        # Create some temp files
        memory_manager._temp_files = ['/tmp/file1', '/tmp/file2']
        
        with patch('os.unlink') as mock_unlink:
            memory_manager.cleanup_temp_files()
            
            assert mock_unlink.call_count == 2
            assert memory_manager._temp_files == []


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
        memory_manager = MemoryManager(config)
        return StreamingSimulator(memory_manager)
    
    def test_initialization(self, streaming_simulator):
        """Test StreamingSimulator initialization."""
        assert isinstance(streaming_simulator.memory_manager, MemoryManager)
    
    def test_simulate_streaming(self, streaming_simulator, sample_portfolio):
        """Test streaming simulation."""
        n_simulations = 5000
        n_policies = len(sample_portfolio)
        
        # Define a simple simulation function
        def simulate_func(batch_size):
            return np.random.exponential(1000, batch_size)
        
        # Collect results from generator
        results = []
        for batch in streaming_simulator.simulate_streaming(
            simulate_func=simulate_func,
            n_simulations=n_simulations,
            n_policies=n_policies
        ):
            results.append(batch)
        
        # Check we got all results
        total_results = sum(len(batch) for batch in results)
        assert total_results == n_simulations
        
        # Check batches are numpy arrays
        for batch in results:
            assert isinstance(batch, np.ndarray)
            assert np.all(batch >= 0)  # Exponential dist is non-negative
    
    def test_calculate_streaming_statistics(self, streaming_simulator):
        """Test streaming statistics calculation."""
        # Create a data generator
        def data_generator():
            for _ in range(5):
                yield np.random.normal(1000, 200, 1000)
        
        stats = streaming_simulator.calculate_streaming_statistics(
            data_generator(),
            confidence_levels=[0.95, 0.99]
        )
        
        # Check statistics structure
        assert 'count' in stats
        assert 'mean' in stats
        assert 'std' in stats
        assert 'variance' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert 'var_95%' in stats
        assert 'var_99%' in stats
        
        # Check values make sense
        assert stats['count'] == 5000  # 5 batches of 1000
        assert 700 < stats['mean'] < 1300  # Should be around 1000
        assert stats['std'] > 0
        assert stats['min'] < stats['mean'] < stats['max']
    
    def test_simulate_streaming_with_output_file(self, streaming_simulator):
        """Test streaming simulation with HDF5 output."""
        # Skip if h5py not available
        try:
            import h5py
        except ImportError:
            pytest.skip("h5py not available")
            
        # Mock HDF5 file and dataset
        with patch('h5py.File') as mock_h5py_file:
            mock_file = Mock()
            mock_dataset = Mock()
            mock_h5py_file.return_value = mock_file
            mock_file.create_dataset.return_value = mock_dataset
        
            def simulate_func(batch_size):
                return np.ones(batch_size) * 100
            
            results = list(streaming_simulator.simulate_streaming(
                simulate_func=simulate_func,
                n_simulations=1000,
                n_policies=10,
                output_file='test.h5'
            ))
            
            # Check HDF5 was used
            mock_h5py_file.assert_called_once_with('test.h5', 'w')
            mock_file.create_dataset.assert_called_once()
            mock_file.close.assert_called_once()
            
            # Check results were yielded
            assert len(results) > 0
    
    def test_simulate_streaming_with_callback(self, streaming_simulator):
        """Test streaming simulation with progress callback."""
        progress_calls = []
        
        def progress_callback(completed, total):
            progress_calls.append((completed, total))
        
        def simulate_func(batch_size):
            return np.random.uniform(0, 1, batch_size)
        
        list(streaming_simulator.simulate_streaming(
            simulate_func=simulate_func,
            n_simulations=1000,
            n_policies=10,
            callback=progress_callback
        ))
        
        # Check callback was called
        assert len(progress_calls) > 0
        # Check final progress is complete
        assert progress_calls[-1][0] == 1000


class TestDemonstrateMemoryManagement:
    """Test the demonstrate_memory_management function."""
    
    @patch('builtins.print')
    def test_demonstration(self, mock_print):
        """Test the demonstration function."""
        # Run demonstration - it should not crash
        demonstrate_memory_management()
        
        # Verify output was printed
        assert mock_print.call_count > 0
        
        # Check that expected sections were printed
        printed_args = [call.args for call in mock_print.call_args_list if call.args]
        printed_text = ' '.join(str(arg[0]) for arg in printed_args if arg)
        assert 'MEMORY MANAGEMENT DEMONSTRATION' in printed_text
        assert 'System memory' in printed_text or 'Max allowed' in printed_text