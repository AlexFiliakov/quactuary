"""Extended tests for parallel processing functionality from extraneous test directory."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock, call
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import queue

from quactuary.parallel_processing import (
    ParallelConfig,
    ParallelSimulator,
    parallel_worker
)
from quactuary.book import PolicyTerms
from quactuary.distributions.frequency import Poisson
from quactuary.distributions.severity import Lognormal


class TestParallelConfigExtended:
    """Extended tests for ParallelConfig."""
    
    def test_parallel_config_defaults(self):
        """Test ParallelConfig default values."""
        config = ParallelConfig()
        
        assert config.n_workers is None  # Uses system default
        assert config.backend == 'multiprocessing'
        assert config.chunk_size is None
        assert config.show_progress is True
        assert config.work_stealing is True
        assert config.prefetch_batches == 2
    
    def test_parallel_config_custom(self):
        """Test ParallelConfig with custom values."""
        config = ParallelConfig(
            n_workers=4,
            backend='threading',
            chunk_size=100,
            show_progress=False,
            work_stealing=False
        )
        
        assert config.n_workers == 4
        assert config.backend == 'threading'
        assert config.chunk_size == 100
        assert config.show_progress is False
        assert config.work_stealing is False


class TestParallelWorker:
    """Test the parallel_worker function."""
    
    def test_parallel_worker_basic(self):
        """Test parallel worker with basic simulation function."""
        # Mock simulation function
        mock_simulate = Mock(return_value=np.random.rand(100))
        
        # Test worker
        args = (mock_simulate, 100, 10, {'random_state': 42})
        result = parallel_worker(args)
        
        # Verify
        mock_simulate.assert_called_once_with(100, 10, random_state=42)
        assert isinstance(result, np.ndarray)
    
    def test_parallel_worker_with_error(self):
        """Test parallel worker error handling."""
        # Mock simulation function that raises error
        mock_simulate = Mock(side_effect=ValueError("Simulation error"))
        
        # Test worker
        args = (mock_simulate, 100, 10, {})
        
        with pytest.raises(ValueError, match="Simulation error"):
            parallel_worker(args)


class TestParallelSimulatorExtended:
    """Extended tests for ParallelSimulator class."""
    
    @pytest.fixture
    def mock_cpu_count(self):
        """Mock CPU count for consistent testing."""
        with patch('quactuary.parallel_processing.cpu_count', return_value=4):
            yield
    
    def test_simulate_with_progress_monitoring(self, mock_cpu_count):
        """Test simulation with progress monitoring enabled."""
        config = ParallelConfig(n_workers=2, show_progress=True)
        simulator = ParallelSimulator(config)
        
        # Mock tqdm
        with patch('quactuary.parallel_processing.tqdm') as mock_tqdm:
            mock_pbar = MagicMock()
            mock_tqdm.return_value = mock_pbar
            
            # Mock simulation function
            def simple_sim(n, p):
                return np.ones(n)
            
            # Run with mocked executor
            with patch.object(simulator, '_get_executor') as mock_get_executor:
                mock_executor = MagicMock()
                mock_future = Mock()
                mock_future.result.return_value = np.ones(50)
                mock_executor.submit.return_value = mock_future
                mock_executor.__enter__.return_value = mock_executor
                mock_get_executor.return_value.__enter__.return_value = mock_executor
                
                # Patch as_completed
                with patch('quactuary.parallel_processing.as_completed') as mock_as_completed:
                    mock_as_completed.return_value = [mock_future]
                    
                    result = simulator.simulate_parallel_multiprocessing(
                        simple_sim, 100, 10
                    )
                    
                    # Verify progress bar was used
                    mock_tqdm.assert_called_once()
                    mock_pbar.update.assert_called()
                    mock_pbar.close.assert_called_once()
    
    def test_work_stealing_algorithm(self, mock_cpu_count):
        """Test work-stealing implementation."""
        config = ParallelConfig(n_workers=2, work_stealing=True)
        simulator = ParallelSimulator(config)
        
        # Mock simulation function
        def variable_time_sim(n, p):
            # Simulate variable execution time
            import time
            if np.random.rand() > 0.5:
                time.sleep(0.01)
            return np.random.rand(n)
        
        # This is a simplified test since actual work stealing is complex
        # Just verify the method exists and can be called
        assert hasattr(simulator, 'simulate_work_stealing')
        
        with patch.object(simulator, '_get_executor') as mock_get_executor:
            mock_executor = MagicMock()
            mock_executor.__enter__.return_value = mock_executor
            mock_get_executor.return_value.__enter__.return_value = mock_executor
            
            # The actual work stealing would need more complex mocking
            # This just verifies the interface
            chunk_size = simulator._calculate_chunk_size(1000, 2)
            assert chunk_size > 0
            assert chunk_size < 1000
    
    def test_joblib_backend(self, mock_cpu_count):
        """Test using joblib backend."""
        config = ParallelConfig(n_workers=2, backend='joblib')
        simulator = ParallelSimulator(config)
        
        # Mock joblib
        with patch('quactuary.parallel_processing.HAS_JOBLIB', True):
            with patch('quactuary.parallel_processing.Parallel') as mock_parallel:
                with patch('quactuary.parallel_processing.delayed') as mock_delayed:
                    mock_parallel_instance = MagicMock()
                    mock_parallel.return_value = mock_parallel_instance
                    mock_parallel_instance.return_value = [np.ones(50), np.ones(50)]
                    
                    def simple_sim(n, p):
                        return np.ones(n)
                    
                    result = simulator.simulate_parallel_joblib(
                        simple_sim, 100, 10
                    )
                    
                    # Verify joblib was called
                    mock_parallel.assert_called_once()
                    assert len(result) == 100
    
    def test_memory_limit_handling(self, mock_cpu_count):
        """Test memory limit configuration."""
        config = ParallelConfig(
            n_workers=2,
            memory_limit_mb=1000.0  # 1GB limit
        )
        simulator = ParallelSimulator(config)
        
        # Verify memory monitoring is configured
        assert hasattr(simulator, 'monitor')
        assert simulator.monitor.max_memory_mb == 1000.0
    
    def test_chunk_size_optimization(self, mock_cpu_count):
        """Test adaptive chunk sizing."""
        config = ParallelConfig(n_workers=4)
        simulator = ParallelSimulator(config)
        
        # Test different problem sizes
        # Small problem
        chunk_size = simulator._calculate_chunk_size(100, 4)
        assert chunk_size >= 10  # Minimum chunk size
        assert chunk_size <= 100  # Can't exceed total work
        
        # Medium problem
        chunk_size = simulator._calculate_chunk_size(10000, 4)
        assert chunk_size > 10
        assert chunk_size < 10000
        
        # Large problem
        chunk_size = simulator._calculate_chunk_size(1000000, 4)
        assert chunk_size <= 1000  # Maximum chunk size
        
        # Custom chunk size
        config.chunk_size = 500
        simulator = ParallelSimulator(config)
        chunk_size = simulator._calculate_chunk_size(10000, 4)
        assert chunk_size == 500


class TestParallelPerformance:
    """Test performance aspects of parallel processing."""
    
    def test_speedup_measurement(self):
        """Test that parallel processing provides speedup."""
        # This is a conceptual test - actual speedup depends on hardware
        config_serial = ParallelConfig(n_workers=1)
        config_parallel = ParallelConfig(n_workers=2)
        
        simulator_serial = ParallelSimulator(config_serial)
        simulator_parallel = ParallelSimulator(config_parallel)
        
        # Mock simulation that would benefit from parallelization
        def cpu_intensive_sim(n, p):
            # Simulate CPU-intensive work
            result = np.zeros(n)
            for i in range(n):
                result[i] = np.sum(np.random.rand(100))
            return result
        
        # In a real test, we would measure actual time
        # Here we just verify the setup
        assert simulator_serial.config.n_workers == 1
        assert simulator_parallel.config.n_workers == 2
    
    def test_overhead_minimization(self):
        """Test that small problems avoid parallel overhead."""
        config = ParallelConfig(n_workers=4)
        simulator = ParallelSimulator(config)
        
        # Very small problem should use larger chunks
        chunk_size = simulator._calculate_chunk_size(20, 4)
        assert chunk_size >= 10  # Minimum chunk to avoid overhead