import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import json
from dataclasses import asdict

from quactuary.benchmarks import (
    BenchmarkResult,
    PerformanceBenchmark,
    run_baseline_profiling
)


class TestBenchmarkResult:
    """Test the BenchmarkResult dataclass."""
    
    def test_benchmark_result_creation(self):
        """Test creating a BenchmarkResult instance."""
        result = BenchmarkResult(
            name="test_benchmark",
            time_seconds=1.5,
            memory_mb=100.5,
            iterations=1000,
            mean_loss=50000.0,
            var_loss=10000.0,
            tvar_loss=75000.0,
            parameters={"n_simulations": 1000}
        )
        
        assert result.name == "test_benchmark"
        assert result.time_seconds == 1.5
        assert result.memory_mb == 100.5
        assert result.iterations == 1000
        assert result.mean_loss == 50000.0
        assert result.var_loss == 10000.0
        assert result.tvar_loss == 75000.0
        assert result.parameters == {"n_simulations": 1000}
    
    def test_benchmark_result_to_dict(self):
        """Test converting BenchmarkResult to dictionary."""
        result = BenchmarkResult(
            name="test",
            time_seconds=1.0,
            memory_mb=50.0,
            iterations=100
        )
        
        result_dict = asdict(result)
        assert result_dict["name"] == "test"
        assert result_dict["time_seconds"] == 1.0
        assert result_dict["memory_mb"] == 50.0
        assert result_dict["iterations"] == 100


class TestPerformanceBenchmark:
    """Test the PerformanceBenchmark class."""
    
    @pytest.fixture
    def mock_psutil(self):
        """Mock psutil for consistent memory measurements."""
        with patch('quactuary.benchmarks.psutil') as mock:
            mock_process = Mock()
            mock_process.memory_info.return_value = Mock(rss=100 * 1024 * 1024)  # 100 MB
            mock.Process.return_value = mock_process
            yield mock
    
    @pytest.fixture
    def benchmark(self, mock_psutil):
        """Create a PerformanceBenchmark instance."""
        return PerformanceBenchmark()
    
    def test_initialization(self, benchmark):
        """Test PerformanceBenchmark initialization."""
        assert benchmark.results == []
        assert benchmark.suite_name is not None
        assert hasattr(benchmark, 'process')
    
    def test_measure_performance_context_manager(self, benchmark, mock_psutil):
        """Test the measure_performance context manager."""
        with benchmark.measure_performance("test_operation") as perf:
            # Simulate some work
            import time
            time.sleep(0.01)
            perf["iterations"] = 100
            perf["mean_loss"] = 50000.0
        
        # Check that result was recorded
        assert len(benchmark.results) == 1
        result = benchmark.results[0]
        assert result.name == "test_operation"
        assert result.time_seconds > 0
        assert result.memory_mb > 0
        assert result.iterations == 100
        assert result.mean_loss == 50000.0
    
    def test_create_test_portfolio(self, benchmark):
        """Test portfolio creation for benchmarking."""
        portfolio = benchmark._create_test_portfolio(size=10)
        
        assert len(portfolio) == 10
        # Check that portfolios have required attributes
        for policy in portfolio:
            assert hasattr(policy, 'policy_id')
            assert hasattr(policy, 'frequency_dist')
            assert hasattr(policy, 'severity_dist')
            assert hasattr(policy, 'policy_terms')
    
    @patch('quactuary.benchmarks.simulate_portfolio')
    def test_run_baseline_benchmark(self, mock_simulate, benchmark):
        """Test running baseline benchmark."""
        # Mock simulation results
        mock_results = Mock()
        mock_results.mean.return_value = 50000.0
        mock_results.var.return_value = 10000.0
        mock_results.percentile.return_value = 75000.0
        mock_simulate.return_value = (mock_results, Mock())
        
        # Run benchmark
        benchmark.run_baseline_benchmark(portfolio_size=10, n_simulations=1000)
        
        # Verify results
        assert len(benchmark.results) == 1
        result = benchmark.results[0]
        assert result.name == "baseline_classical"
        assert result.mean_loss == 50000.0
        assert result.var_loss == 75000.0  # VaR at 95%
        assert result.tvar_loss == 75000.0  # TVaR at 95%
        assert result.parameters["portfolio_size"] == 10
        assert result.parameters["n_simulations"] == 1000
    
    @patch('quactuary.benchmarks.simulate_portfolio')
    def test_run_jit_benchmark(self, mock_simulate, benchmark):
        """Test running JIT benchmark."""
        # Mock simulation results
        mock_results = Mock()
        mock_results.mean.return_value = 50000.0
        mock_results.var.return_value = 10000.0
        mock_results.percentile.return_value = 75000.0
        mock_simulate.return_value = (mock_results, Mock())
        
        # Run benchmark
        benchmark.run_jit_benchmark(portfolio_size=10, n_simulations=1000)
        
        # Verify results
        assert len(benchmark.results) == 1
        result = benchmark.results[0]
        assert result.name == "jit_optimized"
        assert result.parameters["use_jit"] is True
    
    @patch('quactuary.benchmarks.simulate_portfolio')
    def test_run_qmc_benchmark(self, mock_simulate, benchmark):
        """Test running QMC benchmark."""
        # Mock simulation results
        mock_results = Mock()
        mock_results.mean.return_value = 50000.0
        mock_results.var.return_value = 10000.0
        mock_results.percentile.return_value = 75000.0
        mock_simulate.return_value = (mock_results, Mock())
        
        # Run benchmark
        benchmark.run_qmc_benchmark(portfolio_size=10, n_simulations=1000)
        
        # Verify results
        assert len(benchmark.results) == 1
        result = benchmark.results[0]
        assert result.name == "qmc_sobol"
        assert result.parameters["use_qmc"] is True
    
    def test_save_results(self, benchmark):
        """Test saving benchmark results."""
        # Add a test result
        benchmark.results.append(BenchmarkResult(
            name="test",
            time_seconds=1.0,
            memory_mb=50.0,
            iterations=100
        ))
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_filename = f.name
        
        try:
            benchmark.save_results(temp_filename)
            
            # Load and verify
            with open(temp_filename, 'r') as f:
                data = json.load(f)
            
            assert data["suite_name"] == benchmark.suite_name
            assert len(data["results"]) == 1
            assert data["results"][0]["name"] == "test"
        finally:
            os.unlink(temp_filename)
    
    def test_generate_report(self, benchmark):
        """Test report generation."""
        # Add multiple results
        benchmark.results.extend([
            BenchmarkResult(
                name="baseline",
                time_seconds=2.0,
                memory_mb=100.0,
                iterations=1000,
                mean_loss=50000.0
            ),
            BenchmarkResult(
                name="optimized",
                time_seconds=1.0,
                memory_mb=80.0,
                iterations=1000,
                mean_loss=50000.0
            )
        ])
        
        report = benchmark.generate_report()
        
        assert "Performance Benchmark Report" in report
        assert "baseline" in report
        assert "optimized" in report
        assert "2.00s" in report  # baseline time
        assert "1.00s" in report  # optimized time
        assert "Speedup: 2.00x" in report
    
    def test_empty_report(self, benchmark):
        """Test report generation with no results."""
        report = benchmark.generate_report()
        assert "No benchmark results to report" in report


class TestRunBaselineProfiling:
    """Test the run_baseline_profiling function."""
    
    @patch('quactuary.benchmarks.PerformanceBenchmark')
    def test_run_baseline_profiling(self, mock_benchmark_class):
        """Test running baseline profiling."""
        mock_instance = Mock()
        mock_benchmark_class.return_value = mock_instance
        
        # Call function
        run_baseline_profiling(portfolio_size=10, n_simulations=100)
        
        # Verify benchmark methods were called
        mock_instance.run_baseline_benchmark.assert_called_once_with(10, 100)
        mock_instance.run_jit_benchmark.assert_called_once_with(10, 100)
        mock_instance.run_qmc_benchmark.assert_called_once_with(10, 100)
        mock_instance.generate_report.assert_called_once()
    
    @patch('quactuary.benchmarks.PerformanceBenchmark')
    @patch('builtins.print')
    def test_profiling_output(self, mock_print, mock_benchmark_class):
        """Test that profiling prints output."""
        mock_instance = Mock()
        mock_instance.generate_report.return_value = "Test Report"
        mock_benchmark_class.return_value = mock_instance
        
        run_baseline_profiling()
        
        # Verify print was called with report
        mock_print.assert_called_with("Test Report")