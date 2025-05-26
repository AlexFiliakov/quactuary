"""
Intelligent optimization selection system for quactuary.

This module provides automatic selection of optimization strategies based on
portfolio characteristics, hardware capabilities, and runtime conditions.
"""

import os
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil

from quactuary.book import Portfolio


class OptimizationStrategy(Enum):
    """Available optimization strategies."""
    NONE = "none"
    VECTORIZATION = "vectorization"
    JIT = "jit"
    QMC = "qmc"
    PARALLEL = "parallel"
    MEMORY_OPTIMIZATION = "memory_optimization"


@dataclass
class OptimizationProfile:
    """Profile of portfolio characteristics and system capabilities."""
    # Portfolio characteristics
    n_policies: int
    n_simulations: int
    distribution_complexity: float  # 0-1 scale
    has_dependencies: bool
    total_data_points: int
    
    # System capabilities
    available_memory_gb: float
    total_memory_gb: float
    cpu_count: int
    has_gpu: bool
    
    # Estimated requirements
    estimated_memory_gb: float
    estimated_compute_time: float
    
    # Historical performance (if available)
    past_performance: Optional[Dict[str, float]] = None


@dataclass
class UserPreferences:
    """User preferences for optimization trade-offs."""
    prefer_speed: bool = True  # Speed over accuracy
    max_memory_usage: float = 0.8  # Maximum memory usage fraction
    timeout_seconds: Optional[float] = None  # Computation timeout
    min_accuracy: float = 0.95  # Minimum acceptable accuracy (relative)
    allow_approximations: bool = True  # Allow approximate methods
    
    
@dataclass
class OptimizationConfig:
    """Configuration for optimization strategies."""
    use_jit: bool = False
    use_parallel: bool = False
    use_qmc: bool = False
    qmc_method: Optional[str] = None
    use_vectorization: bool = True
    use_memory_optimization: bool = False
    parallel_backend: Optional[str] = None
    batch_size: Optional[int] = None
    n_workers: Optional[int] = None
    fallback_chain: List[str] = field(default_factory=list)  # Fallback strategy names
    user_preferences: Optional[UserPreferences] = None
    
    def to_simulate_params(self) -> Dict[str, Any]:
        """Convert to parameters for PricingModel.simulate()."""
        params = {}
        if self.use_qmc and self.qmc_method:
            params['qmc_method'] = self.qmc_method
        # Other optimizations are handled at strategy level
        return params


class OptimizationSelector:
    """
    Intelligent optimization selector that analyzes portfolio characteristics
    and dynamically selects the best combination of optimization strategies.
    """
    
    # Thresholds for optimization selection
    SMALL_PORTFOLIO = 1000
    MEDIUM_PORTFOLIO = 10000
    LARGE_PORTFOLIO = 100000
    
    SMALL_SIMULATIONS = 1000
    MEDIUM_SIMULATIONS = 10000
    LARGE_SIMULATIONS = 100000
    
    # Memory usage estimates (GB per million data points)
    MEMORY_PER_MILLION = 0.008  # ~8MB per million doubles
    MEMORY_OVERHEAD = 0.5  # Base memory overhead in GB
    
    def __init__(self, enable_ml: bool = False, user_preferences: Optional[UserPreferences] = None):
        """
        Initialize the optimization selector.
        
        Args:
            enable_ml: Whether to enable machine learning for optimization selection
            user_preferences: User preferences for optimization trade-offs
        """
        self.enable_ml = enable_ml
        self.user_preferences = user_preferences or UserPreferences()
        self.performance_history = defaultdict(list)
        self._last_config = None
        self._monitoring_enabled = False
        self._fallback_level = 0  # Track current fallback level
        
    def analyze_portfolio(self, portfolio: Portfolio, n_simulations: int = 10000) -> OptimizationProfile:
        """
        Analyze portfolio characteristics and system capabilities.
        
        Args:
            portfolio: Portfolio to analyze
            n_simulations: Number of simulations planned
            
        Returns:
            OptimizationProfile with portfolio and system characteristics
        """
        # Portfolio characteristics
        n_policies = len(portfolio.policies)
        total_data_points = n_policies * n_simulations
        
        # Analyze distribution complexity (placeholder - could be enhanced)
        distribution_complexity = self._estimate_distribution_complexity(portfolio)
        has_dependencies = self._check_dependencies(portfolio)
        
        # System capabilities
        memory_info = psutil.virtual_memory()
        available_memory_gb = memory_info.available / (1024**3)
        total_memory_gb = memory_info.total / (1024**3)
        
        # Check swap usage to prevent thrashing
        swap_info = psutil.swap_memory()
        swap_used_gb = swap_info.used / (1024**3)
        swap_percent = swap_info.percent
        
        # Adjust available memory if swap is being used
        if swap_percent > 10:  # More than 10% swap usage
            # Reduce available memory to prevent further swap usage
            available_memory_gb = max(0.5, available_memory_gb - swap_used_gb * 0.5)
            
        cpu_count = psutil.cpu_count()
        has_gpu = self._detect_gpu()
        
        # Estimate requirements
        estimated_memory_gb = self._estimate_memory_requirements(
            n_policies, n_simulations
        )
        estimated_compute_time = self._estimate_compute_time(
            n_policies, n_simulations, distribution_complexity
        )
        
        # Get historical performance if available
        past_performance = self._get_historical_performance(
            n_policies, n_simulations
        )
        
        return OptimizationProfile(
            n_policies=n_policies,
            n_simulations=n_simulations,
            distribution_complexity=distribution_complexity,
            has_dependencies=has_dependencies,
            total_data_points=total_data_points,
            available_memory_gb=available_memory_gb,
            total_memory_gb=total_memory_gb,
            cpu_count=cpu_count,
            has_gpu=has_gpu,
            estimated_memory_gb=estimated_memory_gb,
            estimated_compute_time=estimated_compute_time,
            past_performance=past_performance
        )
    
    def estimate_memory_requirements(self, n_policies: int, n_simulations: int) -> float:
        """
        Estimate memory requirements in GB.
        
        Args:
            n_policies: Number of policies
            n_simulations: Number of simulations
            
        Returns:
            Estimated memory requirement in GB
        """
        return self._estimate_memory_requirements(n_policies, n_simulations)
    
    def predict_best_strategy(self, profile: OptimizationProfile) -> OptimizationConfig:
        """
        Predict the best optimization strategy based on the profile.
        
        Args:
            profile: Optimization profile from analyze_portfolio
            
        Returns:
            OptimizationConfig with recommended settings
        """
        config = OptimizationConfig()
        
        # Use ML prediction if enabled and sufficient data
        if self.enable_ml and self._has_sufficient_ml_data():
            ml_config = self._ml_predict(profile)
            if ml_config:
                return ml_config
        
        # Check timeout risk
        if self.check_timeout_risk(profile):
            warnings.warn(f"Computation may exceed timeout ({self.user_preferences.timeout_seconds}s)")
            # Prefer faster methods
            config.use_jit = False  # Skip compilation overhead
            config.use_parallel = True  # Use parallelism for speed
            config.n_workers = min(4, profile.cpu_count)  # Limited workers
        
        # Rule-based selection
        total_points = profile.total_data_points
        memory_ratio = profile.estimated_memory_gb / profile.available_memory_gb
        
        # Memory optimization needed if we're using >max allowed memory
        if memory_ratio > self.user_preferences.max_memory_usage:
            config.use_memory_optimization = True
            config.batch_size = self._calculate_batch_size(profile)
        
        # Small data: Vectorization only
        if total_points < 1e6:  # < 1M points
            config.use_vectorization = True
            config.use_jit = False
            config.use_parallel = False
            config.use_qmc = profile.n_simulations > self.SMALL_SIMULATIONS
            
        # Medium data: JIT + Vectorization, consider QMC
        elif total_points < 1e8:  # < 100M points
            config.use_vectorization = True
            config.use_jit = True
            config.use_parallel = False
            config.use_qmc = profile.n_simulations > self.MEDIUM_SIMULATIONS
            if config.use_qmc:
                config.qmc_method = "sobol"
                
        # Large data: All optimizations
        else:
            config.use_vectorization = True
            config.use_jit = True
            config.use_parallel = True
            config.use_qmc = True
            config.qmc_method = "sobol"
            config.n_workers = min(profile.cpu_count - 1, 8)  # Leave one CPU free
            
        # Adjust based on complexity and dependencies
        if profile.distribution_complexity > 0.7 or profile.has_dependencies:
            # Complex distributions may not benefit from JIT
            config.use_jit = False
            
        # Set parallel backend based on availability
        if config.use_parallel:
            config.parallel_backend = self._select_parallel_backend()
            
        # Set up fallback chain based on selected optimizations
        config.fallback_chain = self._create_fallback_chain(config)
        config.user_preferences = self.user_preferences
        
        # Apply user preference adjustments
        config = self.adjust_for_accuracy_tradeoff(config)
        
        self._last_config = config
        return config
    
    def _create_fallback_chain(self, config: OptimizationConfig) -> List[str]:
        """Create a fallback chain for graceful degradation."""
        chain = []
        
        # Current configuration name
        current = []
        if config.use_parallel:
            current.append("parallel")
        if config.use_jit:
            current.append("jit")
        if config.use_qmc:
            current.append("qmc")
        if config.use_vectorization:
            current.append("vectorized")
        chain.append("_".join(current) if current else "basic")
        
        # Add progressively simpler configurations
        if config.use_parallel:
            # Try without parallel first
            chain.append("jit_qmc_vectorized" if config.use_jit and config.use_qmc else "jit_vectorized")
            
        if config.use_jit:
            # Try without JIT
            chain.append("qmc_vectorized" if config.use_qmc else "vectorized")
            
        if config.use_qmc:
            # Try without QMC
            chain.append("vectorized")
            
        # Always have basic as final fallback
        if "basic" not in chain:
            chain.append("basic")
            
        return chain
    
    def get_fallback_config(self, current_config: OptimizationConfig, 
                           failure_reason: str = "unknown") -> Optional[OptimizationConfig]:
        """
        Get next fallback configuration based on failure reason.
        
        Args:
            current_config: Current configuration that failed
            failure_reason: Reason for failure (memory, timeout, error)
            
        Returns:
            Next fallback configuration or None if no more fallbacks
        """
        if not current_config.fallback_chain:
            return None
            
        self._fallback_level += 1
        
        if self._fallback_level >= len(current_config.fallback_chain):
            return None  # No more fallbacks
            
        next_strategy = current_config.fallback_chain[self._fallback_level]
        
        # Create config based on strategy name
        new_config = OptimizationConfig()
        
        if "parallel" in next_strategy:
            new_config.use_parallel = True
            new_config.n_workers = current_config.n_workers
            
        if "jit" in next_strategy:
            new_config.use_jit = True
            
        if "qmc" in next_strategy:
            new_config.use_qmc = True
            new_config.qmc_method = "sobol"
            
        if "vectorized" in next_strategy:
            new_config.use_vectorization = True
            
        # Adjust based on failure reason
        if failure_reason == "memory":
            new_config.use_memory_optimization = True
            new_config.batch_size = (current_config.batch_size or 10000) // 2
            new_config.use_parallel = False  # Reduce memory usage
            
        elif failure_reason == "timeout":
            # Reduce computational complexity
            new_config.use_jit = False  # Skip compilation overhead
            if current_config.n_workers:
                new_config.n_workers = min(4, current_config.n_workers)
                
        warnings.warn(f"Falling back to {next_strategy} strategy due to {failure_reason}")
        
        return new_config
    
    def monitor_and_adapt(self, runtime_metrics: Dict[str, float]) -> Optional[OptimizationConfig]:
        """
        Monitor runtime metrics and adapt optimization strategy if needed.
        
        Args:
            runtime_metrics: Dictionary with metrics like 'memory_usage', 'cpu_usage', 'time_elapsed'
            
        Returns:
            New OptimizationConfig if adaptation needed, None otherwise
        """
        if not self._monitoring_enabled or not self._last_config:
            return None
            
        # Get current system metrics
        memory_info = psutil.virtual_memory()
        swap_info = psutil.swap_memory()
        
        # Check for swap usage
        if swap_info.percent > 5:  # Any significant swap usage
            warnings.warn(f"Swap usage detected ({swap_info.percent:.1f}%), enabling memory optimization")
            new_config = self._last_config
            new_config.use_memory_optimization = True
            new_config.use_parallel = False  # Reduce parallelism
            new_config.batch_size = self.calculate_dynamic_batch_size(
                n_policies=10000,  # Approximate
                n_simulations=10000,
                current_memory_usage=memory_info.percent / 100
            )
            return new_config
            
        # Check for memory pressure
        if runtime_metrics.get('memory_usage', 0) > 0.9:  # >90% memory used
            warnings.warn("High memory usage detected, adapting optimization strategy")
            new_config = self._last_config
            new_config.use_memory_optimization = True
            new_config.use_parallel = False  # Reduce parallelism to save memory
            return new_config
            
        # Check for slow progress
        if runtime_metrics.get('progress_rate', 1.0) < 0.1:  # Very slow
            warnings.warn("Slow progress detected, disabling some optimizations")
            new_config = self._last_config
            new_config.use_jit = False  # JIT compilation overhead might be too high
            return new_config
            
        return None
    
    def adjust_for_accuracy_tradeoff(self, config: OptimizationConfig, 
                                   target_accuracy: Optional[float] = None) -> OptimizationConfig:
        """
        Adjust configuration based on performance vs accuracy trade-offs.
        
        Args:
            config: Current configuration
            target_accuracy: Target accuracy level (0-1), None uses user preference
            
        Returns:
            Adjusted configuration
        """
        target = target_accuracy or self.user_preferences.min_accuracy
        
        # If user prefers speed and allows approximations
        if self.user_preferences.prefer_speed and self.user_preferences.allow_approximations:
            # Reduce simulation count for speed
            if hasattr(config, 'n_simulations'):
                config.n_simulations = int(config.n_simulations * target)
                
            # Use faster but less accurate methods
            if config.use_qmc:
                config.qmc_scramble = False  # Skip scrambling for speed
                
        # If user prefers accuracy
        elif not self.user_preferences.prefer_speed:
            # Increase precision
            if config.use_qmc:
                config.qmc_method = "sobol"  # Most accurate QMC
                config.qmc_scramble = True
                
            # Disable approximations
            config.use_memory_optimization = False  # Process all at once
            
        return config
    
    def check_timeout_risk(self, profile: OptimizationProfile) -> bool:
        """
        Check if computation might exceed timeout.
        
        Args:
            profile: Optimization profile
            
        Returns:
            True if timeout risk is high
        """
        if not self.user_preferences.timeout_seconds:
            return False
            
        # Add 20% safety margin
        safe_timeout = self.user_preferences.timeout_seconds * 0.8
        
        return profile.estimated_compute_time > safe_timeout
    
    def enable_monitoring(self):
        """Enable runtime monitoring and adaptation."""
        self._monitoring_enabled = True
        
    def disable_monitoring(self):
        """Disable runtime monitoring."""
        self._monitoring_enabled = False
    
    def record_performance(self, profile: OptimizationProfile, config: OptimizationConfig, 
                          performance_metrics: Dict[str, float]):
        """
        Record performance data for future ML training.
        
        Args:
            profile: The optimization profile used
            config: The configuration that was applied
            performance_metrics: Actual performance metrics (time, memory, accuracy)
        """
        record = {
            'profile': profile,
            'config': config,
            'metrics': performance_metrics,
            'timestamp': time.time()
        }
        key = (profile.n_policies, profile.n_simulations)
        self.performance_history[key].append(record)
    
    # Private helper methods
    
    def _estimate_distribution_complexity(self, portfolio: Portfolio) -> float:
        """Estimate distribution complexity on 0-1 scale."""
        complexity_score = 0.0
        complexity_factors = 0
        
        # Check if portfolio has compound distributions
        if hasattr(portfolio, 'compound_distribution') and portfolio.compound_distribution:
            compound = portfolio.compound_distribution
            
            # Heavy-tailed severity distributions are more complex
            if hasattr(compound, 'severity'):
                severity_name = compound.severity.__class__.__name__.lower()
                if 'pareto' in severity_name or 'studentsT' in severity_name:
                    complexity_score += 0.3
                elif 'lognormal' in severity_name or 'gamma' in severity_name:
                    complexity_score += 0.2
                elif 'exponential' in severity_name:
                    complexity_score += 0.1
                complexity_factors += 1
                
            # Overdispersed frequency distributions are more complex
            if hasattr(compound, 'frequency'):
                freq_name = compound.frequency.__class__.__name__.lower()
                if 'negativebinomial' in freq_name or 'mixed' in freq_name:
                    complexity_score += 0.3
                elif 'poisson' in freq_name:
                    complexity_score += 0.1
                elif 'empirical' in freq_name:
                    complexity_score += 0.2
                complexity_factors += 1
                
            # Zero-inflated models add complexity
            if 'zeroinflated' in compound.__class__.__name__.lower():
                complexity_score += 0.2
                complexity_factors += 1
                
        # Check for mixed distributions or hierarchical models
        if hasattr(portfolio, 'policies') and len(portfolio.policies) > 0:
            # Multiple policy types increase complexity
            unique_policy_types = len(set(p.get('type', 'default') for p in portfolio.policies))
            if unique_policy_types > 1:
                complexity_score += min(0.2, unique_policy_types * 0.05)
                complexity_factors += 1
                
        # Normalize to 0-1 scale
        if complexity_factors > 0:
            return min(1.0, complexity_score / complexity_factors * 2)
        return 0.5  # Default moderate complexity
    
    def _check_dependencies(self, portfolio: Portfolio) -> bool:
        """Check if portfolio has dependencies between policies."""
        # Check for explicit dependency indicators
        if hasattr(portfolio, 'correlation_matrix') and portfolio.correlation_matrix is not None:
            return True
            
        if hasattr(portfolio, 'copula') and portfolio.copula is not None:
            return True
            
        # Check for hierarchical or grouped structures
        if hasattr(portfolio, 'policies') and len(portfolio.policies) > 0:
            # Look for group indicators
            groups = set()
            for policy in portfolio.policies:
                if isinstance(policy, dict):
                    group = policy.get('group', policy.get('line_of_business', None))
                    if group:
                        groups.add(group)
                        
            # Multiple groups suggest potential dependencies
            if len(groups) > 1:
                return True
                
        # Check for aggregate exposures
        if hasattr(portfolio, 'aggregate_exposures') and portfolio.aggregate_exposures:
            return True
            
        return False
    
    def _detect_gpu(self) -> bool:
        """Detect if GPU is available."""
        # Check for CUDA availability
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            pass
            
        # Check for OpenCL
        try:
            import pyopencl as cl
            platforms = cl.get_platforms()
            for platform in platforms:
                devices = platform.get_devices(cl.device_type.GPU)
                if devices:
                    return True
        except:
            pass
            
        # Check NVIDIA settings
        nvidia_smi = os.environ.get('NVIDIA_SMI', None)
        if nvidia_smi and os.path.exists(nvidia_smi):
            return True
            
        return False
    
    def _estimate_memory_requirements(self, n_policies: int, n_simulations: int) -> float:
        """Estimate memory requirements in GB."""
        total_points = n_policies * n_simulations
        memory_gb = (total_points / 1e6) * self.MEMORY_PER_MILLION + self.MEMORY_OVERHEAD
        return memory_gb
    
    def _estimate_compute_time(self, n_policies: int, n_simulations: int, 
                              complexity: float) -> float:
        """Estimate compute time in seconds."""
        # Base time per operation (nanoseconds)
        # These are rough estimates - could be calibrated with benchmarks
        base_time_ns = {
            'simple': 100,      # Simple operations like exponential
            'moderate': 500,    # Moderate like gamma, lognormal  
            'complex': 2000,    # Complex like mixed distributions
            'very_complex': 10000  # Very complex with dependencies
        }
        
        # Determine operation complexity
        if complexity < 0.3:
            time_per_op = base_time_ns['simple']
        elif complexity < 0.5:
            time_per_op = base_time_ns['moderate']
        elif complexity < 0.8:
            time_per_op = base_time_ns['complex']
        else:
            time_per_op = base_time_ns['very_complex']
            
        # Total operations
        total_ops = n_policies * n_simulations
        
        # Base time in seconds
        base_time = total_ops * time_per_op * 1e-9
        
        # Adjust for CPU performance (rough estimate)
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        if cpu_freq:
            # Normalize to 3GHz baseline
            cpu_factor = cpu_freq.current / 3000
        else:
            cpu_factor = 1.0
            
        # Memory bandwidth impact for large datasets
        memory_gb = self._estimate_memory_requirements(n_policies, n_simulations)
        if memory_gb > 8:  # Large dataset penalty
            memory_factor = 1 + (memory_gb - 8) * 0.05  # 5% slower per GB over 8GB
        else:
            memory_factor = 1.0
            
        estimated_time = base_time / cpu_factor * memory_factor
        
        # Minimum time to account for overhead
        return max(0.001, estimated_time)
    
    def _get_historical_performance(self, n_policies: int, n_simulations: int) -> Optional[Dict[str, float]]:
        """Get historical performance data for similar workloads."""
        key = (n_policies, n_simulations)
        if key in self.performance_history:
            # Return average of recent performances
            recent = self.performance_history[key][-5:]  # Last 5 runs
            if recent:
                avg_metrics = {}
                for metric in recent[0]['metrics']:
                    avg_metrics[metric] = np.mean([r['metrics'][metric] for r in recent])
                return avg_metrics
        return None
    
    def _calculate_batch_size(self, profile: OptimizationProfile) -> int:
        """Calculate appropriate batch size for memory optimization."""
        # Check current memory pressure
        memory_info = psutil.virtual_memory()
        current_usage = memory_info.percent / 100.0
        
        # Adjust target based on current usage
        if current_usage > 0.8:  # High memory pressure
            target_fraction = 0.2  # Use only 20% of available
        elif current_usage > 0.6:  # Moderate pressure
            target_fraction = 0.3  # Use 30% of available
        else:  # Low pressure
            target_fraction = 0.4  # Use 40% of available
            
        target_memory_gb = profile.available_memory_gb * target_fraction
        
        # Calculate points per batch
        points_per_batch = int(target_memory_gb / self.MEMORY_PER_MILLION * 1e6)
        
        # Ensure batch size is reasonable
        min_batch = max(100, profile.n_policies // 1000)  # At least 0.1% of policies
        max_batch = profile.n_policies // 2  # At most 50% of policies
        
        batch_size = points_per_batch // profile.n_simulations
        batch_size = max(min_batch, min(batch_size, max_batch))
        
        # Round to nice numbers for efficiency
        if batch_size > 10000:
            batch_size = (batch_size // 1000) * 1000
        elif batch_size > 1000:
            batch_size = (batch_size // 100) * 100
            
        return batch_size
    
    def calculate_dynamic_batch_size(self, n_policies: int, n_simulations: int,
                                   current_memory_usage: float) -> int:
        """
        Calculate batch size dynamically based on current memory usage.
        
        Args:
            n_policies: Number of policies
            n_simulations: Number of simulations 
            current_memory_usage: Current memory usage as fraction (0-1)
            
        Returns:
            Recommended batch size
        """
        # Create temporary profile
        available_gb = psutil.virtual_memory().available / (1024**3)
        profile = OptimizationProfile(
            n_policies=n_policies,
            n_simulations=n_simulations,
            distribution_complexity=0.5,
            has_dependencies=False,
            total_data_points=n_policies * n_simulations,
            available_memory_gb=available_gb,
            total_memory_gb=psutil.virtual_memory().total / (1024**3),
            cpu_count=psutil.cpu_count(),
            has_gpu=False,
            estimated_memory_gb=self._estimate_memory_requirements(n_policies, n_simulations),
            estimated_compute_time=0.0
        )
        
        # Override with current usage if provided
        if current_memory_usage > 0:
            memory_info = psutil.virtual_memory()
            memory_info.percent = current_memory_usage * 100
            
        return self._calculate_batch_size(profile)
    
    def _select_parallel_backend(self) -> str:
        """Select the best available parallel backend."""
        # Simple selection - could be enhanced with actual backend detection
        try:
            import joblib
            return "joblib"
        except ImportError:
            try:
                import multiprocessing
                return "multiprocessing"
            except:
                return "threading"
    
    def _has_sufficient_ml_data(self) -> bool:
        """Check if we have enough data for ML predictions."""
        total_records = sum(len(v) for v in self.performance_history.values())
        return total_records >= 100  # Arbitrary threshold
    
    def _ml_predict(self, profile: OptimizationProfile) -> Optional[OptimizationConfig]:
        """Use ML to predict optimal configuration."""
        # Placeholder for ML implementation
        # Would use scikit-learn or similar to train on performance_history
        return None