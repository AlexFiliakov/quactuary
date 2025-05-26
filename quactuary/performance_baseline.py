"""
Adaptive performance baseline system for hardware-independent testing.

This module provides infrastructure for establishing and managing performance baselines
that adapt to different hardware environments while still detecting regressions.
"""

import time
import platform
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import numpy as np
from pathlib import Path
import hashlib
import psutil
try:
    import cpuinfo
    HAS_CPUINFO = True
except ImportError:
    HAS_CPUINFO = False


@dataclass
class HardwareProfile:
    """Hardware profile for environment normalization."""
    cpu_count: int
    cpu_freq_mhz: float
    cpu_model: str
    total_memory_gb: float
    platform_system: str
    platform_machine: str
    python_version: str
    profile_hash: str = field(init=False)
    performance_score: float = field(default=1.0)
    
    def __post_init__(self):
        """Calculate profile hash after initialization."""
        # Create a deterministic hash of hardware characteristics
        hash_data = f"{self.cpu_model}_{self.cpu_count}_{self.cpu_freq_mhz:.0f}_{self.total_memory_gb:.1f}"
        self.profile_hash = hashlib.md5(hash_data.encode()).hexdigest()[:8]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HardwareProfile':
        """Create from dictionary."""
        # Remove computed fields before creating instance
        data = data.copy()
        data.pop('profile_hash', None)
        return cls(**data)
    
    @classmethod
    def get_current(cls) -> 'HardwareProfile':
        """Get hardware profile for current system."""
        # Get CPU information
        cpu_count = psutil.cpu_count(logical=False) or psutil.cpu_count()
        
        # Get CPU frequency
        cpu_freq = psutil.cpu_freq()
        cpu_freq_mhz = cpu_freq.current if cpu_freq else 0.0
        
        # Get CPU model
        cpu_model = "Unknown"
        if HAS_CPUINFO:
            try:
                info = cpuinfo.get_cpu_info()
                cpu_model = info.get('brand_raw', 'Unknown')
            except:
                pass
        
        # Get memory info
        mem_info = psutil.virtual_memory()
        total_memory_gb = mem_info.total / (1024 ** 3)
        
        # Get platform info
        platform_system = platform.system()
        platform_machine = platform.machine()
        python_version = platform.python_version()
        
        profile = cls(
            cpu_count=cpu_count,
            cpu_freq_mhz=cpu_freq_mhz,
            cpu_model=cpu_model,
            total_memory_gb=total_memory_gb,
            platform_system=platform_system,
            platform_machine=platform_machine,
            python_version=python_version
        )
        
        # Calculate performance score
        profile.performance_score = profile.calculate_performance_score()
        
        return profile
    
    def calculate_performance_score(self) -> float:
        """
        Calculate a normalized performance score for the hardware.
        
        Returns a score where 1.0 is the baseline (reference hardware).
        Higher scores indicate faster hardware.
        """
        # Reference hardware (baseline = 1.0)
        # Based on a typical CI server or mid-range development machine
        ref_cpu_count = 4
        ref_cpu_freq = 2500.0  # MHz
        ref_memory = 16.0  # GB
        
        # Calculate component scores
        cpu_count_score = self.cpu_count / ref_cpu_count
        cpu_freq_score = self.cpu_freq_mhz / ref_cpu_freq if self.cpu_freq_mhz > 0 else 1.0
        memory_score = min(self.total_memory_gb / ref_memory, 2.0)  # Cap memory contribution
        
        # Weighted average (CPU performance matters most)
        score = (
            0.4 * cpu_count_score +
            0.4 * cpu_freq_score +
            0.2 * memory_score
        )
        
        return max(0.1, score)  # Minimum score of 0.1


@dataclass
class PerformanceBaseline:
    """Performance baseline data with environment normalization."""
    test_name: str
    timestamp: str
    hardware_profile: HardwareProfile
    raw_time: float  # Actual execution time
    normalized_time: float  # Hardware-normalized time
    sample_size: int
    percentiles: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['hardware_profile'] = self.hardware_profile.to_dict()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceBaseline':
        """Create from dictionary."""
        data = data.copy()
        data['hardware_profile'] = HardwareProfile.from_dict(data['hardware_profile'])
        return cls(**data)


class AdaptiveBaselineManager:
    """
    Manages performance baselines that adapt to hardware differences.
    
    This system:
    1. Profiles hardware capabilities
    2. Normalizes performance measurements
    3. Detects regressions accounting for hardware differences
    4. Maintains historical baselines
    """
    
    def __init__(self, baseline_dir: str = "./performance_baselines"):
        """Initialize baseline manager."""
        self.baseline_dir = Path(baseline_dir)
        self.baseline_dir.mkdir(exist_ok=True)
        self.current_profile = HardwareProfile.get_current()
        self.baselines: Dict[str, List[PerformanceBaseline]] = {}
        self.regression_threshold = 1.2  # 20% slower is a regression
        
        # Load existing baselines
        self._load_baselines()
    
    def _load_baselines(self):
        """Load existing baselines from disk."""
        baseline_file = self.baseline_dir / "baselines.json"
        if baseline_file.exists():
            try:
                with open(baseline_file, 'r') as f:
                    data = json.load(f)
                    for test_name, baseline_list in data.items():
                        self.baselines[test_name] = [
                            PerformanceBaseline.from_dict(b) for b in baseline_list
                        ]
            except Exception as e:
                print(f"Warning: Failed to load baselines: {e}")
    
    def _save_baselines(self):
        """Save baselines to disk."""
        baseline_file = self.baseline_dir / "baselines.json"
        data = {}
        for test_name, baseline_list in self.baselines.items():
            data[test_name] = [b.to_dict() for b in baseline_list]
        
        with open(baseline_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def record_performance(
        self,
        test_name: str,
        execution_time: float,
        sample_size: int,
        percentiles: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> PerformanceBaseline:
        """
        Record a performance measurement and normalize it.
        
        Args:
            test_name: Name of the test
            execution_time: Raw execution time in seconds
            sample_size: Number of samples/iterations
            percentiles: Optional percentile measurements
            metadata: Optional additional metadata
            
        Returns:
            PerformanceBaseline object
        """
        # Normalize time based on hardware performance score
        normalized_time = execution_time / self.current_profile.performance_score
        
        baseline = PerformanceBaseline(
            test_name=test_name,
            timestamp=datetime.now().isoformat(),
            hardware_profile=self.current_profile,
            raw_time=execution_time,
            normalized_time=normalized_time,
            sample_size=sample_size,
            percentiles=percentiles or {},
            metadata=metadata or {}
        )
        
        # Add to baselines
        if test_name not in self.baselines:
            self.baselines[test_name] = []
        self.baselines[test_name].append(baseline)
        
        # Keep only recent baselines (last 100)
        if len(self.baselines[test_name]) > 100:
            self.baselines[test_name] = self.baselines[test_name][-100:]
        
        # Save to disk
        self._save_baselines()
        
        return baseline
    
    def check_regression(
        self,
        test_name: str,
        current_time: float,
        sample_size: int
    ) -> Tuple[bool, Optional[float], str]:
        """
        Check if current performance is a regression.
        
        Returns:
            Tuple of (is_regression, expected_time, message)
        """
        # Normalize current time
        normalized_current = current_time / self.current_profile.performance_score
        
        # Get historical baselines for this test
        if test_name not in self.baselines or not self.baselines[test_name]:
            return False, None, "No historical baseline available"
        
        # Filter baselines with similar sample sizes (within 50%)
        similar_baselines = [
            b for b in self.baselines[test_name]
            if 0.5 <= b.sample_size / sample_size <= 2.0
        ]
        
        if not similar_baselines:
            return False, None, "No baselines with similar sample size"
        
        # Calculate expected time using recent baselines
        recent_baselines = similar_baselines[-10:]  # Last 10 similar runs
        normalized_times = [b.normalized_time for b in recent_baselines]
        
        # Use median as expected value (robust to outliers)
        expected_normalized = np.median(normalized_times)
        
        # Calculate standard deviation for adaptive threshold
        if len(normalized_times) > 3:
            std_dev = np.std(normalized_times)
            # Dynamic threshold: baseline + 2 standard deviations or 20%, whichever is larger
            dynamic_threshold = max(
                expected_normalized + 2 * std_dev,
                expected_normalized * self.regression_threshold
            )
        else:
            dynamic_threshold = expected_normalized * self.regression_threshold
        
        # Check for regression
        is_regression = normalized_current > dynamic_threshold
        
        # Prepare message
        if is_regression:
            slowdown = (normalized_current / expected_normalized - 1) * 100
            message = (
                f"Performance regression detected: {slowdown:.1f}% slower than baseline. "
                f"Expected: {expected_normalized:.3f}s (normalized), "
                f"Got: {normalized_current:.3f}s (normalized)"
            )
        else:
            speedup = (1 - normalized_current / expected_normalized) * 100
            if speedup > 0:
                message = f"Performance improved: {speedup:.1f}% faster than baseline"
            else:
                message = "Performance within acceptable range"
        
        # Convert expected time back to current hardware
        expected_raw = expected_normalized * self.current_profile.performance_score
        
        return is_regression, expected_raw, message
    
    def get_performance_report(self, test_name: str) -> Dict[str, Any]:
        """
        Get a performance report for a specific test.
        
        Returns dictionary with performance trends and statistics.
        """
        if test_name not in self.baselines or not self.baselines[test_name]:
            return {"error": "No baseline data available"}
        
        baselines = self.baselines[test_name]
        
        # Group by hardware profile
        by_hardware = {}
        for b in baselines:
            hw_hash = b.hardware_profile.profile_hash
            if hw_hash not in by_hardware:
                by_hardware[hw_hash] = []
            by_hardware[hw_hash].append(b)
        
        # Calculate statistics
        report = {
            "test_name": test_name,
            "total_runs": len(baselines),
            "hardware_profiles": len(by_hardware),
            "current_hardware": self.current_profile.to_dict(),
            "performance_by_hardware": {}
        }
        
        for hw_hash, hw_baselines in by_hardware.items():
            raw_times = [b.raw_time for b in hw_baselines]
            normalized_times = [b.normalized_time for b in hw_baselines]
            
            hw_profile = hw_baselines[0].hardware_profile
            report["performance_by_hardware"][hw_hash] = {
                "cpu_model": hw_profile.cpu_model,
                "cpu_count": hw_profile.cpu_count,
                "performance_score": hw_profile.performance_score,
                "runs": len(hw_baselines),
                "raw_time_stats": {
                    "mean": np.mean(raw_times),
                    "median": np.median(raw_times),
                    "std": np.std(raw_times),
                    "min": np.min(raw_times),
                    "max": np.max(raw_times)
                },
                "normalized_time_stats": {
                    "mean": np.mean(normalized_times),
                    "median": np.median(normalized_times),
                    "std": np.std(normalized_times)
                }
            }
        
        # Overall normalized statistics
        all_normalized = [b.normalized_time for b in baselines]
        report["overall_normalized_stats"] = {
            "mean": np.mean(all_normalized),
            "median": np.median(all_normalized),
            "std": np.std(all_normalized),
            "trend": "stable"  # Could implement trend detection
        }
        
        return report
    
    def export_baselines(self, output_file: str):
        """Export all baselines to a file."""
        data = {
            "export_date": datetime.now().isoformat(),
            "current_hardware": self.current_profile.to_dict(),
            "baselines": {}
        }
        
        for test_name, baseline_list in self.baselines.items():
            data["baselines"][test_name] = [b.to_dict() for b in baseline_list]
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def import_baselines(self, input_file: str, merge: bool = True):
        """Import baselines from a file."""
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        if not merge:
            self.baselines.clear()
        
        for test_name, baseline_list in data["baselines"].items():
            if test_name not in self.baselines:
                self.baselines[test_name] = []
            
            for b_data in baseline_list:
                baseline = PerformanceBaseline.from_dict(b_data)
                self.baselines[test_name].append(baseline)
        
        self._save_baselines()