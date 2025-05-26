"""
Performance tracking and analysis utilities.

This module provides comprehensive performance tracking, trend analysis,
and regression detection capabilities for the integration test suite.
"""

import json
import os
import time
import psutil
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import warnings


@dataclass
class PerformanceMetric:
    """Single performance measurement."""
    timestamp: str
    test_name: str
    portfolio_size: int
    n_simulations: int
    execution_time: float
    memory_usage_mb: float
    optimization_config: Dict[str, Any]
    numerical_accuracy: Dict[str, float]
    success: bool
    error_message: Optional[str] = None


@dataclass
class TrendAnalysis:
    """Performance trend analysis results."""
    metric_name: str
    time_period_days: int
    data_points: int
    trend_slope: float
    trend_r_squared: float
    is_improving: bool
    regression_detected: bool
    confidence_level: float


class PerformanceTracker:
    """Comprehensive performance tracking and analysis."""
    
    def __init__(self, data_file: str = "performance_history.json"):
        self.data_file = os.path.join(os.path.dirname(__file__), data_file)
        self.current_session = []
        self.baseline_file = os.path.join(os.path.dirname(__file__), "baseline_results.json")
    
    def record_measurement(self, metric: PerformanceMetric):
        """Record a single performance measurement."""
        self.current_session.append(metric)
    
    def save_session(self):
        """Save current session measurements to historical data."""
        if not self.current_session:
            return
        
        # Load existing history
        history = self._load_history()
        
        # Add current session
        for metric in self.current_session:
            history.append(asdict(metric))
        
        # Save updated history
        self._save_history(history)
        
        # Clear current session
        self.current_session = []
    
    def analyze_trends(self, metric_name: str, days: int = 30) -> TrendAnalysis:
        """Analyze performance trends over specified time period."""
        history = self._load_history()
        
        # Filter data by time period and metric
        cutoff_date = datetime.now() - timedelta(days=days)
        
        relevant_data = []
        for record in history:
            if record['test_name'] == metric_name:
                record_date = datetime.fromisoformat(record['timestamp'])
                if record_date >= cutoff_date:
                    relevant_data.append(record)
        
        if len(relevant_data) < 3:
            return TrendAnalysis(
                metric_name=metric_name,
                time_period_days=days,
                data_points=len(relevant_data),
                trend_slope=0.0,
                trend_r_squared=0.0,
                is_improving=False,
                regression_detected=False,
                confidence_level=0.0
            )
        
        # Extract time series data
        timestamps = [datetime.fromisoformat(r['timestamp']) for r in relevant_data]
        values = [r['execution_time'] for r in relevant_data]
        
        # Convert timestamps to days since first measurement
        base_time = min(timestamps)
        time_days = [(t - base_time).total_seconds() / 86400 for t in timestamps]
        
        # Linear regression analysis
        coeffs = np.polyfit(time_days, values, 1)
        slope = coeffs[0]
        
        # Calculate R-squared
        predicted = np.polyval(coeffs, time_days)
        ss_res = np.sum((values - predicted) ** 2)
        ss_tot = np.sum((values - np.mean(values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Determine if improving (negative slope = faster execution)
        is_improving = slope < -0.01  # At least 0.01 seconds improvement per day
        
        # Detect regression (significant performance degradation)
        recent_avg = np.mean(values[-3:])  # Last 3 measurements
        historical_avg = np.mean(values[:-3]) if len(values) > 3 else recent_avg
        regression_detected = recent_avg > historical_avg * 1.5  # 50% slower
        
        # Confidence based on R-squared and data points
        confidence_level = min(r_squared * (len(relevant_data) / 10), 1.0)
        
        return TrendAnalysis(
            metric_name=metric_name,
            time_period_days=days,
            data_points=len(relevant_data),
            trend_slope=slope,
            trend_r_squared=r_squared,
            is_improving=is_improving,
            regression_detected=regression_detected,
            confidence_level=confidence_level
        )
    
    def generate_performance_report(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        history = self._load_history()
        
        if not history:
            return {"error": "No performance history available"}
        
        # Analyze different test categories
        test_categories = set(record['test_name'] for record in history)
        
        report = {
            "generation_time": datetime.now().isoformat(),
            "total_measurements": len(history),
            "test_categories": list(test_categories),
            "time_range": {
                "earliest": min(record['timestamp'] for record in history),
                "latest": max(record['timestamp'] for record in history)
            },
            "trend_analysis": {},
            "current_performance": {},
            "regression_alerts": []
        }
        
        # Analyze trends for each test category
        for test_name in test_categories:
            trend = self.analyze_trends(test_name, days=30)
            report["trend_analysis"][test_name] = asdict(trend)
            
            if trend.regression_detected:
                report["regression_alerts"].append({
                    "test_name": test_name,
                    "severity": "high" if trend.confidence_level > 0.7 else "medium",
                    "message": f"Performance regression detected in {test_name}"
                })
        
        # Current performance summary
        recent_data = [r for r in history if 
                      datetime.fromisoformat(r['timestamp']) > datetime.now() - timedelta(days=7)]
        
        if recent_data:
            report["current_performance"] = {
                "average_execution_time": np.mean([r['execution_time'] for r in recent_data]),
                "average_memory_usage": np.mean([r['memory_usage_mb'] for r in recent_data]),
                "success_rate": sum(1 for r in recent_data if r['success']) / len(recent_data),
                "measurements_last_week": len(recent_data)
            }
        
        # Save report if output file specified
        if output_file:
            report_path = os.path.join(os.path.dirname(__file__), output_file)
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
        
        return report
    
    def plot_performance_trends(self, test_names: List[str], save_path: Optional[str] = None):
        """Create performance trend visualizations."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("Matplotlib not available for plotting")
            return
        
        history = self._load_history()
        
        fig, axes = plt.subplots(len(test_names), 1, figsize=(12, 4 * len(test_names)))
        if len(test_names) == 1:
            axes = [axes]
        
        for i, test_name in enumerate(test_names):
            # Filter data for this test
            test_data = [r for r in history if r['test_name'] == test_name]
            
            if not test_data:
                axes[i].text(0.5, 0.5, f"No data for {test_name}", 
                           ha='center', va='center', transform=axes[i].transAxes)
                continue
            
            # Extract time series
            timestamps = [datetime.fromisoformat(r['timestamp']) for r in test_data]
            execution_times = [r['execution_time'] for r in test_data]
            
            # Plot time series
            axes[i].plot(timestamps, execution_times, 'o-', label='Execution Time')
            
            # Add trend line
            if len(timestamps) > 2:
                time_numeric = [(t - min(timestamps)).total_seconds() for t in timestamps]
                coeffs = np.polyfit(time_numeric, execution_times, 1)
                trend_line = np.polyval(coeffs, time_numeric)
                axes[i].plot(timestamps, trend_line, '--', color='red', label='Trend')
            
            axes[i].set_title(f'Performance Trend: {test_name}')
            axes[i].set_xlabel('Date')
            axes[i].set_ylabel('Execution Time (seconds)')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            
            # Rotate x-axis labels for better readability
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(os.path.dirname(__file__), save_path), 
                       dpi=150, bbox_inches='tight')
        else:
            plt.show()
    
    def compare_with_baseline(self, test_name: str, current_time: float) -> Dict[str, Any]:
        """Compare current performance with baseline expectations."""
        try:
            with open(self.baseline_file, 'r') as f:
                baselines = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"error": "Baseline data not available"}
        
        # Extract relevant baseline data
        speedup_targets = baselines.get('baselines', {}).get('speedup_targets', {})
        
        # Determine portfolio size category from test name
        size_category = None
        if 'small' in test_name.lower():
            size_category = 'small_portfolio'
        elif 'medium' in test_name.lower():
            size_category = 'medium_portfolio'
        elif 'large' in test_name.lower():
            size_category = 'large_portfolio'
        
        if not size_category or size_category not in speedup_targets:
            return {"error": f"No baseline data for {test_name}"}
        
        baseline_data = speedup_targets[size_category]
        expected_time = baseline_data.get('expected_optimized_time_seconds', 0)
        
        if expected_time <= 0:
            return {"error": "Invalid baseline data"}
        
        # Calculate performance metrics
        performance_ratio = current_time / expected_time
        
        return {
            "test_name": test_name,
            "current_time": current_time,
            "expected_time": expected_time,
            "performance_ratio": performance_ratio,
            "meets_baseline": performance_ratio <= 1.5,  # Within 50% of expected
            "performance_level": self._categorize_performance(performance_ratio)
        }
    
    def _categorize_performance(self, ratio: float) -> str:
        """Categorize performance relative to baseline."""
        if ratio <= 0.8:
            return "excellent"
        elif ratio <= 1.2:
            return "good"
        elif ratio <= 2.0:
            return "acceptable"
        elif ratio <= 3.0:
            return "poor"
        else:
            return "unacceptable"
    
    def _load_history(self) -> List[Dict]:
        """Load performance history from file."""
        if not os.path.exists(self.data_file):
            return []
        
        try:
            with open(self.data_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []
    
    def _save_history(self, history: List[Dict]):
        """Save performance history to file."""
        os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
        
        with open(self.data_file, 'w') as f:
            json.dump(history, f, indent=2)


class RegressionDetector:
    """Automated regression detection system."""
    
    def __init__(self, tracker: PerformanceTracker):
        self.tracker = tracker
        self.thresholds = {
            'execution_time_increase': 1.5,  # 50% slower
            'memory_increase': 2.0,          # 100% more memory
            'success_rate_decrease': 0.9     # Below 90% success rate
        }
    
    def check_for_regressions(self, days_back: int = 7) -> List[Dict[str, Any]]:
        """Check for performance regressions in recent measurements."""
        history = self.tracker._load_history()
        
        if len(history) < 10:  # Need sufficient history
            return []
        
        cutoff_date = datetime.now() - timedelta(days=days_back)
        recent_data = [r for r in history if 
                      datetime.fromisoformat(r['timestamp']) > cutoff_date]
        historical_data = [r for r in history if 
                          datetime.fromisoformat(r['timestamp']) <= cutoff_date]
        
        if not recent_data or not historical_data:
            return []
        
        regressions = []
        test_names = set(r['test_name'] for r in recent_data)
        
        for test_name in test_names:
            recent_test_data = [r for r in recent_data if r['test_name'] == test_name]
            historical_test_data = [r for r in historical_data if r['test_name'] == test_name]
            
            if not historical_test_data:
                continue
            
            # Check execution time regression
            recent_avg_time = np.mean([r['execution_time'] for r in recent_test_data])
            historical_avg_time = np.mean([r['execution_time'] for r in historical_test_data])
            
            if recent_avg_time > historical_avg_time * self.thresholds['execution_time_increase']:
                regressions.append({
                    'type': 'execution_time',
                    'test_name': test_name,
                    'current_avg': recent_avg_time,
                    'historical_avg': historical_avg_time,
                    'ratio': recent_avg_time / historical_avg_time,
                    'severity': 'high' if recent_avg_time > historical_avg_time * 2.0 else 'medium'
                })
            
            # Check memory usage regression
            recent_avg_memory = np.mean([r['memory_usage_mb'] for r in recent_test_data])
            historical_avg_memory = np.mean([r['memory_usage_mb'] for r in historical_test_data])
            
            if recent_avg_memory > historical_avg_memory * self.thresholds['memory_increase']:
                regressions.append({
                    'type': 'memory_usage',
                    'test_name': test_name,
                    'current_avg': recent_avg_memory,
                    'historical_avg': historical_avg_memory,
                    'ratio': recent_avg_memory / historical_avg_memory,
                    'severity': 'high' if recent_avg_memory > historical_avg_memory * 3.0 else 'medium'
                })
            
            # Check success rate regression
            recent_success_rate = sum(1 for r in recent_test_data if r['success']) / len(recent_test_data)
            historical_success_rate = sum(1 for r in historical_test_data if r['success']) / len(historical_test_data)
            
            if recent_success_rate < self.thresholds['success_rate_decrease']:
                regressions.append({
                    'type': 'success_rate',
                    'test_name': test_name,
                    'current_rate': recent_success_rate,
                    'historical_rate': historical_success_rate,
                    'severity': 'critical' if recent_success_rate < 0.8 else 'high'
                })
        
        return regressions
    
    def generate_regression_report(self) -> Dict[str, Any]:
        """Generate comprehensive regression analysis report."""
        regressions = self.check_for_regressions()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "regressions_detected": len(regressions),
            "regressions": regressions,
            "summary": {
                "critical": len([r for r in regressions if r.get('severity') == 'critical']),
                "high": len([r for r in regressions if r.get('severity') == 'high']),
                "medium": len([r for r in regressions if r.get('severity') == 'medium'])
            },
            "recommendations": self._generate_recommendations(regressions)
        }
    
    def _generate_recommendations(self, regressions: List[Dict]) -> List[str]:
        """Generate recommendations based on detected regressions."""
        recommendations = []
        
        if any(r['type'] == 'execution_time' for r in regressions):
            recommendations.append("Investigate performance optimizations and profiling")
        
        if any(r['type'] == 'memory_usage' for r in regressions):
            recommendations.append("Check for memory leaks and optimize memory usage")
        
        if any(r['type'] == 'success_rate' for r in regressions):
            recommendations.append("Investigate test failures and error handling")
        
        critical_regressions = [r for r in regressions if r.get('severity') == 'critical']
        if critical_regressions:
            recommendations.append("URGENT: Address critical regressions immediately")
        
        return recommendations


# Convenience functions for easy integration
def track_performance(test_name: str, portfolio_size: int, n_simulations: int,
                     optimization_config: Dict[str, Any]):
    """Decorator for automatic performance tracking."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            tracker = PerformanceTracker()
            process = psutil.Process()
            
            start_time = time.time()
            start_memory = process.memory_info().rss / (1024 * 1024)
            
            try:
                result = func(*args, **kwargs)
                success = True
                error_message = None
            except Exception as e:
                result = None
                success = False
                error_message = str(e)
            
            end_time = time.time()
            end_memory = process.memory_info().rss / (1024 * 1024)
            
            metric = PerformanceMetric(
                timestamp=datetime.now().isoformat(),
                test_name=test_name,
                portfolio_size=portfolio_size,
                n_simulations=n_simulations,
                execution_time=end_time - start_time,
                memory_usage_mb=max(end_memory, start_memory),
                optimization_config=optimization_config,
                numerical_accuracy={},  # To be filled by test
                success=success,
                error_message=error_message
            )
            
            tracker.record_measurement(metric)
            tracker.save_session()
            
            if not success:
                raise Exception(error_message)
            
            return result
        return wrapper
    return decorator