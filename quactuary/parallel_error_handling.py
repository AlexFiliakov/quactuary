"""
Comprehensive error handling and recovery for parallel processing.

This module provides robust error handling, automatic recovery mechanisms,
and detailed diagnostics for parallel computations.
"""

import sys
import time
import logging
import traceback
import warnings
import numpy as np
from typing import Optional, List, Callable, Tuple, Any, Dict, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from collections import defaultdict
import json
import pickle


# Configure logging
logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Types of errors that can occur in parallel processing."""
    SERIALIZATION = "serialization"
    TIMEOUT = "timeout"
    MEMORY = "memory"
    COMPUTATION = "computation"
    WORKER_CRASH = "worker_crash"
    COMMUNICATION = "communication"
    UNKNOWN = "unknown"


@dataclass
class ErrorInfo:
    """Detailed error information."""
    error_type: ErrorType
    error_class: str
    error_message: str
    traceback: str
    timestamp: datetime
    worker_id: Optional[str] = None
    task_id: Optional[str] = None
    retry_count: int = 0
    additional_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'error_type': self.error_type.value,
            'error_class': self.error_class,
            'error_message': self.error_message,
            'traceback': self.traceback,
            'timestamp': self.timestamp.isoformat(),
            'worker_id': self.worker_id,
            'task_id': self.task_id,
            'retry_count': self.retry_count,
            'additional_info': self.additional_info
        }


class ErrorClassifier:
    """Classify errors to determine appropriate recovery strategy."""
    
    @staticmethod
    def classify_error(exception: Exception) -> ErrorType:
        """Classify an exception into an error type."""
        error_name = type(exception).__name__
        error_msg = str(exception).lower()
        
        # Serialization errors
        if any(x in error_name for x in ['Pickle', 'AttributeError']):
            if 'pickle' in error_msg or 'serialize' in error_msg:
                return ErrorType.SERIALIZATION
        
        # Timeout errors
        if 'timeout' in error_name or 'timeout' in error_msg:
            return ErrorType.TIMEOUT
        
        # Memory errors
        if 'memory' in error_name or any(x in error_msg for x in ['memory', 'ram', 'oom']):
            return ErrorType.MEMORY
        
        # Worker crashes
        if any(x in error_msg for x in ['worker', 'process', 'terminated', 'killed']):
            return ErrorType.WORKER_CRASH
        
        # Communication errors
        if any(x in error_msg for x in ['broken pipe', 'connection', 'communication']):
            return ErrorType.COMMUNICATION
        
        # Computation errors
        if any(x in error_name for x in ['ValueError', 'RuntimeError', 'ArithmeticError']):
            return ErrorType.COMPUTATION
        
        return ErrorType.UNKNOWN


class RecoveryStrategy:
    """Base class for recovery strategies."""
    
    def can_recover(self, error_info: ErrorInfo) -> bool:
        """Check if this strategy can handle the error."""
        raise NotImplementedError
    
    def recover(self, error_info: ErrorInfo, task_data: Any) -> Tuple[bool, Any]:
        """Attempt to recover from the error."""
        raise NotImplementedError


class RetryStrategy(RecoveryStrategy):
    """Simple retry strategy with exponential backoff."""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 0.1):
        self.max_retries = max_retries
        self.base_delay = base_delay
    
    def can_recover(self, error_info: ErrorInfo) -> bool:
        """Can retry most errors except serialization."""
        return (
            error_info.error_type != ErrorType.SERIALIZATION and
            error_info.retry_count < self.max_retries
        )
    
    def recover(self, error_info: ErrorInfo, task_data: Any) -> Tuple[bool, Any]:
        """Retry with exponential backoff."""
        delay = self.base_delay * (2 ** error_info.retry_count)
        time.sleep(delay)
        return True, task_data  # Signal to retry


class SerializationFixStrategy(RecoveryStrategy):
    """Fix serialization issues by converting to simpler types."""
    
    def can_recover(self, error_info: ErrorInfo) -> bool:
        """Can handle serialization errors."""
        return error_info.error_type == ErrorType.SERIALIZATION
    
    def recover(self, error_info: ErrorInfo, task_data: Any) -> Tuple[bool, Any]:
        """Try to make data serializable."""
        try:
            # Try cloudpickle if available
            import cloudpickle
            cloudpickle.dumps(task_data)
            return True, task_data
        except:
            pass
        
        # Try to simplify the data
        if isinstance(task_data, dict):
            # Remove non-serializable items
            simplified = {}
            for k, v in task_data.items():
                try:
                    pickle.dumps(v)
                    simplified[k] = v
                except:
                    logger.warning(f"Removing non-serializable key: {k}")
            return True, simplified
        
        # Last resort: convert to string representation
        return True, str(task_data)


class MemoryReductionStrategy(RecoveryStrategy):
    """Reduce memory usage for memory-related errors."""
    
    def can_recover(self, error_info: ErrorInfo) -> bool:
        """Can handle memory errors."""
        return error_info.error_type == ErrorType.MEMORY
    
    def recover(self, error_info: ErrorInfo, task_data: Any) -> Tuple[bool, Any]:
        """Reduce task size to lower memory usage."""
        if isinstance(task_data, dict) and 'chunk_size' in task_data:
            # Reduce chunk size by half
            task_data['chunk_size'] = max(10, task_data['chunk_size'] // 2)
            return True, task_data
        
        return False, "Cannot reduce memory usage"


class FallbackStrategy(RecoveryStrategy):
    """Fallback to alternative computation method."""
    
    def __init__(self, fallback_func: Optional[Callable] = None):
        self.fallback_func = fallback_func
    
    def can_recover(self, error_info: ErrorInfo) -> bool:
        """Can handle any error if fallback is available."""
        return self.fallback_func is not None
    
    def recover(self, error_info: ErrorInfo, task_data: Any) -> Tuple[bool, Any]:
        """Use fallback function."""
        if self.fallback_func:
            try:
                result = self.fallback_func(task_data)
                return True, result
            except Exception as e:
                logger.error(f"Fallback also failed: {e}")
        
        return False, "Fallback failed"


class ErrorRecoveryManager:
    """
    Manages error recovery strategies and tracks error patterns.
    """
    
    def __init__(self):
        self.strategies: List[RecoveryStrategy] = [
            RetryStrategy(),
            SerializationFixStrategy(),
            MemoryReductionStrategy(),
            FallbackStrategy()
        ]
        self.error_history: List[ErrorInfo] = []
        self.error_counts: Dict[ErrorType, int] = defaultdict(int)
        self.recovery_success: Dict[ErrorType, int] = defaultdict(int)
        self.recovery_failure: Dict[ErrorType, int] = defaultdict(int)
    
    def add_strategy(self, strategy: RecoveryStrategy):
        """Add a custom recovery strategy."""
        self.strategies.append(strategy)
    
    def record_error(
        self,
        exception: Exception,
        worker_id: Optional[str] = None,
        task_id: Optional[str] = None,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> ErrorInfo:
        """Record an error and classify it."""
        error_type = ErrorClassifier.classify_error(exception)
        
        error_info = ErrorInfo(
            error_type=error_type,
            error_class=type(exception).__name__,
            error_message=str(exception),
            traceback=traceback.format_exc(),
            timestamp=datetime.now(),
            worker_id=worker_id,
            task_id=task_id,
            additional_info=additional_info or {}
        )
        
        self.error_history.append(error_info)
        self.error_counts[error_type] += 1
        
        # Keep history size manageable
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-500:]
        
        return error_info
    
    def attempt_recovery(
        self,
        error_info: ErrorInfo,
        task_data: Any
    ) -> Tuple[bool, Any, Optional[str]]:
        """
        Attempt to recover from an error.
        
        Returns:
            - success: Whether recovery was successful
            - result: Recovered data or error message
            - strategy_name: Name of strategy used
        """
        for strategy in self.strategies:
            if strategy.can_recover(error_info):
                try:
                    success, result = strategy.recover(error_info, task_data)
                    
                    if success:
                        self.recovery_success[error_info.error_type] += 1
                        return True, result, type(strategy).__name__
                    
                except Exception as e:
                    logger.error(f"Recovery strategy {type(strategy).__name__} failed: {e}")
        
        self.recovery_failure[error_info.error_type] += 1
        return False, f"No recovery strategy available for {error_info.error_type.value}", None
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of errors and recovery attempts."""
        total_errors = sum(self.error_counts.values())
        total_recoveries = sum(self.recovery_success.values())
        total_failures = sum(self.recovery_failure.values())
        
        summary = {
            'total_errors': total_errors,
            'total_recoveries': total_recoveries,
            'total_failures': total_failures,
            'recovery_rate': total_recoveries / total_errors if total_errors > 0 else 0,
            'error_breakdown': dict(self.error_counts),
            'recovery_breakdown': dict(self.recovery_success),
            'recent_errors': [e.to_dict() for e in self.error_history[-10:]]
        }
        
        return summary
    
    def should_abort(self, error_threshold: float = 0.5) -> bool:
        """
        Check if error rate is too high and processing should abort.
        
        Args:
            error_threshold: Maximum acceptable failure rate
            
        Returns:
            True if error rate exceeds threshold
        """
        if len(self.error_history) < 10:
            return False  # Not enough data
        
        recent_errors = self.error_history[-20:]
        unrecoverable = sum(
            1 for e in recent_errors 
            if self.recovery_failure[e.error_type] > self.recovery_success[e.error_type]
        )
        
        failure_rate = unrecoverable / len(recent_errors)
        return failure_rate > error_threshold


class DiagnosticLogger:
    """
    Detailed diagnostic logging for debugging parallel processing issues.
    """
    
    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file
        self.events: List[Dict[str, Any]] = []
        
        # Set up file logging if specified
        if log_file:
            handler = logging.FileHandler(log_file)
            handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
    
    def log_event(
        self,
        event_type: str,
        worker_id: Optional[str] = None,
        task_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log a diagnostic event."""
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'worker_id': worker_id,
            'task_id': task_id,
            'details': details or {}
        }
        
        self.events.append(event)
        
        # Log to file
        logger.debug(f"Event: {json.dumps(event)}")
        
        # Keep event history manageable
        if len(self.events) > 10000:
            self.events = self.events[-5000:]
    
    def get_worker_timeline(self, worker_id: str) -> List[Dict[str, Any]]:
        """Get timeline of events for a specific worker."""
        return [e for e in self.events if e['worker_id'] == worker_id]
    
    def get_task_timeline(self, task_id: str) -> List[Dict[str, Any]]:
        """Get timeline of events for a specific task."""
        return [e for e in self.events if e['task_id'] == task_id]
    
    def export_diagnostics(self, output_file: str):
        """Export diagnostic data for analysis."""
        diagnostics = {
            'events': self.events,
            'worker_summaries': self._get_worker_summaries(),
            'task_summaries': self._get_task_summaries(),
            'error_patterns': self._analyze_error_patterns()
        }
        
        with open(output_file, 'w') as f:
            json.dump(diagnostics, f, indent=2)
    
    def _get_worker_summaries(self) -> Dict[str, Any]:
        """Summarize worker performance."""
        worker_stats = defaultdict(lambda: {
            'tasks_completed': 0,
            'tasks_failed': 0,
            'total_time': 0,
            'errors': []
        })
        
        for event in self.events:
            worker_id = event.get('worker_id')
            if not worker_id:
                continue
            
            if event['event_type'] == 'task_completed':
                worker_stats[worker_id]['tasks_completed'] += 1
                worker_stats[worker_id]['total_time'] += event['details'].get('duration', 0)
            elif event['event_type'] == 'task_failed':
                worker_stats[worker_id]['tasks_failed'] += 1
                worker_stats[worker_id]['errors'].append(event['details'].get('error', 'Unknown'))
        
        return dict(worker_stats)
    
    def _get_task_summaries(self) -> Dict[str, Any]:
        """Summarize task execution."""
        task_stats = defaultdict(lambda: {
            'attempts': 0,
            'success': False,
            'total_time': 0,
            'errors': []
        })
        
        for event in self.events:
            task_id = event.get('task_id')
            if not task_id:
                continue
            
            if event['event_type'] == 'task_started':
                task_stats[task_id]['attempts'] += 1
            elif event['event_type'] == 'task_completed':
                task_stats[task_id]['success'] = True
                task_stats[task_id]['total_time'] += event['details'].get('duration', 0)
            elif event['event_type'] == 'task_failed':
                task_stats[task_id]['errors'].append(event['details'].get('error', 'Unknown'))
        
        return dict(task_stats)
    
    def _analyze_error_patterns(self) -> Dict[str, Any]:
        """Analyze error patterns for insights."""
        error_events = [e for e in self.events if e['event_type'] == 'task_failed']
        
        if not error_events:
            return {}
        
        # Time-based analysis
        error_times = [datetime.fromisoformat(e['timestamp']) for e in error_events]
        if len(error_times) > 1:
            time_diffs = [(error_times[i+1] - error_times[i]).total_seconds() 
                         for i in range(len(error_times)-1)]
            avg_time_between_errors = np.mean(time_diffs)
        else:
            avg_time_between_errors = None
        
        # Error clustering
        error_types = [e['details'].get('error_type', 'unknown') for e in error_events]
        error_type_counts = defaultdict(int)
        for et in error_types:
            error_type_counts[et] += 1
        
        return {
            'total_errors': len(error_events),
            'avg_time_between_errors': avg_time_between_errors,
            'error_type_distribution': dict(error_type_counts),
            'error_burst_detected': avg_time_between_errors < 1.0 if avg_time_between_errors else False
        }


def create_resilient_wrapper(
    func: Callable,
    error_manager: ErrorRecoveryManager,
    diagnostic_logger: Optional[DiagnosticLogger] = None
) -> Callable:
    """
    Create a resilient wrapper around a function with error handling.
    
    Args:
        func: Function to wrap
        error_manager: Error recovery manager
        diagnostic_logger: Optional diagnostic logger
        
    Returns:
        Wrapped function with error handling
    """
    def wrapped(*args, **kwargs):
        task_id = kwargs.get('task_id', f"task_{time.time()}")
        worker_id = kwargs.get('worker_id', 'unknown')
        
        if diagnostic_logger:
            diagnostic_logger.log_event(
                'task_started',
                worker_id=worker_id,
                task_id=task_id,
                details={'args_shape': str(np.array(args).shape) if args else 'none'}
            )
        
        start_time = time.time()
        retry_count = 0
        last_error = None
        
        while retry_count <= 3:  # Maximum retry attempts
            try:
                result = func(*args, **kwargs)
                
                if diagnostic_logger:
                    diagnostic_logger.log_event(
                        'task_completed',
                        worker_id=worker_id,
                        task_id=task_id,
                        details={
                            'duration': time.time() - start_time,
                            'retry_count': retry_count
                        }
                    )
                
                return result
                
            except Exception as e:
                last_error = e
                error_info = error_manager.record_error(
                    e,
                    worker_id=worker_id,
                    task_id=task_id,
                    additional_info={'retry_count': retry_count}
                )
                error_info.retry_count = retry_count
                
                if diagnostic_logger:
                    diagnostic_logger.log_event(
                        'task_failed',
                        worker_id=worker_id,
                        task_id=task_id,
                        details={
                            'error': str(e),
                            'error_type': error_info.error_type.value,
                            'retry_count': retry_count
                        }
                    )
                
                # Attempt recovery
                task_data = {'args': args, 'kwargs': kwargs}
                success, recovered_data, strategy = error_manager.attempt_recovery(
                    error_info, task_data
                )
                
                if success and strategy == 'RetryStrategy':
                    retry_count += 1
                    time.sleep(0.1 * retry_count)  # Exponential backoff
                    continue
                elif success:
                    # Use recovered data
                    if isinstance(recovered_data, dict):
                        args = recovered_data.get('args', args)
                        kwargs = recovered_data.get('kwargs', kwargs)
                    continue
                else:
                    # Recovery failed
                    break
        
        # All retries exhausted
        raise last_error
    
    return wrapped