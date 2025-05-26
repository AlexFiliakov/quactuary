"""
Optimized work distribution algorithms for parallel processing.

This module implements efficient work distribution strategies including
dynamic load balancing and adaptive chunk sizing.
"""

import time
import numpy as np
from typing import List, Tuple, Callable, Optional, Dict, Any
from dataclasses import dataclass
from collections import deque
import threading
import multiprocessing as mp
from queue import Queue, Empty


@dataclass
class WorkUnit:
    """Represents a unit of work."""
    id: int
    start_idx: int
    size: int
    priority: int = 0
    attempts: int = 0
    
    def __lt__(self, other):
        """For priority queue comparison."""
        return self.priority > other.priority  # Higher priority first


@dataclass
class WorkerStats:
    """Track worker performance statistics."""
    worker_id: int
    completed_tasks: int = 0
    total_time: float = 0.0
    failed_tasks: int = 0
    
    @property
    def avg_time_per_task(self) -> float:
        """Average time per completed task."""
        if self.completed_tasks == 0:
            return float('inf')
        return self.total_time / self.completed_tasks


class DynamicLoadBalancer:
    """
    Dynamic load balancer that adjusts work distribution based on worker performance.
    """
    
    def __init__(self, n_workers: int):
        self.n_workers = n_workers
        self.worker_stats = {i: WorkerStats(i) for i in range(n_workers)}
        self.lock = threading.Lock()
    
    def update_worker_stats(
        self, 
        worker_id: int, 
        task_time: float, 
        success: bool = True
    ):
        """Update worker performance statistics."""
        with self.lock:
            stats = self.worker_stats[worker_id]
            if success:
                stats.completed_tasks += 1
                stats.total_time += task_time
            else:
                stats.failed_tasks += 1
    
    def get_worker_weights(self) -> Dict[int, float]:
        """Get worker weights based on performance."""
        with self.lock:
            weights = {}
            
            # Calculate performance scores
            for worker_id, stats in self.worker_stats.items():
                if stats.completed_tasks == 0:
                    # New worker, give average weight
                    weights[worker_id] = 1.0
                else:
                    # Weight based on speed and reliability
                    speed_score = 1.0 / stats.avg_time_per_task
                    reliability = stats.completed_tasks / (stats.completed_tasks + stats.failed_tasks)
                    weights[worker_id] = speed_score * reliability
            
            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                for worker_id in weights:
                    weights[worker_id] /= total_weight
            
            return weights
    
    def suggest_chunk_size(self, worker_id: int, remaining_work: int) -> int:
        """Suggest chunk size for a worker based on performance."""
        weights = self.get_worker_weights()
        worker_weight = weights.get(worker_id, 1.0 / self.n_workers)
        
        # Base chunk size on worker's share of remaining work
        suggested_size = int(remaining_work * worker_weight)
        
        # Apply bounds
        min_size = max(10, remaining_work // (self.n_workers * 10))
        max_size = min(1000, remaining_work // 2)
        
        return max(min_size, min(suggested_size, max_size))


class AdaptiveChunkSizer:
    """
    Adaptive chunk sizing based on task completion times.
    """
    
    def __init__(self, initial_chunk_size: int = 100):
        self.chunk_size = initial_chunk_size
        self.completion_times = deque(maxlen=20)  # Keep last 20 measurements
        self.lock = threading.Lock()
        self.target_time = 1.0  # Target 1 second per chunk
    
    def record_completion(self, chunk_size: int, completion_time: float):
        """Record chunk completion time."""
        with self.lock:
            time_per_unit = completion_time / chunk_size
            self.completion_times.append(time_per_unit)
    
    def get_optimal_chunk_size(self, remaining_work: int) -> int:
        """Get optimal chunk size based on historical data."""
        with self.lock:
            if not self.completion_times:
                return self.chunk_size
            
            # Calculate average time per unit
            avg_time_per_unit = np.mean(self.completion_times)
            
            # Calculate chunk size to hit target time
            optimal_size = int(self.target_time / avg_time_per_unit)
            
            # Apply bounds based on remaining work
            min_size = max(10, remaining_work // 100)
            max_size = min(10000, remaining_work // 4)
            
            # Smooth changes to avoid oscillation
            new_size = int(0.7 * self.chunk_size + 0.3 * optimal_size)
            self.chunk_size = max(min_size, min(new_size, max_size))
            
            return self.chunk_size


class WorkStealingQueue:
    """
    Efficient work-stealing queue implementation.
    """
    
    def __init__(self, n_workers: int):
        self.n_workers = n_workers
        # Each worker has its own deque
        self.worker_queues = [deque() for _ in range(n_workers)]
        self.locks = [threading.Lock() for _ in range(n_workers)]
        self.total_work = 0
        self.completed_work = 0
        self.global_lock = threading.Lock()
    
    def add_work(self, work_unit: WorkUnit, worker_id: Optional[int] = None):
        """Add work to a specific worker or distribute."""
        if worker_id is not None:
            with self.locks[worker_id]:
                self.worker_queues[worker_id].append(work_unit)
        else:
            # Find worker with least work
            min_work = float('inf')
            target_worker = 0
            
            for i in range(self.n_workers):
                with self.locks[i]:
                    queue_size = len(self.worker_queues[i])
                    if queue_size < min_work:
                        min_work = queue_size
                        target_worker = i
            
            with self.locks[target_worker]:
                self.worker_queues[target_worker].append(work_unit)
        
        with self.global_lock:
            self.total_work += work_unit.size
    
    def get_work(self, worker_id: int) -> Optional[WorkUnit]:
        """Get work for a worker, stealing if necessary."""
        # Try own queue first
        with self.locks[worker_id]:
            if self.worker_queues[worker_id]:
                return self.worker_queues[worker_id].popleft()
        
        # Try stealing from others
        for attempt in range(2):  # Two steal attempts
            # Random victim selection
            victim_id = np.random.randint(0, self.n_workers)
            if victim_id == worker_id:
                continue
            
            with self.locks[victim_id]:
                victim_queue = self.worker_queues[victim_id]
                if len(victim_queue) > 1:  # Only steal if victim has multiple items
                    # Steal from the back (newest work)
                    return victim_queue.pop()
        
        return None
    
    def mark_completed(self, work_unit: WorkUnit):
        """Mark work as completed."""
        with self.global_lock:
            self.completed_work += work_unit.size
    
    @property
    def progress(self) -> float:
        """Get completion progress."""
        with self.global_lock:
            if self.total_work == 0:
                return 1.0
            return self.completed_work / self.total_work


def create_balanced_chunks(
    total_work: int,
    n_workers: int,
    min_chunk_size: int = 10,
    heterogeneous_factor: float = 0.0
) -> List[Tuple[int, int]]:
    """
    Create balanced work chunks with optional heterogeneity.
    
    Args:
        total_work: Total amount of work
        n_workers: Number of workers
        min_chunk_size: Minimum chunk size
        heterogeneous_factor: 0.0 for uniform, 1.0 for highly varied chunks
        
    Returns:
        List of (start_idx, size) tuples
    """
    chunks = []
    
    if heterogeneous_factor == 0.0:
        # Uniform distribution
        base_chunk_size = max(min_chunk_size, total_work // (n_workers * 10))
        
        current_idx = 0
        while current_idx < total_work:
            chunk_size = min(base_chunk_size, total_work - current_idx)
            chunks.append((current_idx, chunk_size))
            current_idx += chunk_size
    else:
        # Heterogeneous distribution
        remaining = total_work
        current_idx = 0
        
        while remaining > 0:
            # Vary chunk size based on heterogeneous factor
            variation = 1.0 + heterogeneous_factor * (np.random.rand() - 0.5)
            base_size = max(min_chunk_size, remaining // max(1, n_workers * 5))
            chunk_size = int(base_size * variation)
            chunk_size = max(min_chunk_size, min(chunk_size, remaining))
            
            chunks.append((current_idx, chunk_size))
            current_idx += chunk_size
            remaining -= chunk_size
    
    return chunks


def optimize_chunk_assignment(
    chunks: List[Tuple[int, int]],
    worker_weights: Dict[int, float],
    n_workers: int
) -> Dict[int, List[Tuple[int, int]]]:
    """
    Optimize chunk assignment to workers based on their weights.
    
    Args:
        chunks: List of (start_idx, size) tuples
        worker_weights: Worker performance weights
        n_workers: Number of workers
        
    Returns:
        Dictionary mapping worker_id to assigned chunks
    """
    # Sort chunks by size (largest first)
    sorted_chunks = sorted(chunks, key=lambda x: x[1], reverse=True)
    
    # Initialize worker loads
    worker_loads = {i: 0.0 for i in range(n_workers)}
    worker_chunks = {i: [] for i in range(n_workers)}
    
    # Greedy assignment: assign each chunk to least loaded worker
    for chunk in sorted_chunks:
        start_idx, size = chunk
        
        # Find worker with minimum load/weight ratio
        min_ratio = float('inf')
        best_worker = 0
        
        for worker_id in range(n_workers):
            weight = worker_weights.get(worker_id, 1.0 / n_workers)
            if weight > 0:
                ratio = worker_loads[worker_id] / weight
                if ratio < min_ratio:
                    min_ratio = ratio
                    best_worker = worker_id
        
        # Assign chunk
        worker_chunks[best_worker].append(chunk)
        worker_loads[best_worker] += size
    
    return worker_chunks


class HeterogeneousWorkloadOptimizer:
    """
    Optimizer for heterogeneous workloads where tasks have varying complexity.
    """
    
    def __init__(self, n_workers: int):
        self.n_workers = n_workers
        self.task_complexity_history = deque(maxlen=100)
        self.complexity_estimator = None
    
    def record_task_complexity(
        self, 
        task_features: Dict[str, float], 
        actual_time: float
    ):
        """Record task complexity for learning."""
        self.task_complexity_history.append((task_features, actual_time))
        
        # Simple linear regression for complexity estimation
        if len(self.task_complexity_history) >= 20:
            self._update_complexity_estimator()
    
    def _update_complexity_estimator(self):
        """Update complexity estimation model."""
        # Simple implementation - could be replaced with more sophisticated ML
        features = []
        times = []
        
        for feat_dict, time in self.task_complexity_history:
            # Convert features to vector
            feat_vec = [feat_dict.get('size', 0), feat_dict.get('complexity', 1)]
            features.append(feat_vec)
            times.append(time)
        
        # Store average time per unit size as simple estimator
        features = np.array(features)
        times = np.array(times)
        
        if features[:, 0].sum() > 0:
            self.complexity_estimator = times.sum() / features[:, 0].sum()
    
    def estimate_task_time(self, task_features: Dict[str, float]) -> float:
        """Estimate task completion time."""
        if self.complexity_estimator is None:
            # Default estimate
            return task_features.get('size', 100) * 0.001
        
        return task_features.get('size', 100) * self.complexity_estimator
    
    def optimize_distribution(
        self, 
        tasks: List[Dict[str, Any]]
    ) -> Dict[int, List[int]]:
        """
        Optimize task distribution for heterogeneous workload.
        
        Returns mapping of worker_id to task indices.
        """
        # Estimate task times
        task_times = []
        for task in tasks:
            est_time = self.estimate_task_time(task.get('features', {}))
            task_times.append(est_time)
        
        # Sort tasks by estimated time (longest first)
        task_indices = sorted(range(len(tasks)), key=lambda i: task_times[i], reverse=True)
        
        # Distribute using greedy algorithm
        worker_loads = [0.0] * self.n_workers
        worker_tasks = {i: [] for i in range(self.n_workers)}
        
        for task_idx in task_indices:
            # Assign to least loaded worker
            min_load_worker = np.argmin(worker_loads)
            worker_tasks[min_load_worker].append(task_idx)
            worker_loads[min_load_worker] += task_times[task_idx]
        
        return worker_tasks