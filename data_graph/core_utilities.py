"""
Core utilities for the Data Graph framework.
Contains shared utility classes and functions used across modules.
"""
import time
import warnings
from collections import deque, defaultdict

import numpy as np
from numba import njit, prange

class TimingStats:
    """Utility class to track timing statistics for different operations"""
    def __init__(self):
        self.stats = defaultdict(list)
        self.current_timers = {}
        
    def start(self, operation):
        """Start timing an operation"""
        self.current_timers[operation] = time.time()
        
    def end(self, operation):
        """End timing an operation and record the elapsed time"""
        if operation in self.current_timers:
            elapsed = time.time() - self.current_timers[operation]
            self.stats[operation].append(elapsed)
            del self.current_timers[operation]
            return elapsed
        return None
    
    def get_stats(self, as_dict=False):
        """Get statistics for all operations"""
        result = {}
        for op, times in self.stats.items():
            result[op] = {
                'count': len(times),
                'total': sum(times),
                'mean': sum(times) / len(times) if times else 0,
                'min': min(times) if times else 0,
                'max': max(times) if times else 0
            }
        
        if as_dict:
            return result
            
        # Format as a string for display
        lines = ["Detailed Timing Statistics:"]
        # Sort by total time in descending order
        for op, stats in sorted(result.items(), key=lambda x: x[1]['total'], reverse=True):
            lines.append(f"  • {op}: {stats['total']:.2f}s total, "
                        f"{stats['count']} calls, "
                        f"{stats['mean']:.2f}s avg/call")
        return "\n".join(lines)
    
    def get_operation_total(self, operation):
        """Get total time for a specific operation"""
        if operation in self.stats:
            return sum(self.stats[operation])
        return 0
    
    def report_nested_timing(self, parent_op, indent=2):
        """Report timing as percentage of parent operation"""
        if parent_op not in self.stats:
            return f"No data for parent operation: {parent_op}"
            
        parent_total = sum(self.stats[parent_op])
        if parent_total <= 0:
            return f"Parent operation {parent_op} has no timing data"
            
        lines = [f"Breakdown of {parent_op} ({parent_total:.2f}s total):"]
        
        # Find all operations that might be children (those containing parent_op)
        children = [(op, sum(times)) for op, times in self.stats.items() 
                   if parent_op in op and op != parent_op]
        
        # Sort by total time
        children.sort(key=lambda x: x[1], reverse=True)
        
        # Generate report
        indent_str = " " * indent
        for op, total in children:
            percentage = (total / parent_total) * 100
            short_name = op.replace(f"{parent_op}.", "")  # Remove parent prefix
            lines.append(f"{indent_str}• {short_name}: {total:.2f}s ({percentage:.1f}%)")
            
        other_time = parent_total - sum(t for _, t in children)
        if other_time > 0:
            percentage = (other_time / parent_total) * 100
            lines.append(f"{indent_str}• other operations: {other_time:.2f}s ({percentage:.1f}%)")
            
        return "\n".join(lines)

class BatchStats:
    """
    Efficiently tracks statistics in an online fashion with batch updates.
    Tracks count, mean, variance, min, and max.
    """
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0  # For variance calculation
        self.min_val = float('inf')
        self.max_val = float('-inf')
    
    def update_batch(self, batch_values):
        """
        Update statistics with a batch of values
        
        Parameters:
        -----------
        batch_values : numpy.ndarray
            Array of new values to include in statistics
        """
        if len(batch_values) == 0:
            return
            
        # Calculate batch statistics
        batch_count = len(batch_values)
        batch_mean = np.mean(batch_values)
        batch_min = np.min(batch_values)
        batch_max = np.max(batch_values)
        
        # For variance: sum of squared deviations in batch
        batch_M2 = np.sum((batch_values - batch_mean) ** 2)
        
        # Update global min/max
        self.min_val = min(self.min_val, batch_min)
        self.max_val = max(self.max_val, batch_max)
        
        # Special case for first batch
        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.M2 = batch_M2
            return
        
        # Update mean and variance using batch formula
        old_count = self.count
        new_count = old_count + batch_count
        
        # Calculate delta between batch mean and current mean
        delta = batch_mean - self.mean
        
        # Update mean: M' = (N*M + B*X)/(N+B)
        self.mean = (old_count * self.mean + batch_count * batch_mean) / new_count
        
        # Update M2 for variance calculation
        # M2' = M2 + batch_M2 + delta² * (old_count * batch_count / new_count)
        self.M2 += batch_M2 + delta**2 * (old_count * batch_count) / new_count
        
        # Update count
        self.count = new_count
    
    def get_variance(self):
        """Get the current variance"""
        if self.count < 2:
            return 0.0
        return self.M2 / self.count
    
    def get_std(self):
        """Get the current standard deviation"""
        return np.sqrt(self.get_variance())
    
    def get_stats(self):
        """Get all statistics as a dictionary"""
        return {
            'count': self.count,
            'mean': self.mean,
            'variance': self.get_variance(),
            'std': self.get_std(),
            'min': self.min_val if self.count > 0 else None,
            'max': self.max_val if self.count > 0 else None
        }

def make_parallel_batcher(dist_func):
    """
    Given a user-supplied @njit decorated dist_func(feats, i, j),
    return a njit-compiled, parallel batcher compute_batch(feats, idx_i, idx_j).
    """
    @njit(parallel=True)
    def compute_batch(feats: np.ndarray,
                      idx_i: np.ndarray,
                      idx_j: np.ndarray) -> np.ndarray:
        n = idx_i.shape[0]
        out = np.empty(n, dtype=np.float64)
        for k in prange(n):
            ii = idx_i[k]
            jj = idx_j[k]
            out[k] = dist_func(feats, ii, jj)
        return out
    return compute_batch

def find_knee_point(x, y, S=1.0, use_median_filter=True):
    """
    Kneedle algorithm for distance-based edge pruning.
    
    Parameters:
    -----------
    x : array-like
        X-coordinates (e.g., indices)
    y : array-like
        Y-coordinates (e.g., distances)
    S : float, default=1.0
        Sensitivity parameter
    use_median_filter : bool, default=True
        If True, only consider points above the median value
        
    Returns:
    --------
    int
        Index of the knee point
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    
    if len(x) <= 2:
        return 0
    
    # Optionally filter to only consider points above the median
    if use_median_filter:
        median_y = np.median(y)
        above_median = y >= median_y
        
        # If we have enough points above the median, filter
        if np.sum(above_median) > 2:
            x = x[above_median]
            y = y[above_median]
    
    # Normalize to [0,1]
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    
    x_norm = (x - x_min) / (x_max - x_min) if x_max > x_min else np.zeros_like(x)
    y_norm = (y - y_min) / (y_max - y_min) if y_max > y_min else np.zeros_like(y)
    
    # Calculate line between first and last point
    if x_norm[-1] == x_norm[0]:
        line_y = np.ones_like(y_norm) * y_norm[0]
    else:
        m = (y_norm[-1] - y_norm[0]) / (x_norm[-1] - x_norm[0])
        b = y_norm[0] - m * x_norm[0]
        line_y = m * x_norm + b
    
    # For concave up curves (like sorted distances), 
    # we want the point furthest BELOW the line
    diffs = line_y - y_norm
    
    # Apply sensitivity
    diffs = diffs * S
    
    # Find maximum difference (furthest below the line)
    knee_idx = np.argmax(diffs)
    
    # Map back to the original index space if we filtered
    if use_median_filter and np.sum(above_median) > 2:
        original_indices = np.where(above_median)[0]
        return original_indices[knee_idx]
    
    return knee_idx

def find_2hop_neighbors_efficient(graph, existing_edges):
    """
    Find 2-hop neighbors more efficiently without full matrix multiplication
    
    Parameters:
    -----------
    graph : scipy.sparse.csr_matrix
        Input graph
    existing_edges : set
        Set of existing edges as (i,j) where i < j
        
    Returns:
    --------
    list
        List of new 2-hop edges as (i,j) where i < j
    """
    n = graph.shape[0]
    new_edges = []
    
    # For each node i
    for i in range(n):
        # Get 1-hop neighbors of i
        neighbors_i = set(graph.indices[graph.indptr[i]:graph.indptr[i+1]])
        
        # For each neighbor j of i
        for j in neighbors_i:
            # Get 1-hop neighbors of j (2-hop from i)
            neighbors_j = graph.indices[graph.indptr[j]:graph.indptr[j+1]]
            
            # For each potential 2-hop neighbor k
            for k in neighbors_j:
                # Skip if k is i or already a 1-hop neighbor of i
                if k == i or k in neighbors_i:
                    continue
                
                # Ensure i < k for canonical edge representation
                edge = (i, k) if i < k else (k, i)
                
                # Check if this edge already exists
                if edge not in existing_edges:
                    new_edges.append(edge)
                    # Add to existing_edges to avoid duplicates
                    existing_edges.add(edge)
    
    return new_edges