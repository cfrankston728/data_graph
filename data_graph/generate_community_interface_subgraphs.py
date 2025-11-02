#!/usr/bin/env python3
"""
Optimized Leiden Community Interface Detection Pipeline

Features:
1. High-performance core with Numba acceleration
2. Memory-efficient data structures
3. Cached computations
4. Streamlined I/O
5. Advanced analytics and visualization
6. Parallel processing support
7. Comprehensive timing statistics

This implementation combines the performance optimizations of the streamlined version
with the analytical capabilities of the more complex version.
"""

import sys
import os
import time
import json
import pickle
import multiprocessing as mp
from functools import partial, lru_cache
from collections import defaultdict, Counter
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any, Set

import numpy as np
import pandas as pd
import scipy.sparse as sp
#import igraph as ig # import will be lazy for _create_igraph method
#from igraph import sparse_matrix as _sps
#import leidenalg
from sklearn.metrics import normalized_mutual_info_score
import matplotlib.pyplot as plt
import seaborn as sns
try:
    from tqdm.auto import tqdm
except ImportError:
    # Simple tqdm fallback if not installed
    def tqdm(iterable, **kwargs):
        return iterable
import click
import numba as nb
from numba import njit, prange
from contextlib import contextmanager

from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil, math
import gc
# ============================================================================
# PERFORMANCE MONITORING
# ============================================================================

class PerformanceMonitor:
    """Performance monitoring with minimal overhead."""
    
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.reset()
    
    def reset(self):
        """Reset all timing statistics."""
        self.timing_stats = defaultdict(float)
        self.timing_counts = defaultdict(int)
        self.start_times = {}
        self.total_start_time = time.time()
    
    @contextmanager
    def timed_operation(self, operation_name, verbose=False):
        """Context manager for timing operations with proper nesting."""
        if not self.enabled:
            yield
            return
        
        start_time = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            self.timing_stats[operation_name] += elapsed
            self.timing_counts[operation_name] += 1
            if verbose:
                print(f"  [{operation_name}] completed in {elapsed:.2f} seconds")
    
    def print_timing_summary(self):
        """Print a summary of timing statistics."""
        if not self.enabled:
            return
            
        total_time = time.time() - self.total_start_time
        
        print("\n======== TIMING SUMMARY ========")
        print(f"Total execution time: {total_time:.2f} seconds")
        print("\nBreakdown by operation:")
        
        # Sort operations by time spent (descending)
        sorted_ops = sorted(self.timing_stats.items(), key=lambda x: x[1], reverse=True)
        
        for operation, elapsed in sorted_ops:
            percentage = (elapsed / total_time) * 100
            count = self.timing_counts[operation]
            avg_time = elapsed / count if count > 0 else 0
            print(f"  {operation:<30} {elapsed:10.2f}s ({percentage:6.2f}%)  |  {count} calls, avg {avg_time:.4f}s per call")
        
        print("================================")

# Global performance monitor
perf_monitor = PerformanceMonitor(enabled=True)

# ============================================================================
# NUMBA OPTIMIZED FUNCTIONS
# ============================================================================

import numba as nb
import numpy as np
import scipy.sparse as sp

@nb.njit(cache=True)
def _build_csr_arrays_from_pairs(a, b, w, n):
    # a<b, unique; all dtypes already int32/float32
    deg = np.zeros(n, np.int32)
    m = a.size
    for i in range(m):
        deg[a[i]] += 1
        deg[b[i]] += 1

    indptr = np.empty(n + 1, np.int32)
    indptr[0] = 0
    for i in range(n):
        indptr[i + 1] = indptr[i] + deg[i]

    nnz = int(indptr[-1])
    indices = np.empty(nnz, np.int32)
    data    = np.empty(nnz, np.float32)

    cursor = indptr[:-1].copy()
    for i in range(m):
        u = a[i]; v = b[i]; wt = w[i]
        pu = cursor[u]; indices[pu] = v; data[pu] = wt; cursor[u] = pu + 1
        pv = cursor[v]; indices[pv] = u; data[pv] = wt; cursor[v] = pv + 1

    return indptr, indices, data

def _csr_from_undirected_edges(a, b, w, n_nodes):
    # Ensure dtypes up front (avoids slow implicit casts)
    a = a.astype(np.int32, copy=False)
    b = b.astype(np.int32, copy=False)
    w = w.astype(np.float32, copy=False)
    n = int(n_nodes)

    indptr, indices, data = _build_csr_arrays_from_pairs(a, b, w, n)

    # (Optional but can help Leiden): sort columns within each row.
    # Degrees are small (~k≈50–70), so an insertion-sort per row is cheap.
    _row_sort_inplace(indptr, indices, data)

    return sp.csr_matrix((data, indices, indptr), shape=(n, n), copy=False)

@nb.njit(cache=True)
def _row_sort_inplace(indptr, indices, data):
    for i in range(indptr.size - 1):
        s = indptr[i]; e = indptr[i+1]
        # insertion sort is OK for small degrees
        for j in range(s + 1, e):
            key_idx = indices[j]
            key_val = data[j]
            k = j - 1
            while k >= s and indices[k] > key_idx:
                indices[k + 1] = indices[k]
                data[k + 1]    = data[k]
                k -= 1
            indices[k + 1] = key_idx
            data[k + 1]    = key_val



@nb.njit(cache=True)  # NOTE: no parallel=True
def mutual_nn_coarsening_directed(sources, targets, weights, n_nodes):
    best_neighbor = np.full(n_nodes, -1, dtype=np.int64)
    best_weight   = np.full(n_nodes, -np.inf)
    for e in range(sources.shape[0]):
        u = sources[e]; v = targets[e]; w = weights[e]
        if w > best_weight[u]:
            best_weight[u] = w; best_neighbor[u] = v
        if w > best_weight[v]:
            best_weight[v] = w; best_neighbor[v] = u

    meta_id = np.full(n_nodes, -1, dtype=np.int64)
    next_id = 0
    for i in range(n_nodes):
        j = best_neighbor[i]
        if j > i and j >= 0 and best_neighbor[j] == i and meta_id[i] == -1:
            meta_id[i] = next_id; meta_id[j] = next_id; next_id += 1
    for i in range(n_nodes):
        if meta_id[i] == -1:
            meta_id[i] = next_id; next_id += 1
    return meta_id, next_id

@nb.njit(cache=True)
def _canon_pair(u, v):
    return (u, v) if u < v else (v, u)

@nb.njit(cache=True)
def aggregate_undirected_edges_with_dist(s, t, w, d):
    """
    Take directed edges (s,t) with weights w and distances d.
    Return UNIQUE undirected pairs a<b with:
      - weight = SUM of weights over both directions
      - distance = MIN of distances over both directions
    """
    n = s.shape[0]
    a = np.empty(n, dtype=np.int64)
    b = np.empty(n, dtype=np.int64)
    for i in range(n):
        u = s[i]; v = t[i]
        if u < v:
            a[i] = u; b[i] = v
        else:
            a[i] = v; b[i] = u

    # sort by (a,b)
    # radix-ish: pack into 64 bits (assumes node ids < 2**31)
    keys = (a.astype(np.int64) << 32) | b.astype(np.int64)
    order = np.argsort(keys)

    a = a[order]; b = b[order]
    w = w[order]; d = d[order]

    # single pass to combine groups
    # worst-case len out = n
    out_a = np.empty(n, dtype=np.int64)
    out_b = np.empty(n, dtype=np.int64)
    out_w = np.empty(n, dtype=w.dtype)
    out_d = np.empty(n, dtype=d.dtype)

    out = 0
    i = 0
    while i < n:
        ua = a[i]; ub = b[i]
        sumw = w[i]
        mind = d[i]
        i += 1
        while i < n and a[i] == ua and b[i] == ub:
            sumw += w[i]
            if d[i] < mind:
                mind = d[i]
            i += 1
        out_a[out] = ua
        out_b[out] = ub
        out_w[out] = sumw
        out_d[out] = mind
        out += 1

    return out_a[:out], out_b[:out], out_w[:out], out_d[:out]


@nb.njit(cache=True)
def dedup_undirected_maxw_mind(sel_u, sel_v, sel_w, sel_d):
    """
    Deduplicate a set of *undirected* selections (possibly both u->v and v->u).
    Returns unique a<b with:
      - weight = MAX of weights from either side
      - distance = MIN of distances from either side
    """
    n = sel_u.shape[0]
    a = np.empty(n, dtype=np.int64)
    b = np.empty(n, dtype=np.int64)
    for i in range(n):
        u = sel_u[i]; v = sel_v[i]
        if u < v:
            a[i] = u; b[i] = v
        else:
            a[i] = v; b[i] = u

    keys = (a.astype(np.int64) << 32) | b.astype(np.int64)
    order = np.argsort(keys)
    a = a[order]; b = b[order]
    w = sel_w[order]; d = sel_d[order]

    out_a = np.empty(n, dtype=np.int64)
    out_b = np.empty(n, dtype=np.int64)
    out_w = np.empty(n, dtype=w.dtype)
    out_d = np.empty(n, dtype=d.dtype)

    out = 0
    i = 0
    while i < n:
        ua = a[i]; ub = b[i]
        maxw = w[i]
        mind = d[i]
        i += 1
        while i < n and a[i] == ua and b[i] == ub:
            if w[i] > maxw:
                maxw = w[i]
            if d[i] < mind:
                mind = d[i]
            i += 1
        out_a[out] = ua
        out_b[out] = ub
        out_w[out] = maxw
        out_d[out] = mind
        out += 1

    return out_a[:out], out_b[:out], out_w[:out], out_d[:out]


@nb.njit(cache=True)
def sparsify_knn_undirected(a, b, w, d, n_nodes, k):
    """
    Top-k per node on an undirected graph given unique pairs (a<b).
    Returns unique pairs (a'<b') chosen by OR of endpoint selections.
    """
    m = a.shape[0]
    # degrees
    deg = np.zeros(n_nodes, dtype=np.int64)
    for i in range(m):
        deg[a[i]] += 1
        deg[b[i]] += 1

    # row pointers for local 2m storage
    ptr = np.empty(n_nodes + 1, dtype=np.int64)
    ptr[0] = 0
    for i in range(n_nodes):
        ptr[i+1] = ptr[i] + deg[i]
    total = ptr[-1]  # 2m

    nbr  = np.empty(total, dtype=np.int64)
    wts  = np.empty(total, dtype=np.float32)
    dst  = np.empty(total, dtype=np.float32)

    fill = ptr[:-1].copy()
    for i in range(m):
        u = a[i]; v = b[i]; wt = w[i]; di = d[i]
        pu = fill[u]; nbr[pu] = v; wts[pu] = wt; dst[pu] = di; fill[u] = pu + 1
        pv = fill[v]; nbr[pv] = u; wts[pv] = wt; dst[pv] = di; fill[v] = pv + 1

    # pre-count selections
    sel_count = 0
    for u in range(n_nodes):
        du = ptr[u+1] - ptr[u]
        if du > 0:
            sel_count += k if du > k else du

    cand_u = np.empty(sel_count, dtype=np.int64)
    cand_v = np.empty(sel_count, dtype=np.int64)
    cand_w = np.empty(sel_count, dtype=np.float32)
    cand_d = np.empty(sel_count, dtype=np.float32)

    out = 0
    for u in range(n_nodes):
        start = ptr[u]; end = ptr[u+1]; du = end - start
        if du == 0:
            continue

        if du <= k:
            for j in range(du):
                v = nbr[start + j]
                cand_u[out] = u; cand_v[out] = v
                cand_w[out] = wts[start + j]; cand_d[out] = dst[start + j]
                out += 1
        else:
            # simple O(du*k) selector; after coarsening du is modest
            tmp_w = wts[start:end].copy()
            tmp_i = np.empty(k, dtype=np.int64)
            for t in range(k):
                mi = 0; mw = tmp_w[0]
                for r in range(1, du):
                    if tmp_w[r] > mw:
                        mi = r; mw = tmp_w[r]
                tmp_i[t] = mi; tmp_w[mi] = np.float32(-1e38)
            for t in range(k):
                j = tmp_i[t]
                v = nbr[start + j]
                cand_u[out] = u; cand_v[out] = v
                cand_w[out] = wts[start + j]; cand_d[out] = dst[start + j]
                out += 1

    # OR-of-endpoints, dedup to unique (a<b)
    return dedup_undirected_maxw_mind(cand_u[:out], cand_v[:out], cand_w[:out], cand_d[:out])

# ------------------------------------------------------------------------
# NUMBA KERNEL
# ------------------------------------------------------------------------
@nb.njit(parallel=True)
def _accumulate_stats(
    sources:    np.ndarray,  # full_sources
    targets:    np.ndarray,  # full_targets
    distances:  np.ndarray,  # full_distances
    sims:       np.ndarray,  # similarities
    cidx:       np.ndarray,  # cluster_indices per node
    n_clusters: int
):
    # allocate accumulators
    vol       = np.zeros(n_clusters, dtype=np.float64)
    cut       = np.zeros(n_clusters, dtype=np.float64)
    int_cnt   = np.zeros(n_clusters, dtype=np.int64)
    ext_cnt   = np.zeros(n_clusters, dtype=np.int64)
    sum_d     = np.zeros(n_clusters, dtype=np.float64)
    sumsq_d   = np.zeros(n_clusters, dtype=np.float64)

    for e in nb.prange(sources.shape[0]):
        u = sources[e]
        v = targets[e]
        du = cidx[u]
        dv = cidx[v]
        w  = sims[e]
        d  = distances[e]

        # volume (weighted degree)
        vol[du] += w
        vol[dv] += w

        if du == dv:
            # internal edge
            int_cnt[du]   += 1
            sum_d[du]     += d
            sumsq_d[du]   += d * d
        else:
            # cross‐cluster edge (cut)
            cut[du]       += w
            cut[dv]       += w
            ext_cnt[du]   += 1
            ext_cnt[dv]   += 1

    return vol, cut, int_cnt, ext_cnt, sum_d, sumsq_d

@nb.njit(parallel=True, cache=True)
def detect_interface_edges(sources: np.ndarray, targets: np.ndarray, 
                          clusters: np.ndarray) -> np.ndarray:
    """Fast interface edge detection."""
    n_edges = len(sources)
    is_interface = np.zeros(n_edges, dtype=np.bool_)
    
    for i in prange(n_edges):
        if clusters[sources[i]] != clusters[targets[i]]:
            is_interface[i] = True
    
    return is_interface

@nb.njit(parallel=True, cache=True)
def identify_interface_edges_detailed(sources: np.ndarray, targets: np.ndarray, 
                                    distances: np.ndarray, similarities: np.ndarray,
                                    clusters: np.ndarray, pruned_clusters: np.ndarray) -> Tuple:
    """
    Detailed interface edge detection with support for pruned clusters.
    Returns complete information needed for analysis.
    """
    n_edges = len(sources)
    
    # Pre-allocate result arrays
    is_interface = np.zeros(n_edges, dtype=nb.boolean)
    edge_types = np.zeros(n_edges, dtype=nb.int8)
    source_clusters = np.zeros(n_edges, dtype=np.int64)
    target_clusters = np.zeros(n_edges, dtype=np.int64)
    
    # Create set-like structure for pruned clusters
    max_cluster = 0
    for c in pruned_clusters:
        if c > max_cluster:
            max_cluster = c
    
    is_pruned = np.zeros(max_cluster + 1, dtype=nb.boolean)
    for c in pruned_clusters:
        is_pruned[c] = True
    
    # Process each edge
    interface_count = 0
    cross_count = 0
    pruned_count = 0
    
    for idx in range(n_edges):
        i = sources[idx]
        j = targets[idx]
        
        ci = clusters[i]
        cj = clusters[j]
        
        source_clusters[idx] = ci
        target_clusters[idx] = cj
        
        # Determine edge type
        if ci != cj:
            # Cross-community edge
            is_interface[idx] = True
            edge_types[idx] = 0  # 0 = cross_community
            interface_count += 1
            cross_count += 1
        elif (ci <= max_cluster and is_pruned[ci]) or (cj <= max_cluster and is_pruned[cj]):
            # Edge within a pruned community
            is_interface[idx] = True
            edge_types[idx] = 1  # 1 = pruned_community
            interface_count += 1
            pruned_count += 1
    
    return (is_interface, edge_types, source_clusters, target_clusters, 
            interface_count, cross_count, pruned_count)

@nb.njit(parallel=True, cache=True)
def sparsify_knn_fast(sources: np.ndarray, targets: np.ndarray, 
                     weights: np.ndarray, n_nodes: int, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fast k-NN sparsification using parallel sorting."""
    # Count edges per node
    degrees = np.zeros(n_nodes, dtype=np.int64)
    for i in range(len(sources)):
        degrees[sources[i]] += 1
    
    # Build offsets
    offsets = np.zeros(n_nodes + 1, dtype=np.int64)
    for i in range(n_nodes):
        offsets[i+1] = offsets[i] + degrees[i]
    
    # Store edges per node
    total_edges = len(sources)
    edge_targets = np.empty(total_edges, dtype=np.int64)
    edge_weights = np.empty(total_edges, dtype=np.float64)
    edge_indices = np.empty(total_edges, dtype=np.int64)
    
    # Reset degrees for filling
    degrees.fill(0)
    for e in range(total_edges):
        src = sources[e]
        pos = offsets[src] + degrees[src]
        edge_targets[pos] = targets[e]
        edge_weights[pos] = weights[e]
        edge_indices[pos] = e
        degrees[src] += 1
    
    # Count total edges to keep
    new_edge_count = 0
    for node in range(n_nodes):
        deg = offsets[node+1] - offsets[node]
        if deg > 0:
            new_edge_count += min(k, deg)
    
    # Allocate output arrays
    sparse_sources = np.empty(new_edge_count, dtype=np.int64)
    sparse_targets = np.empty(new_edge_count, dtype=np.int64)
    sparse_weights = np.empty(new_edge_count, dtype=np.float64)
    sparse_orig_idx = np.empty(new_edge_count, dtype=np.int64)
    
    # Fill output arrays
    out = 0
    for node in range(n_nodes):
        start = offsets[node]
        end = offsets[node+1]
        deg = end - start
        
        if deg == 0:
            continue
            
        # Get edges for this node
        wts = edge_weights[start:end]
        idx = edge_indices[start:end]
        
        if deg <= k:
            # Keep all edges
            for j in range(deg):
                sparse_sources[out] = node
                sparse_targets[out] = edge_targets[start + j]
                sparse_weights[out] = wts[j]
                sparse_orig_idx[out] = idx[j]
                out += 1
        else:
            # Keep top k edges
            temp_w = wts.copy()
            best_j = np.empty(k, dtype=np.int64)
            
            # Find top k
            for j in range(k):
                max_i = 0
                max_w = temp_w[0]
                for l in range(1, deg):
                    if temp_w[l] > max_w:
                        max_i = l
                        max_w = temp_w[l]
                best_j[j] = max_i
                temp_w[max_i] = -np.inf
            
            # Store top k
            for j in range(k):
                i0 = best_j[j]
                sparse_sources[out] = node
                sparse_targets[out] = edge_targets[start + i0]
                sparse_weights[out] = wts[i0]
                sparse_orig_idx[out] = idx[i0]
                out += 1
    
    return sparse_sources, sparse_targets, sparse_weights, sparse_orig_idx
                    
@nb.njit(fastmath=True, cache=True)
def find_knee_point(x: np.ndarray, y: np.ndarray, S: float = 1.0, 
                   use_median_filter: bool = False) -> int:
    """
    Optimized kneedle algorithm for finding the knee point in a curve.
    Used for determining cluster size thresholds.
    """
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    
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
            original_indices = np.where(above_median)[0]
    
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
        return original_indices[knee_idx]
    
    return knee_idx

# ============================================================================
# OPTIMIZED GRAPH LOADER
# ============================================================================

# assume perf_monitor is already imported and configured
# from your performance monitoring utilities

class OptimizedGraphLoader:
    """Memory-efficient graph loader with on-disk caching and lazy I/O."""

    def __init__(self, input_dir: str):
        self.input_dir = input_dir
        self._metadata = None
        self._node_df = None
        self._component_labels = None
        self._embedding = None
        self._full_embedding = None
        self._umap_results = None
        self._means = None
        self._sigmas = None
        self._adjacency = None
        self._edge_arrays = None

        # Pre-compute or load CSR offsets/indices to speed up interface detection
        csr_cache = os.path.join(input_dir, "csr_cache.npz")
        if os.path.exists(csr_cache):
            with perf_monitor.timed_operation("Load CSR cache"):
                data = np.load(csr_cache)
                self.csr_offsets = data["offsets"]
                self.csr_indices = data["indices"]
        else:
            with perf_monitor.timed_operation("Build+cache CSR"):
                adj = self.adjacency.tocsr()
                self.csr_offsets = adj.indptr
                self.csr_indices = adj.indices
                np.savez_compressed(csr_cache,
                                    offsets=self.csr_offsets,
                                    indices=self.csr_indices)
        
    @property
    def metadata(self) -> Dict:
        if self._metadata is None:
            with perf_monitor.timed_operation("Load metadata"):
                path = os.path.join(self.input_dir, "metadata.json")
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        self._metadata = json.load(f)
                else:
                    self._metadata = {}
        return self._metadata

    @property
    def node_df(self) -> pd.DataFrame:
        if self._node_df is None:
            with perf_monitor.timed_operation("Load node dataframe"):
                self._node_df = pd.read_parquet(
                    os.path.join(self.input_dir, "node_df.parquet")
                )
        return self._node_df

    @property
    def component_labels(self) -> np.ndarray:
        if self._component_labels is None:
            with perf_monitor.timed_operation("Load component labels"):
                path = os.path.join(self.input_dir, "component_labels.npy")
                if os.path.exists(path):
                    self._component_labels = np.load(path)
                else:
                    # default: everyone in one component
                    self._component_labels = np.zeros(len(self.node_df), dtype=np.int64)
        return self._component_labels

    @property
    def means(self) -> Optional[np.ndarray]:
        if self._means is None and self.metadata.get("has_means", False):
            with perf_monitor.timed_operation("Load means"):
                self._means = np.load(os.path.join(self.input_dir, "means.npy"))
        return self._means

    @property
    def sigmas(self) -> Optional[np.ndarray]:
        if self._sigmas is None and self.metadata.get("has_sigmas", False):
            with perf_monitor.timed_operation("Load sigmas"):
                self._sigmas = np.load(os.path.join(self.input_dir, "sigmas.npy"))
        return self._sigmas

    @property
    def embedding(self) -> Optional[np.ndarray]:
        if self._embedding is None:
            with perf_monitor.timed_operation("Load embedding"):
                path = os.path.join(self.input_dir, "embedding.npy")
                if os.path.exists(path):
                    self._embedding = np.load(path)
        return self._embedding

    @property
    def full_embedding(self) -> Optional[np.ndarray]:
        if self._full_embedding is None:
            with perf_monitor.timed_operation("Load full embedding"):
                path = os.path.join(self.input_dir, "full_embedding.npy")
                if os.path.exists(path):
                    self._full_embedding = np.load(path)
        return self._full_embedding

    @property
    def umap_results(self) -> Optional[Dict]:
        if self._umap_results is None and self.embedding is not None:
            with perf_monitor.timed_operation("Build UMAP results"):
                self._umap_results = {
                    "original_embedding": self.embedding,
                    "parameters": self.metadata.get("umap_parameters", {})
                }
                if self.full_embedding is not None:
                    self._umap_results["full_embedding"] = self.full_embedding
                # optional extra info
                dummy = os.path.join(self.input_dir, "dummy_info.pkl")
                if os.path.exists(dummy):
                    with open(dummy, 'rb') as f:
                        self._umap_results["dummy_info"] = pickle.load(f)
        return self._umap_results

    @property
    def adjacency(self) -> sp.spmatrix:
        if self._adjacency is None:
            with perf_monitor.timed_operation("Load adjacency matrix"):
                path = os.path.join(self.input_dir, "graph_adjacency.npz")
                self._adjacency = sp.load_npz(path)
        return self._adjacency

    @property
    def edge_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(sources, targets, distances) as contiguous arrays, cached to disk."""
        if self._edge_arrays is None:
            with perf_monitor.timed_operation("Load edge arrays"):
                cache = os.path.join(self.input_dir, "edge_arrays.npz")
                if os.path.exists(cache):
                    data = np.load(cache, mmap_mode='r')
                    src = data["sources"].astype(np.int64)
                    tgt = data["targets"].astype(np.int64)
                    dist = data["distances"].astype(np.float64)
                else:
                    coo = self.adjacency.tocoo()
                    src = coo.row.astype(np.int64)
                    tgt = coo.col.astype(np.int64)
                    dist = coo.data.astype(np.float64)
                    np.savez_compressed(cache,
                                        sources=src.astype(np.int32),
                                        targets=tgt.astype(np.int32),
                                        distances=dist.astype(np.float32))
                # ensure C-contiguous
                self._edge_arrays = (
                    np.ascontiguousarray(src),
                    np.ascontiguousarray(tgt),
                    np.ascontiguousarray(dist)
                )
        return self._edge_arrays

    @property
    def n_nodes(self) -> int:
        return len(self.node_df)

    @property
    def n_edges(self) -> int:
        return len(self.edge_arrays[0])

    def build_graph_wrapper(self, include_embedding: bool = True):
        """
        Returns a tiny wrapper exposing .n_nodes, .node_df and .graph.get_edge_list()
        so you can plug into DataGraph or similar.
        """
        class GraphWrapper:
            def __init__(self, loader):
                self.loader = loader
                self.node_df = loader.node_df.copy()
                if include_embedding and loader.embedding is not None and loader.embedding.shape[1] >= 2:
                    if "UMAP1" not in self.node_df:
                        self.node_df["UMAP1"] = loader.embedding[:, 0]
                        self.node_df["UMAP2"] = loader.embedding[:, 1]

                # a minimal graph API
                self.graph = type("GraphObj", (), {
                    "n_nodes": loader.n_nodes,
                    "get_edge_list": lambda: [
                        (int(u), int(v), float(d))
                        for u, v, d in zip(*loader.edge_arrays)
                    ]
                })
        return GraphWrapper(self)

# ============================================================================
# OPTIMIZED COMMUNITY ANALYZER
# ============================================================================

class OptimizedCommunityAnalyzer:
    """Optimized community detection with minimal memory footprint."""
    
    def __init__(self, graph_loader: OptimizedGraphLoader,
                 coarsen: bool = True,
                 coarsen_levels: int = 1,
                 sparsify: bool = True,
                 sparsify_pre_k: int = 60,
                 sparsify_post_k: int = 60,
                 similarity_function=None,
                 verbose: bool = True):
        # Warm up numba function to prepare for bottleneck        
        self.coarsen = coarsen
        self.coarsen_levels = coarsen_levels
        self.sparsify = sparsify
        self.sparsify_pre_k = sparsify_pre_k
        self.sparsify_post_k = sparsify_post_k
        self.verbose = verbose
        self._csr_offsets = None
        self._csr_indices = None

        # Default similarity function
        if similarity_function is None:
            self.similarity_function = lambda d, s=None: self._gaussian_similarity(d, s)
        else:
            self.similarity_function = similarity_function

        # Storage for results
        self.runs = {}
        self.interface_edges = {}
        self.combined_edges = None
        self.visualizations = {}

        # Cache for similarity computations
        self._similarity_cache = {}

        # Flags for lazy initialization
        self._initialized = False
        self._csr_built = False
        self._graph_prepared = False

        # Load edge data once
        self.loader = graph_loader
        self.sources, self.targets, self.distances = self.loader.edge_arrays
        total_mem = (
            self.sources.nbytes +
            self.targets.nbytes +
            self._compute_similarities(self.distances).nbytes
        )
        avail   = psutil.virtual_memory().available
        cpu_cnt = max(1, psutil.cpu_count(logical=True) or 1)
        n_workers = min(cpu_cnt, max(1, int(avail * 0.8) // total_mem))
        self._pool = ThreadPoolExecutor(max_workers=n_workers)
        # Store full arrays for interface detection
        self.full_sources = self.sources.copy()
        self.full_targets = self.targets.copy()
        self.full_distances = self.distances.copy()

        # Defaults after lazy init
        self.coarsened = False
        self.n_nodes_final = self.loader.n_nodes
        self.meta_id = None

        # even if we never coarsen, we still need these attrs for the CSR path
        self.coarsened_sources = self.sources
        self.coarsened_targets = self.targets
        self.coarsened_n_nodes = self.n_nodes_final
        # for Louvain‐CSR we need weights; use your default similarity‐function here
        self.coarsened_weights = self._compute_similarities(self.distances)

        if self.verbose:
            print(f"Analyzer initialized (lazy mode)")
            print(f"  Nodes: {self.loader.n_nodes:,}")
            print(f"  Edges: {len(self.sources):,}")

    def close(self):
        """Shut down the thread‐pool exactly once when you’re done."""
        if hasattr(self, '_pool'):
            self._pool.shutdown(wait=True)
            if self.verbose:
                print("Coarsening pool closed.")
                
    @property
    def csr_offsets(self):
        return self._csr_offsets

    @property
    def csr_indices(self):
        return self._csr_indices

    @property
    def n_edges(self) -> int:
        return len(self.edge_arrays[0])
    
    def _ensure_csr_built(self):
        if not self._csr_built:
            if self.verbose:
                print("Loading CSR graph structure from cache...")
            # assign to the private attributes
            self._csr_offsets = self.loader.csr_offsets
            self._csr_indices = self.loader.csr_indices
            self._csr_built = True

    
    def _ensure_prepared(self):
        """Ensure graph is prepared—coarsening before sparsification—when needed."""
        if not self._graph_prepared:
            if self.verbose:
                print("Preparing graph…")

            # 1) Apply pre-coarsening sparsification so edge aggregation is less demanding
            if self.sparsify:
                if self.verbose:
                    print("  Pre-coarsening sparsification step")
                self._apply_pre_coarsening_sparsification()

            # 2) Apply coarsening so everything downstream sees the reduced graph
            if self.coarsen:
                if self.verbose:
                    print("  → Coarsening step")
                self._apply_coarsening()
               
            # NOTE: Conversion to igraph is unnecessary...
            if False:
                # 4) Finally, build the igraph structure for Leiden/Louvain
                #if self.verbose and self.algorithm:
                #    print("  → Constructing igraph")
                #self._create_igraph()
                pass

            # Mark as done
            self._graph_prepared = True
            self._initialized     = True
    
    def _apply_pre_coarsening_sparsification(self):
        if self.verbose:
            print(f"Sparsifying pre-coarsened graph (k={self.sparsify_pre_k})…")
        with perf_monitor.timed_operation("Pre-coarsened graph sparsification"):
            sims = self._compute_similarities(self.full_distances, scale="adaptive")
            s, t, w, orig_idx = sparsify_knn_fast(
                self.sources, self.targets, sims, self.loader.n_nodes, int(self.sparsify_pre_k)
            )
            # keep DIRECTED here
            self.sources   = s.astype(np.int64, copy=False)
            self.targets   = t.astype(np.int64, copy=False)
            self.distances = self.full_distances[orig_idx].astype(np.float32, copy=False)
            if self.verbose:
                print(f"  Sparsified pre-coarsened graph to {len(self.sources):,} directed edges")
                
    def _apply_post_coarsened_sparsification(self):
        """Undirected top-k on coarsened unique pairs (a<b) — stays (a<b)."""
        if self.verbose:
            print(f"Sparsifying post-coarsened graph (k={self.sparsify_post_k})…")

        with perf_monitor.timed_operation("Graph sparsification"):
            a  = self.coarsened_sources.astype(np.int64,  copy=False)
            b  = self.coarsened_targets.astype(np.int64,  copy=False)
            d  = self.distances.astype(np.float32, copy=False)
            n  = int(self.n_nodes_final)

            # recompute sims on current distances
            w  = self._compute_similarities(d, scale="adaptive").astype(np.float32, copy=False)

            # undirected top-k (OR across endpoints) → unique (a'<b')
            a2, b2, w2, d2 = sparsify_knn_undirected(a, b, w, d, n, int(self.sparsify_post_k))

            # write back (still unique a<b)
            self.coarsened_sources  = a2.astype(np.int32,  copy=False)
            self.coarsened_targets  = b2.astype(np.int32,  copy=False)
            self.coarsened_weights  = w2.astype(np.float32, copy=False)
            self.distances          = d2.astype(np.float32, copy=False)

            if self.verbose:
                print(f"  Sparsified to {len(self.coarsened_sources):,} unique undirected edges")

    def _gaussian_similarity(self, distances: np.ndarray, scale: Union[float, str] = 'adaptive') -> np.ndarray:
        """Gaussian similarity function with adaptive scaling."""
        if isinstance(scale, str) and scale == 'adaptive':
            scale = float(np.median(distances))
        elif scale is None:
            scale = float(np.median(distances))
        
        return np.exp(-(distances/scale)**2/2)
        
    def _compute_similarities(self, distances: np.ndarray, scale: Union[float, str] = 'adaptive') -> np.ndarray:
        """Compute similarities with caching."""
        # Create cache key
        if isinstance(scale, str) and scale == 'adaptive':
            scale = float(np.median(distances))
        
        cache_key = (id(distances), scale)
        
        if cache_key not in self._similarity_cache:
            # Use the similarity function
            self._similarity_cache[cache_key] = self.similarity_function(distances, scale)
        
        return self._similarity_cache[cache_key]
    
    def _apply_coarsening(self):
        """Directed reciprocal 1-NN → aggregate to unique UNDIRECTED meta-edges (a<b).
        Track distances (min) and weights (sum). Then run undirected post-k.
        """
        if self.verbose:
            print(f"Applying {self.coarsen_levels} levels of coarsening…")

        # Start from directed graph
        current_sources   = self.sources.astype(np.int64,  copy=False)
        current_targets   = self.targets.astype(np.int64,  copy=False)
        current_distances = self.distances.astype(np.float32, copy=False)
        current_weights   = self._compute_similarities(current_distances).astype(np.float32, copy=False)
        current_n_nodes   = int(self.loader.n_nodes)

        cumulative_mapping = np.arange(current_n_nodes, dtype=np.int64)
        self.coarsening_hierarchy = []

        for level in range(self.coarsen_levels):
            if self.verbose:
                print(f"  Level {level+1}: {current_n_nodes:,} nodes")

            # mutual 1-NN on directed edges (safe sequential version)
            meta_id, n_meta = mutual_nn_coarsening_directed(
                current_sources, current_targets, current_weights, current_n_nodes
            )
            ratio = n_meta / current_n_nodes
            self.coarsening_hierarchy.append({
                'level': level,
                'original_nodes': current_n_nodes,
                'coarsened_nodes': int(n_meta),
                'reduction_ratio': float(ratio)
            })
            if self.verbose:
                print(f"    → {n_meta:,} meta-nodes (ratio: {ratio:.3f})")

            # Early stop if weak reduction or very small
            if n_meta < 1000 or ratio > 0.95:
                # still produce unique (a<b) at this level
                ms = meta_id[current_sources]
                mt = meta_id[current_targets]
                keep = (ms != mt)
                if np.any(keep):
                    a, b, w, d = aggregate_undirected_edges_with_dist(
                        ms[keep], mt[keep],
                        current_weights[keep], current_distances[keep]
                    )
                    current_sources   = a.astype(np.int64,  copy=False)
                    current_targets   = b.astype(np.int64,  copy=False)
                    current_weights   = w.astype(np.float32, copy=False)
                    current_distances = d.astype(np.float32, copy=False)
                else:
                    # degenerate case: no edges
                    current_sources = current_sources[:0]
                    current_targets = current_targets[:0]
                    current_weights = current_weights[:0]
                    current_distances = current_distances[:0]

                cumulative_mapping = meta_id[cumulative_mapping]
                current_n_nodes = int(n_meta)
                break


            # Map to meta-nodes, still directed; drop self loops
            ms = meta_id[current_sources]
            mt = meta_id[current_targets]
            keep = (ms != mt)
            ms = ms[keep]; mt = mt[keep]
            mw = current_weights[keep]; md = current_distances[keep]

            # Aggregate to UNIQUE undirected pairs (a<b): sum weights, min dist
            a, b, w, d = aggregate_undirected_edges_with_dist(ms, mt, mw, md)

            # Prepare for next level
            cumulative_mapping = meta_id[cumulative_mapping]
            current_n_nodes    = int(n_meta)
            current_sources    = a.astype(np.int64,  copy=False)
            current_targets    = b.astype(np.int64,  copy=False)
            current_weights    = w.astype(np.float32, copy=False)
            current_distances  = d.astype(np.float32, copy=False)

            if self.verbose:
                print(f"    → Aggregated to {len(current_sources):,} unique undirected edges")

        # Store final state (unique undirected a<b)
        self.coarsened         = True
        self.meta_id           = cumulative_mapping
        self.n_nodes_final     = int(current_n_nodes)
        self.coarsening_ratio  = float(self.n_nodes_final / self.loader.n_nodes)
        self.coarsened_sources = current_sources.astype(np.int32,  copy=False)
        self.coarsened_targets = current_targets.astype(np.int32,  copy=False)
        self.coarsened_weights = current_weights.astype(np.float32, copy=False)
        self.distances         = current_distances.astype(np.float32, copy=False)

        # Scale post-k target if desired
        prev = getattr(self, "sparsify_post_k", None)
        if prev is None:
            self.sparsify_post_k = max(1, int(self.sparsify_pre_k * self.coarsening_ratio))
        else:
            self.sparsify_post_k = max(1, int(prev * self.coarsening_ratio))

        # >>> IMPORTANT: run undirected post-k here <<<
        if self.sparsify and self.sparsify_post_k > 0:
            self._apply_post_coarsened_sparsification()

    def _create_igraph(self):
        """Enhanced igraph creation with detailed diagnostics."""
        import inspect
        import igraph as ig
        
        print("Creating igraph...")
        print(f"Using igraph version: {ig.__version__}")  # Check actual version
        
        n_nodes = self.n_nodes_final
        n_edges = len(self.coarsened_sources)
        
        print(f"Graph size: {n_nodes} nodes, {n_edges} edges")
        print(f"Available memory: {psutil.virtual_memory().available / 1e9:.2f} GB")
        
        # 1. Try CSR first (fastest, most memory efficient)
        if hasattr(ig.Graph, "from_scipy_sparse_matrix"):
            print("Method 1: CSR conversion is available in this igraph version")
            try:
                if self.verbose:
                    print("→ fast path: CSR→igraph")
                csr = sp.csr_matrix(
                    (self.coarsened_weights,
                     (self.coarsened_sources, self.coarsened_targets)),
                    shape=(n_nodes, n_nodes)
                )
                print(f"  CSR matrix created: shape={csr.shape}, nnz={csr.nnz}")
                try:
                    # Check actual method signature
                    print(f"  Method signature: {inspect.signature(ig.Graph.from_scipy_sparse_matrix)}")
                    self.igraph = ig.Graph.from_scipy_sparse_matrix(
                        csr, directed=False, edge_attrs=["weight"]
                    )
                    print("  CSR conversion successful!")
                    return
                except Exception as e:
                    print(f"  CSR conversion failed with error: {type(e).__name__}: {str(e)}")
                    import traceback
                    traceback.print_exc()
            except Exception as e:
                print(f"  CSR matrix creation failed: {type(e).__name__}: {str(e)}")
                import traceback
                traceback.print_exc()
        else:
            print("Method 1: CSR conversion is NOT available in this igraph version")
        
        # 1b. Try COO as an alternative
        print("\nMethod 1b: Trying COO format instead of CSR")
        try:
            coo = sp.coo_matrix(
                (self.coarsened_weights,
                 (self.coarsened_sources, self.coarsened_targets)),
                shape=(n_nodes, n_nodes)
            )
            print(f"  COO matrix created: shape={coo.shape}, nnz={coo.nnz}")
            
            if hasattr(ig.Graph, "from_scipy_sparse_matrix"):
                try:
                    self.igraph = ig.Graph.from_scipy_sparse_matrix(
                        coo, directed=False, edge_attrs=["weight"]
                    )
                    print("  COO conversion successful!")
                    return
                except Exception as e:
                    print(f"  COO conversion failed: {type(e).__name__}: {str(e)}")
                    import traceback
                    traceback.print_exc()
        except Exception as e:
            print(f"  COO matrix creation failed: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # 2. Try direct creation (fast, but needs more memory)
        print("\nMethod 2: Trying direct edge list creation")
        edge_list_memory = n_edges * 32  # More realistic: 2 ints + overhead
        available_memory = psutil.virtual_memory().available
        
        print(f"  Estimated edge list memory: {edge_list_memory / 1e9:.2f} GB")
        print(f"  Available memory: {available_memory / 1e9:.2f} GB")
        print(f"  Memory requirement: {edge_list_memory / available_memory * 100:.1f}% of available")
        
        if available_memory > edge_list_memory * 1.5:  # 50% safety margin
            try:
                print("  Memory check passed, attempting direct creation")
                
                # Try numpy array approach first (newer igraph versions)
                try:
                    print("  Method 2a: Using numpy array approach")
                    # This avoids ALL Python object creation
                    edge_array = np.column_stack((self.coarsened_sources, self.coarsened_targets))
                    print(f"  Edge array created: shape={edge_array.shape}")
                    
                    try:
                        start_time = time.time()
                        self.igraph = ig.Graph(n=n_nodes, edges=edge_array, directed=False)
                        creation_time = time.time() - start_time
                        print(f"  Graph created in {creation_time:.2f} seconds")
                        
                        # Set weights
                        start_time = time.time()
                        self.igraph.es["weight"] = self.coarsened_weights
                        weight_time = time.time() - start_time
                        print(f"  Weights set in {weight_time:.2f} seconds")
                        
                        print("  Numpy array approach successful!")
                        return
                    except Exception as e:
                        print(f"  Numpy array graph creation failed: {type(e).__name__}: {str(e)}")
                        import traceback
                        traceback.print_exc()
                except Exception as e:
                    print(f"  Edge array creation failed: {type(e).__name__}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                
                # Fallback to list of tuples (older igraph versions)
                try:
                    print("  Method 2b: Using list of tuples approach")
                    start_time = time.time()
                    edges = list(zip(self.coarsened_sources.tolist(), 
                                    self.coarsened_targets.tolist()))
                    edges_time = time.time() - start_time
                    print(f"  Edge list created in {edges_time:.2f} seconds: {len(edges)} edges")
                    
                    try:
                        start_time = time.time()
                        self.igraph = ig.Graph(n=n_nodes, edges=edges, directed=False)
                        creation_time = time.time() - start_time
                        print(f"  Graph created in {creation_time:.2f} seconds")
                        
                        # Set weights
                        start_time = time.time()
                        self.igraph.es["weight"] = self.coarsened_weights.tolist()
                        weight_time = time.time() - start_time
                        print(f"  Weights set in {weight_time:.2f} seconds")
                        
                        print("  List of tuples approach successful!")
                        return
                    except Exception as e:
                        print(f"  Tuple list graph creation failed: {type(e).__name__}: {str(e)}")
                        import traceback
                        traceback.print_exc()
                except Exception as e:
                    print(f"  Tuple list creation failed: {type(e).__name__}: {str(e)}")
                    import traceback
                    traceback.print_exc()
            except MemoryError as e:
                print(f"  Memory error during direct creation: {str(e)}")
                import traceback
                traceback.print_exc()
        else:
            print("  Memory check failed, skipping direct creation")
        
        # 3. Fallback to chunked approach (always works, but slower)
        print("\nMethod 3: Using chunked edge addition (memory-safe)")
        
        try:
            # Create empty graph
            start_time = time.time()
            self.igraph = ig.Graph(n=n_nodes, directed=False)
            creation_time = time.time() - start_time
            print(f"  Empty graph created in {creation_time:.2f} seconds")
            
            # Determine optimal chunk size
            mem_per_edge = 24  # Estimated bytes per edge in Python representation
            chunk_size = min(10_000_000, max(1000, int(available_memory * 0.2 / mem_per_edge)))
            total_chunks = (n_edges + chunk_size - 1) // chunk_size
            print(f"  Processing in {total_chunks} chunks of {chunk_size} edges")
            
            # Process in chunks
            weights_list = self.coarsened_weights.tolist()
            total_time_edges = 0
            total_time_weights = 0
            
            for chunk_idx in range(total_chunks):
                start = chunk_idx * chunk_size
                end = min((chunk_idx + 1) * chunk_size, n_edges)
                
                # Create edge chunk
                chunk_start_time = time.time()
                edges = list(zip(
                    self.coarsened_sources[start:end].tolist(),
                    self.coarsened_targets[start:end].tolist()
                ))
                edge_time = time.time() - chunk_start_time
                total_time_edges += edge_time
                
                # Add edges
                add_start_time = time.time()
                edge_start_id = self.igraph.ecount()
                self.igraph.add_edges(edges)
                add_time = time.time() - add_start_time
                
                # Add weights
                weight_start_time = time.time()
                for i, w in enumerate(weights_list[start:end]):
                    self.igraph.es[edge_start_id + i]["weight"] = w
                weight_time = time.time() - weight_start_time
                total_time_weights += weight_time
                
                print(f"  Chunk {chunk_idx + 1}/{total_chunks}: {len(edges)} edges processed in {edge_time + add_time + weight_time:.2f}s")
                print(f"    Edge prep: {edge_time:.2f}s, Add: {add_time:.2f}s, Weights: {weight_time:.2f}s")
                
            print(f"  All chunks processed. Total edge prep time: {total_time_edges:.2f}s, Weight assignment: {total_time_weights:.2f}s")
            print(f"  Final graph: {self.igraph.vcount()} vertices, {self.igraph.ecount()} edges")
            return
            
        except Exception as e:
            print(f"  Chunked approach failed: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            raise RuntimeError("All graph creation methods failed!")
    
    def run_leiden_igraph(self,
               resolution: float,
               run_id: str = None,
               scale: Union[float, str] = 'adaptive',
               initial_membership: Optional[np.ndarray] = None,
               rank_stat_col: Optional[str] = None,
               prune_small_clusters: bool = False,
               min_cluster_size: Optional[int] = None,
               knee_sensitivity: float = 1.0,
               normalize_rank_stat: bool = True,
               reassign_pruned: bool = False,
               output_prefix: Optional[str] = None
              ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Run Leiden with optional Louvain cold‐start and warm‐start refinement."""
        import igraph as ig
        import leidenalg
        # Ensure graph is prepared
        self._ensure_prepared()
    
        # Generate run_id and output prefix if needed
        if run_id is None:
            run_id = f"leiden_res{resolution:.3f}"
        if output_prefix is None:
            output_prefix = f"leiden_{run_id}_"
    
        if self.verbose:
            print(f"\n--- Running Leiden with resolution={resolution} ---")
    
        with perf_monitor.timed_operation(f"Leiden clustering (res={resolution})"):
            # 0) Cold‐start: run Louvain if no initial_membership provided
            if initial_membership is None:
                if self.verbose:
                    print("  Cold‐start Louvain…")
                louvain_part = self.igraph.community_multilevel(
                    weights=self.igraph.es['weight']
                )
                initial_membership = np.array(louvain_part.membership, dtype=np.int64)
                leiden_iters = 5
            else:
                leiden_iters = 5
    
            # 1) Set edge weights if graph is not coarsened
            if not self.coarsened:
                sims = self._compute_similarities(self.distances, scale)
                self.igraph.es['weight'] = sims.tolist()
    
            # 2) Handle projection of warm-start into coarsened space
            # Only project when we actually have a full-graph labeling (warm start),
            # not when we just cold-started via Louvain on the coarsened graph.
            if self.coarsened \
               and initial_membership is not None \
               and initial_membership.shape[0] == self.loader.n_nodes:
            
                coarse_init = np.zeros(self.n_nodes_final, dtype=np.int64)
                for i in range(self.loader.n_nodes):
                    coarse_init[self.meta_id[i]] = initial_membership[i]
                initial_membership = coarse_init
    
            # 3) Run Leiden with the chosen number of iterations
            partition_kwargs = {
                'resolution_parameter': float(resolution),
                'weights': 'weight',
                'n_iterations': leiden_iters
            }
            if initial_membership is not None:
                partition_kwargs['initial_membership'] = initial_membership.tolist()
    
            partition = leidenalg.find_partition(
                self.igraph,
                leidenalg.RBConfigurationVertexPartition,
                **partition_kwargs
            )
            labels = np.array(partition.membership, dtype=np.int64)
    
            # 4) Project labels back to full graph if coarsened
            if self.coarsened:
                full_labels = np.zeros(self.loader.n_nodes, dtype=np.int64)
                for i in range(self.loader.n_nodes):
                    full_labels[i] = labels[self.meta_id[i]]
                labels = full_labels
    
            if self.verbose:
                n_clusters = len(np.unique(labels))
                print(f"Found {n_clusters} communities")
    
        # Process results: attach labels, compute stats, detect interfaces, store
        with perf_monitor.timed_operation("Process cluster labels"):
            df = self.loader.node_df.copy()
            cluster_col = f'{output_prefix}cluster'
            rank_col = f'{output_prefix}rank'
            df[cluster_col] = labels
    
        with perf_monitor.timed_operation("Process cluster statistics"):
            cluster_stats, pruning_info = self._process_cluster_stats(
                df, labels, cluster_col, rank_col, rank_stat_col,
                normalize_rank_stat, prune_small_clusters,
                min_cluster_size, knee_sensitivity, reassign_pruned
            )
    
        with perf_monitor.timed_operation("Identify interface edges"):
            interface_edges_df = self._identify_interface_edges(
                df, cluster_col, pruning_info.get('pruned_clusters', []), scale
            )
    
        with perf_monitor.timed_operation("Store run results"):
            self.runs[run_id] = {
                'df': df,
                'cluster_stats': cluster_stats,
                'pruning_info': pruning_info,
                'resolution': resolution,
                'cluster_col': cluster_col,
                'rank_col': rank_col,
                'similarity_scale': scale,
                'labels': labels,
                'coarsened': self.coarsened,
                'coarsening_ratio': self.coarsening_ratio if self.coarsened else None
            }
            self.interface_edges[run_id] = interface_edges_df
    
        return cluster_stats, labels

    def run_louvain_csr(self,
                   resolution: float,
                   run_id: str = None,
                   scale: Union[float, str] = 'adaptive',
                   initial_membership: Optional[np.ndarray] = None,
                   rank_stat_col: Optional[str] = None,
                   prune_small_clusters: bool = False,
                   min_cluster_size: Optional[int] = None,
                   knee_sensitivity: float = 1.0,
                   normalize_rank_stat: bool = True,
                   reassign_pruned: bool = False,
                   output_prefix: Optional[str] = None
                  ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Run CSR-native Louvain community detection (No warm-start support yet)"""
        try:
            from sknetwork.clustering import Louvain
        except ImportError:
            raise ImportError("To use CSR-native Louvain, install scikit-network: pip install scikit-network")
        
        # Ensure CSR is built but skip igraph creation
        if not self._csr_built:
            if self.verbose:
                print("Loading CSR graph structure...")
            # assign to the private attributes
            self._csr_offsets = self.loader.csr_offsets
            self._csr_indices = self.loader.csr_indices
            self._csr_built = True
        
        # Do sparsification/coarsening if needed but don't create igraph
        if not self._graph_prepared:
            self._ensure_prepared()
            # Mark as prepared but don't create igraph
            #self._graph_prepared = True
            self._initialized = True
        
        # Generate run_id and output prefix if needed
        if run_id is None:
            run_id = f"louvain_res{resolution:.3f}"
        if output_prefix is None:
            output_prefix = f"louvain_{run_id}_"
        
        if self.verbose:
            print(f"\n--- Running CSR-native Louvain with resolution={resolution} ---")
        
        with perf_monitor.timed_operation(f"Louvain clustering (res={resolution})"):
            # Create CSR matrix directly
            csr = sp.csr_matrix(
                (self.coarsened_weights,
                 (self.coarsened_sources, self.coarsened_targets)),
                shape=(self.n_nodes_final, self.n_nodes_final)
            )
            
            # Handle warm-start initial membership if provided
            if initial_membership is not None:
                # Project to coarsened space if needed
                if self.coarsened \
                   and initial_membership is not None \
                   and initial_membership.shape[0] == self.loader.n_nodes:
                    
                    coarse_init = np.zeros(self.n_nodes_final, dtype=np.int64)
                    for i in range(self.loader.n_nodes):
                        coarse_init[self.meta_id[i]] = initial_membership[i]
                    initial_membership = coarse_init
                
                # Configure Louvain with initial labels
                louvain = Louvain(
                    resolution=float(resolution),
                    random_state=42,
                    modularity='newman',  # Most compatible with Leiden's default
                    return_probs=False#,       # ← here
                    #initial_labels=initial_membership  # Warm start #Sadly, this implementation of Louvain does not enable intiializing the labels...
                )
            else:
                # Cold start - run standard Louvain
                louvain = Louvain(
                    resolution=float(resolution),
                    random_state=42,
                    modularity='newman',
                    return_probs=False      # ← here
                )
            
            # Run the algorithm
            result = louvain.fit_transform(csr)
            # 2) extract a proper 1‐D np.int64 array of labels
            if hasattr(louvain, 'labels_'):
                labels = np.asarray(louvain.labels_, dtype=np.int64)
    
            elif isinstance(result, np.ndarray):
                # some versions return a dense array
                labels = result.astype(np.int64, copy=False).ravel()
    
            elif sp.issparse(result):
                # some return a 1×n sparse matrix
                labels = np.asarray(result.toarray(), dtype=np.int64).ravel()
    
            else:
                raise ValueError(f"Cannot interpret Louvain result of type {type(result)}")
    
            # 3) sanity‐check
            if labels.ndim != 1 or labels.shape[0] != self.n_nodes_final:
                raise ValueError(
                    f"Louvain labels have wrong shape {labels.shape}, "
                    f"expected ({self.n_nodes_final},)"
                )
            
            # Project labels back to full graph if coarsened
            if self.coarsened:
                full_labels = np.zeros(self.loader.n_nodes, dtype=np.int64)
                for i in range(self.loader.n_nodes):
                    full_labels[i] = labels[self.meta_id[i]]
                labels = full_labels
            
            if self.verbose:
                n_clusters = len(np.unique(labels))
                print(f"Found {n_clusters} communities")
        
        # Process results: attach labels, compute stats, detect interfaces, store
        # (This part is identical to run_leiden)
        with perf_monitor.timed_operation("Process cluster labels"):
            df = self.loader.node_df.copy()
            cluster_col = f'{output_prefix}cluster'
            rank_col = f'{output_prefix}rank'
            df[cluster_col] = labels
    
        with perf_monitor.timed_operation("Process cluster statistics"):
            cluster_stats, pruning_info = self._process_cluster_stats(
                df, labels, cluster_col, rank_col, rank_stat_col,
                normalize_rank_stat, prune_small_clusters,
                min_cluster_size, knee_sensitivity, reassign_pruned
            )
    
        with perf_monitor.timed_operation("Identify interface edges"):
            interface_edges_df = self._identify_interface_edges(
                df, cluster_col, pruning_info.get('pruned_clusters', []), scale
            )
    
        with perf_monitor.timed_operation("Store run results"):
            self.runs[run_id] = {
                'df': df,
                'cluster_stats': cluster_stats,
                'pruning_info': pruning_info,
                'resolution': resolution,
                'cluster_col': cluster_col,
                'rank_col': rank_col,
                'similarity_scale': scale,
                'labels': labels,
                'coarsened': self.coarsened,
                'coarsening_ratio': getattr(self, 'coarsening_ratio', None),
                'algorithm': 'louvain_csr'  # Mark the algorithm used
            }
            self.interface_edges[run_id] = interface_edges_df
    
        return cluster_stats, labels

    def run_leiden_csr(self, resolution: float, run_id: str = None,
                   scale: Union[float, str] = 'adaptive',
                   initial_membership: Optional[np.ndarray] = None,
                   rank_stat_col: Optional[str] = None,
                   prune_small_clusters: bool = False,
                   min_cluster_size: Optional[int] = None,
                   knee_sensitivity: float = 1.0,
                   normalize_rank_stat: bool = True,
                   reassign_pruned: bool = False,
                   output_prefix: Optional[str] = None) -> Tuple[pd.DataFrame, np.ndarray]:
        """Run CSR-native Leiden on a symmetric CSR built DIRECTLY from unique undirected pairs (a<b).
        No COO conversions, no array mirroring. Optional warm-start projection supported.
        """
        try:
            from sknetwork.clustering import Leiden
        except ImportError:
            raise ImportError("Install scikit-network: pip install scikit-network")

        # Ensure CSR structures (offsets/indices) cached if you use them elsewhere
        if not self._csr_built:
            self._ensure_csr_built()

        # Ensure sparsify/coarsen has been executed
        if not self._graph_prepared:
            self._ensure_prepared()
            self._initialized = True

        # Names
        if run_id is None:
            run_id = f"leiden_res{resolution:.3f}"
        if output_prefix is None:
            output_prefix = f"leiden_{run_id}_"
        
        if self.verbose:
            print(f"\n--- Building csr from undirected edges ---")
        with perf_monitor.timed_operation(f"\n--- Building csr from undirected edges ---"):
            # We have UNIQUE undirected pairs (a<b). Build symmetric CSR without COO.
            a = self.coarsened_sources.astype(np.int32,  copy=False)
            b = self.coarsened_targets.astype(np.int32,  copy=False)
            w = self.coarsened_weights.astype(np.float32, copy=False)
            n = int(self.n_nodes_final)

            csr = _csr_from_undirected_edges(a, b, w, n)

        if self.verbose:
            print(f"\n--- Running CSR-native Leiden with resolution={resolution} ---")

        with perf_monitor.timed_operation(f"Leiden clustering (res={resolution})"):
            # Optional: degree-cap in CSR (keeps symmetry by construction)
            # If you want to cap further, do it BEFORE CSR with an undirected top-k selector,
            # or implement a symmetric row-top-k that ORs selections from both endpoints.
            # For now, we skip additional degree caps to avoid symmetry breakage.
            # If you still want it and have a symmetric top-k utility, call it here.

            # Warm start: project fine → coarse if needed
            if initial_membership is not None:
                if self.coarsened and initial_membership.shape[0] == self.loader.n_nodes:
                    initial_membership = self._project_labels_to_coarse_mode(initial_membership)
                elif initial_membership.shape[0] != n:
                    raise ValueError(
                        f"initial_membership has length {initial_membership.shape[0]}, expected "
                        f"{self.loader.n_nodes} (fine) or {n} (coarse)."
                    )

            # Run Leiden (weights taken from csr.data)
            leiden = Leiden(
                resolution=float(resolution),
                random_state=42,
                modularity='newman',
                return_probs=False
            )
            result = leiden.fit_transform(csr)

            # Extract labels
            if hasattr(leiden, 'labels_'):
                labels = np.asarray(leiden.labels_, dtype=np.int32)
            elif isinstance(result, np.ndarray):
                labels = result.astype(np.int32).ravel()
            elif sp.issparse(result):
                labels = np.asarray(result.toarray(), dtype=np.int32).ravel()
            else:
                raise ValueError(f"Cannot interpret Leiden result of type {type(result)}")

            if labels.ndim != 1 or labels.shape[0] != n:
                raise ValueError(f"Leiden labels have wrong shape {labels.shape}")

            # Project labels back to fine level if coarsened
            if self.coarsened:
                full_labels = np.empty(self.loader.n_nodes, dtype=np.int32)
                # meta_id maps fine node -> coarse node id
                for i in range(self.loader.n_nodes):
                    full_labels[i] = labels[self.meta_id[i]]
                labels = full_labels

            if self.verbose:
                n_clusters = len(np.unique(labels))
                print(f"Found {n_clusters} communities")

        # ---- Post-processing (unchanged) ----
        with perf_monitor.timed_operation("Process cluster labels"):
            df = self.loader.node_df.copy()
            cluster_col = f'{output_prefix}cluster'
            rank_col    = f'{output_prefix}rank'
            df[cluster_col] = labels

        with perf_monitor.timed_operation("Process cluster statistics"):
            cluster_stats, pruning_info = self._process_cluster_stats(
                df, labels, cluster_col, rank_col, rank_stat_col,
                normalize_rank_stat, prune_small_clusters,
                min_cluster_size, knee_sensitivity, reassign_pruned
            )

        with perf_monitor.timed_operation("Identify interface edges"):
            interface_edges_df = self._identify_interface_edges(
                df, cluster_col, pruning_info.get('pruned_clusters', []), scale
            )

        with perf_monitor.timed_operation("Store run results"):
            self.runs[run_id] = {
                'df': df,
                'cluster_stats': cluster_stats,
                'pruning_info': pruning_info,
                'resolution': resolution,
                'cluster_col': cluster_col,
                'rank_col': rank_col,
                'similarity_scale': scale,
                'labels': labels,
                'coarsened': self.coarsened,
                'coarsening_ratio': getattr(self, 'coarsening_ratio', None),
                'algorithm': 'leiden_csr'
            }
            self.interface_edges[run_id] = interface_edges_df

        return cluster_stats, labels

    def _process_cluster_stats(self, df, labels, cluster_col, rank_col, rank_stat_col, 
                          normalize_rank_stat, prune_small_clusters, 
                          min_cluster_size, knee_sensitivity, reassign_pruned):
        """Process cluster statistics and handle pruning."""
        if self.verbose:
            print(f"Calculating cluster statistics...")
        #labels = np.asarray(labels)
        
        # 1) power-transform (unchanged)
        transformed_rank_stat_col = None
        if normalize_rank_stat and rank_stat_col and rank_stat_col in df.columns:
            try:
                from sklearn.preprocessing import PowerTransformer
                if self.verbose:
                    print(f"Applying standardized Yeo-Johnson transform to '{rank_stat_col}'...")
                values = df[rank_stat_col].values.reshape(-1, 1)
                if not np.all(values == values[0]):
                    pt = PowerTransformer(method='yeo-johnson', standardize=True)
                    transformed = pt.fit_transform(values).flatten()
                    transformed_rank_stat_col = f"{rank_stat_col}_transformed"
                    df[transformed_rank_stat_col] = transformed
                    if self.verbose:
                        print(f"Created transformed column '{transformed_rank_stat_col}'")
                else:
                    if self.verbose:
                        print(f"Warning: Column '{rank_stat_col}' is constant; skipping transform.")
            except ImportError:
                if self.verbose:
                    print("Warning: scikit-learn not available for transformation.")
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Transformation failed: {e}")
        
        # 2) choose which column (or None)
        stat_col = transformed_rank_stat_col or rank_stat_col
        if stat_col and stat_col in df.columns:
            stat_array = df[stat_col].to_numpy()
        else:
            stat_array = None
        
        # 3) aggregate per cluster
        stats = []
        unique_clusters = np.unique(labels)
        for cid in unique_clusters:
            mask = (labels == cid)
            size = int(mask.sum())
            
            if stat_array is not None:
                vals = stat_array[mask]
                mean_stat = float(vals.mean()) if vals.size else 0.0
                ranking = mean_stat * np.sqrt(size)
            else:
                mean_stat = 0.0
                ranking = float(size)
            
            stats.append({
                'cluster_id': int(cid),
                'size': size,
                'rank_stat_mean': mean_stat if stat_array is not None else None,
                'ranking_stat': ranking
            })
        
        # 4) build & sort DataFrame, assign ranks
        cluster_stats = pd.DataFrame(stats)
        cluster_stats = cluster_stats.sort_values('ranking_stat', ascending=False)
        cluster_stats['rank'] = np.arange(1, len(cluster_stats) + 1)
        
        # 5) initialize & apply pruning
        pruning_info = {
            'pruning_applied': False,
            'knee_point_index': None,
            'size_threshold': None,
            'pruned_clusters': [],
            'kept_clusters': list(unique_clusters)
        }
        if prune_small_clusters:
            pruning_info = self._apply_pruning(
                df, cluster_stats, cluster_col, labels,
                min_cluster_size, knee_sensitivity, reassign_pruned
            )
        
        # 6) map ranks back into df
        rank_map = {row['cluster_id']: row['rank'] 
                    for _, row in cluster_stats.iterrows()}
        df[rank_col] = df[cluster_col].map(rank_map)
        
        return cluster_stats, pruning_info


    def _apply_pruning(self, df, cluster_stats, cluster_col, labels, 
                      min_cluster_size, knee_sensitivity, reassign_pruned):
        """Apply pruning to small clusters."""
        # Sort by size for knee detection
        size_sorted_stats = cluster_stats.sort_values('size', ascending=False).reset_index(drop=True)
        
        # Determine pruning threshold
        if min_cluster_size is not None:
            idxs = size_sorted_stats.index[size_sorted_stats['size'] >= min_cluster_size]
            knee_idx = int(idxs.max()) if len(idxs) > 0 else -1
            size_threshold = min_cluster_size
        else:
            # Use knee detection
            x = np.arange(len(size_sorted_stats))
            y = size_sorted_stats['size'].values
            knee_idx = find_knee_point(x, y, S=knee_sensitivity)
            size_threshold = size_sorted_stats.iloc[knee_idx]['size']
        
        # Get lists of kept and pruned clusters
        kept_clusters = size_sorted_stats.iloc[:knee_idx+1]['cluster_id'].tolist()
        pruned_clusters = size_sorted_stats.iloc[knee_idx+1:]['cluster_id'].tolist()
        
        # Update pruning info
        pruning_info = {
            'pruning_applied': True,
            'knee_point_index': knee_idx,
            'size_threshold': size_threshold,
            'kept_clusters': kept_clusters,
            'pruned_clusters': pruned_clusters
        }
        
        if self.verbose:
            print(f"\nPruning small clusters:")
            print(f"- Size threshold: {size_threshold} (knee point at index {knee_idx})")
            print(f"- Kept {len(kept_clusters)} clusters, pruned {len(pruned_clusters)} clusters")
        
        # Add 'pruned' flag to cluster stats
        cluster_stats['pruned'] = cluster_stats['cluster_id'].apply(
            lambda x: x in pruned_clusters
        )
        
        # For visualization, add original cluster column regardless of reassignment
        orig_cluster_col = f"{cluster_col}_original"
        df[orig_cluster_col] = labels.copy()
        
        # Reassign nodes from pruned clusters if requested
        if reassign_pruned and pruned_clusters:
            self._reassign_pruned_nodes(df, cluster_col, pruned_clusters, kept_clusters)
        elif pruned_clusters and self.verbose:
            print("Not reassigning pruned clusters (reassign_pruned=False)")
            
        return pruning_info
    
    def _reassign_pruned_nodes(self, df, cluster_col, pruned_clusters, kept_clusters):
        """
        Reassign nodes from pruned clusters to the most common non-pruned neighbor cluster.

        Key properties:
        - Position-based only (no df.loc with node ids).
        - Works even if df has a non-RangeIndex.
        - Iterative (few passes) to reduce stranded nodes.
        - Safe fallbacks and clear logging.

        Assumptions:
        - self.csr_offsets / self.csr_indices correspond to the *same node ordering*
            as df rows (i.e., positions 0..n-1).
        """
        if self.verbose:
            print("Reassigning nodes from pruned clusters...")

        # Ensure CSR is available
        if not hasattr(self, "csr_offsets") or self.csr_offsets is None:
            # try to build/load once
            if hasattr(self, "_ensure_csr_built"):
                self._ensure_csr_built()
        if self.csr_offsets is None or self.csr_indices is None:
            raise RuntimeError("CSR structure not available; cannot reassign pruned nodes.")

        # Basic checks
        if cluster_col not in df.columns:
            raise KeyError(f"Column '{cluster_col}' not found in dataframe.")

        n_nodes = len(df)
        if self.csr_offsets.shape[0] != n_nodes + 1:
            raise ValueError(
                f"CSR size mismatch: indptr={self.csr_offsets.shape[0]} vs nodes={n_nodes}."
            )

        # Convert inputs to arrays (positional, no index alignment)
        node_to_cluster = df[cluster_col].to_numpy(copy=True)
        pruned_clusters = np.asarray(pruned_clusters, dtype=node_to_cluster.dtype)

        if pruned_clusters.size == 0:
            if self.verbose:
                print("No pruned clusters to reassign.")
            return

        # Mark pruned nodes with a temporary label (guaranteed not to collide with >=0 cluster ids)
        temp_label = np.int64(-1)
        pruned_mask = np.isin(node_to_cluster, pruned_clusters, assume_unique=False)
        pruned_indices = np.flatnonzero(pruned_mask)

        if pruned_indices.size == 0:
            if self.verbose:
                print("No nodes belong to pruned clusters.")
            return

        node_to_cluster[pruned_indices] = temp_label

        # Helper to (re)assign one node from neighbors
        def _assign_from_neighbors(idx) -> bool:
            start = self.csr_offsets[idx]
            end = self.csr_offsets[idx + 1]
            if start == end:
                return False  # no neighbors

            neigh = self.csr_indices[start:end]
            # Filter out neighbors that are still temp (i.e., pruned/unassigned)
            neigh_labels = node_to_cluster[neigh]
            valid = neigh_labels != temp_label
            if not np.any(valid):
                return False

            neigh_labels = neigh_labels[valid]
            # Vote by frequency; ties broken by smallest cluster id for determinism
            uniq, counts = np.unique(neigh_labels, return_counts=True)
            winner = uniq[np.argmax(counts)]
            node_to_cluster[idx] = winner
            return True

        # Iterative passes to let assignments propagate through regions of pruned nodes
        max_passes = 3
        total_reassigned = 0
        for it in range(max_passes):
            changes = 0
            # Only process nodes still unassigned
            todo = pruned_indices[node_to_cluster[pruned_indices] == temp_label]
            if todo.size == 0:
                break
            for idx in todo:
                if _assign_from_neighbors(idx):
                    changes += 1
            total_reassigned += changes
            if self.verbose:
                print(f"  pass {it+1}/{max_passes}: reassigned {changes} nodes")
            if changes == 0:
                break  # no more progress

        # Fallback: any still-temp nodes go to the largest kept cluster
        still_temp = (node_to_cluster == temp_label)
        stranded = int(still_temp.sum())
        if stranded > 0:
            if not kept_clusters:
                # If we truly have no "kept" clusters, pick the modal non-temp label
                non_temp = node_to_cluster[node_to_cluster != temp_label]
                if non_temp.size > 0:
                    vals, cnts = np.unique(non_temp, return_counts=True)
                    fallback = vals[np.argmax(cnts)]
                else:
                    # last resort, make a single cluster 0
                    fallback = np.int64(0)
            else:
                # kept_clusters is sorted by size desc upstream; pick the first
                fallback = np.int64(kept_clusters[0])

            node_to_cluster[still_temp] = fallback
            if self.verbose:
                print(f"  fallback: assigned {stranded} stranded nodes to cluster {int(fallback)}")

        # Write back once
        df[cluster_col] = node_to_cluster

        if self.verbose:
            print(f"Reassignment complete. Total reassigned from neighbors: {total_reassigned}")

    
    def _identify_interface_edges(self, df, cluster_col, pruned_clusters, scale):
        """Identify interface edges between communities."""
        if self.verbose:
            print(f"Identifying interface edges...")
        
        # Get cluster assignments
        clusters = df[cluster_col].values
        
        # Use full graph for interface detection
        sources = self.full_sources
        targets = self.full_targets
        distances = self.full_distances
        
        # Calculate similarities for analysis
        similarities = self._compute_similarities(distances, scale)
        
        # Convert pruned_clusters to array
        pruned_clusters_array = np.array(list(pruned_clusters), dtype=np.int64)
        
        # Run interface detection with detailed info
        (is_interface, edge_types, source_clusters, target_clusters, 
        interface_count, cross_count, pruned_count) = identify_interface_edges_detailed(
            sources, targets, distances, similarities, 
            clusters, pruned_clusters_array
        )
        
        # Get indices of interface edges
        interface_indices = np.where(is_interface)[0]
        
        if self.verbose:
            print(f"Found {interface_count} interface edges:")
            print(f"  - {cross_count} cross-community edges")
            print(f"  - {pruned_count} edges in pruned communities")
        
        if len(interface_indices) > 0:
            # Extract data for interface edges only
            interface_sources = sources[interface_indices]
            interface_targets = targets[interface_indices]
            interface_distances = distances[interface_indices]
            interface_similarities = similarities[interface_indices]
            interface_source_clusters = source_clusters[interface_indices]
            interface_target_clusters = target_clusters[interface_indices]
            interface_edge_types = edge_types[interface_indices]
            
            # Convert edge type codes to strings
            edge_type_map = {0: "cross_community", 1: "pruned_community"}
            edge_type_strings = [edge_type_map[t] for t in interface_edge_types]
            
            # Create DataFrame
            interface_edges_df = pd.DataFrame({
                'source': interface_sources,
                'target': interface_targets,
                'distance': interface_distances,
                'similarity': interface_similarities,
                'source_cluster': interface_source_clusters,
                'target_cluster': interface_target_clusters,
                'edge_type': edge_type_strings
            })
        else:
            # Empty DataFrame
            interface_edges_df = pd.DataFrame(columns=[
                'source', 'target', 'distance', 'similarity', 
                'source_cluster', 'target_cluster', 'edge_type'
            ])
        
        return interface_edges_df
    
    def extract_interface_edges(self, labels: np.ndarray) -> Dict[str, Any]:
        """Extract interface edges efficiently."""
        # Use original edges for interface detection
        sources, targets, distances = self.loader.edge_arrays
        
        # Detect interfaces
        is_interface = detect_interface_edges(sources, targets, labels)
        interface_indices = np.where(is_interface)[0]
        
        # Extract interface data
        if len(interface_indices) > 0:
            return {
                'sources': sources[interface_indices],
                'targets': targets[interface_indices],
                'distances': distances[interface_indices],
                'source_clusters': labels[sources[interface_indices]],
                'target_clusters': labels[targets[interface_indices]],
                'count': len(interface_indices)
            }
        else:
            return {'count': 0}
    
    def get_combined_edge_data(self, run_ids=None) -> pd.DataFrame:
        """
        Combine interface edges from multiple runs into a single dataframe.
        
        Parameters:
            run_ids: List of run IDs to include (default: all runs)
            
        Returns:
            DataFrame with combined edge data
        """
        if run_ids is None:
            run_ids = list(self.runs.keys())
            
        if not run_ids:
            raise ValueError("No runs available to combine")
        
        with perf_monitor.timed_operation("Combine edge data"):
            # Get all unique edges across runs
            all_edges = set()
            for run_id in run_ids:
                if run_id not in self.interface_edges:
                    continue
                    
                edges_df = self.interface_edges[run_id]
                for _, row in edges_df.iterrows():
                    source = min(row['source'], row['target'])
                    target = max(row['source'], row['target'])
                    all_edges.add((source, target))
            
            # Create a map of edge types for each run
            edge_types = {run_id: {} for run_id in run_ids}
            for run_id in run_ids:
                if run_id not in self.interface_edges:
                    continue
                    
                edges_df = self.interface_edges[run_id]
                for _, row in edges_df.iterrows():
                    source = min(row['source'], row['target'])
                    target = max(row['source'], row['target'])
                    edge_types[run_id][(source, target)] = row['edge_type']
            
            # Build the combined dataframe
            combined_data = []
            for source, target in all_edges:
                # Get edge properties
                edge_idx = None
                for idx, (s, t) in enumerate(zip(self.full_sources, self.full_targets)):
                    if (min(s, t), max(s, t)) == (source, target):
                        edge_idx = idx
                        break
                        
                if edge_idx is None:
                    continue
                    
                # Basic edge data
                edge_data = {
                    'source': source,
                    'target': target,
                    'distance': float(self.full_distances[edge_idx])
                }
                
                # Add data for each run
                in_any_run = False
                for run_id in run_ids:
                    if run_id in edge_types and (source, target) in edge_types[run_id]:
                        edge_data[f'in_{run_id}'] = True
                        edge_data[f'type_{run_id}'] = edge_types[run_id][(source, target)]
                        in_any_run = True
                    else:
                        edge_data[f'in_{run_id}'] = False
                        edge_data[f'type_{run_id}'] = 'not_interface'
                
                if in_any_run:
                    # Count how many runs included this edge
                    edge_data['run_count'] = sum(1 for run_id in run_ids 
                                               if f'in_{run_id}' in edge_data and edge_data[f'in_{run_id}'])
                    combined_data.append(edge_data)
            
            # Create dataframe
            self.combined_edges = pd.DataFrame(combined_data)
            
            if self.verbose and len(self.combined_edges) > 0:
                run_count_stats = self.combined_edges['run_count'].value_counts().sort_index()
                print("Edge counts by number of runs:")
                for count, freq in run_count_stats.items():
                    print(f"  - In {count}/{len(run_ids)} runs: {freq} edges")
                    
            return self.combined_edges
    
    def add_community_statistics(self, run_id, recalculate=False) -> pd.DataFrame:
        """
        Calculate and add community statistics like conductance for a specific run.
        Uses a single, parallel Numba pass over edges for all clusters.
        """
        if run_id not in self.runs:
            raise ValueError(f"Run '{run_id}' not found")
    
        run_data = self.runs[run_id]
        stats_df = run_data['cluster_stats']
    
        # skip if already present
        if 'conductance' in stats_df.columns and not recalculate:
            return stats_df
    
        with perf_monitor.timed_operation(f"Calculate community statistics (run={run_id})"):
            if self.verbose:
                print(f"Calculating community statistics for run '{run_id}'...")
    
            # 1) prepare inputs
            df          = run_data['df']
            col         = run_data['cluster_col']
            labels      = df[col].values
            unique_ids  = np.sort(stats_df['cluster_id'].unique())
            n_clusters  = unique_ids.shape[0]
    
            # map cluster_id → [0..n_clusters)
            id_to_idx = {int(cid): i for i, cid in enumerate(unique_ids)}
            # cluster index per node
            cluster_idx = np.array([id_to_idx[int(l)] for l in labels], dtype=np.int64)
    
            # full edge lists
            S = self.full_sources
            T = self.full_targets
            D = self.full_distances
            sims = self._compute_similarities(D, run_data.get('similarity_scale', 'adaptive'))
    
            # 2) accumulate everything in one parallel pass
            vol, cut, int_cnt, ext_cnt, sum_d, sumsq_d = _accumulate_stats(
                S, T, D, sims, cluster_idx, n_clusters
            )
    
            # 3) post‐process into final metrics
            # mean & std for internal distances
            mean_int = np.zeros(n_clusters, dtype=np.float64)
            std_int  = np.zeros(n_clusters, dtype=np.float64)
            for i in range(n_clusters):
                cnt = int_cnt[i]
                if cnt > 0:
                    mean_int[i] = sum_d[i] / cnt
                    # sample‐std
                    var = (sumsq_d[i] - (sum_d[i]**2)/cnt) / max(1, cnt - 1)
                    std_int[i]  = np.sqrt(var) if var > 0 else 0.0
    
            # conductance = cut / vol
            cond = np.zeros_like(vol)
            nz = vol > 0
            cond[nz] = cut[nz] / vol[nz]
    
            # node counts per cluster
            node_counts = np.bincount(cluster_idx, minlength=n_clusters)
            # possible edges per cluster
            max_edges = node_counts * (node_counts - 1) / 2
            density  = np.zeros(n_clusters, dtype=np.float64)
            nonzero  = max_edges > 0
            density[nonzero] = int_cnt[nonzero] / max_edges[nonzero]
    
            # edge‐to‐node ratio
            e2n = np.zeros(n_clusters, dtype=np.float64)
            nonz = node_counts > 0
            e2n[nonz] = int_cnt[nonz] / node_counts[nonz]
    
            # 4) write back into stats_df
            # create a helper map idx→cluster_id
            idx_to_id = {i: cid for i, cid in enumerate(unique_ids)}
    
            # prepare columns
            new_cols = defaultdict(list)
            for i in range(n_clusters):
                cid = idx_to_id[i]
                new_cols['conductance'].append(cond[i])
                new_cols['internal_edges'].append(int_cnt[i])
                new_cols['external_edges'].append(ext_cnt[i])
                new_cols['mean_edge_dist'].append(mean_int[i])
                new_cols['std_edge_dist'].append(std_int[i])
                new_cols['edge_density'].append(density[i])
                new_cols['edge_to_node_ratio'].append(e2n[i])
    
            # assign by matching cluster_id order
            # stats_df is already sorted by cluster_id
            for colname, vals in new_cols.items():
                stats_df[colname] = vals
    
            # coefficient of variation
            stats_df['coef_var'] = stats_df['std_edge_dist'] / stats_df['mean_edge_dist']
    
            # store and return
            self.runs[run_id]['cluster_stats'] = stats_df
            return stats_df
    
    def compare_runs(self, run_id1, run_id2) -> Dict:
        """
        Compare two runs using normalized mutual information and edge overlap.
        
        Parameters:
            run_id1: First run ID
            run_id2: Second run ID
            
        Returns:
            dict with comparison metrics
        """
        if run_id1 not in self.runs or run_id2 not in self.runs:
            raise ValueError(f"Both runs must exist")
        
        with perf_monitor.timed_operation(f"Compare runs {run_id1} vs {run_id2}"):
            run1 = self.runs[run_id1]
            run2 = self.runs[run_id2]
            
            # Get cluster assignments
            clusters1 = run1['df'][run1['cluster_col']].values
            clusters2 = run2['df'][run2['cluster_col']].values
            
            # Calculate NMI
            nmi = normalized_mutual_info_score(clusters1, clusters2)
            
            # Get interface edges for both runs
            if run_id1 not in self.interface_edges or run_id2 not in self.interface_edges:
                self.get_combined_edge_data([run_id1, run_id2])
                
            edges1 = self.interface_edges[run_id1]
            edges2 = self.interface_edges[run_id2]
            
            # Create sets of edge pairs
            edge_pairs1 = set([(min(row['source'], row['target']), max(row['source'], row['target'])) 
                             for _, row in edges1.iterrows()])
            edge_pairs2 = set([(min(row['source'], row['target']), max(row['source'], row['target'])) 
                             for _, row in edges2.iterrows()])
            
            # Find overlap and exclusive edges
            common_edges = edge_pairs1.intersection(edge_pairs2)
            only_in_run1 = edge_pairs1 - edge_pairs2
            only_in_run2 = edge_pairs2 - edge_pairs1
            
            # Calculate Jaccard similarity (intersection over union)
            jaccard_similarity = len(common_edges) / len(edge_pairs1.union(edge_pairs2)) if edge_pairs1 or edge_pairs2 else 0
            
            # Return comparison metrics
            comparison = {
                'nmi': nmi,
                'jaccard_similarity': jaccard_similarity,
                'common_edges': len(common_edges),
                'only_in_run1': len(only_in_run1),
                'only_in_run2': len(only_in_run2),
                'total_edges1': len(edge_pairs1),
                'total_edges2': len(edge_pairs2)
            }
            
            if self.verbose:
                print(f"Comparison between '{run_id1}' and '{run_id2}':")
                print(f"  - Normalized Mutual Information: {nmi:.4f}")
                print(f"  - Interface edge Jaccard similarity: {jaccard_similarity:.4f}")
                print(f"  - Common interface edges: {len(common_edges)}")
                print(f"  - Edges only in '{run_id1}': {len(only_in_run1)}")
                print(f"  - Edges only in '{run_id2}': {len(only_in_run2)}")
            
            return comparison
    
    def plot_community_comparison(self, run_ids=None, plot_type='scatter', 
                                 figsize=(12, 10), include_stats=True, 
                                 max_edges=1000, alpha=0.3, s=2,
                                 cmap='tab20', random_seed=None) -> plt.Figure:
        """
        Create visualization comparing communities across runs.
        
        Parameters:
            run_ids: List of run IDs to include (default: all runs)
            plot_type: Type of plot ('scatter', 'edge_heatmap', 'community_overlap')
            figsize: Figure size as (width, height)
            include_stats: Whether to include statistics in the plot
            max_edges: Maximum number of edges to plot (for performance)
            alpha: Transparency for nodes and edges
            s: Size of points in scatter plot
            cmap: Colormap for clusters
            random_seed: Random seed for sampling edges
            
        Returns:
            matplotlib figure
        """
        if run_ids is None:
            run_ids = list(self.runs.keys())
            
        if not run_ids:
            raise ValueError("No runs available to plot")
        
        with perf_monitor.timed_operation(f"Plot {plot_type} for {len(run_ids)} runs"):
            # Ensure we have combined edge data
            if self.combined_edges is None or not all(f'in_{run_id}' in self.combined_edges.columns for run_id in run_ids):
                self.get_combined_edge_data(run_ids)
                
            # Set random seed if provided
            if random_seed is not None:
                np.random.seed(random_seed)
                
            # Create figure
            fig = plt.figure(figsize=figsize)
            
            if plot_type == 'scatter':
                # UMAP scatter with interface edges
                if 'UMAP1' not in self.loader.node_df.columns or 'UMAP2' not in self.loader.node_df.columns:
                    # Check if we have embedding
                    if self.loader.embedding is not None and self.loader.embedding.shape[1] >= 2:
                        # Add UMAP coordinates
                        self.loader.node_df['UMAP1'] = self.loader.embedding[:, 0]
                        self.loader.node_df['UMAP2'] = self.loader.embedding[:, 1]
                    else:
                        raise ValueError("UMAP coordinates not found and no embedding available")
                    
                ax = fig.add_subplot(111)
                
                # Use first run for node coloring
                run_id = run_ids[0]
                run_data = self.runs[run_id]
                cluster_col = run_data['cluster_col']
                
                # Plot nodes with community colors
                scatter = ax.scatter(
                    self.loader.node_df['UMAP1'], 
                    self.loader.node_df['UMAP2'],
                    c=run_data['df'][cluster_col], 
                    cmap=cmap, 
                    s=s, 
                    alpha=alpha
                )
                
                # Create edge subsets
                edge_sets = {}
                for run_id in run_ids:
                    run_edges = self.combined_edges[self.combined_edges[f'in_{run_id}'] == True]
                    if len(run_edges) > max_edges:
                        run_edges = run_edges.sample(max_edges, random_state=random_seed)
                    edge_sets[run_id] = run_edges
                
                # Plot edges for each run
                colors = ['black', 'blue', 'red', 'green', 'purple', 'orange', 'cyan']
                for i, run_id in enumerate(run_ids):
                    color = colors[i % len(colors)]
                    edges = edge_sets[run_id]
                    
                    for _, edge in edges.iterrows():
                        source_idx = edge['source']
                        target_idx = edge['target']
                        source_pos = (self.loader.node_df.iloc[source_idx]['UMAP1'], 
                                     self.loader.node_df.iloc[source_idx]['UMAP2'])
                        target_pos = (self.loader.node_df.iloc[target_idx]['UMAP1'], 
                                     self.loader.node_df.iloc[target_idx]['UMAP2'])
                        ax.plot([source_pos[0], target_pos[0]], [source_pos[1], target_pos[1]], 
                               color=color, alpha=alpha, linewidth=0.5)
                
                # Add legend
                from matplotlib.lines import Line2D
                legend_elements = [Line2D([0], [0], color=colors[i % len(colors)], lw=2, label=run_id) 
                                 for i, run_id in enumerate(run_ids)]
                ax.legend(handles=legend_elements, title="Interface Edges")
                
                ax.set_title(f'Community Comparison: {", ".join(run_ids)}')
                ax.set_xlabel('UMAP1')
                ax.set_ylabel('UMAP2')
                
            elif plot_type == 'edge_heatmap':
                # Heatmap of edge overlap
                ax = fig.add_subplot(111)
                
                # Create matrix of runs vs edges
                edge_matrix = np.zeros((len(run_ids), len(self.combined_edges)))
                for i, run_id in enumerate(run_ids):
                    edge_matrix[i] = self.combined_edges[f'in_{run_id}'].astype(int)
                
                # Calculate similarity matrix
                similarity_matrix = np.zeros((len(run_ids), len(run_ids)))
                for i in range(len(run_ids)):
                    for j in range(len(run_ids)):
                        # Jaccard similarity
                        intersection = np.sum(edge_matrix[i] & edge_matrix[j])
                        union = np.sum(edge_matrix[i] | edge_matrix[j])
                        similarity_matrix[i, j] = intersection / union if union > 0 else 0
                
                # Plot heatmap
                sns.heatmap(similarity_matrix, annot=True, fmt='.3f', cmap='viridis',
                          xticklabels=run_ids, yticklabels=run_ids, ax=ax)
                ax.set_title('Interface Edge Similarity (Jaccard)')
                
            elif plot_type == 'community_overlap':
                # Community overlap analysis
                # Create contingency tables between all pairs of runs
                ax = fig.add_subplot(111)
                
                # Calculate NMI between all pairs
                nmi_matrix = np.zeros((len(run_ids), len(run_ids)))
                for i, run_id1 in enumerate(run_ids):
                    for j, run_id2 in enumerate(run_ids):
                        if i == j:
                            nmi_matrix[i, j] = 1.0
                        else:
                            clusters1 = self.runs[run_id1]['df'][self.runs[run_id1]['cluster_col']].values
                            clusters2 = self.runs[run_id2]['df'][self.runs[run_id2]['cluster_col']].values
                            nmi_matrix[i, j] = normalized_mutual_info_score(clusters1, clusters2)
                
                # Plot heatmap
                sns.heatmap(nmi_matrix, annot=True, fmt='.3f', cmap='viridis',
                          xticklabels=run_ids, yticklabels=run_ids, ax=ax)
                ax.set_title('Community Structure Similarity (NMI)')
                
            elif plot_type == 'cluster_stats':
                # Compare cluster statistics across runs
                if include_stats:
                    # Ensure we have stats for all runs
                    for run_id in run_ids:
                        if run_id in self.runs and 'conductance' not in self.runs[run_id]['cluster_stats'].columns:
                            self.add_community_statistics(run_id)
                
                # Create grid of plots
                fig, axes = plt.subplots(2, 2, figsize=figsize)
                fig.suptitle('Cluster Statistics Comparison', fontsize=16)
                axes = axes.flatten()
                
                # Plot different statistics
                stat_cols = ['conductance', 'edge_density', 'mean_edge_dist', 'coef_var']
                titles = ['Conductance', 'Edge Density', 'Mean Edge Distance', 'Coefficient of Variation']
                
                for i, (stat, title) in enumerate(zip(stat_cols, titles)):
                    ax = axes[i]
                    
                    for run_id in run_ids:
                        stats_df = self.runs[run_id]['cluster_stats']
                        if stat in stats_df.columns:
                            # Plot stat vs cluster size
                            ax.scatter(stats_df['size'], stats_df[stat], 
                                      alpha=0.7, label=run_id)
                    
                    ax.set_xlabel('Cluster Size')
                    ax.set_ylabel(title)
                    ax.set_xscale('log')
                    if stat in ['conductance', 'edge_density']:
                        ax.set_ylim(0, 1)
                
                # Add legend to last plot
                axes[-1].legend(title='Run ID')
                plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
            
            plt.tight_layout()
            
            # Store the figure
            self.visualizations[f"{plot_type}_{'-'.join(run_ids)}"] = fig
            
            return fig
    
    def update_graph_wrapper(self, run_id=None):
        """
        Update graph wrapper with a specific run's community assignments.
        
        Parameters:
            run_id: Run ID to use (default: most recent run)
            
        Returns:
            Updated graph wrapper
        """
        if run_id is None:
            if not self.runs:
                raise ValueError("No runs available")
            run_id = list(self.runs.keys())[-1]
            
        if run_id not in self.runs:
            raise ValueError(f"Run '{run_id}' not found")
        
        with perf_monitor.timed_operation(f"Update graph with run {run_id}"):
            run_data = self.runs[run_id]
            df = run_data['df']
            
            # Create a new graph wrapper
            graph_wrapper = self.loader.build_graph_wrapper()
            
            # Update node dataframe with run data
            columns_to_copy = [col for col in df.columns if col not in graph_wrapper.node_df.columns]
            for col in columns_to_copy:
                graph_wrapper.node_df[col] = df[col]
                
            if self.verbose:
                print(f"Updated graph with data from run '{run_id}'")
                print(f"Added columns: {columns_to_copy}")
                
            return graph_wrapper

# ============================================================================
# OUTPUT FUNCTIONS
# ============================================================================

def save_interface_edges_efficient(interface_data: Dict[str, Any], 
                                 output_file: str,
                                 metadata: Dict[str, Any]):
    """Save interface edges in efficient format."""
    with perf_monitor.timed_operation("Save interface edges"):
        if interface_data['count'] == 0:
            # Save empty result
            metadata['edge_count'] = 0
            with open(output_file + '_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            return
        
        # Create sparse matrix for edges
        s = interface_data['sources']
        t = interface_data['targets']
        
        # Canonical form
        canonical_s = np.minimum(s, t)
        canonical_t = np.maximum(s, t)
        
        n_nodes = metadata['n_nodes']
        edge_matrix = sp.coo_matrix(
            (np.ones(len(canonical_s), dtype=bool), (canonical_s, canonical_t)),
            shape=(n_nodes, n_nodes)
        ).tocsr()
        
        # Save sparse matrix
        edge_file = output_file + '_edges.npz'
        sp.save_npz(edge_file, edge_matrix)
        
        # Save attributes
        attr_file = output_file + '_attributes.npz'
        np.savez_compressed(
            attr_file,
            source_cluster=interface_data['source_clusters'].astype(np.int32),
            target_cluster=interface_data['target_clusters'].astype(np.int32),
            distance=interface_data['distances'].astype(np.float32)
        )
        
        # Update and save metadata
        metadata['edge_count'] = interface_data['count']
        metadata['files'] = {
            'edges': os.path.basename(edge_file),
            'attributes': os.path.basename(attr_file)
        }
        
        with open(output_file + '_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

def load_interface_edges(output_dir, run_id):
    """
    Load interface edges saved in optimized format.
    
    Returns:
        edge_list: List of (source, target) tuples
        attributes: Dictionary of edge attributes
        metadata: Metadata dictionary
    """
    with perf_monitor.timed_operation(f"Load interface edges for {run_id}"):
        # Load metadata
        metadata_file = os.path.join(output_dir, f"{run_id}_metadata.json")
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Get file paths
        edge_file = os.path.join(output_dir, metadata['files']['edges'])
        attr_file = os.path.join(output_dir, metadata['files']['attributes'])
        
        # Load sparse matrix
        edge_matrix = sp.load_npz(edge_file)
        
        # Convert to COO format to get edge list
        edge_matrix_coo = edge_matrix.tocoo()
        edge_list = list(zip(edge_matrix_coo.row, edge_matrix_coo.col))
        
        # Load attributes
        attributes = dict(np.load(attr_file))
        
        return edge_list, attributes, metadata

# ============================================================================
# PARALLEL PROCESSING
# ============================================================================

def process_single_resolution(resolution, analyzer, output_dir, run_name, 
                             scale, min_cluster_size, rank_stat_col, 
                             prev_labels=None, warm_start=True,
                             save_outputs=True, algorithm="leiden"):
    """Process a single resolution value."""
    if prev_labels is not None and warm_start:
        # Warm-start from previous resolution
        initial_membership = prev_labels
    else:
        initial_membership = None
    
    # Generate run ID
    run_id = f"{run_name}_res{resolution}"
    
    # Run community detection with the selected algorithm
    if algorithm.lower() == "leiden_igraph":
        # Run Leiden community detection
        cluster_stats, labels = analyzer.run_leiden_igraph(
            resolution=resolution,
            run_id=run_id,
            scale=scale,
            initial_membership=initial_membership,
            rank_stat_col=rank_stat_col,
            prune_small_clusters=True,
            min_cluster_size=min_cluster_size
        )
    elif algorithm.lower() == "louvain_csr":
        # Run CSR-native Louvain community detection
        cluster_stats, labels = analyzer.run_louvain_csr(
            resolution=resolution,
            run_id=run_id,
            scale=scale,
            initial_membership=initial_membership,
            rank_stat_col=rank_stat_col,
            prune_small_clusters=True,
            min_cluster_size=min_cluster_size
        )
    elif algorithm.lower() == "leiden_csr":
        # Run CSR-native Louvain community detection
        cluster_stats, labels = analyzer.run_leiden_csr(
            resolution=resolution,
            run_id=run_id,
            scale=scale,
            initial_membership=initial_membership,
            rank_stat_col=rank_stat_col,
            prune_small_clusters=True,
            min_cluster_size=min_cluster_size
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Use 'leiden' or 'louvain_csr'")
    
    # Calculate community statistics
    analyzer.add_community_statistics(run_id)
    
    # Save results if requested
    if save_outputs:
        # Create resolution-specific output directory
        res_dir = os.path.join(output_dir, f"res_{resolution}")
        os.makedirs(res_dir, exist_ok=True)
        
        # Extract interface edges
        interface_data = analyzer.extract_interface_edges(labels)
        
        # Create metadata
        metadata = {
            'run_name': run_name,
            'resolution': resolution,
            'scale': scale if isinstance(scale, (int, float)) else 'adaptive',
            'n_nodes': analyzer.loader.n_nodes,
            'timestamp': datetime.now().isoformat(),
            'coarsened': analyzer.coarsened,
            'coarsening_ratio': getattr(analyzer, 'coarsening_ratio', None),
            'algorithm': algorithm
        }
        
        # Save interface edges
        output_file = os.path.join(res_dir, f"{run_name}_res{resolution}")
        save_interface_edges_efficient(interface_data, output_file, metadata)
        
        # Save cluster statistics
        stats_file = os.path.join(res_dir, f"{run_name}_res{resolution}_stats.csv")
        cluster_stats.to_csv(stats_file, index=False)
    
    return labels

def prepare_shared_data(analyzer, output_dir):
    """Prepare data for shared memory access across processes."""
    # Create temporary files
    node_df_path = os.path.join(output_dir, "temp_node_df.pkl")
    edges_path = os.path.join(output_dir, "temp_edges.npz")
    csr_path = os.path.join(output_dir, "temp_csr.npz")
    
    # Save to disk
    analyzer.loader.node_df.to_pickle(node_df_path)
    
    # Save edge arrays
    np.savez_compressed(
        edges_path,
        sources=analyzer.full_sources.astype(np.int32),
        targets=analyzer.full_targets.astype(np.int32),
        distances=analyzer.full_distances.astype(np.float32)
    )
    
    # Save CSR structure
    np.savez_compressed(
        csr_path,
        offsets=analyzer.csr_offsets.astype(np.int32),
        indices=analyzer.csr_indices.astype(np.int32)
    )
    
    # Return paths
    return {
        'node_df': node_df_path,
        'edges': edges_path,
        'csr': csr_path,
        'n_nodes': analyzer.loader.n_nodes
    }

def process_resolution_shared(resolution, shared_data, output_dir, run_name, 
                            scale, min_cluster_size, rank_stat_col,
                            coarsen=True, coarsen_levels=1,
                            sparsify=True, sparsify_post_k=60,
                            algorithm="louvain_csr"):  # Default to CSR Louvain
    """Process a single resolution using shared data."""
    # Load shared data
    node_df = pd.read_pickle(shared_data['node_df'])
    
    # Load edge arrays
    edge_data = np.load(shared_data['edges'])
    sources = np.asarray(edge_data['sources'], dtype=np.int64)
    targets = np.asarray(edge_data['targets'], dtype=np.int64)
    distances = np.asarray(edge_data['distances'], dtype=np.float64)
    
    # Load CSR structure
    csr_data = np.load(shared_data['csr'])
    offsets = np.asarray(csr_data['offsets'], dtype=np.int64)
    indices = np.asarray(csr_data['indices'], dtype=np.int64)
    
    # Create mini graph loader
    class MiniLoader:
        def __init__(self):
            self.node_df = node_df
            self.n_nodes = shared_data['n_nodes']
            self._edge_arrays = (sources, targets, distances)
        
        @property
        def edge_arrays(self):
            return self._edge_arrays
    
    # Create analyzer
    analyzer = OptimizedCommunityAnalyzer(
        MiniLoader(),
        coarsen=coarsen,
        coarsen_levels=coarsen_levels,
        sparsify=sparsify,
        sparsify_post_k=sparsify_post_k,
        verbose=True
    )
    
    # Restore CSR structure
    analyzer.csr_offsets = offsets
    analyzer.csr_indices = indices
    
    # Run the selected community detection algorithm
    run_id = f"{run_name}_res{resolution}"
    
    if algorithm.lower() == "leiden":
        # Run Leiden (requires igraph conversion)
        cluster_stats, labels = analyzer.run_leiden_igraph(
            resolution=resolution,
            run_id=run_id,
            scale=scale,
            rank_stat_col=rank_stat_col,
            prune_small_clusters=True,
            min_cluster_size=min_cluster_size
        )
    else:
        # Run CSR-native Louvain (default, avoids igraph conversion)
        cluster_stats, labels = analyzer.run_louvain_csr(
            resolution=resolution,
            run_id=run_id,
            scale=scale,
            rank_stat_col=rank_stat_col,
            prune_small_clusters=True,
            min_cluster_size=min_cluster_size
        )
    
    # Calculate statistics
    analyzer.add_community_statistics(run_id)
    
    # Save results
    res_dir = os.path.join(output_dir, f"res_{resolution}")
    os.makedirs(res_dir, exist_ok=True)
    
    # Extract interface edges
    interface_data = analyzer.extract_interface_edges(labels)
    
    # Create metadata
    metadata = {
        'run_name': run_name,
        'resolution': resolution,
        'scale': scale if isinstance(scale, (int, float)) else 'adaptive',
        'n_nodes': analyzer.loader.n_nodes,
        'timestamp': datetime.now().isoformat(),
        'coarsened': analyzer.coarsened,
        'coarsening_ratio': getattr(analyzer, 'coarsening_ratio', None),
        'algorithm': algorithm
    }
    
    # Save interface edges
    output_file = os.path.join(res_dir, f"{run_name}_res{resolution}")
    save_interface_edges_efficient(interface_data, output_file, metadata)
    
    # Save cluster statistics
    stats_file = os.path.join(res_dir, f"{run_name}_res{resolution}_stats.csv")
    cluster_stats.to_csv(stats_file, index=False)
    
    return resolution

# ============================================================================
# MAIN PIPELINE
# ============================================================================

@click.command()
@click.option('--location', type=click.Path(exists=True), required=True,
              help="Location of the input graph directory.")
@click.option('--output-dir', type=click.Path(), required=True,
              help="Directory to save output files.")
@click.option('--run-name', type=str, required=True,
              help="Base name for this analysis run.")
@click.option('--resolutions', type=str, required=True,
              help="Comma-separated resolutions to process.")
@click.option('--similarity-scale', type=float, default=None,
              help="Scale for similarity function (default: adaptive median).")
@click.option('--coarsen/--no-coarsen', default=False,
              help="Coarsen the graph before community detection.")
@click.option('--coarsen-levels', type=int, default=1,
              help="Number of coarsening levels (default: 1).")
@click.option('--sparsify/--no-sparsify', default=False,
              help="Sparsify the graph before community detection.")
@click.option('--sparsify-pre-k', type=int, default=60,
              help="Number of neighbors to keep for pre-coarsening sparsification.")
@click.option('--sparsify-post-k', type=int, default=None,
              help="Number of neighbors to keep for post-coarsening sparsification.")
@click.option('--warm-start/--no-warm-start', default=True,
              help="Use warm-start for sequential resolutions.")
@click.option('--min-cluster-size', type=int, default=None,
              help="Minimum cluster size to keep.")
@click.option('--rank-stat-col', type=str, default=None,
              help="Column to use for ranking clusters.")
@click.option('--parallel/--no-parallel', default=False,
              help="Run multiple resolutions in parallel.")
@click.option('--n-jobs', type=int, default=-1,
              help="Number of parallel jobs (-1 = all cores).")
@click.option('--shared-memory/--no-shared-memory', default=True,
              help="Use shared memory for parallel processing.")
@click.option('--timing/--no-timing', default=True,
              help="Enable detailed timing statistics.")
@click.option('--algorithm', type=click.Choice(['leiden_igraph', 'louvain_csr', 'leiden_csr']), default='louvain_csr',
              help="Community detection algorithm to use (default: louvain_csr)")
def main(location, output_dir, run_name, resolutions, similarity_scale,
         coarsen, coarsen_levels, sparsify, sparsify_pre_k, sparsify_post_k, warm_start,
         min_cluster_size, rank_stat_col, parallel, n_jobs, shared_memory, 
         timing, algorithm):
    """Optimized community detection pipeline."""
    # Configure performance monitoring
    global perf_monitor
    perf_monitor.enabled = timing
    perf_monitor.reset()
    
    print("Starting optimized pipeline...")
    
    # Parse resolutions
    resolution_values = sorted([float(r) for r in resolutions.split(',')])
    print(f"Processing {len(resolution_values)} resolutions: {resolution_values}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize graph loader
    print("\nLoading graph...")
    with perf_monitor.timed_operation("Graph loading", verbose=True):
        loader = OptimizedGraphLoader(location)
        print(f"  Nodes: {loader.n_nodes:,}")
        print(f"  Edges: {loader.n_edges:,}")
    
    # Initialize analyzer (with lazy initialization)
    print("\nInitializing analyzer...")
    with perf_monitor.timed_operation("Analyzer initialization", verbose=True):
        analyzer = OptimizedCommunityAnalyzer(
            loader,
            coarsen=coarsen,
            coarsen_levels=coarsen_levels,
            sparsify=sparsify,
            sparsify_pre_k=sparsify_pre_k,
            sparsify_post_k=sparsify_post_k,
            verbose=True
        )
        
        # Pre-build CSR for interface detection
        print("\nPre-building CSR for interface detection...")
        analyzer._ensure_csr_built()
    
    # Process resolutions
    scale = similarity_scale if similarity_scale else 'adaptive'
    
    # Process resolutions
    scale = similarity_scale if similarity_scale else 'adaptive'
    
    if not parallel:
        # Sequential processing with warm-start
        print("\nProcessing resolutions sequentially...")
        prev_labels = None
        
        for resolution in resolution_values:
            with perf_monitor.timed_operation(f"Process resolution {resolution}", verbose=True):
                labels = process_single_resolution(
                    resolution, analyzer, output_dir, run_name, scale,
                    min_cluster_size, rank_stat_col, 
                    prev_labels=prev_labels if warm_start else None,
                    algorithm=algorithm  # Pass algorithm choice
                )
                
                # Store labels for next resolution if warm-start enabled
                if warm_start:
                    prev_labels = labels
    else:
        # Parallel processing
        if n_jobs == -1:
            n_jobs = mp.cpu_count()
        
        print(f"\nProcessing resolutions in parallel with {n_jobs} processes...")
        
        if shared_memory:
            # Use shared memory
            with perf_monitor.timed_operation("Prepare shared data", verbose=True):
                shared_data = prepare_shared_data(analyzer, output_dir)
            
            # Process resolutions in parallel
            with perf_monitor.timed_operation("Parallel processing", verbose=True):
                pool = mp.Pool(processes=n_jobs)
                
                # Create partial function
                worker_func = partial(
                    process_resolution_shared,
                    shared_data=shared_data,
                    output_dir=output_dir,
                    run_name=run_name,
                    scale=scale,
                    min_cluster_size=min_cluster_size,
                    rank_stat_col=rank_stat_col,
                    coarsen=coarsen,
                    coarsen_levels=coarsen_levels,
                    sparsify=sparsify,
                    sparsify_post_k=sparsify_post_k,
                    algorithm=algorithm
                )
                
                # Run parallel jobs
                results = pool.map(worker_func, resolution_values)
                pool.close()
                pool.join()
            
            # Clean up temporary files
            with perf_monitor.timed_operation("Clean up temporary files", verbose=True):
                for key, path in shared_data.items():
                    if isinstance(path, str) and os.path.exists(path):
                        os.remove(path)
                        print(f"  Removed temporary file: {path}")
        else:
            # Each process loads its own data
            with perf_monitor.timed_operation("Parallel processing (independent)", verbose=True):
                pool = mp.Pool(processes=n_jobs)
                
                # Define worker function for independent processing
                def independent_worker(resolution):
                    # Load graph
                    loader = OptimizedGraphLoader(location)
                    
                    # Initialize analyzer
                    analyzer = OptimizedCommunityAnalyzer(
                        loader,
                        coarsen=coarsen,
                        coarsen_levels=coarsen_levels,
                        sparsify=sparsify,
                        sparsify_pre_k=sparsify_pre_k,
                        sparsify_post_k=sparsify_post_k,
                        verbose=False  # Reduce output in parallel mode
                    )
                    
                    # Process resolution
                    return process_single_resolution(
                        resolution, analyzer, output_dir, run_name, scale,
                        min_cluster_size, rank_stat_col,
                        prev_labels=None,  # No warm-start in independent mode
                        save_outputs=True
                    )
                
                # Run parallel jobs
                results = pool.map(independent_worker, resolution_values)
                pool.close()
                pool.join()
    
    # Summary
    print("\nPipeline complete!")
    print(f"Processed {len(resolution_values)} resolutions: {resolution_values}")
    print(f"Results saved to: {output_dir}")
    
    # Print timing summary
    if timing:
        perf_monitor.print_timing_summary()

    print("\nClosing analyzer pool...")
    analyzer.close()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_results(output_dir, run_name=None, resolution=None):
    """
    Load results from a previous run.
    
    Parameters:
        output_dir: Directory containing results
        run_name: Optional run name filter
        resolution: Optional resolution filter
        
    Returns:
        Dictionary of loaded results
    """
    results = {}
    
    # Find all resolution directories
    res_dirs = [d for d in os.listdir(output_dir) if d.startswith('res_') and 
               os.path.isdir(os.path.join(output_dir, d))]
    
    for res_dir in res_dirs:
        # Extract resolution from directory name
        try:
            res_value = float(res_dir.split('_')[1])
        except (IndexError, ValueError):
            continue
            
        # Apply resolution filter if specified
        if resolution is not None and res_value != resolution:
            continue
            
        # Find metadata files in this directory
        meta_files = [f for f in os.listdir(os.path.join(output_dir, res_dir)) 
                     if f.endswith('_metadata.json')]
        
        for meta_file in meta_files:
            # Apply run name filter if specified
            if run_name is not None and not meta_file.startswith(run_name):
                continue
                
            # Load metadata
            meta_path = os.path.join(output_dir, res_dir, meta_file)
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
                
            # Extract run ID
            run_id = metadata.get('run_name', '') + f"_res{res_value}"
            
            # Load interface edges
            try:
                edge_list, attributes, _ = load_interface_edges(os.path.join(output_dir, res_dir), run_id)
                
                # Load cluster statistics if available
                stats_file = os.path.join(output_dir, res_dir, f"{run_id}_stats.csv")
                cluster_stats = None
                if os.path.exists(stats_file):
                    cluster_stats = pd.read_csv(stats_file)
                
                # Store results
                results[run_id] = {
                    'metadata': metadata,
                    'edge_list': edge_list,
                    'attributes': attributes,
                    'cluster_stats': cluster_stats,
                    'resolution': res_value
                }
            except Exception as e:
                print(f"Error loading results for {run_id}: {e}")
    
    return results

def visualize_results(results, figsize=(10, 8), plot_type='scatter', **kwargs):
    """
    Visualize results from a previous run.
    
    Parameters:
        results: Dictionary of results from load_results
        figsize: Figure size
        plot_type: Type of plot ('scatter', 'edge_heatmap', 'community_overlap', 'cluster_stats')
        **kwargs: Additional arguments passed to plotting function
        
    Returns:
        matplotlib figure
    """
    import matplotlib.pyplot as plt
    
    if not results:
        raise ValueError("No results to visualize")
    
    fig = plt.figure(figsize=figsize)
    
    if plot_type == 'resolution_comparison':
        # Compare statistics across resolutions
        run_ids = list(results.keys())
        resolutions = [results[run_id]['resolution'] for run_id in run_ids]
        
        # Extract metrics
        metrics = {
            'n_clusters': [],
            'edge_count': [],
            'coarsening_ratio': []
        }
        
        for run_id in run_ids:
            result = results[run_id]
            metadata = result['metadata']
            
            # Get number of clusters
            if result['cluster_stats'] is not None:
                metrics['n_clusters'].append(len(result['cluster_stats']))
            else:
                metrics['n_clusters'].append(None)
            
            # Get edge count
            metrics['edge_count'].append(metadata.get('edge_count', 0))
            
            # Get coarsening ratio
            metrics['coarsening_ratio'].append(metadata.get('coarsening_ratio', 1.0))
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        # Sort data by resolution
        sorted_indices = np.argsort(resolutions)
        sorted_resolutions = [resolutions[i] for i in sorted_indices]
        
        # Plot metrics
        for i, (metric, values) in enumerate(metrics.items()):
            ax = axes[i]
            sorted_values = [values[i] for i in sorted_indices if values[i] is not None]
            sorted_res = [sorted_resolutions[i] for i in sorted_indices if values[i] is not None]
            
            if sorted_values:
                ax.plot(sorted_res, sorted_values, 'o-')
                ax.set_xlabel('Resolution')
                ax.set_ylabel(metric.replace('_', ' ').title())
                ax.set_title(f'{metric.replace("_", " ").title()} vs Resolution')
                
                # Add grid
                ax.grid(True, linestyle='--', alpha=0.7)
        
        # Fourth plot - interface edge count histogram for highest resolution
        if metrics['edge_count']:
            max_idx = np.argmax([metrics['edge_count'][i] for i in sorted_indices])
            run_id = run_ids[sorted_indices[max_idx]]
            
            # Get edge list
            edge_list = results[run_id]['edge_list']
            
            # Count edges per node
            node_edges = defaultdict(int)
            for src, tgt in edge_list:
                node_edges[src] += 1
                node_edges[tgt] += 1
            
            # Plot histogram
            ax = axes[3]
            ax.hist(list(node_edges.values()), bins=30)
            ax.set_xlabel('Interface Edges per Node')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Interface Edge Distribution (res={resolutions[sorted_indices[max_idx]]})')
            ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
    elif plot_type == 'cluster_stats_comparison':
        # Compare cluster statistics across resolutions
        run_ids = list(results.keys())
        stats_runs = [run_id for run_id in run_ids if results[run_id]['cluster_stats'] is not None]
        
        if not stats_runs:
            raise ValueError("No cluster statistics available")
        
        # Get available statistics columns
        all_columns = set()
        for run_id in stats_runs:
            all_columns.update(results[run_id]['cluster_stats'].columns)
        
        # Remove non-numeric columns
        numeric_columns = []
        for col in all_columns:
            if any(results[run_id]['cluster_stats'][col].dtype.kind in 'fib' 
                  for run_id in stats_runs if col in results[run_id]['cluster_stats']):
                numeric_columns.append(col)
        
        # Select columns to plot (max 4)
        plot_columns = ['size', 'conductance', 'edge_density', 'mean_edge_dist']
        plot_columns = [col for col in plot_columns if col in numeric_columns][:4]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        # Plot metrics
        for i, col in enumerate(plot_columns):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            for run_id in stats_runs:
                if col in results[run_id]['cluster_stats'].columns:
                    stats_df = results[run_id]['cluster_stats']
                    ax.scatter(stats_df['size'], stats_df[col], alpha=0.7, 
                               label=f"res={results[run_id]['resolution']}")
            
            ax.set_xlabel('Cluster Size')
            ax.set_ylabel(col.replace('_', ' ').title())
            ax.set_title(f'{col.replace("_", " ").title()} vs Size')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Log scale for size
            ax.set_xscale('log')
            
            # Add legend
            ax.legend()
        
        plt.tight_layout()
        
    elif plot_type == 'edge_overlap':
        # Visualize edge overlap between runs
        run_ids = list(results.keys())
        
        if len(run_ids) < 2:
            raise ValueError("Need at least 2 runs to compare edge overlap")
        
        # Create edge sets for each run
        edge_sets = {}
        for run_id in run_ids:
            edge_list = results[run_id]['edge_list']
            canonical_edges = set()
            for src, tgt in edge_list:
                canonical_edges.add((min(src, tgt), max(src, tgt)))
            edge_sets[run_id] = canonical_edges
        
        # Create overlap matrix
        n_runs = len(run_ids)
        overlap_matrix = np.zeros((n_runs, n_runs))
        
        for i, run_id1 in enumerate(run_ids):
            for j, run_id2 in enumerate(run_ids):
                # Jaccard similarity
                intersection = len(edge_sets[run_id1] & edge_sets[run_id2])
                union = len(edge_sets[run_id1] | edge_sets[run_id2])
                overlap_matrix[i, j] = intersection / union if union > 0 else 0
        
        # Plot heatmap
        import seaborn as sns
        ax = plt.gca()
        
        # Format resolution values for labels
        labels = [f"res={results[run_id]['resolution']}" for run_id in run_ids]
        
        # Plot heatmap
        sns.heatmap(overlap_matrix, annot=True, fmt='.3f', cmap='viridis',
                  xticklabels=labels, yticklabels=labels, ax=ax)
        
        plt.title('Interface Edge Overlap (Jaccard Similarity)')
        plt.tight_layout()
        
    elif plot_type == 'network_view':
        # Network visualization of communities and interface edges
        import networkx as nx
        
        # Select a run to visualize
        if 'run_id' in kwargs:
            run_id = kwargs['run_id']
            if run_id not in results:
                raise ValueError(f"Run ID '{run_id}' not found")
        else:
            # Use the first run
            run_id = list(results.keys())[0]
        
        result = results[run_id]
        edge_list = result['edge_list']
        
        # Create graph
        G = nx.Graph()
        
        # Add edges
        for src, tgt in edge_list:
            G.add_edge(src, tgt)
        
        # Create layout
        if 'pos' in kwargs:
            pos = kwargs['pos']
        else:
            # Use spring layout
            pos = nx.spring_layout(G, seed=42)
        
        # Plot
        ax = plt.gca()
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=20, node_color='blue', alpha=0.7, ax=ax)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.5, ax=ax)
        
        plt.title(f'Interface Network for {run_id} (Resolution: {result["resolution"]})')
        plt.axis('off')
        
    else:
        plt.text(0.5, 0.5, f"Plot type '{plot_type}' not implemented", 
                ha='center', va='center', fontsize=14)
    
    return fig

def merge_interface_edges(output_dir, run_ids, output_file, min_overlap=1):
    """
    Merge interface edges from multiple runs into a single output.
    
    Parameters:
        output_dir: Directory containing results
        run_ids: List of run IDs to merge
        output_file: Output file path
        min_overlap: Minimum number of runs an edge must appear in (default: 1)
        
    Returns:
        Dictionary with merge statistics
    """
    # Load results
    results = load_results(output_dir)
    
    # Filter results
    selected_results = {run_id: results[run_id] for run_id in run_ids if run_id in results}
    
    if not selected_results:
        raise ValueError("No matching results found")
    
    # Collect all edges with counts
    edge_counts = defaultdict(int)
    edge_appearances = defaultdict(set)
    
    for run_id, result in selected_results.items():
        edge_list = result['edge_list']
        
        # Convert to canonical form (smaller index first)
        for src, tgt in edge_list:
            edge = (min(src, tgt), max(src, tgt))
            edge_counts[edge] += 1
            edge_appearances[edge].add(run_id)
    
    # Filter by minimum overlap
    filtered_edges = {edge: count for edge, count in edge_counts.items() if count >= min_overlap}
    
    # Create a matrix representation
    if filtered_edges:
        n_nodes = max(max(src, tgt) for src, tgt in filtered_edges.keys()) + 1
        merged_matrix = sp.lil_matrix((n_nodes, n_nodes), dtype=np.int16)
        
        for (src, tgt), count in filtered_edges.items():
            merged_matrix[src, tgt] = count
            merged_matrix[tgt, src] = count
        
        # Convert to CSR for efficient storage
        merged_matrix = merged_matrix.tocsr()
        
        # Save merged edges
        sp.save_npz(output_file + '_edges.npz', merged_matrix)
    else:
        n_nodes = 0
    
    # Save metadata
    metadata = {
        'source_runs': list(selected_results.keys()),
        'min_overlap': min_overlap,
        'edge_counts': {
            'total_unique_edges': len(edge_counts),
            'filtered_edges': len(filtered_edges),
            'per_run': {run_id: len(result['edge_list']) for run_id, result in selected_results.items()}
        },
        'n_nodes': n_nodes,
        'timestamp': datetime.now().isoformat(),
        'overlap_distribution': {str(count): len([e for e, c in edge_counts.items() if c == count]) 
                                for count in range(1, len(selected_results) + 1)}
    }
    
    with open(output_file + '_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Create edge appearance mapping
    if filtered_edges:
        appearance_data = {}
        for edge, runs in edge_appearances.items():
            if len(runs) >= min_overlap:
                src, tgt = edge
                key = f"{src}-{tgt}"
                appearance_data[key] = list(runs)
        
        with open(output_file + '_appearances.json', 'w') as f:
            json.dump(appearance_data, f)
    
    return metadata

def create_spatial_visualization(output_dir, run_id, coordinates_file, output_file=None, 
                               point_size=5, edge_width=0.5, alpha=0.7, cmap='tab20',
                               max_edges=5000, seed=42):
    """
    Create a spatial visualization of communities and interface edges.
    
    Parameters:
        output_dir: Directory containing results
        run_id: Run ID to visualize
        coordinates_file: File containing spatial coordinates (CSV or NPY)
        output_file: Output file path (default: None, display only)
        point_size: Size of points in the plot
        edge_width: Width of edges in the plot
        alpha: Transparency of points and edges
        cmap: Colormap for communities
        max_edges: Maximum number of edges to plot (for performance)
        seed: Random seed for edge sampling
        
    Returns:
        matplotlib figure
    """
    import matplotlib.pyplot as plt
    
    # Load results
    results = load_results(output_dir)
    
    if run_id not in results:
        raise ValueError(f"Run ID '{run_id}' not found")
    
    result = results[run_id]
    edge_list = result['edge_list']
    
    # Load coordinates
    if coordinates_file.endswith('.npy'):
        coordinates = np.load(coordinates_file)
    elif coordinates_file.endswith('.csv'):
        coordinates = pd.read_csv(coordinates_file).values
    else:
        raise ValueError("Coordinates file must be .npy or .csv")
    
    # Load cluster assignments if available
    cluster_file = os.path.join(output_dir, f"res_{result['resolution']}", f"{run_id}_stats.csv")
    if os.path.exists(cluster_file):
        cluster_stats = pd.read_csv(cluster_file)
        cluster_data = True
    else:
        cluster_data = False
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot nodes
    if coordinates.shape[1] >= 2:
        # Use first two columns as coordinates
        x, y = coordinates[:, 0], coordinates[:, 1]
        
        if cluster_data:
            # Color by cluster
            clusters = result['attributes']['source_cluster']
            scatter = ax.scatter(x, y, s=point_size, c=clusters, cmap=cmap, alpha=alpha)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Cluster ID')
        else:
            # Single color
            ax.scatter(x, y, s=point_size, color='blue', alpha=alpha)
    
    # Sample edges if needed
    np.random.seed(seed)
    if len(edge_list) > max_edges:
        indices = np.random.choice(len(edge_list), max_edges, replace=False)
        edge_list = [edge_list[i] for i in indices]
    
    # Plot edges
    for src, tgt in edge_list:
        if src < len(coordinates) and tgt < len(coordinates):
            ax.plot([coordinates[src, 0], coordinates[tgt, 0]],
                   [coordinates[src, 1], coordinates[tgt, 1]],
                   color='black', linewidth=edge_width, alpha=alpha*0.5)
    
    # Set title and labels
    ax.set_title(f'Spatial Visualization of Interface Edges (Resolution: {result["resolution"]})')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    
    # Set equal aspect ratio
    ax.set_aspect('equal')
    
    # Save if requested
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    return fig

def compare_across_resolutions(output_dir, run_name, similarity_metric='jaccard'):
    """
    Compare interface edges across different resolutions.
    
    Parameters:
        output_dir: Directory containing results
        run_name: Run name to filter results
        similarity_metric: 'jaccard' or 'overlap' for comparison
        
    Returns:
        Dictionary with comparison results
    """
    # Load results
    results = load_results(output_dir, run_name=run_name)
    
    if not results:
        raise ValueError(f"No results found for run name '{run_name}'")
    
    # Sort runs by resolution
    run_ids = sorted(results.keys(), key=lambda x: results[x]['resolution'])
    resolutions = [results[run_id]['resolution'] for run_id in run_ids]
    
    # Create edge sets for each run
    edge_sets = {}
    for run_id in run_ids:
        edge_list = results[run_id]['edge_list']
        canonical_edges = set()
        for src, tgt in edge_list:
            canonical_edges.add((min(src, tgt), max(src, tgt)))
        edge_sets[run_id] = canonical_edges
    
    # Compare adjacent resolutions
    comparisons = []
    
    for i in range(len(run_ids) - 1):
        run_id1 = run_ids[i]
        run_id2 = run_ids[i + 1]
        
        edges1 = edge_sets[run_id1]
        edges2 = edge_sets[run_id2]
        
        # Calculate similarity
        intersection = len(edges1 & edges2)
        
        if similarity_metric == 'jaccard':
            union = len(edges1 | edges2)
            similarity = intersection / union if union > 0 else 0
        else:  # overlap
            similarity = intersection / min(len(edges1), len(edges2)) if min(len(edges1), len(edges2)) > 0 else 0
        
        # Store comparison
        comparisons.append({
            'run_id1': run_id1,
            'run_id2': run_id2,
            'resolution1': results[run_id1]['resolution'],
            'resolution2': results[run_id2]['resolution'],
            'edges1': len(edges1),
            'edges2': len(edges2),
            'intersection': intersection,
            'similarity': similarity
        })
    
    # Calculate stability measure (average similarity)
    avg_similarity = np.mean([comp['similarity'] for comp in comparisons]) if comparisons else 0
    
    # Return results
    return {
        'run_name': run_name,
        'resolutions': resolutions,
        'comparisons': comparisons,
        'avg_similarity': avg_similarity,
        'similarity_metric': similarity_metric
    }

def extract_edge_list(output_dir, run_id, output_file=None, format='edgelist'):
    """
    Extract edge list in various formats for use with other tools.
    
    Parameters:
        output_dir: Directory containing results
        run_id: Run ID to extract edges from
        output_file: Output file path (default: None, return only)
        format: Output format ('edgelist', 'gexf', 'graphml', 'csv')
        
    Returns:
        Edge list in the specified format or path to saved file
    """
    # Load results
    results = load_results(output_dir)
    
    if run_id not in results:
        raise ValueError(f"Run ID '{run_id}' not found")
    
    result = results[run_id]
    edge_list = result['edge_list']
    
    # Process according to format
    if format == 'edgelist':
        # Simple edge list format: source target [weight]
        output = '\n'.join(f"{src} {tgt}" for src, tgt in edge_list)
        
    elif format == 'gexf' or format == 'graphml':
        # Use networkx to create graph format
        import networkx as nx
        
        G = nx.Graph()
        
        # Add edges
        for src, tgt in edge_list:
            G.add_edge(src, tgt)
        
        # Add edge attributes if available
        if 'distance' in result['attributes']:
            distances = result['attributes']['distance']
            for i, (src, tgt) in enumerate(edge_list):
                if i < len(distances):
                    G[src][tgt]['distance'] = float(distances[i])
        
        # Add node cluster information if available
        if 'source_cluster' in result['attributes']:
            clusters = result['attributes']['source_cluster']
            source_nodes = set(src for src, _ in edge_list)
            for node in source_nodes:
                idx = next((i for i, (src, _) in enumerate(edge_list) if src == node), None)
                if idx is not None and idx < len(clusters):
                    G.nodes[node]['cluster'] = int(clusters[idx])
        
        # Export to requested format
        if output_file:
            if format == 'gexf':
                nx.write_gexf(G, output_file)
            else:  # graphml
                nx.write_graphml(G, output_file)
            output = output_file
        else:
            # Return string representation
            import io
            buffer = io.StringIO()
            if format == 'gexf':
                nx.write_gexf(G, buffer)
            else:  # graphml
                nx.write_graphml(G, buffer)
            output = buffer.getvalue()
            
    elif format == 'csv':
        # CSV format with headers
        headers = ['source', 'target']
        rows = [[src, tgt] for src, tgt in edge_list]
        
        # Add attributes if available
        if 'distance' in result['attributes']:
            headers.append('distance')
            distances = result['attributes']['distance']
            for i, row in enumerate(rows):
                if i < len(distances):
                    row.append(float(distances[i]))
        
        if 'source_cluster' in result['attributes'] and 'target_cluster' in result['attributes']:
            headers.extend(['source_cluster', 'target_cluster'])
            source_clusters = result['attributes']['source_cluster']
            target_clusters = result['attributes']['target_cluster']
            for i, row in enumerate(rows):
                if i < len(source_clusters) and i < len(target_clusters):
                    row.extend([int(source_clusters[i]), int(target_clusters[i])])
        
        # Create CSV content
        import csv
        if output_file:
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(rows)
            output = output_file
        else:
            # Return string representation
            import io
            buffer = io.StringIO()
            writer = csv.writer(buffer)
            writer.writerow(headers)
            writer.writerows(rows)
            output = buffer.getvalue()
    
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    return output

# Run the pipeline if executed directly
if __name__ == "__main__":
    main()