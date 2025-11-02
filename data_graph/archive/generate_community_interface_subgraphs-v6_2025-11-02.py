#!/usr/bin/env python3
"""
Optimized Community Interface Detection Pipeline

Features:
1. High-performance core with Numba acceleration
2. Memory-efficient data structures (int32/float32)
3. Cached computations
4. Streamlined I/O
5. Advanced analytics and visualization
6. Numba-based parallel processing
7. Comprehensive timing statistics

This implementation focuses on single-process performance with Numba acceleration.
"""

import sys
import os
import time
import json
import pickle
from functools import lru_cache
from collections import defaultdict, Counter
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any, Set

import os, multiprocessing as mp
for k in ["OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS",
          "NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"]:
    os.environ.setdefault(k, "1")
try:
    mp.set_start_method("forkserver")
except RuntimeError:
    pass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import normalized_mutual_info_score
import click
import numba as nb
from numba import njit, prange
from contextlib import contextmanager
import psutil
import math
import gc

# Optional imports - loaded only when needed
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

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
@nb.njit(parallel=True, cache=True)
def _mask_upper_cross_labels(s: np.ndarray, t: np.ndarray, labels: np.ndarray) -> np.ndarray:
    n = s.size
    m = np.zeros(n, dtype=np.uint8)
    for i in nb.prange(n):
        u = s[i]; v = t[i]
        if u < v and labels[u] != labels[v]:
            m[i] = 1
    return m

@nb.njit(cache=True, parallel=False)
def mutual_nn_coarsening(sources, targets, weights, n_nodes):
    deg = np.zeros(n_nodes, dtype=np.int32)
    m = sources.shape[0]
    for e in range(m):
        u = sources[e]; v = targets[e]
        if u >= v:  # process undirected edge once
            continue
        deg[u] += 1; deg[v] += 1

    indptr = np.zeros(n_nodes + 1, dtype=np.int32)
    for i in range(n_nodes):
        indptr[i+1] = indptr[i] + deg[i]

    nbrs = np.empty(indptr[-1], dtype=np.int32)
    wts  = np.empty(indptr[-1], dtype=np.float32)
    fill = np.zeros(n_nodes, dtype=np.int32)

    for e in range(m):
        u = sources[e]; v = targets[e]; w = weights[e]
        if u >= v:
            continue
        p = indptr[u] + fill[u]; nbrs[p] = v; wts[p] = w; fill[u] += 1
        p = indptr[v] + fill[v]; nbrs[p] = u; wts[p] = w; fill[v] += 1
    
    # Best neighbor per node (parallel, independent writes)
    best_neighbor = np.full(n_nodes, -1, dtype=np.int32)
    for u in nb.prange(n_nodes):
        start = indptr[u]; end = indptr[u+1]
        bw = -np.inf; bn = -1
        for p in range(start, end):
            w = wts[p]; v = nbrs[p]
            if w > bw:
                bw = w; bn = v
        best_neighbor[u] = bn

    # Assign meta IDs (sequential, deterministic)
    meta_id = np.full(n_nodes, -1, dtype=np.int32)
    next_meta = 0
    for i in range(n_nodes):
        j = best_neighbor[i]
        if j >= 0 and j > i and best_neighbor[j] == i and meta_id[i] == -1:
            meta_id[i] = next_meta; meta_id[j] = next_meta; next_meta += 1
    for i in range(n_nodes):
        if meta_id[i] == -1:
            meta_id[i] = next_meta; next_meta += 1
    return meta_id, next_meta

from numba import atomic

@nb.njit(cache=True, parallel=True)
def _accumulate_stats(sources, targets, distances, sims, cidx, n_clusters):
    vol    = np.zeros(n_clusters, dtype=np.float64)
    cut    = np.zeros(n_clusters, dtype=np.float64)
    int_cnt= np.zeros(n_clusters, dtype=np.int64)
    ext_cnt= np.zeros(n_clusters, dtype=np.int64)
    sum_d  = np.zeros(n_clusters, dtype=np.float64)
    sumsq_d= np.zeros(n_clusters, dtype=np.float64)

    for e in nb.prange(sources.shape[0]):
        u = sources[e]; v = targets[e]
        if u >= v:      # keep only one orientation of an undirected edge
            continue

        du = cidx[u]; dv = cidx[v]
        w64 = np.float64(sims[e])
        d64 = np.float64(distances[e])

        # volume: add each endpoint once (since we only see the edge once)
        atomic.add(vol, du, w64)
        atomic.add(vol, dv, w64)

        if du == dv:
            atomic.add(int_cnt, du, 1)
            atomic.add(sum_d, du, d64)
            atomic.add(sumsq_d, du, d64 * d64)
        else:
            atomic.add(cut, du, w64)
            atomic.add(cut, dv, w64)
            atomic.add(ext_cnt, du, 1)
            atomic.add(ext_cnt, dv, 1)

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
    """Detailed interface edge detection with support for pruned clusters."""
    n_edges = len(sources)
    
    # Pre-allocate result arrays
    is_interface = np.zeros(n_edges, dtype=np.bool_)
    edge_types = np.zeros(n_edges, dtype=nb.int8)
    source_clusters = np.zeros(n_edges, dtype=np.int32)
    target_clusters = np.zeros(n_edges, dtype=np.int32)
    
    # Create set-like structure for pruned clusters
    max_cluster = 0
    for c in pruned_clusters:
        if c > max_cluster:
            max_cluster = c
    
    is_pruned = np.zeros(max_cluster + 1, dtype=np.bool_)
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

@nb.njit(cache=True)
def sparsify_knn_fast(sources: np.ndarray, targets: np.ndarray, 
                      weights: np.ndarray, n_nodes: int, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Count deg
    degrees = np.zeros(n_nodes, dtype=np.int32)
    for i in range(sources.shape[0]):
        degrees[sources[i]] += 1

    # Offsets
    offsets = np.zeros(n_nodes + 1, dtype=np.int32)
    for i in range(n_nodes):
        offsets[i+1] = offsets[i] + degrees[i]

    # Buckets
    total = sources.shape[0]
    edge_targets = np.empty(total, dtype=np.int32)
    edge_weights = np.empty(total, dtype=np.float32)
    edge_indices = np.empty(total, dtype=np.int32)

    fill = np.zeros(n_nodes, dtype=np.int32)
    for e in range(total):
        u = sources[e]
        pos = offsets[u] + fill[u]
        edge_targets[pos] = targets[e]
        edge_weights[pos] = weights[e]
        edge_indices[pos] = e
        fill[u] += 1

    # Count kept edges
    new_edge_count = 0
    for u in range(n_nodes):
        deg = offsets[u+1] - offsets[u]
        if deg > 0:
            new_edge_count += min(k, deg)

    sparse_sources = np.empty(new_edge_count, dtype=np.int32)
    sparse_targets = np.empty(new_edge_count, dtype=np.int32)
    sparse_weights = np.empty(new_edge_count, dtype=np.float32)
    sparse_orig_idx = np.empty(new_edge_count, dtype=np.int32)

    out = 0
    for u in range(n_nodes):
        start = offsets[u]; end = offsets[u+1]
        deg = end - start
        if deg == 0:
            continue
        if deg <= k:
            for j in range(deg):
                sparse_sources[out] = u
                sparse_targets[out] = edge_targets[start + j]
                sparse_weights[out] = edge_weights[start + j]
                sparse_orig_idx[out] = edge_indices[start + j]
                out += 1
        else:
            # indices of top-k weights
            idx_top = np.argpartition(edge_weights[start:end], deg - k)[deg - k:]  # length k
            for j in range(k):
                i0 = idx_top[j]
                sparse_sources[out] = u
                sparse_targets[out] = edge_targets[start + i0]
                sparse_weights[out] = edge_weights[start + i0]
                sparse_orig_idx[out] = edge_indices[start + i0]
                out += 1
    return sparse_sources, sparse_targets, sparse_weights, sparse_orig_idx

@nb.njit(fastmath=True, cache=True)
def find_knee_point(x: np.ndarray, y: np.ndarray, S: float = 1.0, 
                   use_median_filter: bool = False) -> int:
    """Optimized kneedle algorithm for finding the knee point in a curve."""
    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
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
    
    # For concave up curves, find point furthest below the line
    diffs = line_y - y_norm
    diffs = diffs * S
    
    # Find maximum difference
    knee_idx = np.argmax(diffs)
    
    # Map back to original index space if we filtered
    if use_median_filter and np.sum(above_median) > 2:
        return original_indices[knee_idx]
    
    return knee_idx

# ============================================================================
# OPTIMIZED GRAPH LOADER
# ============================================================================

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
                self.csr_offsets = data["offsets"].astype(np.int32)
                self.csr_indices = data["indices"].astype(np.int32)
        else:
            with perf_monitor.timed_operation("Build+cache CSR"):
                adj = self.adjacency
                if not sp.isspmatrix_csr(adj):
                    adj = adj.tocsr()
                self.csr_offsets = adj.indptr.astype(np.int32, copy=False)
                self.csr_indices = adj.indices.astype(np.int32, copy=False)
                np.savez_compressed(csr_cache, offsets=self.csr_offsets, indices=self.csr_indices)

        
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
                    self._component_labels = np.load(path).astype(np.int32)
                else:
                    self._component_labels = np.zeros(len(self.node_df), dtype=np.int32)
        return self._component_labels

    @property
    def means(self) -> Optional[np.ndarray]:
        if self._means is None and self.metadata.get("has_means", False):
            with perf_monitor.timed_operation("Load means"):
                self._means = np.load(os.path.join(self.input_dir, "means.npy")).astype(np.float32)
        return self._means

    @property
    def sigmas(self) -> Optional[np.ndarray]:
        if self._sigmas is None and self.metadata.get("has_sigmas", False):
            with perf_monitor.timed_operation("Load sigmas"):
                self._sigmas = np.load(os.path.join(self.input_dir, "sigmas.npy")).astype(np.float32)
        return self._sigmas

    @property
    def embedding(self) -> Optional[np.ndarray]:
        if self._embedding is None:
            with perf_monitor.timed_operation("Load embedding"):
                path = os.path.join(self.input_dir, "embedding.npy")
                if os.path.exists(path):
                    self._embedding = np.load(path).astype(np.float32)
        return self._embedding

    @property
    def full_embedding(self) -> Optional[np.ndarray]:
        if self._full_embedding is None:
            with perf_monitor.timed_operation("Load full embedding"):
                path = os.path.join(self.input_dir, "full_embedding.npy")
                if os.path.exists(path):
                    self._full_embedding = np.load(path).astype(np.float32)
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
                    src = data["sources"].astype(np.int32)
                    tgt = data["targets"].astype(np.int32)
                    dist = data["distances"].astype(np.float32)
                else:
                    coo = self.adjacency.tocoo()
                    src = coo.row.astype(np.int32)
                    tgt = coo.col.astype(np.int32)
                    dist = coo.data.astype(np.float32)
                    np.savez_compressed(cache, sources=src, targets=tgt, distances=dist)
                # Ensure C-contiguous
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
        """Returns a wrapper exposing .n_nodes, .node_df and .graph.get_edge_list()"""
        class GraphWrapper:
            def __init__(self, loader):
                self.loader = loader
                self.node_df = loader.node_df.copy()
                if include_embedding and loader.embedding is not None and loader.embedding.shape[1] >= 2:
                    if "UMAP1" not in self.node_df:
                        self.node_df["UMAP1"] = loader.embedding[:, 0]
                        self.node_df["UMAP2"] = loader.embedding[:, 1]

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
def _fmt_res(x: float) -> str:
    return f"{x:.6g}"

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
        self._median_cache = {}

        # Flags for lazy initialization
        self._initialized = False
        self._csr_built = False
        self._graph_prepared = False

        # Load edge data once
        self.loader = graph_loader
        self.sources, self.targets, self.distances = self.loader.edge_arrays
        
        # Store full arrays for interface detection
        self.full_sources = self.sources.copy()
        self.full_targets = self.targets.copy()
        self.full_distances = self.distances.copy()

        # Defaults after lazy init
        self.coarsened = False
        self.n_nodes_final = self.loader.n_nodes
        self.meta_id = None

        # For CSR path we need weights
        self.coarsened_sources = self.sources
        self.coarsened_targets = self.targets
        self.coarsened_n_nodes = self.n_nodes_final
        self.coarsened_weights = self._compute_similarities(self.distances)

        if self.verbose:
            print(f"Analyzer initialized (lazy mode)")
            print(f"  Nodes: {self.loader.n_nodes:,}")
            print(f"  Edges: {len(self.sources):,}")

    @property
    def csr_offsets(self):
        return self._csr_offsets

    @property
    def csr_indices(self):
        return self._csr_indices

    @property
    def n_edges(self) -> int:
        return len(self.sources)
    
    def _ensure_csr_built(self):
        if not self._csr_built:
            if self.verbose:
                print("Loading CSR graph structure from cache...")
            self._csr_offsets = self.loader.csr_offsets
            self._csr_indices = self.loader.csr_indices
            self._csr_built = True
    
    def _ensure_prepared(self):
        """Ensure graph is prepared with coarsening/sparsification."""
        if not self._graph_prepared:
            if self.verbose:
                print("Preparing graph…")

            # Apply pre-coarsening sparsification
            if self.sparsify:
                if self.verbose:
                    print("  Pre-coarsening sparsification step")
                self._apply_pre_coarsening_sparsification()

            # Apply coarsening
            if self.coarsen:
                if self.verbose:
                    print("  → Coarsening step")
                self._apply_coarsening()

            self._graph_prepared = True
            self._initialized = True
    
    def _apply_pre_coarsening_sparsification(self):
        if self.verbose:
            print(f"Sparsifying pre-coarsened graph (k={self.sparsify_pre_k})…")

        with perf_monitor.timed_operation("Pre-coarsened graph sparsification"):
            sims = self._compute_similarities(self.full_distances, scale="adaptive")

            s, t, w, orig_idx = sparsify_knn_fast(
                self.sources, self.targets, sims, self.loader.n_nodes, self.sparsify_pre_k
            )

            # ---- replace mirror section with undirected-unique aggregation ----
            a = np.minimum(s, t).astype(np.int64)
            b = np.maximum(s, t).astype(np.int64)
            pair = (a << 32) | b
            order = np.argsort(pair)
            uniq = np.empty_like(pair, dtype=bool)
            uniq[0] = True
            uniq[1:] = pair[order][1:] != pair[order][:-1]
            gid = np.cumsum(uniq) - 1
            ng = int(gid[-1]) + 1

            # segment max on similarity weights
            w_sorted = w[order]
            w_agg = np.full(ng, -np.inf, dtype=np.float32)
            np.maximum.at(w_agg, gid, w_sorted)

            keep = order[uniq]
            self.sources = a[keep].astype(np.int32)
            self.targets = b[keep].astype(np.int32)
            self.edge_sim = w_agg
            self.distances = np.sqrt(
                -2.0 * (np.median(self.full_distances) ** 2)
                * np.log(np.clip(w_agg, 1e-12, 1.0))
            ).astype(np.float32)

            if self.verbose:
                print(f"  Sparsified to {len(self.sources):,} undirected edges")
        
    def _apply_post_coarsened_sparsification(self):
        """Sparsify the coarsened graph with undirected-unique kNN in similarity space."""
        if self.verbose:
            print(f"Sparsifying post-coarsened graph (k={self.sparsify_post_k})…")

        with perf_monitor.timed_operation("Graph sparsification"):
            src_arr = self.coarsened_sources
            tgt_arr = self.coarsened_targets
            w_sim = self.coarsened_weights
            n_nodes = self.n_nodes_final

            # --- k-NN sparsification in similarity space ---
            s, t, w, orig_idx = sparsify_knn_fast(
                src_arr, tgt_arr, w_sim, n_nodes, self.sparsify_post_k
            )

            # --- Canonicalize & deduplicate (undirected-unique) ---
            a = np.minimum(s, t).astype(np.int64)
            b = np.maximum(s, t).astype(np.int64)
            pair = (a << 32) | b
            order = np.argsort(pair)

            uniq = np.empty_like(pair, dtype=bool)
            uniq[0] = True
            uniq[1:] = pair[order][1:] != pair[order][:-1]
            gid = np.cumsum(uniq) - 1
            ng = int(gid[-1]) + 1

            w_sorted = w[order]
            w_agg = np.full(ng, -np.inf, dtype=np.float32)
            np.maximum.at(w_agg, gid, w_sorted)

            keep = order[uniq]
            new_src = a[keep].astype(np.int32)
            new_tgt = b[keep].astype(np.int32)
            new_w = w_agg

            # --- Update coarsened graph state ---
            self.coarsened_sources = new_src
            self.coarsened_targets = new_tgt
            self.coarsened_weights = new_w
            self.distances = np.sqrt(-2.0 * np.log(np.clip(new_w, 1e-12, 1.0))).astype(np.float32)

            if self.verbose:
                print(f"  Sparsified to {len(new_src):,} edges (unique undirected)")

    def _gaussian_similarity(self, distances, scale='adaptive'):
        if scale == 'adaptive' or scale is None:
            scale_val = float(np.median(distances)) if distances.size else 1.0
        else:
            scale_val = float(scale)
        if not np.isfinite(scale_val) or scale_val <= 0.0:
            scale_val = 1e-12  # epsilon
        return np.exp(-(distances/scale_val)**2/2).astype(np.float32)
        
    def _compute_similarities(self, distances: np.ndarray, scale: Union[float, str] = 'adaptive') -> np.ndarray:
        """Compute similarities with caching."""
        # Use simpler cache key based on array characteristics
        cache_key = (distances.ctypes.data, distances.size, str(scale))
        
        if cache_key not in self._similarity_cache:
            self._similarity_cache[cache_key] = self.similarity_function(distances, scale)
        
        return self._similarity_cache[cache_key]
    
    def _apply_coarsening(self):
        """Apply multi-level coarsening with undirected, similarity-consistent edges."""
        if self.verbose:
            print(f"Applying {self.coarsen_levels} levels of coarsening…")

        # Initialize working arrays
        current_sources = self.sources
        current_targets = self.targets
        current_weights = self._compute_similarities(self.distances)   # similarities
        current_n_nodes = self.loader.n_nodes
        cumulative_mapping = np.arange(current_n_nodes, dtype=np.int32)
        self.coarsening_hierarchy = []

        for level in range(self.coarsen_levels):
            if self.verbose:
                print(f"  Level {level + 1}: {current_n_nodes:,} nodes")

            # --- Mutual-NN coarsening ---
            meta_id, n_meta = mutual_nn_coarsening(
                current_sources, current_targets, current_weights, current_n_nodes
            )
            ratio = n_meta / current_n_nodes
            self.coarsening_hierarchy.append({
                "level": level,
                "original_nodes": current_n_nodes,
                "coarsened_nodes": n_meta,
                "reduction_ratio": ratio,
            })
            if self.verbose:
                print(f"    → {n_meta:,} meta-nodes (ratio: {ratio:.3f})")

            if n_meta < 1000 or ratio > 0.95:
                if self.verbose:
                    print("    Stopping early (too few or little reduction)")
                break

            # --- Build meta-edges ---
            if self.verbose:
                print("    Building meta-edges...")

            ms = meta_id[current_sources]
            mt = meta_id[current_targets]
            keep = ms != mt
            ms, mt = ms[keep], mt[keep]
            mw = current_weights[keep]

            # --- Canonicalize (min,max) for undirected graph ---
            a = np.minimum(ms, mt).astype(np.int64)
            b = np.maximum(ms, mt).astype(np.int64)
            pair = (a << 32) | b
            order = np.argsort(pair)

            uniq = np.empty_like(pair, dtype=bool)
            uniq[0] = True
            uniq[1:] = pair[order][1:] != pair[order][:-1]
            gid = np.cumsum(uniq) - 1
            ng = int(gid[-1]) + 1

            # --- Aggregate similarities (max pooling for stability) ---
            w_sorted = mw[order]
            w_agg = np.full(ng, -np.inf, dtype=np.float32)
            np.maximum.at(w_agg, gid, w_sorted)

            keep = order[uniq]
            curr_src = a[keep].astype(np.int32)
            curr_tgt = b[keep].astype(np.int32)
            curr_w = w_agg

            # --- Update mappings ---
            cumulative_mapping = meta_id[cumulative_mapping]
            current_sources = curr_src
            current_targets = curr_tgt
            current_weights = curr_w
            current_n_nodes = n_meta

            if self.verbose:
                print(f"    → Aggregated to {len(curr_src):,} edges")

        # --- Store final coarsened graph state ---
        self.coarsened = True
        self.meta_id = cumulative_mapping
        self.n_nodes_final = current_n_nodes
        self.coarsening_ratio = current_n_nodes / self.loader.n_nodes
        self.coarsened_sources = current_sources
        self.coarsened_targets = current_targets
        self.coarsened_weights = current_weights
        # Optionally derive distances only for logging/plots
        self.distances = np.sqrt(-2.0 * np.log(np.clip(current_weights, 1e-12, 1.0))).astype(np.float32)

        # --- Adaptive post-coarsen sparsify ---
        self.previous_sparsify_post_k = self.sparsify_post_k
        if self.previous_sparsify_post_k is None:
            self.sparsify_post_k = max(1, int(self.sparsify_pre_k * self.coarsening_ratio))
        else:
            self.sparsify_post_k = max(1, int(self.previous_sparsify_post_k * self.coarsening_ratio))
        self._apply_post_coarsened_sparsification()

    def _project_labels_to_coarse_mode(self, fine_labels: np.ndarray) -> np.ndarray:
        """
        Project fine-level labels (size = |V|) to the current coarse graph (size = |V'|)
        by taking the mode (majority label) over each meta-node's members.

        - Deterministic: ties break toward the smallest label (via np.unique sorting).
        - If the graph is not coarsened, returns a copy of the input.
        """
        if not self.coarsened or self.meta_id is None:
            return fine_labels.copy()

        if fine_labels.shape[0] != self.loader.n_nodes:
            raise ValueError(
                f"fine_labels has length {fine_labels.shape[0]}, "
                f"but loader.n_nodes is {self.loader.n_nodes}."
            )

        n_meta = int(self.n_nodes_final)
        coarse = np.empty(n_meta, dtype=np.int32)

        # Group once by meta_id (sorted), then take a mode per group.
        order = np.argsort(self.meta_id)
        meta_sorted = self.meta_id[order]
        labels_sorted = fine_labels[order]

        # Boundaries where meta id changes, with sentinels at ends.
        boundaries = np.flatnonzero(
            np.concatenate((
                np.array([True]),
                meta_sorted[1:] != meta_sorted[:-1],
                np.array([True])
            ))
        )

        for g in range(boundaries.shape[0] - 1):
            start = boundaries[g]
            end = boundaries[g + 1]
            vals, counts = np.unique(labels_sorted[start:end], return_counts=True)
            # np.unique sorts vals ascending; argmax picks first max → deterministic tie-break
            coarse[meta_sorted[start]] = np.int32(vals[np.argmax(counts)])

        return coarse

    def _create_igraph(self):
        """Create igraph from current graph data."""
        import igraph as ig  # Lazy import
        
        if self.verbose:
            print("Creating igraph...")
        
        n_nodes = self.n_nodes_final
        n_edges = len(self.coarsened_sources)
        
        # Try direct edge list creation
        try:
            edge_array = np.column_stack((self.coarsened_sources, self.coarsened_targets))
            self.igraph = ig.Graph(n=n_nodes, edges=edge_array, directed=False)
            self.igraph.es["weight"] = self.coarsened_weights
            if self.verbose:
                print("  Created igraph using numpy array approach")
        except:
            # Fallback to list of tuples
            edges = list(zip(self.coarsened_sources.tolist(), self.coarsened_targets.tolist()))
            self.igraph = ig.Graph(n=n_nodes, edges=edges, directed=False)
            self.igraph.es["weight"] = self.coarsened_weights.tolist()
            if self.verbose:
                print("  Created igraph using list approach")
    
    def run_leiden_igraph(self, resolution: float, run_id: str = None, 
                      scale: Union[float, str] = 'adaptive',
                      initial_membership: Optional[np.ndarray] = None,
                      rank_stat_col: Optional[str] = None,
                      prune_small_clusters: bool = False,
                      min_cluster_size: Optional[int] = None,
                      knee_sensitivity: float = 1.0,
                      normalize_rank_stat: bool = True,
                      reassign_pruned: bool = False,
                      output_prefix: Optional[str] = None) -> Tuple[pd.DataFrame, np.ndarray]:
        """Run Leiden with optional warm start (Louvain cold-start fallback)."""
        import igraph as ig
        import leidenalg

        # Ensure graph is prepared and exists
        self._ensure_prepared()
        if not hasattr(self, 'igraph'):
            self._create_igraph()

        # Run / output naming
        if run_id is None:
            run_id = f"leiden_res{_fmt_res(resolution)}"
        if output_prefix is None:
            output_prefix = f"leiden_{run_id}_"

        if self.verbose:
            print(f"\n--- Running Leiden with resolution={_fmt_res(resolution)} ---")

        with perf_monitor.timed_operation(f"Leiden clustering (res={_fmt_res(resolution)})"):
            # Always set the weights for the *current* graph (fine or coarsened)
            sims = self._compute_similarities(self.distances, scale)
            self.igraph.es['weight'] = sims.tolist()

            # Prepare warm start:
            # - If provided at fine-level and we are coarsened, project via mode.
            # - If provided at coarse-level, length must match current graph.
            # - If not provided, use Louvain memberships as a cold start.
            if initial_membership is not None:
                if self.coarsened and initial_membership.shape[0] == self.loader.n_nodes:
                    initial_membership = self._project_labels_to_coarse_mode(initial_membership)
                elif initial_membership.shape[0] != self.n_nodes_final:
                    raise ValueError(
                        f"initial_membership has length {initial_membership.shape[0]}, "
                        f"expected {self.loader.n_nodes} (fine) or {self.n_nodes_final} (coarse)."
                    )
            else:
                if self.verbose:
                    print("  Cold-start Louvain…")
                louvain_part = self.igraph.community_multilevel(weights=self.igraph.es['weight'])
                initial_membership = np.array(louvain_part.membership, dtype=np.int32)

            # Leiden with warm-start
            partition = leidenalg.find_partition(
                self.igraph,
                leidenalg.RBConfigurationVertexPartition,
                resolution_parameter=float(resolution),
                weights='weight',
                n_iterations=5,
                initial_membership=initial_membership.tolist()
            )
            labels = np.array(partition.membership, dtype=np.int32)

            # If we ran on a coarsened graph, project labels back to fine level
            if self.coarsened:
                full_labels = np.zeros(self.loader.n_nodes, dtype=np.int32)
                for i in range(self.loader.n_nodes):
                    full_labels[i] = labels[self.meta_id[i]]
                labels = full_labels

            if self.verbose:
                n_clusters = len(np.unique(labels))
                print(f"Found {n_clusters} communities")

        # Process results
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
                df, cluster_col, pruning_info.get('pruned_clusters', []), scale, include_similarity=False
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


    def run_louvain_csr(self, resolution: float, run_id: str = None,
                       scale: Union[float, str] = 'adaptive',
                       initial_membership: Optional[np.ndarray] = None,
                       rank_stat_col: Optional[str] = None,
                       prune_small_clusters: bool = False,
                       min_cluster_size: Optional[int] = None,
                       knee_sensitivity: float = 1.0,
                       normalize_rank_stat: bool = True,
                       reassign_pruned: bool = False,
                       output_prefix: Optional[str] = None) -> Tuple[pd.DataFrame, np.ndarray]:
        """Run CSR-native Louvain community detection."""
        try:
            from sknetwork.clustering import Louvain
        except ImportError:
            raise ImportError("Install scikit-network: pip install scikit-network")
        
        # Ensure CSR is built
        if not self._csr_built:
            self._ensure_csr_built()
        
        # Do sparsification/coarsening if needed
        if not self._graph_prepared:
            self._ensure_prepared()
            self._initialized = True
        
        # Generate run_id and output prefix
        if run_id is None:
            run_id = f"louvain_res{resolution:.3f}"
        if output_prefix is None:
            output_prefix = f"louvain_{run_id}_"
        
        if self.verbose:
            print(f"\n--- Running CSR-native Louvain with resolution={resolution} ---")
        
        with perf_monitor.timed_operation(f"Louvain clustering (res={resolution})"):
            # --- Mirror edges for symmetric CSR (undirected modularity) ---
            src = self.coarsened_sources.astype(np.int32, copy=False)
            tgt = self.coarsened_targets.astype(np.int32, copy=False)
            w   = self.coarsened_weights.astype(np.float32, copy=False)

            mask = src != tgt
            src_mir = np.concatenate([src, tgt[mask]])
            tgt_mir = np.concatenate([tgt, src[mask]])
            w_mir   = np.concatenate([w, w[mask]])

            # Optional: aggregate duplicates (min/max mean) to enforce symmetry
            # Convert to COO first for built-in deduplication
            csr = sp.csr_matrix(
                (w_mir, (src_mir, tgt_mir)),
                shape=(self.n_nodes_final, self.n_nodes_final)
            )
            csr = 0.5 * (csr + csr.T)
            
            # Handle warm-start if provided
            if initial_membership is not None:
                if self.coarsened and initial_membership.shape[0] == self.loader.n_nodes:
                    initial_membership = self._project_labels_to_coarse_mode(initial_membership)
                elif initial_membership.shape[0] != self.n_nodes_final:
                    raise ValueError(
                        f"initial_membership has length {initial_membership.shape[0]}, "
                        f"expected {self.loader.n_nodes} (fine) or {self.n_nodes_final} (coarse)."
                    )
            
            # Run Louvain
            louvain = Louvain(
                resolution=float(resolution),
                random_state=42,
                modularity='newman',
                return_probs=False
            )
            
            result = louvain.fit_transform(csr)
            
            # Extract labels
            if hasattr(louvain, 'labels_'):
                labels = np.asarray(louvain.labels_, dtype=np.int32)
            elif isinstance(result, np.ndarray):
                labels = result.astype(np.int32).ravel()
            elif sp.issparse(result):
                labels = np.asarray(result.toarray(), dtype=np.int32).ravel()
            else:
                raise ValueError(f"Cannot interpret Louvain result of type {type(result)}")
            
            # Sanity check
            if labels.ndim != 1 or labels.shape[0] != self.n_nodes_final:
                raise ValueError(f"Louvain labels have wrong shape {labels.shape}")
            
            # Project labels back if coarsened
            if self.coarsened:
                full_labels = np.zeros(self.loader.n_nodes, dtype=np.int32)
                for i in range(self.loader.n_nodes):
                    full_labels[i] = labels[self.meta_id[i]]
                labels = full_labels
            
            if self.verbose:
                n_clusters = len(np.unique(labels))
                print(f"Found {n_clusters} communities")
        
        # Process results (same as Leiden)
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
                'algorithm': 'louvain_csr'
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
        """Run CSR-native Leiden community detection."""
        try:
            from sknetwork.clustering import Leiden
        except ImportError:
            raise ImportError("Install scikit-network: pip install scikit-network")
        
        # Ensure CSR is built
        if not self._csr_built:
            self._ensure_csr_built()
        
        # Do sparsification/coarsening if needed
        if not self._graph_prepared:
            self._ensure_prepared()
            self._initialized = True
        
        # Generate run_id and output prefix
        if run_id is None:
            run_id = f"leiden_res{resolution:.3f}"
        if output_prefix is None:
            output_prefix = f"leiden_{run_id}_"
        
        if self.verbose:
            print(f"\n--- Running CSR-native Leiden with resolution={resolution} ---")
        
        with perf_monitor.timed_operation(f"Leiden clustering (res={resolution})"):
            src = self.coarsened_sources.astype(np.int32, copy=False)
            tgt = self.coarsened_targets.astype(np.int32, copy=False)
            w   = self.coarsened_weights.astype(np.float32, copy=False)

            mask = src != tgt
            src_mir = np.concatenate([src, tgt[mask]])
            tgt_mir = np.concatenate([tgt, src[mask]])
            w_mir   = np.concatenate([w, w[mask]])

            # Optional: aggregate duplicates (min/max mean) to enforce symmetry
            # Convert to COO first for built-in deduplication
            csr = sp.csr_matrix(
                (w_mir, (src_mir, tgt_mir)),
                shape=(self.n_nodes_final, self.n_nodes_final)
            )
            csr = 0.5 * (csr + csr.T)
            
            # Handle warm-start if provided
            if initial_membership is not None:
                if self.coarsened and initial_membership.shape[0] == self.loader.n_nodes:
                    initial_membership = self._project_labels_to_coarse_mode(initial_membership)
                elif initial_membership.shape[0] != self.n_nodes_final:
                    raise ValueError(
                        f"initial_membership has length {initial_membership.shape[0]}, "
                        f"expected {self.loader.n_nodes} (fine) or {self.n_nodes_final} (coarse)."
                    )
            
            # Run Leiden
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
            
            # Sanity check
            if labels.ndim != 1 or labels.shape[0] != self.n_nodes_final:
                raise ValueError(f"Leiden labels have wrong shape {labels.shape}")
            
            # Project labels back if coarsened
            if self.coarsened:
                full_labels = np.zeros(self.loader.n_nodes, dtype=np.int32)
                for i in range(self.loader.n_nodes):
                    full_labels[i] = labels[self.meta_id[i]]
                labels = full_labels
            
            if self.verbose:
                n_clusters = len(np.unique(labels))
                print(f"Found {n_clusters} communities")
        
        # Process results (same as Leiden/Louvain)
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
        
        # Power transform if requested
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
        
        # Choose stat column
        stat_col = transformed_rank_stat_col or rank_stat_col
        if stat_col and stat_col in df.columns:
            stat_array = df[stat_col].to_numpy()
        else:
            stat_array = None
        
        # Aggregate per cluster
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
        
        # Build DataFrame and assign ranks
        cluster_stats = pd.DataFrame(stats)
        cluster_stats = cluster_stats.sort_values('ranking_stat', ascending=False)
        cluster_stats['rank'] = np.arange(1, len(cluster_stats) + 1)
        
        # Initialize pruning info
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
        
        # Map ranks back
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
        
        # Add 'pruned' flag
        cluster_stats['pruned'] = cluster_stats['cluster_id'].apply(
            lambda x: x in pruned_clusters
        )
        
        # Add original cluster column
        orig_cluster_col = f"{cluster_col}_original"
        df[orig_cluster_col] = labels.copy()
        
        # Reassign nodes if requested
        if reassign_pruned and pruned_clusters:
            self._reassign_pruned_nodes(df, cluster_col, pruned_clusters, kept_clusters)
        elif pruned_clusters and self.verbose:
            print("Not reassigning pruned clusters (reassign_pruned=False)")
            
        return pruning_info
    
    def _reassign_pruned_nodes(self, df, cluster_col, pruned_clusters, kept_clusters):
        if self.verbose:
            print("Reassigning nodes from pruned clusters...")

        col_idx = df.columns.get_loc(cluster_col)
        node_to_cluster = df[cluster_col].to_numpy(copy=True)

        temp_label = -1
        pruned_mask = np.isin(node_to_cluster, np.array(pruned_clusters, dtype=node_to_cluster.dtype))
        pruned_indices = np.flatnonzero(pruned_mask)

        node_to_cluster[pruned_indices] = temp_label

        reassigned = 0
        for node_idx in pruned_indices:
            start = self.csr_offsets[node_idx]; end = self.csr_offsets[node_idx+1]
            nbr = self.csr_indices[start:end]
            nbr_cl = node_to_cluster[nbr]
            nbr_cl = nbr_cl[nbr_cl != temp_label]
            if nbr_cl.size:
                # majority vote
                vals, counts = np.unique(nbr_cl, return_counts=True)
                node_to_cluster[node_idx] = vals[np.argmax(counts)]
                reassigned += 1

        if self.verbose:
            print(f"Reassigned {reassigned} of {pruned_indices.size} nodes from pruned clusters")

        # Fallback for any still -1
        if np.any(node_to_cluster == temp_label):
            node_to_cluster[node_to_cluster == temp_label] = kept_clusters[0]

        # write-back once
        df.iloc[:, col_idx] = node_to_cluster

    
    def _identify_interface_edges(self,
                              df: pd.DataFrame,
                              cluster_col: str,
                              pruned_clusters,
                              scale,
                              include_similarity: bool = False) -> pd.DataFrame:
        """
        Identify interface edges between communities.

        - Processes only one orientation (s < t).
        - Builds a compact result directly (two-pass).
        - Skips similarity computation unless include_similarity=True.
        """
        if self.verbose:
            print("Identifying interface edges...")

        # Inputs
        clusters = df[cluster_col].values.astype(np.int32, copy=False)
        s_all = self.full_sources
        t_all = self.full_targets
        d_all = self.full_distances

        # Pass 1: mask cross-community on upper triangle
        mask = _mask_upper_cross_labels(s_all, t_all, clusters)
        idx = np.flatnonzero(mask)  # compact indices

        if idx.size == 0:
            return pd.DataFrame(columns=[
                'source', 'target', 'distance', 'similarity',
                'source_cluster', 'target_cluster', 'edge_type'
            ])

        # Pass 2: gather compact arrays
        s = s_all[idx]
        t = t_all[idx]
        d = d_all[idx]
        sc = clusters[s]
        tc = clusters[t]

        # Only compute similarities if requested
        if include_similarity:
            sim = self._compute_similarities(d, scale).astype(np.float32, copy=False)
        else:
            sim = np.zeros_like(d, dtype=np.float32)

        # Edge types (cross vs pruned_community)
        if pruned_clusters:
            # vectorized, safe: use isin() on int arrays
            pruned = np.array(list(pruned_clusters), dtype=np.int32)
            is_pruned = np.isin(sc, pruned) | np.isin(tc, pruned)
            edge_type = np.where(is_pruned, "pruned_community", "cross_community")
            edge_type = pd.Categorical(edge_type, categories=["cross_community", "pruned_community"])
        else:
            edge_type = np.full(s.shape[0], "cross_community", dtype=object)
            edge_type = pd.Categorical(edge_type, categories=["cross_community", "pruned_community"])

        return pd.DataFrame({
            'source': s.astype(np.int32, copy=False),
            'target': t.astype(np.int32, copy=False),
            'distance': d.astype(np.float32, copy=False),
            'similarity': sim,  # zeros unless include_similarity=True
            'source_cluster': sc.astype(np.int32, copy=False),
            'target_cluster': tc.astype(np.int32, copy=False),
            'edge_type': edge_type
        })

    def extract_interface_edges(self, labels: np.ndarray) -> Dict[str, Any]:
        s, t, d = self.loader.edge_arrays
        mask = _mask_upper_cross_labels(s, t, labels)
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return {'count': 0}
        return {
            'sources': s[idx],
            'targets': t[idx],
            'distances': d[idx],
            'source_clusters': labels[s[idx]],
            'target_clusters': labels[t[idx]],
            'count': int(idx.size),
        }
    
    def get_combined_edge_data(self, run_ids=None) -> pd.DataFrame:
        if run_ids is None:
            run_ids = list(self.runs.keys())
        if not run_ids:
            raise ValueError("No runs available to combine")

        with perf_monitor.timed_operation("Combine edge data"):
            # edge -> first index map on full graph (canonicalized)
            full_map = {}
            for idx in range(self.full_sources.shape[0]):
                s = int(self.full_sources[idx]); t = int(self.full_targets[idx])
                if s == t:
                    continue
                key = (s, t) if s < t else (t, s)
                if key not in full_map:
                    full_map[key] = idx

            # union of all interface edges
            all_edges = set()
            per_run_type = {rid: {} for rid in run_ids}
            for rid in run_ids:
                if rid not in self.interface_edges:
                    continue
                df = self.interface_edges[rid]
                for i in range(len(df)):
                    a = int(df.iloc[i]['source']); b = int(df.iloc[i]['target'])
                    key = (a, b) if a < b else (b, a)
                    all_edges.add(key)
                    per_run_type[rid][key] = df.iloc[i]['edge_type']

            combined = []
            for key in all_edges:
                idx = full_map.get(key, -1)
                if idx < 0:
                    continue
                row = {
                    'source': key[0],
                    'target': key[1],
                    'distance': float(self.full_distances[idx])
                }
                in_any = False
                for rid in run_ids:
                    present = key in per_run_type[rid]
                    row[f'in_{rid}'] = present
                    row[f'type_{rid}'] = per_run_type[rid].get(key, 'not_interface')
                    if present:
                        in_any = True
                if in_any:
                    row['run_count'] = sum(1 for rid in run_ids if row[f'in_{rid}'])
                    combined.append(row)
            self.combined_edges = pd.DataFrame(combined)

            if self.verbose and len(self.combined_edges) > 0:
                vc = self.combined_edges['run_count'].value_counts().sort_index()
                print("Edge counts by number of runs:")
                for k, v in vc.items():
                    print(f"  - In {k}/{len(run_ids)} runs: {v} edges")
            return self.combined_edges

    
    def add_community_statistics(self, run_id, recalculate=False) -> pd.DataFrame:
        """Calculate community statistics using Numba acceleration."""
        if run_id not in self.runs:
            raise ValueError(f"Run '{run_id}' not found")
    
        run_data = self.runs[run_id]
        stats_df = run_data['cluster_stats']
    
        # Skip if already present
        if 'conductance' in stats_df.columns and not recalculate:
            return stats_df
    
        with perf_monitor.timed_operation(f"Calculate community statistics (run={run_id})"):
            if self.verbose:
                print(f"Calculating community statistics for run '{run_id}'...")
    
            # Prepare inputs
            df = run_data['df']
            col = run_data['cluster_col']
            labels = df[col].values
            unique_ids = np.sort(stats_df['cluster_id'].unique())
            n_clusters = unique_ids.shape[0]
    
            # Map cluster_id to index
            id_to_idx = {int(cid): i for i, cid in enumerate(unique_ids)}
            cluster_idx = np.array([id_to_idx[int(l)] for l in labels], dtype=np.int32)
    
            # Full edge lists
            S = self.full_sources
            T = self.full_targets
            D = self.full_distances
            sims = self._compute_similarities(D, run_data.get('similarity_scale', 'adaptive'))
    
            # Accumulate stats using Numba
            mask = S < T
            vol, cut, int_cnt, ext_cnt, sum_d, sumsq_d = _accumulate_stats(
                S[mask], T[mask], D[mask], sims[mask], cluster_idx, n_clusters
            )
    
            # Post-process
            mean_int = np.zeros(n_clusters, dtype=np.float64)
            std_int = np.zeros(n_clusters, dtype=np.float64)
            for i in range(n_clusters):
                cnt = int_cnt[i]
                if cnt > 0:
                    mean_int[i] = sum_d[i] / cnt
                    var = (sumsq_d[i] - (sum_d[i]**2)/cnt) / max(1, cnt - 1)
                    std_int[i] = np.sqrt(var) if var > 0 else 0.0
    
            # Conductance
            cond = np.zeros_like(vol)
            nz = vol > 0
            cond[nz] = cut[nz] / vol[nz]
    
            # Density
            node_counts = np.bincount(cluster_idx, minlength=n_clusters)
            max_edges = node_counts * (node_counts - 1) / 2
            density = np.zeros(n_clusters, dtype=np.float64)
            nonzero = max_edges > 0
            density[nonzero] = int_cnt[nonzero] / max_edges[nonzero]
    
            # Edge-to-node ratio
            e2n = np.zeros(n_clusters, dtype=np.float64)
            nonz = node_counts > 0
            e2n[nonz] = int_cnt[nonz] / node_counts[nonz]
    
            # Update stats dataframe
            idx_to_id = {i: cid for i, cid in enumerate(unique_ids)}
            
            new_cols = defaultdict(list)
            for i in range(n_clusters):
                new_cols['conductance'].append(cond[i])
                new_cols['internal_edges'].append(int_cnt[i])
                new_cols['external_edges'].append(ext_cnt[i])
                new_cols['mean_edge_dist'].append(mean_int[i])
                new_cols['std_edge_dist'].append(std_int[i])
                new_cols['edge_density'].append(density[i])
                new_cols['edge_to_node_ratio'].append(e2n[i])
    
            for colname, vals in new_cols.items():
                stats_df[colname] = vals
    
            # Coefficient of variation
            stats_df['coef_var'] = stats_df['std_edge_dist'] / stats_df['mean_edge_dist']
    
            # Store and return
            self.runs[run_id]['cluster_stats'] = stats_df
            return stats_df
    
    def compare_runs(self, run_id1, run_id2) -> Dict:
        """Compare two runs using NMI and edge overlap."""
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
            
            # Get interface edges
            if run_id1 not in self.interface_edges or run_id2 not in self.interface_edges:
                self.get_combined_edge_data([run_id1, run_id2])
                
            edges1 = self.interface_edges[run_id1]
            edges2 = self.interface_edges[run_id2]
            
            # Create edge sets
            edge_pairs1 = set([(min(row['source'], row['target']), max(row['source'], row['target'])) 
                             for _, row in edges1.iterrows()])
            edge_pairs2 = set([(min(row['source'], row['target']), max(row['source'], row['target'])) 
                             for _, row in edges2.iterrows()])
            
            # Find overlap
            common_edges = edge_pairs1.intersection(edge_pairs2)
            only_in_run1 = edge_pairs1 - edge_pairs2
            only_in_run2 = edge_pairs2 - edge_pairs1
            
            # Jaccard similarity
            jaccard_similarity = len(common_edges) / len(edge_pairs1.union(edge_pairs2)) if edge_pairs1 or edge_pairs2 else 0
            
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
                                 cmap='tab20', random_seed=None) -> "plt.Figure":
        """Create visualization comparing communities across runs."""
        # Lazy import matplotlib
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if run_ids is None:
            run_ids = list(self.runs.keys())
            
        if not run_ids:
            raise ValueError("No runs available to plot")
        
        with perf_monitor.timed_operation(f"Plot {plot_type} for {len(run_ids)} runs"):
            # Ensure we have combined edge data
            if self.combined_edges is None or not all(f'in_{run_id}' in self.combined_edges.columns for run_id in run_ids):
                self.get_combined_edge_data(run_ids)
                
            # Set random seed
            if random_seed is not None:
                np.random.seed(random_seed)
                
            # Create figure
            fig = plt.figure(figsize=figsize)
            
            if plot_type == 'scatter':
                # UMAP scatter with interface edges
                if 'UMAP1' not in self.loader.node_df.columns or 'UMAP2' not in self.loader.node_df.columns:
                    if self.loader.embedding is not None and self.loader.embedding.shape[1] >= 2:
                        # Add UMAP coordinates
                        self.loader.node_df['UMAP1'] = self.loader.embedding[:, 0]
                        self.loader.node_df['UMAP2'] = self.loader.embedding[:, 1]
                    else:
                        raise ValueError("UMAP coordinates not found")
                    
                ax = fig.add_subplot(111)
                
                # Use first run for node coloring
                run_id = run_ids[0]
                run_data = self.runs[run_id]
                cluster_col = run_data['cluster_col']
                
                # Plot nodes
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
                
                # Plot edges
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
                
                # Create matrix
                edge_matrix = np.zeros((len(run_ids), len(self.combined_edges)))
                for i, run_id in enumerate(run_ids):
                    edge_matrix[i] = self.combined_edges[f'in_{run_id}'].astype(int)
                
                # Calculate similarity matrix
                similarity_matrix = np.zeros((len(run_ids), len(run_ids)))
                for i in range(len(run_ids)):
                    for j in range(len(run_ids)):
                        intersection = np.sum(edge_matrix[i] & edge_matrix[j])
                        union = np.sum(edge_matrix[i] | edge_matrix[j])
                        similarity_matrix[i, j] = intersection / union if union > 0 else 0
                
                # Plot heatmap
                sns.heatmap(similarity_matrix, annot=True, fmt='.3f', cmap='viridis',
                          xticklabels=run_ids, yticklabels=run_ids, ax=ax)
                ax.set_title('Interface Edge Similarity (Jaccard)')
                
            elif plot_type == 'community_overlap':
                # NMI matrix
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
                # Compare cluster statistics
                if include_stats:
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
        """Update graph wrapper with community assignments."""
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
# OUTPUT FUNCTIONS (rewritten for speed + smaller I/O)
#   - Write canonical (s,t) uint32 pairs + side arrays (no CSR build)
#   - Deduplicate once via 64-bit key sort (optional)
#   - Support legacy loader fallback
# ============================================================================

from typing import Dict, Any, Tuple, List
import os, json
import numpy as np
import scipy.sparse as sp  # only used by legacy loader branch


def save_interface_edges_efficient(interface_data: Dict[str, Any],
                                   output_file: str,
                                   metadata: Dict[str, Any],
                                   *,
                                   dedup: bool = True,
                                   compress: bool = False) -> None:
    """
    Save interface edges fast as canonical (s,t) pairs with side arrays.
    - Avoids COO→CSR construction and deflate; much faster on large graphs.
    - Keeps signature backward-compatible; extra args are keyword-only.

    Parameters
    ----------
    interface_data : dict
        Expect keys:
          'sources' (int32), 'targets' (int32), 'distances' (float32),
          'source_clusters' (int32), 'target_clusters' (int32), 'count' (int)
    output_file : str
        Base path (without suffix). Files written:
          <output_file>_pairs.npz
          <output_file>_metadata.json
    metadata : dict
        Must contain 'n_nodes'. Will be updated with 'edge_count' and 'files'.
    dedup : bool, optional
        If True (default), remove duplicate undirected edges after canonicalizing.
    compress : bool, optional
        If True, use np.savez_compressed (slower, smaller). Default False (faster).
    """
    with perf_monitor.timed_operation("Save interface edges"):
        count = int(interface_data.get('count', 0))
        if count == 0:
            md = dict(metadata)
            md['edge_count'] = 0
            md['files'] = {'pairs': None}
            with open(output_file + '_metadata.json', 'w') as f:
                json.dump(md, f, indent=2)
            return

        # Required arrays
        s = np.asarray(interface_data['sources'], dtype=np.uint32)
        t = np.asarray(interface_data['targets'], dtype=np.uint32)
        d = np.asarray(interface_data['distances'], dtype=np.float32)
        sc = np.asarray(interface_data['source_clusters'], dtype=np.int32)
        tc = np.asarray(interface_data['target_clusters'], dtype=np.int32)

        # Canonicalize to undirected orientation: s <= t
        s_c = np.minimum(s, t, dtype=np.uint32)
        t_c = np.maximum(s, t, dtype=np.uint32)

        # Optional dedup (stable mergesort to keep deterministic order)
        if dedup and s_c.size:
            keys = (s_c.astype(np.uint64) << 32) | t_c.astype(np.uint64)
            order = np.argsort(keys, kind='mergesort')
            ks = keys[order]
            keep = np.empty(ks.size, dtype=bool)
            keep[0] = True
            if ks.size > 1:
                keep[1:] = ks[1:] != ks[:-1]
            idx = order[keep]
            s_c, t_c = s_c[idx], t_c[idx]
            d, sc, tc = d[idx], sc[idx], tc[idx]

        # Persist quickly (no deflate by default)
        out_pairs = output_file + '_pairs.npz'
        saver = np.savez_compressed if compress else np.savez
        saver(out_pairs, s=s_c, t=t_c, d=d, sc=sc, tc=tc)

        # Update metadata
        md = dict(metadata)
        md['edge_count'] = int(s_c.size)
        md['files'] = {'pairs': os.path.basename(out_pairs)}
        with open(output_file + '_metadata.json', 'w') as f:
            json.dump(md, f, indent=2)


def load_interface_edges(output_dir: str, run_id: str) -> Tuple[List[Tuple[int, int]], Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Load interface edges saved by `save_interface_edges_efficient`.
    Backward compatible with the older CSR+attributes format.

    Returns
    -------
    edge_list : list[tuple[int,int]]
        Canonical undirected pairs (s, t) with s <= t
    attributes : dict[str, np.ndarray]
        Keys: 'source_cluster' ('sc'), 'target_cluster' ('tc'), 'distance' ('d')
    metadata : dict
        Metadata JSON for the run.
    """
    with perf_monitor.timed_operation(f"Load interface edges for {run_id}"):
        meta_path = os.path.join(output_dir, f"{run_id}_metadata.json")
        with open(meta_path, 'r') as f:
            metadata = json.load(f)

        files = metadata.get('files', {})

        # ---- New fast-pairs format ----
        if 'pairs' in files and files['pairs'] is not None:
            pair_path = os.path.join(output_dir, files['pairs'])
            # Load without mmap to keep portability; switch to mmap_mode='r' if needed
            data = np.load(pair_path)
            s = data['s'].astype(np.int32, copy=False)
            t = data['t'].astype(np.int32, copy=False)
            # Support both long and short keys for attributes
            if 'd' in data:
                d = data['d'].astype(np.float32, copy=False)
            else:
                d = data['distance'].astype(np.float32, copy=False)
            sc = (data['sc'] if 'sc' in data else data['source_cluster']).astype(np.int32, copy=False)
            tc = (data['tc'] if 'tc' in data else data['target_cluster']).astype(np.int32, copy=False)

            edge_list = list(zip(s.tolist(), t.tolist()))
            attributes = {'source_cluster': sc, 'target_cluster': tc, 'distance': d}
            return edge_list, attributes, metadata

        # ---- Legacy CSR+attributes format (backward compatibility) ----
        # Expect: { 'edges': '<file>.npz', 'attributes': '<file>.npz' }
        edge_file = files.get('edges', None)
        attr_file = files.get('attributes', None)
        if edge_file is None or attr_file is None:
            raise ValueError(f"Metadata for run '{run_id}' does not contain recognized file entries: {files}")

        edge_matrix = sp.load_npz(os.path.join(output_dir, edge_file))
        coo = edge_matrix.tocoo()
        edge_list = list(zip(coo.row.tolist(), coo.col.tolist()))

        attrs = dict(np.load(os.path.join(output_dir, attr_file)))
        # Normalize keys to the new names for callers
        attributes = {
            'source_cluster': attrs.get('source_cluster', attrs.get('sc')).astype(np.int32, copy=False),
            'target_cluster': attrs.get('target_cluster', attrs.get('tc')).astype(np.int32, copy=False),
            'distance': attrs.get('distance', attrs.get('d')).astype(np.float32, copy=False),
        }
        return edge_list, attributes, metadata


# ============================================================================
# OUTPUT HELPERS FOR RESOLUTION RUN USAGE
#   - save_interface_edges_pairs_df: write from a DataFrame (fast pairs format)
#   - process_single_resolution: reuse per-run iface DF; fall back to dict path
# ============================================================================

import os, json
from datetime import datetime
from typing import Dict, Any
import numpy as np

def save_interface_edges_pairs_df(
    iface_df,
    output_file: str,
    metadata: Dict[str, Any],
    *,
    dedup: bool = True,
    compress: bool = False,
) -> None:
    """
    Save interface edges from a pandas DataFrame to fast pairs format.

    Expects columns (at minimum):
      - 'source' (int), 'target' (int), 'distance' (float),
      - 'source_cluster' (int), 'target_cluster' (int)

    Writes:
      <output_file>_pairs.npz  (arrays: s, t, d, sc, tc)
      <output_file>_metadata.json
    """
    with perf_monitor.timed_operation("Save interface edges"):
        # Handle empty
        if iface_df is None or len(iface_df) == 0:
            md = dict(metadata)
            md["edge_count"] = 0
            md["files"] = {"pairs": None}
            with open(output_file + "_metadata.json", "w") as f:
                json.dump(md, f, indent=2)
            return

        # Pull arrays (ignore 'similarity' if present to avoid heavy recomputation)
        s = iface_df["source"].to_numpy(dtype=np.uint32, copy=False)
        t = iface_df["target"].to_numpy(dtype=np.uint32, copy=False)
        d = iface_df["distance"].to_numpy(dtype=np.float32, copy=False)

        # Prefer precomputed clusters if present; else map from a single run df if needed
        if "source_cluster" in iface_df.columns and "target_cluster" in iface_df.columns:
            sc = iface_df["source_cluster"].to_numpy(dtype=np.int32, copy=False)
            tc = iface_df["target_cluster"].to_numpy(dtype=np.int32, copy=False)
        else:
            # Safe fallback to zeros if columns missing (shouldn't happen with current pipeline)
            sc = np.zeros_like(s, dtype=np.int32)
            tc = np.zeros_like(t, dtype=np.int32)

        # Canonical orientation: s <= t
        s_c = np.minimum(s, t, dtype=np.uint32)
        t_c = np.maximum(s, t, dtype=np.uint32)

        # Optional dedup via 64-bit key
        if dedup and s_c.size:
            keys = (s_c.astype(np.uint64) << 32) | t_c.astype(np.uint64)
            order = np.argsort(keys, kind="mergesort")
            ks = keys[order]
            keep = np.empty(ks.size, dtype=bool)
            keep[0] = True
            if ks.size > 1:
                keep[1:] = ks[1:] != ks[:-1]
            idx = order[keep]
            s_c, t_c = s_c[idx], t_c[idx]
            d, sc, tc = d[idx], sc[idx], tc[idx]

        # Persist (np.savez is faster than compressed)
        out_pairs = output_file + "_pairs.npz"
        saver = np.savez_compressed if compress else np.savez
        saver(out_pairs, s=s_c, t=t_c, d=d, sc=sc, tc=tc)

        # Metadata
        md = dict(metadata)
        md["edge_count"] = int(s_c.size)
        md["files"] = {"pairs": os.path.basename(out_pairs)}
        with open(output_file + "_metadata.json", "w") as f:
            json.dump(md, f, indent=2)

def process_single_resolution(
    resolution,
    analyzer,
    output_dir,
    run_name,
    scale,
    min_cluster_size,
    rank_stat_col,
    prev_labels=None,
    warm_start=True,
    save_outputs=True,
    algorithm="leiden_igraph",
):
    tag = _fmt_res(resolution)
    run_id = f"{run_name}_res{tag}"

    initial_membership = prev_labels if (prev_labels is not None and warm_start) else None

    # --- community detection ---
    if algorithm.lower() == "leiden_igraph":
        cluster_stats, labels = analyzer.run_leiden_igraph(
            resolution=resolution,
            run_id=run_id,
            scale=scale,
            initial_membership=initial_membership,
            rank_stat_col=rank_stat_col,
            prune_small_clusters=True,
            min_cluster_size=min_cluster_size,
        )
    elif algorithm.lower() == "louvain_csr":
        cluster_stats, labels = analyzer.run_louvain_csr(
            resolution=resolution,
            run_id=run_id,
            scale=scale,
            initial_membership=initial_membership,
            rank_stat_col=rank_stat_col,
            prune_small_clusters=True,
            min_cluster_size=min_cluster_size,
        )
    elif algorithm.lower() == "leiden_csr":
        cluster_stats, labels = analyzer.run_leiden_csr(
            resolution=resolution,
            run_id=run_id,
            scale=scale,
            initial_membership=initial_membership,
            rank_stat_col=rank_stat_col,
            prune_small_clusters=True,
            min_cluster_size=min_cluster_size,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    analyzer.add_community_statistics(run_id)

    if save_outputs:
        res_dir = os.path.join(output_dir, f"res_{tag}")
        os.makedirs(res_dir, exist_ok=True)

        metadata = {
            "run_name": run_name,
            "resolution": float(resolution),
            "scale": (scale if isinstance(scale, (int, float)) else "adaptive"),
            "n_nodes": analyzer.loader.n_nodes,
            "timestamp": datetime.now().isoformat(),
            "coarsened": analyzer.coarsened,
            "coarsening_ratio": getattr(analyzer, "coarsening_ratio", None),
            "algorithm": algorithm,
        }

        out_base = os.path.join(res_dir, run_id)

        # Prefer reusing the already-built interface df
        iface_df = analyzer.interface_edges.get(run_id, None)
        if iface_df is not None and len(iface_df) > 0:
            save_interface_edges_pairs_df(iface_df, out_base, metadata, dedup=True, compress=False)
        else:
            interface_data = analyzer.extract_interface_edges(analyzer.runs[run_id]["labels"])
            save_interface_edges_efficient(interface_data, out_base, metadata, dedup=True, compress=False)

        stats_file = os.path.join(res_dir, f"{run_id}_stats.csv")
        cluster_stats.to_csv(stats_file, index=False)

    return labels


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
@click.option('--timing/--no-timing', default=True,
              help="Enable detailed timing statistics.")
@click.option('--algorithm', type=click.Choice(['leiden_igraph', 'louvain_csr', 'leiden_csr']), 
              default='louvain_csr',
              help="Community detection algorithm to use (default: louvain_csr)")
def main(location, output_dir, run_name, resolutions, similarity_scale,
         coarsen, coarsen_levels, sparsify, sparsify_pre_k, sparsify_post_k, warm_start,
         min_cluster_size, rank_stat_col, timing, algorithm):
    """Optimized community detection pipeline."""
    # Configure performance monitoring
    np.random.seed(42)
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
    
    # Initialize analyzer
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
    
    # Sequential processing with warm-start
    print("\nProcessing resolutions sequentially...")
    prev_labels = None
    
    for resolution in resolution_values:
        with perf_monitor.timed_operation(f"Process resolution {resolution}", verbose=True):
            labels = process_single_resolution(
                resolution, analyzer, output_dir, run_name, scale,
                min_cluster_size, rank_stat_col, 
                prev_labels=prev_labels if warm_start else None,
                algorithm=algorithm
            )
            
            # Store labels for next resolution if warm-start enabled
            if warm_start:
                prev_labels = labels
    
    # Summary
    print("\nPipeline complete!")
    print(f"Processed {len(resolution_values)} resolutions: {resolution_values}")
    print(f"Results saved to: {output_dir}")
    
    # Print timing summary
    if timing:
        perf_monitor.print_timing_summary()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_results(output_dir, run_name=None, resolution=None):
    """Load results from a previous run."""
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
            run_id = metadata.get('run_name', '') + f"_res{_fmt_res(res_value)}"

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
    """Visualize results from a previous run."""
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
                ax.grid(True, linestyle='--', alpha=0.7)
        
        # Fourth plot - interface edge count histogram
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
        
    return fig

def merge_interface_edges(output_dir, run_ids, output_file, min_overlap=1):
    """Merge interface edges from multiple runs."""
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
        
        # Convert to canonical form
        for src, tgt in edge_list:
            edge = (min(src, tgt), max(src, tgt))
            edge_counts[edge] += 1
            edge_appearances[edge].add(run_id)
    
    # Filter by minimum overlap
    filtered_edges = {edge: count for edge, count in edge_counts.items() if count >= min_overlap}
    
    # Create matrix representation
    if filtered_edges:
        n_nodes = max(max(src, tgt) for src, tgt in filtered_edges.keys()) + 1
        merged_matrix = sp.lil_matrix((n_nodes, n_nodes), dtype=np.int16)
        
        for (src, tgt), count in filtered_edges.items():
            merged_matrix[src, tgt] = count
            merged_matrix[tgt, src] = count
        
        # Convert to CSR
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

def compare_across_resolutions(output_dir, run_name, similarity_metric='jaccard'):
    """Compare interface edges across different resolutions."""
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
    
    # Calculate stability measure
    avg_similarity = np.mean([comp['similarity'] for comp in comparisons]) if comparisons else 0
    
    return {
        'run_name': run_name,
        'resolutions': resolutions,
        'comparisons': comparisons,
        'avg_similarity': avg_similarity,
        'similarity_metric': similarity_metric
    }

# Run the pipeline if executed directly
if __name__ == "__main__":
    main()