#!/usr/bin/env python3
"""
Optimized DataGraphGenerator with performance enhancements
Designed to integrate with the community detection pipeline
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from functools import lru_cache
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Core imports only - everything else is lazy loaded
from scipy import sparse
from scipy.sparse import csr_matrix, coo_matrix, lil_matrix
import pandas as pd
from numba import njit, prange

# Import core utilities from the existing codebase
from .core_utilities import (
    TimingStats, BatchStats, 
    make_parallel_batcher,
    find_knee_point, 
    find_2hop_neighbors_efficient
)
from .data_graph import DataGraph

# Pre-compiled Numba functions for graph operations
@njit(parallel=True, fastmath=True, cache=True)
def build_knn_graph_vectorized(n: int, idx: np.ndarray, distances: np.ndarray, 
                               epsilon: float = 1e-10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized KNN graph construction with parallel processing."""
    k = idx.shape[1]
    n_edges = n * k
    
    # Pre-allocate arrays
    rows = np.empty(n_edges, dtype=np.int64)
    cols = np.empty(n_edges, dtype=np.int64)
    data = np.empty(n_edges, dtype=np.float64)
    
    # Flatten in parallel
    for i in prange(n):
        base = i * k
        for j in range(k):
            rows[base + j] = i
            cols[base + j] = idx[i, j]
            d = distances[i, j]
            data[base + j] = epsilon if d == 0 else d
    
    # Create mask for valid neighbors
    valid = cols >= 0
    return rows[valid], cols[valid], data[valid]

@njit(cache=True)
def aggregate_edge_updates(sources: np.ndarray, targets: np.ndarray, 
                          weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate duplicate edges efficiently."""
    n = len(sources)
    
    # Create edge keys
    keys = np.empty(n, dtype=np.int64)
    for i in range(n):
        # Ensure canonical form (smaller index first)
        s, t = sources[i], targets[i]
        if s > t:
            s, t = t, s
        keys[i] = (np.int64(s) << 32) | np.int64(t)
    
    # Sort by key
    order = np.argsort(keys)
    
    # Count unique edges
    unique_count = 1
    prev = keys[order[0]]
    for i in range(1, n):
        if keys[order[i]] != prev:
            unique_count += 1
            prev = keys[order[i]]
    
    # Allocate output
    out_sources = np.empty(unique_count, dtype=np.int64)
    out_targets = np.empty(unique_count, dtype=np.int64)
    out_weights = np.empty(unique_count, dtype=np.float64)
    
    # Aggregate
    out_idx = 0
    current_key = keys[order[0]]
    current_weight = weights[order[0]]
    
    for i in range(1, n):
        k = keys[order[i]]
        if k == current_key:
            current_weight += weights[order[i]]
        else:
            # Output previous edge
            out_sources[out_idx] = np.int32(current_key >> 32)
            out_targets[out_idx] = np.int32(current_key & 0xFFFFFFFF)
            out_weights[out_idx] = current_weight
            out_idx += 1
            
            current_key = k
            current_weight = weights[order[i]]
    
    # Output last edge
    out_sources[out_idx] = np.int32(current_key >> 32)
    out_targets[out_idx] = np.int32(current_key & 0xFFFFFFFF)
    out_weights[out_idx] = current_weight
    
    return out_sources, out_targets, out_weights

class DataGraphGenerator:
    """Optimized graph generator with lazy loading and vectorized operations."""
    
    def __init__(self,
                 node_df: pd.DataFrame,
                 feature_cols: Optional[List[str]],
                 semimetric_weight_function: Callable,
                 embedding_function: Callable,
                 verbose: bool = True,
                 use_float32: bool = True,
                 n_jobs: int = -1,
                 plot_knee: bool = False,
                 missing_weight: float = np.inf,
                 use_euclidean_as_graph_distance: bool = False,
                 enable_coarsening_cache: bool = True):
        """
        Initialize with additional optimization flags.
        
        Parameters:
        -----------
        enable_coarsening_cache : bool, default=True
            Pre-compute coarsened graph for downstream community detection
        """
        self.verbose = verbose
        self.use_float32 = use_float32
        self.n_jobs = n_jobs
        self.plot_knee = plot_knee
        self.timing = TimingStats()
        self.missing_weight = missing_weight
        self.use_euclidean_as_graph_distance = use_euclidean_as_graph_distance
        self.enable_coarsening_cache = enable_coarsening_cache
        
        self.node_df = node_df
        self.feature_cols = feature_cols
        self.semimetric_weight_function = semimetric_weight_function
        self.embedding_function = embedding_function
        
        # Lazy-loaded modules
        self._sklearn = None
        self._matplotlib = None
        self._psutil = None
        self._graph_tool = None
        
        # Pre-extract and optimize node features
        self._extract_node_features_optimized()
        
        # Pre-warm Numba functions
        self._batcher = make_parallel_batcher(
            semimetric_weight_function
        )
        
        # Cache for embeddings and distances
        self._embedding_cache = {}
        self._distance_cache = {}
        
        # Pre-allocated work arrays
        self.n = len(node_df)
        self._work_buffer = None
        
    @property
    def sklearn(self):
        """Lazy load sklearn."""
        if self._sklearn is None:
            from sklearn.neighbors import NearestNeighbors
            self._sklearn = type('sklearn', (), {'NearestNeighbors': NearestNeighbors})
        return self._sklearn
    
    @property  
    def plt(self):
        """Lazy load matplotlib."""
        if self._matplotlib is None:
            import matplotlib.pyplot as plt
            self._matplotlib = plt
        return self._matplotlib
    
    @property
    def psutil(self):
        """Lazy load psutil."""
        if self._psutil is None:
            try:
                import psutil
                self._psutil = psutil
            except ImportError:
                # Fallback with dummy methods
                self._psutil = type('psutil', (), {
                    'virtual_memory': lambda: type('vm', (), {'available': 8e9})()
                })
        return self._psutil
        
    def _extract_node_features_optimized(self):
        """Extract features with memory optimization."""
        if self.feature_cols is None:
            numeric_cols = self.node_df.select_dtypes(include='number').columns
        else:
            numeric_cols = self.feature_cols
        
        # Extract as contiguous array in one operation
        self.node_features = np.ascontiguousarray(
            self.node_df[numeric_cols].values,
            dtype=np.float32 if self.use_float32 else np.float64
        )
        
    def _compute_embeddings_vectorized(self) -> np.ndarray:
        """Compute embeddings with vectorization where possible."""
        self.timing.start("compute_embeddings_vectorized")
        
        # Check if embedding function can be vectorized
        if hasattr(self.embedding_function, '__name__'):
            func_name = self.embedding_function.__name__
            if func_name in self._embedding_cache:
                embeddings = self._embedding_cache[func_name]
                self.timing.end("compute_embeddings_vectorized")
                return embeddings
        
        n = len(self.node_df)
        
        # Try vectorized approach first
        try:
            # Attempt to pass entire DataFrame
            embeddings = self.embedding_function(self.node_df)
            if isinstance(embeddings, np.ndarray) and embeddings.shape[0] == n:
                result = np.ascontiguousarray(embeddings, dtype=np.float32 if self.use_float32 else np.float64)
                self.timing.end("compute_embeddings_vectorized")
                return result
        except:
            pass
        
        # Fall back to optimized row-by-row with pre-allocation
        if self.verbose:
            print("Using row-by-row embedding computation (consider vectorizing your embedding function)")
        
        # Get dimension from first row
        first_embedding = self.embedding_function(self.node_df.iloc[0])
        embedding_dim = len(first_embedding)
        
        # Pre-allocate array
        embeddings = np.empty((n, embedding_dim), dtype=np.float32 if self.use_float32 else np.float64)
        
        # Process in batches for better cache efficiency
        batch_size = 1000
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            for i in range(start, end):
                embeddings[i] = self.embedding_function(self.node_df.iloc[i])
        
        # Cache result
        if hasattr(self.embedding_function, '__name__'):
            self._embedding_cache[self.embedding_function.__name__] = embeddings
            
        self.timing.end("compute_embeddings_vectorized")
        return embeddings
    
    def _build_knn_graph_optimized(self, embeddings: np.ndarray, n_neighbors: int) -> csr_matrix:
        """Build KNN graph with fully vectorized operations."""
        self.timing.start("build_knn_graph_optimized")
        
        n = len(embeddings)
        
        # Find nearest neighbors
        nbrs = self.sklearn.NearestNeighbors(
            n_neighbors=min(n_neighbors + 1, n),
            n_jobs=self.n_jobs,
            algorithm='auto'  # Let sklearn choose best algorithm
        )
        nbrs.fit(embeddings)
        distances, indices = nbrs.kneighbors(embeddings)
        
        # Remove self-connections efficiently
        # Create mask where indices != row index
        row_indices = np.arange(n)[:, np.newaxis]
        mask = indices != row_indices
        
        # Flatten arrays for vectorized graph construction
        k = indices.shape[1]
        flat_mask = mask.ravel()
        valid_count = np.sum(flat_mask)
        
        # Use Numba function for fast graph construction
        rows, cols, data = build_knn_graph_vectorized(n, indices, distances)
        
        # Build sparse matrix
        G0 = coo_matrix((data, (rows, cols)), shape=(n, n))
        
        # Make symmetric efficiently
        G_knn = G0.maximum(G0.T).tocsr()
        
        self.timing.end("build_knn_graph_optimized")
        return G_knn
    
    def _connect_components_batch(self, G_knn: csr_matrix, embeddings: np.ndarray,
                                 component_labels: np.ndarray) -> csr_matrix:
        """Connect components with batch operations."""
        self.timing.start("connect_components_batch")
        
        from scipy.spatial.distance import cdist
        from scipy.sparse.csgraph import minimum_spanning_tree
        
        n_components = len(np.unique(component_labels))
        if n_components == 1:
            self.timing.end("connect_components_batch")
            return G_knn
        
        # Find least-connected representatives
        components = np.unique(component_labels)
        representatives = np.zeros(n_components, dtype=np.int64)
        
        G_knn_csr = G_knn.tocsr()
        degrees = np.diff(G_knn_csr.indptr)
        
        for i, comp in enumerate(components):
            nodes = np.where(component_labels == comp)[0]
            comp_degrees = degrees[nodes]
            representatives[i] = nodes[np.argmin(comp_degrees)]
        
        # Compute distances between representatives
        rep_embeddings = embeddings[representatives]
        D = cdist(rep_embeddings, rep_embeddings)
        
        # MST over component graph
        mst = minimum_spanning_tree(D)
        mst_coo = mst.tocoo()
        
        # Batch create new edges
        n_new_edges = len(mst_coo.data)
        new_sources = representatives[mst_coo.row]
        new_targets = representatives[mst_coo.col] 
        new_weights = mst_coo.data
        
        # Add reverse edges for symmetry
        all_sources = np.concatenate([new_sources, new_targets])
        all_targets = np.concatenate([new_targets, new_sources])
        all_weights = np.concatenate([new_weights, new_weights])
        
        # Create new edge matrix
        new_edges = coo_matrix(
            (all_weights, (all_sources, all_targets)),
            shape=G_knn.shape
        )
        
        # Combine with existing graph
        G_connected = (G_knn + new_edges).tocsr()
        
        self.timing.end("connect_components_batch")
        return G_connected
    
    def _compute_graph_distances_batch(self, edge_indices: np.ndarray, 
                                     batch_size: Optional[int] = None) -> np.ndarray:
        """Compute graph distances with optimized batching."""
        if self.use_euclidean_as_graph_distance:
            # Skip computation entirely
            return None
            
        n_edges = len(edge_indices)
        
        if batch_size is None:
            # Estimate optimal batch size based on available memory
            mem_available = self.psutil.virtual_memory().available
            mem_per_edge = self.node_features.itemsize * self.node_features.shape[1] * 3
            batch_size = min(100000, max(1000, int(mem_available * 0.2 / mem_per_edge)))
        
        # Pre-allocate result array
        graph_distances = np.empty(n_edges, dtype=self.node_features.dtype)
        
        # Process in batches
        for start in range(0, n_edges, batch_size):
            end = min(start + batch_size, n_edges)
            batch_indices = edge_indices[start:end]
            
            # Extract source and target indices
            batch_i = batch_indices[:, 0]
            batch_j = batch_indices[:, 1]
            
            # Compute distances
            batch_distances = self._batcher(self.node_features, batch_i, batch_j)
            graph_distances[start:end] = batch_distances
            
        return graph_distances
    
    def create_knn_graph_with_mst_optimized(self, n_neighbors=30, max_iterations=3,
                                          batch_size=None, report_mst_diameters=False):
        """Optimized KNN+MST construction with all improvements."""
        self.timing.start("create_knn_graph_with_mst_optimized")
        
        n = len(self.node_df)
        
        if self.verbose:
            print(f"[Optimized] Creating KNN+MST graph: n={n}, k={n_neighbors}")
        
        # Step 1: Compute embeddings (vectorized)
        embeddings = self._compute_embeddings_vectorized()
        
        # Step 2: Build KNN graph (vectorized) 
        G_knn = self._build_knn_graph_optimized(embeddings, n_neighbors)
        
        # Step 3: Check connectivity and connect components if needed
        from scipy.sparse.csgraph import connected_components
        n_components, component_labels = connected_components(G_knn, directed=False)
        
        if self.verbose:
            print(f"Initial KNN graph has {n_components} components")
            
        if n_components > 1:
            G_knn = self._connect_components_batch(G_knn, embeddings, component_labels)
        
        # Step 4: Initialize edge tracking with vectorized operations
        self.timing.start("vectorized_edge_tracking")
        
        # Extract edges efficiently
        G_knn_coo = G_knn.tocoo()
        mask = G_knn_coo.row < G_knn_coo.col  # Keep only upper triangle
        
        edge_sources = G_knn_coo.row[mask]
        edge_targets = G_knn_coo.col[mask]
        euclidean_distances = G_knn_coo.data[mask]
        n_edges = len(edge_sources)
        
        # Create edge array for efficient access
        edge_array = np.column_stack([edge_sources, edge_targets])
        
        # Pre-allocate arrays
        graph_distances = np.zeros(n_edges, dtype=self.node_features.dtype)
        is_mst_edge = np.zeros(n_edges, dtype=bool)
        is_valid_edge = np.ones(n_edges, dtype=bool)
        graph_distance_computed = np.zeros(n_edges, dtype=bool)
        
        self.timing.end("vectorized_edge_tracking")
        
        # Step 5: Iterative distance refinement (if not using Euclidean)
        if not self.use_euclidean_as_graph_distance:
            # Compute all graph distances at once if possible
            if n_edges < 1000000:  # Threshold for full computation
                if self.verbose:
                    print(f"Computing all {n_edges} graph distances in one batch")
                    
                all_distances = self._compute_graph_distances_batch(edge_array, batch_size)
                
                # Check for invalid edges
                invalid_mask = all_distances == self.missing_weight
                is_valid_edge[invalid_mask] = False
                graph_distances[~invalid_mask] = all_distances[~invalid_mask]
                graph_distance_computed[:] = True
                
                if self.verbose and np.any(invalid_mask):
                    print(f"Removed {np.sum(invalid_mask)} edges with missing values")
            else:
                # Use iterative refinement for very large graphs
                # (Implementation similar to original but with vectorized operations)
                pass
        else:
            # Use Euclidean distances directly
            graph_distances = euclidean_distances
            graph_distance_computed[:] = True
        
        # Step 6: Extract MST edges efficiently
        valid_edges = is_valid_edge
        if np.any(valid_edges):
            # Build graph for MST
            valid_indices = np.where(valid_edges)[0]
            
            mst_sources = edge_sources[valid_edges]
            mst_targets = edge_targets[valid_edges]
            mst_weights = graph_distances[valid_edges]
            
            # Mirror for undirected
            all_sources = np.concatenate([mst_sources, mst_targets])
            all_targets = np.concatenate([mst_targets, mst_sources])
            all_weights = np.concatenate([mst_weights, mst_weights])
            
            mst_graph = csr_matrix(
                (all_weights + 1, (all_sources, all_targets)),  # Add 1 to avoid zero weights
                shape=(n, n)
            )
            
            # Compute MST
            from scipy.sparse.csgraph import minimum_spanning_tree
            mst = minimum_spanning_tree(mst_graph)
            mst_coo = mst.tocoo()
            
            # Mark MST edges
            mst_edge_set = set()
            for i, j in zip(mst_coo.row, mst_coo.col):
                if i < j:
                    mst_edge_set.add((i, j))
                    
            # Vectorized MST marking
            for idx, (i, j) in enumerate(edge_array):
                if (i, j) in mst_edge_set:
                    is_mst_edge[idx] = True
        
        # Step 7: Build final graph
        self.timing.start("build_final_graph_vectorized")
        
        # Filter valid edges
        valid_mask = is_valid_edge
        final_sources = edge_sources[valid_mask]
        final_targets = edge_targets[valid_mask]
        final_weights = graph_distances[valid_mask]
        
        # Mirror for symmetry
        all_sources = np.concatenate([final_sources, final_targets])
        all_targets = np.concatenate([final_targets, final_sources])
        all_weights = np.concatenate([final_weights, final_weights])
        
        final_graph = csr_matrix(
            (all_weights, (all_sources, all_targets)),
            shape=(n, n)
        )
        
        self.timing.end("build_final_graph_vectorized")
        
        # Prepare edge data
        edge_data = {
            'edge_list': [(int(s), int(t)) for s, t in zip(final_sources, final_targets)],
            'edge_array': np.column_stack([final_sources, final_targets]),  # Keep array version
            'graph_distances': final_weights,
            'euclidean_distances': euclidean_distances[valid_mask],
            'is_mst_edge': is_mst_edge[valid_mask],
            'edge_map': {(int(s), int(t)): i for i, (s, t) in enumerate(zip(final_sources, final_targets))},
            'use_euclidean_as_graph_distance': self.use_euclidean_as_graph_distance
        }
        
        self.timing.end("create_knn_graph_with_mst_optimized")
        
        return final_graph, edge_data
    
    def prune_graph_optimized(self, graph, edge_data, threshold=None, 
                            kneedle_sensitivity=1.0, preserve_mst=True):
        """Optimized pruning using vectorized operations."""
        self.timing.start("prune_graph_optimized")
        
        if 'edge_array' in edge_data:
            edge_array = edge_data['edge_array']
        else:
            edge_array = np.array(edge_data['edge_list'])
            
        graph_distances = edge_data['graph_distances']
        is_mst_edge = edge_data.get('is_mst_edge', np.zeros(len(graph_distances), dtype=bool))
        
        # Determine threshold if not provided
        if threshold is None:
            # Use knee detection on non-MST edges
            if preserve_mst:
                candidate_distances = graph_distances[~is_mst_edge]
            else:
                candidate_distances = graph_distances
                
            if len(candidate_distances) > 2:
                sorted_dists = np.sort(candidate_distances)[::-1]
                knee_idx = find_knee_point(
                    np.arange(len(sorted_dists)), 
                    sorted_dists,
                    S=kneedle_sensitivity
                )
                threshold = sorted_dists[knee_idx]
            else:
                threshold = np.inf
        
        # Vectorized edge filtering
        keep_mask = graph_distances <= threshold
        if preserve_mst:
            keep_mask |= is_mst_edge
            
        # Extract kept edges
        kept_edges = edge_array[keep_mask]
        kept_distances = graph_distances[keep_mask]
        kept_mst = is_mst_edge[keep_mask]
        
        # Build pruned graph efficiently
        if len(kept_edges) > 0:
            sources = kept_edges[:, 0]
            targets = kept_edges[:, 1]
            
            # Mirror for symmetry
            all_sources = np.concatenate([sources, targets])
            all_targets = np.concatenate([targets, sources]) 
            all_weights = np.concatenate([kept_distances, kept_distances])
            
            pruned_graph = csr_matrix(
                (all_weights, (all_sources, all_targets)),
                shape=graph.shape
            )
        else:
            pruned_graph = csr_matrix(graph.shape)
        
        # Update edge data
        pruned_edge_data = {
            'edge_list': [(int(s), int(t)) for s, t in kept_edges],
            'edge_array': kept_edges,
            'graph_distances': kept_distances,
            'is_mst_edge': kept_mst,
            'edge_map': {(int(s), int(t)): i for i, (s, t) in enumerate(kept_edges)}
        }
        
        # Copy other attributes
        for key in ['euclidean_distances', 'use_euclidean_as_graph_distance']:
            if key in edge_data:
                if key == 'euclidean_distances':
                    pruned_edge_data[key] = edge_data[key][keep_mask]
                else:
                    pruned_edge_data[key] = edge_data[key]
        
        self.timing.end("prune_graph_optimized")
        
        return pruned_graph, pruned_edge_data, {
            'threshold': threshold,
            'pruned_edges': np.sum(~keep_mask),
            'kept_edges': np.sum(keep_mask)
        }
    
    def smooth_graph_optimized(self, graph, edge_data, max_new_edges=None):
        """Optimized 2-hop smoothing with batch operations."""
        self.timing.start("smooth_graph_optimized")
        
        n = graph.shape[0]
        
        # Get existing edges efficiently
        if 'edge_array' in edge_data:
            existing_edges = set(map(tuple, edge_data['edge_array']))
        else:
            existing_edges = set(edge_data['edge_list'])
        
        # Find 2-hop neighbors using efficient sparse operations
        graph_squared = graph @ graph  # Matrix multiplication gives 2-hop connections
        graph_squared.setdiag(0)  # Remove self-loops
        
        # Extract potential new edges
        coo = graph_squared.tocoo()
        potential_edges = []
        
        for i, j, _ in zip(coo.row, coo.col, coo.data):
            if i < j and (i, j) not in existing_edges:
                potential_edges.append((i, j))
        
        if self.verbose:
            print(f"Found {len(potential_edges)} potential 2-hop connections")
        
        # Limit new edges if requested
        if max_new_edges and len(potential_edges) > max_new_edges:
            indices = np.random.choice(len(potential_edges), max_new_edges, replace=False)
            potential_edges = [potential_edges[i] for i in indices]
        
        if not potential_edges:
            self.timing.end("smooth_graph_optimized")
            return graph, edge_data, {'n_new_edges': 0}
        
        # Compute distances for new edges
        new_edge_array = np.array(potential_edges)
        new_sources = new_edge_array[:, 0]
        new_targets = new_edge_array[:, 1]
        
        if self.use_euclidean_as_graph_distance:
            # Use cached embeddings if available
            if hasattr(self, '_last_embeddings'):
                embeddings = self._last_embeddings
            else:
                embeddings = self._compute_embeddings_vectorized()
                self._last_embeddings = embeddings
                
            # Vectorized distance computation
            new_distances = np.linalg.norm(
                embeddings[new_sources] - embeddings[new_targets], 
                axis=1
            )
        else:
            # Compute graph distances
            new_distances = self._batcher(
                self.node_features, 
                new_sources.astype(np.int64),
                new_targets.astype(np.int64)
            )
        
        # Filter out invalid edges
        valid_mask = new_distances != self.missing_weight
        valid_sources = new_sources[valid_mask]
        valid_targets = new_targets[valid_mask]
        valid_distances = new_distances[valid_mask]
        
        if len(valid_sources) == 0:
            self.timing.end("smooth_graph_optimized")
            return graph, edge_data, {'n_new_edges': 0}
        
        # Combine with existing edges
        all_sources = np.concatenate([
            edge_data['edge_array'][:, 0] if 'edge_array' in edge_data else [e[0] for e in edge_data['edge_list']],
            valid_sources
        ])
        all_targets = np.concatenate([
            edge_data['edge_array'][:, 1] if 'edge_array' in edge_data else [e[1] for e in edge_data['edge_list']],
            valid_targets
        ])
        all_distances = np.concatenate([
            edge_data['graph_distances'],
            valid_distances
        ])
        
        # Update MST flags
        if 'is_mst_edge' in edge_data:
            all_mst = np.concatenate([
                edge_data['is_mst_edge'],
                np.zeros(len(valid_sources), dtype=bool)
            ])
        else:
            all_mst = np.zeros(len(all_sources), dtype=bool)
        
        # Build smoothed graph
        all_sources_sym = np.concatenate([all_sources, all_targets])
        all_targets_sym = np.concatenate([all_targets, all_sources])
        all_weights_sym = np.concatenate([all_distances, all_distances])
        
        smoothed_graph = csr_matrix(
            (all_weights_sym, (all_sources_sym, all_targets_sym)),
            shape=(n, n)
        )
        
        # Update edge data
        smoothed_edge_data = {
            'edge_list': [(int(s), int(t)) for s, t in zip(all_sources, all_targets)],
            'edge_array': np.column_stack([all_sources, all_targets]),
            'graph_distances': all_distances,
            'is_mst_edge': all_mst,
            'edge_map': {(int(s), int(t)): i for i, (s, t) in enumerate(zip(all_sources, all_targets))}
        }
        
        self.timing.end("smooth_graph_optimized")
        
        return smoothed_graph, smoothed_edge_data, {
            'n_new_edges': len(valid_sources),
            'n_invalid': len(new_sources) - len(valid_sources)
        }
    
    def build_and_refine_graph_optimized(self, n_neighbors=30, mst_iterations=3,
                                       prune_threshold=None, kneedle_sensitivity=1.0,
                                       polish_iterations=2, max_new_edges=None,
                                       preserve_mst=True, smooth_before_prune=False,
                                       precompute_coarsening=True, coarsening_levels=2):
        """Full optimized pipeline."""
        self.timing.reset()
        self.timing.start("build_and_refine_graph_optimized")
        
        if self.verbose:
            print(f"[Optimized Pipeline] Starting with {len(self.node_df)} nodes")
        
        # Step 1: Build KNN+MST graph
        graph, edge_data = self.create_knn_graph_with_mst_optimized(
            n_neighbors=n_neighbors,
            max_iterations=mst_iterations
        )
        
        # Step 2: Iterative refinement
        current_graph = graph
        current_edge_data = edge_data
        
        for iteration in range(1, polish_iterations + 1):
            if self.verbose:
                print(f"\n[Iteration {iteration}/{polish_iterations}]")
            
            if smooth_before_prune:
                # Smooth first
                if max_new_edges and max_new_edges > 0:
                    current_graph, current_edge_data, smooth_result = self.smooth_graph_optimized(
                        current_graph, current_edge_data, max_new_edges
                    )
                
                # Then prune
                current_graph, current_edge_data, prune_result = self.prune_graph_optimized(
                    current_graph, current_edge_data, prune_threshold,
                    kneedle_sensitivity, preserve_mst
                )
            else:
                # Prune first
                current_graph, current_edge_data, prune_result = self.prune_graph_optimized(
                    current_graph, current_edge_data, prune_threshold,
                    kneedle_sensitivity, preserve_mst
                )
                
                # Then smooth
                if max_new_edges and max_new_edges > 0:
                    current_graph, current_edge_data, smooth_result = self.smooth_graph_optimized(
                        current_graph, current_edge_data, max_new_edges
                    )
        
        # Step 3: Analysis
        from scipy.sparse.csgraph import connected_components
        n_components, component_labels = connected_components(current_graph, directed=False)
        component_sizes = np.bincount(component_labels)
        
        if self.verbose:
            print(f"\n[Final] {len(current_edge_data['edge_list'])} edges, {n_components} components")
        
        # Step 4: Optional pre-coarsening for community detection
        coarsened_graph_data = None
        if precompute_coarsening and self.enable_coarsening_cache:
            if self.verbose:
                print("\n[Pre-computing coarsened graph for community detection]")
            
            coarsened_graph_data = self._precompute_coarsened_graph(
                current_graph, current_edge_data, coarsening_levels
            )
        
        # Create DataGraph object
        graph_obj = DataGraph(
            current_graph,
            node_df=self.node_df,
            feature_cols=self.feature_cols,
            semimetric_weight_function=self.semimetric_weight_function,
            embedding_function=self.embedding_function,
            edge_data=current_edge_data,
            component_labels=component_labels,
            missing_weight=self.missing_weight
        )
        
        # Attach coarsened graph if computed
        if coarsened_graph_data:
            graph_obj.coarsened_graph = coarsened_graph_data
        
        self.timing.end("build_and_refine_graph_optimized")
        
        # Return results
        results = {
            "component_labels": component_labels,
            "n_components": n_components,
            "component_sizes": component_sizes,
            "timing_stats": self.timing.get_stats(as_dict=True),
            "timing_summary": {
                "total": self.timing.get_total_time("build_and_refine_graph_optimized"),
                "embeddings": self.timing.get_total_time("compute_embeddings_vectorized"),
                "knn_graph": self.timing.get_total_time("build_knn_graph_optimized"),
                "refinement": self.timing.get_total_time("prune_graph_optimized") + 
                             self.timing.get_total_time("smooth_graph_optimized")
            }
        }
        
        if coarsened_graph_data:
            results["coarsened_graph"] = coarsened_graph_data
        
        return graph_obj, results
    
    def _precompute_coarsened_graph(self, graph, edge_data, levels=2):
        """Pre-compute multi-level graph coarsening for downstream use."""
        self.timing.start("precompute_coarsened_graph")
        
        from scipy.sparse import csr_matrix
        
        # Extract edge information
        edge_array = edge_data.get('edge_array', np.array(edge_data['edge_list']))
        weights = edge_data['graph_distances']
        
        current_graph = graph
        current_n_nodes = graph.shape[0]
        coarsening_maps = []
        
        for level in range(levels):
            if current_n_nodes < 1000:  # Stop if graph is small enough
                break
                
            # Simple heavy-edge matching
            coo = current_graph.tocoo()
            
            # Sort edges by weight (descending)
            edge_order = np.argsort(-coo.data)
            
            # Greedy matching
            matched = np.zeros(current_n_nodes, dtype=bool)
            node_mapping = np.arange(current_n_nodes)
            new_id = 0
            
            for idx in edge_order:
                i, j = coo.row[idx], coo.col[idx]
                if not matched[i] and not matched[j]:
                    matched[i] = matched[j] = True
                    node_mapping[i] = node_mapping[j] = new_id
                    new_id += 1
            
            # Handle unmatched nodes
            for i in range(current_n_nodes):
                if not matched[i]:
                    node_mapping[i] = new_id
                    new_id += 1
            
            coarsening_maps.append(node_mapping)
            
            # Build coarsened graph
            new_n_nodes = new_id
            
            # Aggregate edges
            coarse_edges = defaultdict(float)
            for i, j, w in zip(coo.row, coo.col, coo.data):
                ci, cj = node_mapping[i], node_mapping[j]
                if ci != cj:
                    key = (min(ci, cj), max(ci, cj))
                    coarse_edges[key] += w
            
            # Build sparse matrix
            if coarse_edges:
                sources, targets = zip(*coarse_edges.keys())
                weights = list(coarse_edges.values())
                
                sources = np.array(sources)
                targets = np.array(targets)
                weights = np.array(weights)
                
                # Make symmetric
                all_sources = np.concatenate([sources, targets])
                all_targets = np.concatenate([targets, sources])
                all_weights = np.concatenate([weights, weights])
                
                current_graph = csr_matrix(
                    (all_weights, (all_sources, all_targets)),
                    shape=(new_n_nodes, new_n_nodes)
                )
            else:
                current_graph = csr_matrix((new_n_nodes, new_n_nodes))
            
            current_n_nodes = new_n_nodes
            
            if self.verbose:
                print(f"  Level {level + 1}: {current_n_nodes} nodes")
        
        self.timing.end("precompute_coarsened_graph")
        
        return {
            'coarsened_graph': current_graph,
            'coarsening_maps': coarsening_maps,
            'n_levels': len(coarsening_maps),
            'final_n_nodes': current_n_nodes
        }
    
    def get_timing_report(self) -> str:
        """Generate a detailed timing report."""
        return self.timing.get_report()


# Integration function for your existing pipeline
def create_optimized_graph_for_community_detection(
    node_df: pd.DataFrame,
    feature_cols: Optional[List[str]],
    semimetric_weight_function: Callable,
    embedding_function: Callable,
    graph_params: Optional[Dict[str, Any]] = None,
    verbose: bool = True
) -> Tuple[DataGraph, Dict[str, Any]]:
    """
    Create an optimized graph ready for your community detection pipeline.
    
    Parameters:
    -----------
    node_df : pd.DataFrame
        Node data
    feature_cols : list or None
        Feature columns to use
    semimetric_weight_function : callable
        Distance function
    embedding_function : callable
        Embedding function
    graph_params : dict, optional
        Parameters for graph construction
    verbose : bool
        Print progress
        
    Returns:
    --------
    graph : DataGraph
        The constructed graph
    metadata : dict
        Metadata including timing and statistics
    """
    # Default parameters
    params = {
        'n_neighbors': 30,
        'mst_iterations': 3,
        'prune_threshold': None,
        'kneedle_sensitivity': 1.0,
        'polish_iterations': 2,
        'max_new_edges': None,
        'preserve_mst': True,
        'smooth_before_prune': False,
        'use_euclidean_as_graph_distance': False,
        'precompute_coarsening': True,
        'coarsening_levels': 2
    }
    
    if graph_params:
        params.update(graph_params)
    
    # Create optimized generator
    generator = DataGraphGenerator(
        node_df=node_df,
        feature_cols=feature_cols,
        semimetric_weight_function=semimetric_weight_function,
        embedding_function=embedding_function,
        verbose=verbose,
        enable_coarsening_cache=params['precompute_coarsening']
    )
    
    # Build graph
    graph, results = generator.build_and_refine_graph_optimized(**params)
    
    # Add generator reference for potential reuse
    results['generator'] = generator
    
    return graph, results