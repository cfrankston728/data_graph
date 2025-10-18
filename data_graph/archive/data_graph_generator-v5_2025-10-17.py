"""
DataGraphGenerator - Creates and refines a sparse graph over data according to a premetric weight function
Optimized version with array-based edge storage and in-place operations
"""
import numpy as np
import time

from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from sklearn.neighbors import kneighbors_graph

import pandas as pd

from numba import njit, prange

from .core_utilities import (
    TimingStats, BatchStats, 
    make_parallel_batcher,
    find_knee_point, 
    find_2hop_neighbors_efficient
)
from .data_graph import DataGraph

# Optional GPU MST (needs rapids/cuGraph)
try:
    import cugraph
    import cudf
    _has_cugraph = True
except ImportError:
    _has_cugraph = False

def prewarm_numba_functions(dist_func, feature_dim=4):
    """
    Pre-warm the numba batcher for dist_func so that the first real call
    doesn't pay the compilation hit.
    """
    # 1) build your parallel batcher
    compute_batch = make_parallel_batcher(dist_func)

    # 2) make a tiny dummy input
    dummy_feats = np.zeros((2, feature_dim), dtype=np.float64)
    dummy_idx   = np.array([0, 1],             dtype=np.int64)

    # 3) call it once so numba compiles it
    _ = compute_batch(dummy_feats, dummy_idx, dummy_idx)

    # return it so callers can stash it if they like
    return compute_batch

from scipy.sparse.csgraph import breadth_first_order

def check_graph_connectivity(csr_graph):
    """
    Return True iff csr_graph (scipy.sparse.csr_matrix) is connected.
    """
    # start BFS from node 0
    order = breadth_first_order(csr_graph, i_start=0, return_predecessors=False)
    # If we reached every node, it's connected
    return order.size == csr_graph.shape[0]

import numpy as np
from numba import njit, prange, config, get_num_threads, get_thread_id

@njit
def _union_find_parent(parent, i):
    """Path compression find for union-find data structure."""
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i

@njit
def _union_by_rank(parent, rank, x, y):
    """Union by rank for union-find data structure."""
    xroot = _union_find_parent(parent, x)
    yroot = _union_find_parent(parent, y)
    if xroot == yroot:
        return False
    if rank[xroot] < rank[yroot]:
        parent[xroot] = yroot
    elif rank[xroot] > rank[yroot]:
        parent[yroot] = xroot
    else:
        parent[yroot] = xroot
        rank[xroot] += 1
    return True

@njit(parallel=True)
def boruvka_mst_parallel_correct(n, rows, cols, weights):
    """
    Correct parallel Borůvka's algorithm using thread-local reduction.
    
    This implementation avoids race conditions by having each thread
    maintain its own cheapest edge array, then merging them serially.
    
    Parameters:
    -----------
    n : int
        Number of vertices
    rows : array of int32
        Source vertices of edges
    cols : array of int32
        Destination vertices of edges
    weights : array of float
        Edge weights
        
    Returns:
    --------
    selected : array of bool
        Boolean mask indicating which edges are in the MST
    """
    # Get number of threads (Numba will set this based on CPU cores)
    nt = config.NUMBA_NUM_THREADS
    
    # Initialize union-find structure
    parent = np.arange(n, dtype=np.int32)
    rank = np.zeros(n, dtype=np.int32)
    comp_count = n
    m = rows.shape[0]
    selected = np.zeros(m, dtype=np.bool_)
    
    # Thread-local storage for cheapest edges
    # Each thread gets its own row to avoid race conditions
    cheapest_per_thread = np.full((nt, n), -1, dtype=np.int32)
    
    # Global cheapest edges (for serial reduction)
    cheapest = np.empty(n, dtype=np.int32)
    
    while comp_count > 1:
        # Reset thread-local arrays
        cheapest_per_thread.fill(-1)
        
        # Phase 1: Parallel edge scanning
        # Each thread finds cheapest edges for components it encounters
        for e in prange(m):
            tid = get_thread_id()
            u = rows[e]
            v = cols[e]
            w = weights[e]
            
            # Find roots
            ru = _union_find_parent(parent, u)
            rv = _union_find_parent(parent, v)
            
            # Skip if already in same component
            if ru == rv:
                continue
            
            # Update thread-local cheapest for ru
            ce = cheapest_per_thread[tid, ru]
            if ce < 0 or w < weights[ce]:
                cheapest_per_thread[tid, ru] = e
            
            # Update thread-local cheapest for rv
            ce = cheapest_per_thread[tid, rv]
            if ce < 0 or w < weights[ce]:
                cheapest_per_thread[tid, rv] = e
        
        # Phase 2: Serial reduction - merge thread results
        cheapest.fill(-1)
        for t in range(nt):
            for comp in range(n):
                e = cheapest_per_thread[t, comp]
                if e >= 0:
                    # Check if this edge is cheaper than current global cheapest
                    ce = cheapest[comp]
                    if ce < 0 or weights[e] < weights[ce]:
                        cheapest[comp] = e
        
        # Phase 3: Merge components using cheapest edges
        merged = 0
        for comp in range(n):
            e = cheapest[comp]
            if e >= 0:
                u = rows[e]
                v = cols[e]
                # Try to union the components
                if _union_by_rank(parent, rank, u, v):
                    selected[e] = True
                    comp_count -= 1
                    merged += 1
        
        # Early exit if no merges (disconnected graph)
        if merged == 0:
            break
    
    return selected


# For even better performance on very large graphs, we can use a hybrid approach
@njit(parallel=True)
def boruvka_mst_hybrid(n, rows, cols, weights, parallel_threshold=50000):
    """
    Hybrid Borůvka that switches between parallel and sequential based on graph size.
    
    For smaller graphs or when the number of components becomes small,
    sequential execution is often faster due to reduced overhead.
    """
    # Start with parallel for large graphs
    if rows.shape[0] > parallel_threshold:
        return boruvka_mst_parallel_correct(n, rows, cols, weights)
    else:
        return boruvka_mst_sequential(n, rows, cols, weights)


# Optional: Active component tracking for additional optimization
@njit(parallel=True)
def boruvka_mst_parallel_active(n, rows, cols, weights):
    """
    Parallel Borůvka with both thread-local reduction and active component tracking.
    
    This version tracks which components were modified in the last iteration
    to reduce unnecessary edge scans.
    """
    nt = config.NUMBA_NUM_THREADS
    
    parent = np.arange(n, dtype=np.int32)
    rank = np.zeros(n, dtype=np.int32)
    comp_count = n
    m = rows.shape[0]
    selected = np.zeros(m, dtype=np.bool_)
    
    cheapest_per_thread = np.full((nt, n), -1, dtype=np.int32)
    cheapest = np.empty(n, dtype=np.int32)
    
    # Track active components
    active = np.ones(n, dtype=np.bool_)
    next_active = np.zeros(n, dtype=np.bool_)
    
    while comp_count > 1:
        cheapest_per_thread.fill(-1)
        next_active.fill(False)
        
        # Phase 1: Parallel edge scanning (only check edges with active components)
        for e in prange(m):
            tid = get_thread_id()
            u = rows[e]
            v = cols[e]
            
            # Find roots
            ru = _union_find_parent(parent, u)
            rv = _union_find_parent(parent, v)
            
            if ru == rv:
                continue
            
            # Skip if neither component is active
            if not (active[ru] or active[rv]):
                continue
            
            w = weights[e]
            
            # Update thread-local cheapest
            ce = cheapest_per_thread[tid, ru]
            if ce < 0 or w < weights[ce]:
                cheapest_per_thread[tid, ru] = e
            
            ce = cheapest_per_thread[tid, rv]
            if ce < 0 or w < weights[ce]:
                cheapest_per_thread[tid, rv] = e
        
        # Phase 2: Serial reduction
        cheapest.fill(-1)
        for t in range(nt):
            for comp in range(n):
                if not active[comp]:
                    continue
                e = cheapest_per_thread[t, comp]
                if e >= 0:
                    ce = cheapest[comp]
                    if ce < 0 or weights[e] < weights[ce]:
                        cheapest[comp] = e
        
        # Phase 3: Merge components
        merged = 0
        for comp in range(n):
            e = cheapest[comp]
            if e >= 0:
                u = rows[e]
                v = cols[e]
                if _union_by_rank(parent, rank, u, v):
                    selected[e] = True
                    comp_count -= 1
                    merged += 1
                    # Mark the new root as active
                    new_root = _union_find_parent(parent, u)
                    next_active[new_root] = True
        
        # No active edges found
        if merged == 0:
            break
        
        # Swap active arrays
        active, next_active = next_active, active
    
    return selected


# Sequential version for comparison/fallback
@njit
def boruvka_mst_sequential(n, rows, cols, weights):
    """
    Sequential Borůvka's algorithm - no parallelism, no races.
    Often faster for smaller graphs due to lower overhead.
    """
    parent = np.arange(n, dtype=np.int32)
    rank = np.zeros(n, dtype=np.int32)
    comp_count = n
    m = rows.shape[0]
    selected = np.zeros(m, dtype=np.bool_)
    cheapest = np.empty(n, dtype=np.int32)
    
    while comp_count > 1:
        cheapest.fill(-1)
        
        # Find cheapest outgoing edges
        for e in range(m):
            u = rows[e]
            v = cols[e]
            w = weights[e]
            
            ru = _union_find_parent(parent, u)
            rv = _union_find_parent(parent, v)
            
            if ru == rv:
                continue
            
            # Update cheapest for each component
            if cheapest[ru] < 0 or w < weights[cheapest[ru]]:
                cheapest[ru] = e
            if cheapest[rv] < 0 or w < weights[cheapest[rv]]:
                cheapest[rv] = e
        
        # Merge components
        merged = 0
        for comp in range(n):
            e = cheapest[comp]
            if e >= 0:
                u = rows[e]
                v = cols[e]
                if _union_by_rank(parent, rank, u, v):
                    selected[e] = True
                    comp_count -= 1
                    merged += 1
        
        if merged == 0:
            break
    
    return selected
# Keep original version as fallback
boruvka_mst_parallel = boruvka_mst_parallel_active

@njit(parallel=True)
def extract_upper_triangle_edges_from_csr_numba(indptr, indices, data, n):
    """
    Numba-accelerated extraction of upper-triangle edges from CSR.
    Returns (rows, cols, weights) as numpy arrays.
    """
    # Count edges first
    edge_count = 0
    for i in range(n):
        for j_idx in range(indptr[i], indptr[i+1]):
            j = indices[j_idx]
            if i < j:
                edge_count += 1
    
    # Allocate output arrays
    rows = np.empty(edge_count, dtype=np.int32)
    cols = np.empty(edge_count, dtype=np.int32)
    weights = np.empty(edge_count, dtype=data.dtype)
    
    # Fill arrays in parallel
    edge_idx = 0
    for i in range(n):
        for j_idx in range(indptr[i], indptr[i+1]):
            j = indices[j_idx]
            if i < j:
                rows[edge_idx] = i
                cols[edge_idx] = j
                weights[edge_idx] = data[j_idx]
                edge_idx += 1
                
    return rows, cols, weights

def extract_upper_triangle_edges_from_csr(graph):
    """
    Extract upper-triangle edge arrays from CSR matrix using numba acceleration.
    Returns (rows, cols, weights) as numpy int32/float arrays.
    """
    return extract_upper_triangle_edges_from_csr_numba(
        graph.indptr, graph.indices, graph.data, graph.shape[0]
    )

@njit
def count_edges_per_row(indptr, indices, n, upper_only):
    counts = np.zeros(n, dtype=np.int32)
    for i in range(n):
        for j_idx in range(indptr[i], indptr[i+1]):
            j = indices[j_idx]
            if not upper_only or i < j:
                counts[i] += 1
    return counts

@njit
def prefix_sum(arr):
    total = 0
    out = np.empty_like(arr)
    for i in range(arr.shape[0]):
        out[i] = total
        total += arr[i]
    return out, total

@njit(parallel=True)
def extract_all_edges_from_csr_numba(indptr, indices, data, n, upper_only):
    # Phase 1: how many edges from each row?
    per_row = count_edges_per_row(indptr, indices, n, upper_only)
    prefix, edge_count = prefix_sum(per_row)

    # Allocate output
    edge_arr = np.empty((edge_count, 2), dtype=np.int32)
    weights  = np.empty(edge_count,     dtype=data.dtype)

    # Phase 2: fill in parallel, one row per thread
    for i in prange(n):
        insert_pos = prefix[i]
        for j_idx in range(indptr[i], indptr[i+1]):
            j = indices[j_idx]
            if not upper_only or i < j:
                edge_arr[insert_pos, 0] = i
                edge_arr[insert_pos, 1] = j
                weights[insert_pos]     = data[j_idx]
                insert_pos += 1

    return edge_arr, weights

def extract_all_edges_from_csr(graph, upper_only=True):
    """
    Extract edges directly from CSR without converting to COO.
    Returns edge_arr (n_edges, 2) and weights array.
    """
    return extract_all_edges_from_csr_numba(
        graph.indptr, graph.indices, graph.data, graph.shape[0], upper_only
    )

def find_2hop_neighbors_sparse(graph):
    """
    Find 2-hop neighbors using sparse matrix multiplication.
    Much more efficient than iterating through edges.
    
    Returns:
    --------
    edge_arr : np.ndarray of shape (n_edges, 2)
        Array of new 2-hop edges
    """
    # Compute A^2 to get 2-hop paths
    A2 = graph @ graph
    
    # Remove direct edges (already in graph)
    A2 = A2 - graph
    
    # Remove self-loops
    A2.setdiag(0)
    
    # Remove any negative values that might have resulted from subtraction
    A2.data[A2.data < 0] = 0
    A2.eliminate_zeros()
    
    # Extract new edges as array
    edge_arr, _ = extract_all_edges_from_csr(A2, upper_only=True)
    return edge_arr

    
class DataGraphGenerator:
    def __init__(self,
                 node_df,
                 feature_cols,
                 premetric_weight_function,
                 embedding_function,
                 verbose=True, 
                 use_float32=True,
                 n_jobs=-1, 
                 plot_knee=False,
                 missing_weight=np.inf,
                 use_euclidean_as_graph_distance=False,
                 use_gpu_mst=False):
        """
        Graph generator for a custom distance graph over data
        
        Parameters:
        -----------
        verbose : bool, default=True
            Whether to print progress messages
        use_float32 : bool, default=True
            Whether to use float32 precision instead of float64 for memory efficiency
        n_jobs : int, default=-1
            Number of parallel jobs for parallel operations
        plot_knee : bool, default=False
            Whether to generate a plot of the knee point detection
        missing_weight : float, default=np.inf
            Value returned by premetric function to indicate missing/invalid edge
        use_euclidean_as_graph_distance : bool, default=False
            If True, skip graph distance computation and use Euclidean distances directly.
            Set to True when you know the premetric function returns the same values
            as Euclidean distance in embedding space.
        """
        self.use_gpu_mst = use_gpu_mst and _has_cugraph
        
        self.verbose = verbose
        self.use_float32 = use_float32
        self.n_jobs = n_jobs
        self.plot_knee = plot_knee
        self.timing = TimingStats()  # Initialize timing stats
        self.missing_weight = missing_weight
        self.use_euclidean_as_graph_distance = use_euclidean_as_graph_distance

        self.node_df = node_df
        self.feature_cols = feature_cols
        self.premetric_weight_function = premetric_weight_function
        self.embedding_function = embedding_function
        
        self._extract_node_features()
        
        # Lazy initialization of batcher
        self._batcher = None
        self._batcher_initialized = False
        
        # Edge array cache
        self._edge_arr_cache = {}
        
        # Sorted edge cache for pruning
        self._sorted_edge_cache = {}

    def _extract_node_features(self):
        # pick your columns
        if self.feature_cols is None:
            df = self.node_df.select_dtypes(include='number')
        else:
            df = self.node_df[self.feature_cols]
    
        # convert to float32 and then force C-order
        arr = df.to_numpy(dtype=np.float32)
        arr = np.ascontiguousarray(arr)
    
        self.node_features = arr
    
    def _get_batcher(self):
        """Lazy initialization of the numba batcher."""
        if not self._batcher_initialized:
            self._batcher = prewarm_numba_functions(
                self.premetric_weight_function,
                feature_dim=len(self.feature_cols or
                              self.node_df.select_dtypes(include='number').columns)
            )
            self._batcher_initialized = True
        return self._batcher

    def _suggest_euclidean_optimization(self, graph_distances, euclidean_distances, threshold=0.01):
        """
        Check if graph distances are essentially the same as Euclidean distances
        
        Parameters:
        -----------
        threshold : float
            Relative difference threshold to consider distances "the same"
        
        Returns:
        --------
        bool
            True if distances are essentially the same
        """
        if len(graph_distances) == 0 or len(euclidean_distances) == 0:
            return False
            
        # Compute relative differences for non-zero distances
        valid_mask = (euclidean_distances > 0) & (graph_distances > 0)
        if np.sum(valid_mask) == 0:
            return False
            
        valid_graph = graph_distances[valid_mask]
        valid_euclidean = euclidean_distances[valid_mask]
        
        # Check relative differences
        rel_diff = np.abs(valid_graph - valid_euclidean) / valid_euclidean
        mean_rel_diff = np.mean(rel_diff)
        max_rel_diff = np.max(rel_diff)
        
        if self.verbose and mean_rel_diff < threshold:
            print(f"\n         SUGGESTION: Graph distances are very similar to Euclidean distances")
            print(f"         Mean relative difference: {mean_rel_diff:.2%}, Max: {max_rel_diff:.2%}")
            print(f"         Consider setting use_euclidean_as_graph_distance=True for better performance\n")
        
        return mean_rel_diff < threshold
        
    def estimate_optimal_batch_size(self, n_pairs, feature_dim):
        """Estimate optimal batch size based on data dimensions and available memory"""
        try:
            import psutil
            # Memory per pair (2 points × feature_dim × bytes per float)
            bytes_per_pair = 2 * feature_dim * (4 if self.use_float32 else 8)
            
            # Include overhead for intermediate calculations
            bytes_per_pair *= 1.5  # 50% overhead estimate
            
            # Use at most 20% of available memory
            available_memory = psutil.virtual_memory().available * 0.2
            optimal_pairs = int(available_memory / bytes_per_pair)
            
            # Reasonable bounds
            return max(1000, min(optimal_pairs, n_pairs))
        except ImportError:
            # Default if psutil not available
            return min(100000, n_pairs)
    
    def estimate_graph_building_batch_size(self, n_edges, max_edges_per_batch=np.inf):
        """Estimate optimal batch size for graph building operations"""
        try:
            import psutil
            
            # Memory per edge for graph building:
            # - Input: 2 int32 (edge endpoints) + 1 float (distance)
            # - Output: 2 * (2 int32 + 1 float) for bidirectional
            # - Temporary arrays during processing
            
            bytes_per_edge = (
                2 * 4 +  # edge endpoints (int32)
                1 * (4 if self.use_float32 else 8) +  # distance
                2 * (2 * 4 + (4 if self.use_float32 else 8))  # output arrays
            )
            
            # Add overhead for list operations and numpy internals
            bytes_per_edge *= 2  # 100% overhead for safety
            
            # Use at most 10% of available memory (more conservative for graph ops)
            available_memory = psutil.virtual_memory().available * 0.1
            optimal_edges = int(available_memory / bytes_per_edge)
            
            # Reasonable bounds: at least 1000, at most 2M edges per batch
            print(f'Optimal edges per batch: {optimal_edges}')
            print(f'Edges per batch capped at: {max_edges_per_batch}')
            return max(1000, min(optimal_edges, max_edges_per_batch, n_edges))
        except ImportError:
            # Default if psutil not available
            return min(10000, n_edges)
    
    def _build_csr_from_edge_arrays(self, rows, cols, data, n):
        """Build CSR matrix from edge arrays with deduplication and sorting."""
        M = csr_matrix((data, (rows, cols)), shape=(n,n))
        M.sum_duplicates()
        M.sort_indices()
        return M

    def build_graph_from_edges_memory_efficient(self, edge_arr, distances, is_valid_edge, n, batch_size=None):
        """
        Build CSR matrix with minimal memory footprint, de‑duplicated and sorted.
        Now accepts edge_arr as (n_edges, 2) array instead of list of tuples.
        """
        valid_indices = np.where(is_valid_edge)[0]
        n_valid = len(valid_indices)

        if batch_size is None:
            batch_size = self.estimate_graph_building_batch_size(n_valid)
            if self.verbose:
                print(f"         Using batch size {batch_size:,} for {n_valid:,} valid edges")

        all_rows = []
        all_cols = []
        all_data = []

        for start in range(0, n_valid, batch_size):
            end = min(start + batch_size, n_valid)
            batch_idx = valid_indices[start:end]

            # Direct array slicing instead of list comprehension
            batch_edges = edge_arr[batch_idx]  # shape (batch, 2)
            batch_dists = distances[batch_idx]

            k = len(batch_idx)
            row_block = np.empty(2 * k, dtype=np.int32)
            col_block = np.empty(2 * k, dtype=np.int32)
            data_block = np.repeat(batch_dists, 2)

            # Use array slicing for efficiency
            row_block[0::2] = batch_edges[:, 0]
            row_block[1::2] = batch_edges[:, 1]
            col_block[0::2] = batch_edges[:, 1]
            col_block[1::2] = batch_edges[:, 0]

            all_rows.append(row_block)
            all_cols.append(col_block)
            all_data.append(data_block)

        final_rows = np.concatenate(all_rows)
        final_cols = np.concatenate(all_cols)
        final_data = np.concatenate(all_data)

        # now use your helper to get a deduped, sorted CSR
        return self._build_csr_from_edge_arrays(final_rows, final_cols, final_data, n)

    def extract_mst_edges_boruvka(self, graph, n):
        """
        Extract MST edges via streaming Borůvka MST (avoids CSR->CSC conversion).
        Returns a boolean mask array aligned with the graph's upper triangle edges,
        plus the row/col arrays for reference.
        """
        self.timing.start("extract_mst_edges")
    
        # 1) Flatten CSR to edge lists
        rows, cols, weights = extract_upper_triangle_edges_from_csr(graph)
    
        # 2) Compute MST mask
        selected = boruvka_mst_parallel(n, rows, cols, weights)
    
        self.timing.end("extract_mst_edges")
        return selected, rows, cols
        
    def create_knn_graph_with_mst(self, 
                                  n_neighbors=30, 
                                  max_iterations=3, 
                                  batch_size=None, 
                                  report_mst_diameters=False,
                                  force_mst=False,
                                  use_approximate_nn=False
                                 ):
        """
        Create KNN graph with MST and iteratively refine it using custom distances
        
        Parameters:
        -----------
        """
        # Initialize scale_factor early
        scale_factor = 1.0
        
        self.timing.start("create_knn_graph_with_mst")
        node_df = self.node_df
        embedding_function = self.embedding_function
        
        n = len(node_df)
        
        if self.verbose:
            print(f"[Step 1] Creating KNN+MST graph: n={n}, k={n_neighbors}")
            if self.use_euclidean_as_graph_distance:
                print(f"         Using Euclidean distances as graph distances (optimization mode)")
        
        # Step 1: Create embeddings using the embedding function
        self.timing.start("create_knn_graph_with_mst.compute_embeddings")
        
        # More efficient: use swifter.apply instead of iterrows
        import swifter
        swifter.set_defaults(progress_bar=False, verbose=True)
        embeddings = np.array(node_df.swifter.apply(embedding_function, axis=1).tolist())
        
        if self.use_float32:
            embeddings = embeddings.astype(np.float32)
            
        if self.verbose:
            print(f"         Computed embeddings with shape: {embeddings.shape}")
        
        self.timing.end("create_knn_graph_with_mst.compute_embeddings")
        
        # Step 2: Find nearest neighbors in embedding space
        self.timing.start("create_knn_graph_with_mst.find_nearest_neighbors")
        # Step 2: Directly build the CSR KNN graph
        print("Building CSR KNN graph with kneighbors_graph...")
        if use_approximate_nn:
            print("Using approximate KNN (nmslib)")
            import nmslib
        
            # 1) Initialize the index
            n, d = embeddings.shape
            idx = nmslib.init(method='hnsw', space='l2')
            idx.addDataPointBatch(embeddings)
            idx.createIndex({'post': 2}, print_progress=False)
        
            # 2) Query k neighbors for each point in batch
            nbrs = idx.knnQueryBatch(
                embeddings, 
                k=n_neighbors, 
                num_threads=self.n_jobs
            )
            # nbrs is a list of (labels, distances) tuples
            labels, distances = zip(*nbrs)
            labels    = np.vstack(labels).astype(np.int32)
            distances = np.vstack(distances).astype(embeddings.dtype)
        
            # 3) Build symmetric CSR just like before
            rows = np.repeat(np.arange(n, dtype=np.int32), n_neighbors)
            cols = labels.ravel()
            data = distances.ravel()
            G0 = csr_matrix((data, (rows, cols)), shape=(n, n))
            G_knn = G0.maximum(G0.T)
            G_knn.sort_indices()
        else:
            G0 = kneighbors_graph(
                embeddings,
                n_neighbors,
                mode='distance',
                include_self=False,
                n_jobs=self.n_jobs
            )
            G_knn = G0.maximum(G0.T)
            G_knn.sort_indices()
        
        # Check if KNN graph is connected
        self.timing.start("create_knn_graph_with_mst.check_connectivity")
        is_knn_connected = check_graph_connectivity(G_knn)
        if not is_knn_connected:
            n_components, component_labels_knn = connected_components(G_knn, directed=False)
        self.timing.end("create_knn_graph_with_mst.check_connectivity")
        
        if self.verbose:
            if is_knn_connected:
                print(f"         KNN graph is connected")
            else:
                print(f"         KNN graph has {n_components} connected components")

        if not is_knn_connected:
            from collections import Counter
            counts = Counter(component_labels_knn)
            total = sum(counts.values())
            
            print("Total nodes:", total)
            print("Largest 5 components (label → size):")
            for lbl, sz in counts.most_common(5):
                print(f"  {lbl:5d} → {sz:8d}")
            
            # Maybe also see how many "tiny" islands:
            tiny = sum(1 for sz in counts.values() if sz < 100)
            print(f"{tiny} components of size < 100")
            
        # If KNN graph has multiple components, we need to add edges to connect them
        if not is_knn_connected:
            self.timing.start("create_knn_graph_with_mst.connect_components")
            if self.verbose:
                print(f"         Adding edges to connect all components")
            
            # Find representatives for each component
            print('Selecting least connected representatives of each connected component representatives...')
            component_reps = {}
            for comp in np.unique(component_labels_knn):
                nodes = np.where(component_labels_knn == comp)[0]
                degrees = np.diff(G_knn.indptr)[nodes]
                boundary = nodes[np.argmin(degrees)]
                component_reps[comp] = boundary
            
            # Create a minimally connected graph between component representatives
            print('Computing MST of connected component representatives (in embedding)...')
            # 2. Compute all-pairs Euclidean distances between representatives
            reps = np.array(list(component_reps.values()))
            D = cdist(embeddings[reps], embeddings[reps])
            
            # 3. MST over meta-graph
            mst = minimum_spanning_tree(D).toarray()
            edges_to_add = np.argwhere(mst > 0)
            
            # 4. Efficiently batch-update graph (switch to LIL)
            print('Converting disconnected KNN graph to LIL format...')
            G_knn = G_knn.tolil()
            for i, j in edges_to_add:
                u = reps[i]
                v = reps[j]
                dist = D[i, j]
                G_knn[u, v] = dist
                G_knn[v, u] = dist  # For undirected graphs
            print('Converting LIL graph back to CSR format...')
            G_knn = G_knn.tocsr()
            G_knn.sort_indices()
                        
            # Verify the graph is now connected
            is_connected_after = check_graph_connectivity(G_knn)
            if self.verbose:
                if is_connected_after:
                    print(f"         After adding edges: graph is connected")
                else:
                    print(f"         WARNING: Graph still has disconnected components after attempted repair")
            self.timing.end("create_knn_graph_with_mst.connect_components")
        
        # Initialize edge tracking
        print("Tracking edges for graph distance computation...")
        # CSR‐only edge initialization (no COO conversion)
        self.timing.start("create_knn_graph_with_mst.initialize_edge_tracking")
        print("Extracting upper‑triangle edges directly from CSR buffers...")
        
        # Use the optimized extraction function
        edge_arr, euclidean_distances_list = extract_all_edges_from_csr(G_knn, upper_only=True)
        n_edges = len(edge_arr)
        print(f"Tracking {n_edges} unique edges from CSR")
        
        # Cache the edge array
        graph_id = id(G_knn)
        self._edge_arr_cache[graph_id] = edge_arr
        
        # Pre‐allocate all trackers at once
        dtype = np.float32 if self.use_float32 else np.float64
        graph_distances          = np.zeros(n_edges, dtype=dtype)
        graph_distance_computed  = np.zeros(n_edges, dtype=bool)
        is_mst_edge              = np.zeros(n_edges, dtype=bool)
        is_valid_edge            = np.ones(n_edges, dtype=bool)
        
        self.timing.end("create_knn_graph_with_mst.initialize_edge_tracking")
        
        # Initialize statistics accumulators
        sum_graph_x = 0.0
        sum_graph_x2 = 0.0
        count_graph = 0
        sum_euclidean_x = 0.0
        sum_euclidean_x2 = 0.0
        count_euclidean = 0
        
        # NEW: If using Euclidean optimization, skip graph distance computation
        if self.use_euclidean_as_graph_distance:
            if self.verbose:
                print(f"         Skipping graph distance computation (using Euclidean distances)")
            
            # Copy Euclidean distances as graph distances
            graph_distances[:] = euclidean_distances_list
            graph_distance_computed[:] = True
            
            # Update statistics
            valid_mask = euclidean_distances_list > 0
            if np.sum(valid_mask) > 0:
                valid_dists = euclidean_distances_list[valid_mask]
                sum_graph_x = np.sum(valid_dists)
                sum_graph_x2 = np.sum(valid_dists ** 2)
                count_graph = len(valid_dists)
                sum_euclidean_x = sum_graph_x
                sum_euclidean_x2 = sum_graph_x2
                count_euclidean = count_graph
            
            scale_factor = 1.0  # No scaling needed
            
        else:
            # Get the batcher lazily
            batcher = self._get_batcher()
            
            # Original iterative computation logic
            for iteration in range(max_iterations):
                self.timing.start(f"create_knn_graph_with_mst.iteration_{iteration+1}")
                if self.verbose:
                    print(f"         Iteration {iteration+1}/{max_iterations} of distance refinement")
                
                # Find edges that don't have graph distances computed yet
                uncomputed_indices = np.where(~graph_distance_computed)[0]
                
                if len(uncomputed_indices) == 0:
                    if self.verbose:
                        print(f"         All edges have graph distances computed. Stopping iteration.")
                    
                    self.timing.end(f"create_knn_graph_with_mst.iteration_{iteration+1}")
                    break
                
                # Get edges that need graph distances using array indexing
                new_edge_indices = uncomputed_indices
                new_edges = edge_arr[uncomputed_indices]  # shape (n_uncomputed, 2)
                new_i = new_edges[:, 0]
                new_j = new_edges[:, 1]
                
                if self.verbose:
                    print(f"         Computing {len(new_i)} new graph distances")
                
                # Compute optimal batch size if needed
                if batch_size is None:
                    embedding_dim = embeddings.shape[1]
                    batch_size = self.estimate_optimal_batch_size(len(new_i), embedding_dim)
                
                if self.verbose and len(new_i) > batch_size:
                    print(f"         Processing in {len(new_i) // batch_size + 1} batches of size {batch_size}")
                    
                self.timing.start(f"create_knn_graph_with_mst.iteration_{iteration+1}.compute_graph_distances")
                
                removed_count = 0  # Track removed edges
                
                for batch_idx in range(0, len(new_i), batch_size):
                    if self.verbose and len(new_i) > batch_size and batch_idx % (10 * batch_size) == 0:
                        print(f"         Processing batch {batch_idx//batch_size + 1}/{(len(new_i)-1)//batch_size + 1}")
                    
                    end_idx = min(batch_idx + batch_size, len(new_i))
                    batch_i = new_i[batch_idx:end_idx]
                    batch_j = new_j[batch_idx:end_idx]
                    batch_indices = new_edge_indices[batch_idx:end_idx]
                    
                    # Compute distances for this batch
                    batch_distances = batcher(
                        self.node_features, batch_i, batch_j
                    )
                    
                    # Vectorized missing value handling
                    graph_distances[batch_indices] = np.where(
                        batch_distances == self.missing_weight,
                        np.inf,
                        batch_distances
                    )
                    is_valid_edge[batch_indices] = batch_distances != self.missing_weight
                    removed_count += np.sum(batch_distances == self.missing_weight)

                    # Get the corresponding Euclidean distances for this batch
                    batch_euclidean = euclidean_distances_list[batch_indices]
                    
                    # Update computed flags
                    graph_distance_computed[batch_indices] = True
                    
                    # Find valid pairs for statistics (positive values and not missing)
                    valid_mask = (batch_distances > 0) & (batch_euclidean > 0) & (batch_distances != self.missing_weight)
                    if np.sum(valid_mask) > 0:
                        valid_graph = batch_distances[valid_mask]
                        valid_euclidean = batch_euclidean[valid_mask]
                        
                        # Update running sums for statistics
                        sum_graph_x += np.sum(valid_graph)
                        sum_graph_x2 += np.sum(valid_graph ** 2)
                        count_graph += len(valid_graph)
                        
                        sum_euclidean_x += np.sum(valid_euclidean)
                        sum_euclidean_x2 += np.sum(valid_euclidean ** 2)
                        count_euclidean += len(valid_euclidean)
                
                if removed_count > 0 and self.verbose:
                    print(f"         Removed {removed_count} edges with missing values")
                
                self.timing.end(f"create_knn_graph_with_mst.iteration_{iteration+1}.compute_graph_distances")
        
                # Calculate scaling factor between graph and Euclidean distances using running statistics
                if count_graph > 0:
                    graph_distance_mean = sum_graph_x / count_graph
                    euclidean_mean = sum_euclidean_x / count_euclidean
                    scale_factor = euclidean_mean / graph_distance_mean if graph_distance_mean > 0 else 1.0
                    
                    # Calculate standard deviations
                    graph_variance = (sum_graph_x2 / count_graph) - graph_distance_mean**2
                    graph_std = np.sqrt(max(0, graph_variance))
                    
                    if self.verbose:
                        print(f"         Scale factor: {scale_factor:.4g} "
                              f"(graph mean: {graph_distance_mean:.4g}, Euclidean mean: {euclidean_mean:.4g})")
                        print(f"         graph stats: min={np.min(graph_distances[graph_distance_computed]):.4g}, "
                              f"max={np.max(graph_distances[graph_distance_computed]):.4g}, "
                              f"std={graph_std:.4g}, count={count_graph}")
                    
                    # Check if we should suggest Euclidean optimization
                    if iteration == 0:  # Only check on first iteration
                        self._suggest_euclidean_optimization(graph_distances[graph_distance_computed], 
                                                           euclidean_distances_list[graph_distance_computed])
                else:
                    scale_factor = 1.0
                    if self.verbose:
                        print(f"         No completed graph distances yet. Using scale factor of 1.0")
                
                # Create hybrid distances for MST computation
                self.timing.start(f"create_knn_graph_with_mst.iteration_{iteration+1}.build_hybrid_graph")
                hybrid_distances = np.where(
                    ~is_valid_edge,
                    np.inf,  # Invalid edges get infinite distance
                    np.where(
                        graph_distance_computed,
                        graph_distances * scale_factor,  # Use scaled graph distances when available
                        euclidean_distances_list      # Fall back to Euclidean
                    )
                )
                
                # Create hybrid graph using memory-efficient function
                hybrid_graph = self.build_graph_from_edges_memory_efficient(
                    edge_arr,  # Now passing array instead of list
                    hybrid_distances, 
                    is_valid_edge, 
                    n
                )
                self.timing.end(f"create_knn_graph_with_mst.iteration_{iteration+1}.build_hybrid_graph")
                
                                    
                # Extract MST from hybrid graph
                if force_mst:
                    self.timing.start(f"create_knn_graph_with_mst.iteration_{iteration+1}.extract_mst")
                    mst_mask, mst_rows, mst_cols = self.extract_mst_edges_boruvka(hybrid_graph, n)
                    self.timing.end(f"create_knn_graph_with_mst.iteration_{iteration+1}.extract_mst")
                    
                    # Update MST edge flags efficiently using the returned mask
                    self.timing.start(f"create_knn_graph_with_mst.iteration_{iteration+1}.identify_mst_edges")
                    
                    # Create a mapping from upper triangle edges to the full edge array
                    # First normalize all edges in edge_arr to ensure i < j
                    edge_i = np.minimum(edge_arr[:, 0], edge_arr[:, 1])
                    edge_j = np.maximum(edge_arr[:, 0], edge_arr[:, 1])
                    
                    # Build edge to index mapping using a dictionary
                    edge_to_idx = {(i, j): idx for idx, (i, j) in enumerate(zip(edge_i, edge_j))}
                    
                    # Map MST edges to the full edge array
                    mst_edge_indices = np.where(mst_mask)[0]
                    for idx in mst_edge_indices:
                        edge = (mst_rows[idx], mst_cols[idx])
                        if edge in edge_to_idx:
                            is_mst_edge[edge_to_idx[edge]] = True
                    
                    self.timing.end(f"create_knn_graph_with_mst.iteration_{iteration+1}.identify_mst_edges")
                    
                    # Report MST diameters if requested
                    if report_mst_diameters:
                        # Calculate MST diameter using unscaled graph distances
                        # Create a graph containing only the MST edges with unscaled graph distances
                        mst_graph = np.zeros((n, n), dtype=np.float32 if self.use_float32 else np.float64)
                        mst_edge_count = 0
                        mst_edge_indices = np.where(is_mst_edge)[0]
                        for idx in mst_edge_indices:
                            if graph_distance_computed[idx] and is_valid_edge[idx]:
                                i, j = edge_arr[idx]
                                mst_graph[i, j] = graph_distances[idx]  # Use unscaled graph distances
                                mst_graph[j, i] = graph_distances[idx]
                                mst_edge_count += 1
                        
                        if mst_edge_count > 0:
                            # Use Floyd-Warshall algorithm to compute all-pairs shortest paths
                            from scipy.sparse.csgraph import floyd_warshall
                            dist_matrix = floyd_warshall(csr_matrix(mst_graph))
                            
                            # Find the diameter (maximum finite distance)
                            finite_dists = dist_matrix[np.isfinite(dist_matrix)]
                            if len(finite_dists) > 0:
                                diameter = np.max(finite_dists)
                                print(f"         MST diameter (unscaled graph distances): {diameter:.4g}, with {mst_edge_count} MST edges having graph distances")
                            else:
                                print(f"         MST is disconnected or has no finite paths with graph distances")
                        else:
                            print(f"         No MST edges have valid graph distances computed")
                else:
                    # No MST computation - all edges marked as non-MST
                    is_mst_edge[:] = False
                
                self.timing.end(f"create_knn_graph_with_mst.iteration_{iteration+1}")
        
        # Create the final graph with graph distances (excluding invalid edges)
        self.timing.start("create_knn_graph_with_mst.build_final_graph")
        final_graph = self.build_graph_from_edges_memory_efficient(
            edge_arr,
            graph_distances,
            is_valid_edge,
            n
        )
        self.timing.end("create_knn_graph_with_mst.build_final_graph")
        
        # Final verification that the graph is connected
        self.timing.start("create_knn_graph_with_mst.final_verification")
        is_final_connected = check_graph_connectivity(final_graph)
        self.timing.end("create_knn_graph_with_mst.final_verification")
        
        if self.verbose:
            valid_edge_count = np.sum(is_valid_edge)
            mst_edge_count = np.sum(is_mst_edge)
            print(f"         Combined graph has {valid_edge_count} valid edges (of {len(edge_arr)} total), "
                  f"including {mst_edge_count} MST edges")
            if is_final_connected:
                print(f"         Final graph is connected")
            else:
                n_components_final, _ = connected_components(final_graph, directed=False)
                print(f"         WARNING: Final graph has {n_components_final} connected components!")
        
        self.timing.end("create_knn_graph_with_mst")
        
        # Filter edge data to only include valid edges
        valid_indices = np.where(is_valid_edge)[0]
        filtered_edge_arr = edge_arr[valid_indices]
        filtered_graph_distances = graph_distances[valid_indices]
        filtered_euclidean_distances = euclidean_distances_list[valid_indices]
        filtered_is_mst_edge = is_mst_edge[valid_indices]
        
        # Compute final statistics if we have data
        graph_distance_stats = None
        euclidean_stats = None
        
        if count_graph > 0:
            graph_mean = sum_graph_x / count_graph
            graph_variance = (sum_graph_x2 / count_graph) - graph_mean**2
            graph_std = np.sqrt(max(0, graph_variance))
            
            valid_graph_dists = graph_distances[graph_distance_computed & is_valid_edge]
            graph_distance_stats = {
                'mean': graph_mean,
                'std': graph_std,
                'min': np.min(valid_graph_dists) if len(valid_graph_dists) > 0 else 0,
                'max': np.max(valid_graph_dists) if len(valid_graph_dists) > 0 else 0,
                'count': count_graph
            }
            
        if count_euclidean > 0:
            euclidean_mean = sum_euclidean_x / count_euclidean
            euclidean_variance = (sum_euclidean_x2 / count_euclidean) - euclidean_mean**2
            euclidean_std = np.sqrt(max(0, euclidean_variance))
            
            euclidean_stats = {
                'mean': euclidean_mean,
                'std': euclidean_std,
                'min': np.min(filtered_euclidean_distances) if len(filtered_euclidean_distances) > 0 else 0,
                'max': np.max(filtered_euclidean_distances) if len(filtered_euclidean_distances) > 0 else 0,
                'count': count_euclidean
            }
        
        return final_graph, {
            'edge_arr': filtered_edge_arr,  # Now returning array instead of list
            'graph_distances': filtered_graph_distances,
            'euclidean_distances': filtered_euclidean_distances,
            'is_mst_edge': filtered_is_mst_edge,
            'graph_distance_scalar': scale_factor,
            # Add statistics to the returned data
            'graph_distance_stats': graph_distance_stats,
            'euclidean_stats': euclidean_stats,
            'n_removed_edges': len(edge_arr) - len(filtered_edge_arr),  # Track removed edges
            'use_euclidean_as_graph_distance': self.use_euclidean_as_graph_distance,
            'mst_computed': force_mst
        }
    
    def prune_graph_by_threshold(self, graph, edge_data, threshold=None, kneedle_sensitivity=1.0, 
                               max_components=None, use_median_filter=True, preserve_mst=True):
        """
        Prune a graph by removing edges with distances above a threshold.
        Now uses in-place edge removal (B.6 optimization) and direct MST mask.
        
        NOTE: Edges that exceed the threshold are already removed during computation,
        so this method primarily handles automatic threshold detection and additional pruning.
        
        Parameters:
        -----------
        graph : scipy.sparse.csr_matrix
            Graph with edge weights
        edge_data : dict
            Dictionary with edge arrays and distances
        threshold : float, optional
            Distance threshold for pruning edges. If None, determined automatically.
        kneedle_sensitivity : float, default=1.0
            Sensitivity for kneedle algorithm if using automatic threshold
        max_components : int, optional
            Maximum number of connected components allowed after pruning.
            If pruning would exceed this, adjust the threshold to maintain connectivity.
        use_median_filter : bool, default=True
            If True, only consider edges above the median distance for knee point detection
        preserve_mst : bool, default=True
            If True, preserve all edges in the minimum spanning tree to maintain connectivity
            
        Returns:
        --------
        scipy.sparse.csr_matrix
            Pruned graph
        dict
            Updated edge data
        dict
            Additional results including pruned edges and threshold
        """
        self.timing.start("prune_graph_by_threshold")
        
        n = graph.shape[0]
        edge_arr = edge_data['edge_arr']  # Now using array
        graph_distances = edge_data['graph_distances']
        
        # OPTIMIZATION: Early return if no pruning needed
        # Case 1: Threshold is infinity
        if threshold == np.inf:
            if self.verbose:
                print(f"[Pruning] No pruning needed: threshold is infinity")
            
            self.timing.end("prune_graph_by_threshold")
            return graph, edge_data, {
                'pruned_edges': np.array([], dtype=np.int32).reshape(0, 2),
                'pruned_distances': np.array([]),
                'threshold': threshold
            }
        
        # Case 2: Threshold exceeds max distance
        max_dist = None
        if 'graph_distance_stats' in edge_data and edge_data['graph_distance_stats'] is not None:
            # Use precomputed statistics if available
            max_dist = edge_data['graph_distance_stats']['max']
        elif graph_distances.size > 0:
            # Otherwise compute it directly
            max_dist = graph_distances.max()
            
        if threshold is not None and max_dist is not None and threshold >= max_dist:
            if self.verbose:
                print(f"[Pruning] No pruning needed: threshold {threshold:.4g} >= max distance {max_dist:.4g}")
            
            self.timing.end("prune_graph_by_threshold")
            return graph, edge_data, {
                'pruned_edges': np.array([], dtype=np.int32).reshape(0, 2),
                'pruned_distances': np.array([]),
                'threshold': threshold
            }
        
        if self.verbose:
            print(f"[Pruning] Pruning graph edges by threshold")
        
        # If we're preserving the MST, get the MST edge mask
        self.timing.start("prune_graph_by_threshold.prepare_mst")
        mst_edge_mask = None
        if preserve_mst:
            if threshold is not None:
                # 1) Build mask of edges to keep
                keep_mask = graph_distances <= threshold
        
                # 2) Build the pruned graph once
                pruned = self.build_graph_from_edges_memory_efficient(
                    edge_arr, graph_distances, keep_mask, graph.shape[0]
                )
        
                # 3) Check connectivity
                if check_graph_connectivity(pruned):
                    if self.verbose:
                        print("         Pruned graph stays connected—no MST edges lost")
                    # Use existing MST edge information if available
                    if edge_data.get('mst_computed') and 'is_mst_edge' in edge_data:
                        mst_edge_mask = edge_data['is_mst_edge']
                else:
                    if self.verbose:
                        print("         Pruned graph is disconnected—recomputing MST")
                    # Compute MST and get the boolean mask directly
                    mst_mask, mst_rows, mst_cols = self.extract_mst_edges_boruvka(graph, n)
                    
                    # Map MST mask to edge_arr indices
                    edge_i = np.minimum(edge_arr[:, 0], edge_arr[:, 1])
                    edge_j = np.maximum(edge_arr[:, 0], edge_arr[:, 1])
                    edge_to_idx = {(i, j): idx for idx, (i, j) in enumerate(zip(edge_i, edge_j))}
                    
                    mst_edge_mask = np.zeros(len(edge_arr), dtype=bool)
                    mst_edge_indices = np.where(mst_mask)[0]
                    for idx in mst_edge_indices:
                        edge = (mst_rows[idx], mst_cols[idx])
                        if edge in edge_to_idx:
                            mst_edge_mask[edge_to_idx[edge]] = True
        
            else:
                # No threshold ⇒ automatic pruning; always ensure MST
                if edge_data.get('mst_computed') and 'is_mst_edge' in edge_data:
                    mst_edge_mask = edge_data['is_mst_edge']
                    if self.verbose:
                        print(f"         Using precomputed MST ({np.sum(mst_edge_mask)} edges)")
                else:
                    if self.verbose:
                        print("         Computing MST edges for automatic threshold pruning")
                    # Compute MST and get the boolean mask directly
                    mst_mask, mst_rows, mst_cols = self.extract_mst_edges_boruvka(graph, n)
                    
                    # Map MST mask to edge_arr indices
                    edge_i = np.minimum(edge_arr[:, 0], edge_arr[:, 1])
                    edge_j = np.maximum(edge_arr[:, 0], edge_arr[:, 1])
                    edge_to_idx = {(i, j): idx for idx, (i, j) in enumerate(zip(edge_i, edge_j))}
                    
                    mst_edge_mask = np.zeros(len(edge_arr), dtype=bool)
                    mst_edge_indices = np.where(mst_mask)[0]
                    for idx in mst_edge_indices:
                        edge = (mst_rows[idx], mst_cols[idx])
                        if edge in edge_to_idx:
                            mst_edge_mask[edge_to_idx[edge]] = True
        
            if self.verbose and mst_edge_mask is not None:
                print(f"         Preserving {np.sum(mst_edge_mask)} MST edges to maintain connectivity")
        else:
            if self.verbose:
                print("         MST preservation disabled—pruning without connectivity guarantee")
                
        self.timing.end("prune_graph_by_threshold.prepare_mst")
        
        # Check if we have cached sorted edges
        graph_id = id(graph)
        cache_key = (graph_id, 'sorted')
        
        if cache_key in self._sorted_edge_cache:
            sorted_indices = self._sorted_edge_cache[cache_key]['indices']
            sorted_edges = self._sorted_edge_cache[cache_key]['edges']
            sorted_distances = self._sorted_edge_cache[cache_key]['distances']
        else:
            # Sort edges by distance (in descending order)
            self.timing.start("prune_graph_by_threshold.sort_edges")
            sorted_indices = np.argsort(graph_distances)[::-1]
            sorted_edges = edge_arr[sorted_indices]  # Direct array indexing
            sorted_distances = graph_distances[sorted_indices]
            
            # Cache the sorted data
            self._sorted_edge_cache[cache_key] = {
                'indices': sorted_indices,
                'edges': sorted_edges,
                'distances': sorted_distances
            }
            self.timing.end("prune_graph_by_threshold.sort_edges")
        
        if self.verbose:
            # Use precomputed statistics if available
            if 'graph_distance_stats' in edge_data and edge_data['graph_distance_stats'] is not None:
                stats = edge_data['graph_distance_stats']
                print(f"         Edge distance stats: min={stats['min']:.4g}, "
                      f"mean={stats['mean']:.4g}, max={stats['max']:.4g}")
            else:
                print(f"         Edge distance stats: min={graph_distances.min():.4g}, "
                      f"mean={graph_distances.mean():.4g}, max={graph_distances.max():.4g}")
                
            # Use standard median calculation
            median_graph = np.median(graph_distances)
            print(f"         Median distance: {median_graph:.4g}")
        
        # Filter out MST edges from consideration for pruning
        self.timing.start("prune_graph_by_threshold.filter_prunable")
        if preserve_mst and mst_edge_mask is not None:
            # Use the boolean mask directly
            sorted_mst_mask = mst_edge_mask[sorted_indices]
            prunable_mask = ~sorted_mst_mask
            prunable_edges = sorted_edges[prunable_mask]
            prunable_distances = sorted_distances[prunable_mask]
            
            if self.verbose:
                print(f"         {len(prunable_edges)} edges are candidates for pruning "
                      f"(non-MST edges)")
        else:
            prunable_edges = sorted_edges
            prunable_distances = sorted_distances
        self.timing.end("prune_graph_by_threshold.filter_prunable")
        
        # Determine which edges to prune
        self.timing.start("prune_graph_by_threshold.determine_edges")
        if threshold is not None:
            # Prune edges above the provided threshold
            edges_to_prune_mask = prunable_distances > threshold
            edges_to_prune = prunable_edges[edges_to_prune_mask]
            prune_distances = prunable_distances[edges_to_prune_mask]
            
            if self.verbose:
                print(f"         Pruning {len(edges_to_prune)} edges above threshold {threshold:.4g}")
        
        else:
            if len(prunable_edges) < 2:
                if self.verbose:
                    print(f"         Not enough prunable edges. Skipping pruning.")
                
                self.timing.end("prune_graph_by_threshold.determine_edges")
                self.timing.end("prune_graph_by_threshold")
                return graph, edge_data, {
                    'pruned_edges': np.array([], dtype=np.int32).reshape(0, 2),
                    'pruned_distances': np.array([]),
                    'threshold': None
                }
            
            # Use kneedle algorithm to find the optimal threshold
            knee_idx = find_knee_point(np.arange(len(prunable_distances)), prunable_distances, 
                                      S=kneedle_sensitivity, use_median_filter=use_median_filter)
            threshold = prunable_distances[knee_idx]
            edges_to_prune = prunable_edges[:knee_idx+1]
            prune_distances = prunable_distances[:knee_idx+1]
            
            if self.verbose:
                print(f"         Kneedle algorithm: found threshold at {threshold:.4g} "
                      f"(index {knee_idx}/{len(prunable_distances)})")
                print(f"         Pruning {len(edges_to_prune)} edges above threshold")
            
            # Optionally plot the knee point
            if self.plot_knee:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 6))
                plt.plot(np.arange(len(prunable_distances)), prunable_distances, 'b-', linewidth=2)
                plt.axvline(x=knee_idx, color='r', linestyle='--', 
                          label=f'Knee point: {threshold:.4g}')
                plt.axhline(y=np.median(prunable_distances), color='g', linestyle=':', 
                          label=f'Median: {np.median(prunable_distances):.4g}')
                plt.xlabel('Sorted Edge Index')
                plt.ylabel('Graph Distance')
                plt.title('Knee Point Detection for Graph Pruning')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig('graph_knee_point_detection.png', dpi=300)
                plt.close()
                print(f"         Knee point visualization saved to 'graph_knee_point_detection.png'")
        self.timing.end("prune_graph_by_threshold.determine_edges")
        
        # If max_components is specified, ensure we don't create too many components
        if max_components is not None and not preserve_mst:
            self.timing.start("prune_graph_by_threshold.check_components")
            # Use in-place edge removal (B.6 optimization)
            test_graph = graph.copy()
            
            # Try removing edges one by one, from highest to lowest distance
            current_components = 1  # Assume we start with a connected graph
            
            for i, edge in enumerate(edges_to_prune):
                # Remove the edge in-place
                test_graph[edge[0], edge[1]] = 0
                test_graph[edge[1], edge[0]] = 0
                
                # Check if we've reached the component limit (do this every few edges for efficiency)
                if i % 100 == 0 or i == len(edges_to_prune) - 1:
                    test_graph.eliminate_zeros()  # Clean up zeros
                    n_components, _ = connected_components(test_graph, directed=False)
                    
                    if n_components > max_components:
                        # We've created too many components, revert to the previous threshold
                        if i > 0:
                            threshold = prunable_distances[i-1]
                            edges_to_prune = edges_to_prune[:i]
                            prune_distances = prune_distances[:i]
                        else:
                            # Don't prune any edges if even the first one creates too many components
                            threshold = prunable_distances[0] + 1e-6
                            edges_to_prune = np.array([], dtype=edges_to_prune.dtype).reshape(0, 2)
                            prune_distances = np.array([])
                        
                        if self.verbose:
                            print(f"         Limiting pruning to maintain max {max_components} components")
                            print(f"         Adjusted threshold: {threshold:.4g}, pruning {len(edges_to_prune)} edges")
                        
                        break
                    
                    current_components = n_components
            self.timing.end("prune_graph_by_threshold.check_components")
        
        # Apply pruning using optimized in-place modification
        self.timing.start("prune_graph_by_threshold.apply_pruning")
        
        # Create a mask for edges to keep
        edges_to_remove_set = set(map(tuple, edges_to_prune))
        keep_mask = np.array([tuple(edge) not in edges_to_remove_set for edge in edge_arr])
        
        # Build pruned graph directly with the keep mask
        pruned_graph = self.build_graph_from_edges_memory_efficient(
            edge_arr, graph_distances, keep_mask, n
        )
        
        self.timing.end("prune_graph_by_threshold.apply_pruning")
        
        # Create new edge data for the pruned graph more efficiently
        self.timing.start("prune_graph_by_threshold.create_edge_data")
        
        # Extract edges directly from CSR without converting to COO
        pruned_edge_arr, pruned_weights = extract_all_edges_from_csr(pruned_graph, upper_only=True)
        
        # Preserve MST edge information if available
        if 'is_mst_edge' in edge_data:
            # Create a mapping of edges to MST status from original data
            original_mst_map = {}
            for idx, edge in enumerate(edge_arr):
                original_mst_map[tuple(edge)] = edge_data['is_mst_edge'][idx]
            
            # Map to new edges
            pruned_is_mst_edge = np.array([
                original_mst_map.get(tuple(edge), False)
                for edge in pruned_edge_arr
            ])
        else:
            pruned_is_mst_edge = None
        
        # Create updated statistics for the pruned graph
        pruned_graph_distance_stats = None
        if len(pruned_weights) > 0:
            pruned_graph_distance_stats = {
                'mean': np.mean(pruned_weights),
                'std': np.std(pruned_weights),
                'min': np.min(pruned_weights),
                'max': np.max(pruned_weights),
                'count': len(pruned_weights)
            }
        
        pruned_edge_data = {
            'edge_arr': pruned_edge_arr,  # Now storing as array
            'graph_distances': pruned_weights
        }
        
        # Add MST edge information if it was available
        if pruned_is_mst_edge is not None:
            pruned_edge_data['is_mst_edge'] = pruned_is_mst_edge
        
        # Add statistics if we calculated them
        if pruned_graph_distance_stats is not None:
            pruned_edge_data['graph_distance_stats'] = pruned_graph_distance_stats
        
        # Copy over other statistics if they were available
        if 'euclidean_stats' in edge_data:
            pruned_edge_data['euclidean_stats'] = edge_data['euclidean_stats']
            
        # Copy over optimization flag
        if 'use_euclidean_as_graph_distance' in edge_data:
            pruned_edge_data['use_euclidean_as_graph_distance'] = edge_data['use_euclidean_as_graph_distance']
            
        self.timing.end("prune_graph_by_threshold.create_edge_data")
        
        # Verify the pruned graph
        self.timing.start("prune_graph_by_threshold.verify")
        if self.verbose:
            is_pruned_connected = check_graph_connectivity(pruned_graph)
            print(f"         Pruned graph has {len(pruned_edge_arr)} edges")
            if is_pruned_connected:
                print(f"         Pruned graph is connected")
            else:
                n_components, _ = connected_components(pruned_graph, directed=False)
                print(f"         Pruned graph has {n_components} connected components")
            
            # Verify that all MST edges are still present
            if 'is_mst_edge' in pruned_edge_data and preserve_mst:
                mst_edge_count = np.sum(pruned_edge_data['is_mst_edge'])
                expected_mst_edges = n - 1
                if mst_edge_count != expected_mst_edges:
                    print(f"         WARNING: After pruning, found {mst_edge_count} MST edges, expected {expected_mst_edges}")
        self.timing.end("prune_graph_by_threshold.verify")
        self.timing.end("prune_graph_by_threshold")
        
        return pruned_graph, pruned_edge_data, {
            'pruned_edges': edges_to_prune,
            'pruned_distances': prune_distances,
            'threshold': threshold
        }

    def smooth_graph_with_2hop(self, graph, edge_data, node_df, premetric_weight_function, 
                              max_new_edges=None, batch_size=None):
        """
        Smooth the graph by adding connections between 2-hop neighbors.
        
        NOTE: New edges that have missing values (self.missing_weight) are not added.
        
        Parameters:
        -----------
        graph : scipy.sparse.csr_matrix
            Current graph
        edge_data : dict
            Dictionary with edge arrays and distances
        max_new_edges : int, optional
            Maximum number of new edges to add. If None, add all valid 2-hop connections.
        batch_size : int, default=automatic
            Batch size for computing graph distances
            
        Returns:
        --------
        scipy.sparse.csr_matrix
            Smoothed graph with new edges
        dict
            Updated edge data
        dict
            Additional results
        """
        self.timing.start("smooth_graph_with_2hop")
        
        if self.verbose:
            print(f"[Smoothing] Adding connections between 2-hop neighbors")
        
        n = graph.shape[0]
        edge_arr = edge_data['edge_arr']  # Now using array
        graph_distances = edge_data['graph_distances']
        
        # Check if using Euclidean optimization
        use_euclidean = edge_data.get('use_euclidean_as_graph_distance', False)
        
        # Create a set of existing edges for fast lookup
        existing_edges = set(map(tuple, edge_arr))
        
        # Find 2-hop neighbors using sparse matrix operations
        self.timing.start("smooth_graph_with_2hop.find_2hop")
        new_edge_arr = find_2hop_neighbors_sparse(graph)
        
        # Filter out existing edges
        new_edges_mask = np.array([
            tuple(edge) not in existing_edges 
            for edge in new_edge_arr
        ])
        new_edge_arr = new_edge_arr[new_edges_mask]
        self.timing.end("smooth_graph_with_2hop.find_2hop")
        
        if max_new_edges is not None and len(new_edge_arr) > max_new_edges:
            if self.verbose:
                print(f"         Found {len(new_edge_arr)} potential 2-hop connections, limiting to {max_new_edges}")
            # Randomly sample a subset of new edges
            random_indices = np.random.choice(len(new_edge_arr), max_new_edges, replace=False)
            new_edge_arr = new_edge_arr[random_indices]
        
        if self.verbose:
            print(f"         Adding up to {len(new_edge_arr)} new 2-hop connections")
        
        # If no new edges, return early
        if len(new_edge_arr) == 0:
            if self.verbose:
                print(f"         No new 2-hop connections to add")
            
            self.timing.end("smooth_graph_with_2hop")
            return graph, edge_data, {
                'n_new_edges': 0,
                'new_edges': np.array([], dtype=np.int32).reshape(0, 2),
                'new_distances': np.array([])
            }
        
        new_i = new_edge_arr[:, 0]
        new_j = new_edge_arr[:, 1]
        
        # Compute graph distances for new edges in batches
        valid_new_edges = []
        new_edge_distances = []
        
        # If using Euclidean optimization, compute embeddings once
        if use_euclidean:
            if self.verbose:
                print(f"         Using Euclidean distances for 2-hop edges (optimization mode)")
            embeddings = np.array(node_df.apply(self.embedding_function, axis=1).tolist())
            if self.use_float32:
                embeddings = embeddings.astype(np.float32)
        
        # Process in batches to reduce memory pressure
        self.timing.start("smooth_graph_with_2hop.compute_distances")
        n_items = len(new_i)
        if batch_size is None:
            # Estimate based on dataframe size
            batch_size = min(100000, n_items)
                    
        n_batches = (len(new_edge_arr) - 1) // batch_size + 1
        if self.verbose and len(new_edge_arr) > batch_size:
            print(f"         Processing in {n_batches} batches of size {batch_size}")
        
        removed_count = 0  # Track edges with missing values
        
        # Get the batcher lazily
        batcher = self._get_batcher() if not use_euclidean else None
        
        for batch_idx in range(0, len(new_edge_arr), batch_size):
            if self.verbose and len(new_edge_arr) > batch_size and batch_idx % (10 * batch_size) == 0:
                print(f"         Processing batch {batch_idx//batch_size + 1}/{n_batches}")
                
            end_idx = min(batch_idx + batch_size, len(new_edge_arr))
            batch_i = new_i[batch_idx:end_idx]
            batch_j = new_j[batch_idx:end_idx]
            batch_edges = new_edge_arr[batch_idx:end_idx]
            
            if use_euclidean:
                # Compute Euclidean distances directly (vectorized)
                batch_distances = np.linalg.norm(
                    embeddings[batch_i] - embeddings[batch_j], 
                    axis=1
                )
            else:
                # Compute distances for this batch using the custom function
                batch_distances = batcher(
                    self.node_features, batch_i, batch_j
                )
            
            # Filter out edges with missing values
            valid_mask = batch_distances != self.missing_weight
            valid_batch_edges = batch_edges[valid_mask]
            valid_batch_distances = batch_distances[valid_mask]
            
            removed_count += np.sum(~valid_mask)
            
            # Add valid edges to results
            if len(valid_batch_edges) > 0:
                valid_new_edges.extend(valid_batch_edges)
                new_edge_distances.extend(valid_batch_distances)
                    
        self.timing.end("smooth_graph_with_2hop.compute_distances")
        
        # Convert to arrays
        if len(valid_new_edges) > 0:
            valid_new_edges = np.array(valid_new_edges, dtype=np.int32)
            new_edge_distances = np.array(new_edge_distances)
        else:
            valid_new_edges = np.array([], dtype=np.int32).reshape(0, 2)
            new_edge_distances = np.array([])
        
        if removed_count > 0 and self.verbose:
            print(f"         Skipped {removed_count} 2-hop edges with missing values")
            print(f"         Adding {len(valid_new_edges)} valid new edges")
        
        # If no valid edges remain, return early
        if len(valid_new_edges) == 0:
            if self.verbose:
                print(f"         No valid 2-hop connections to add after filtering")
            
            self.timing.end("smooth_graph_with_2hop")
            return graph, edge_data, {
                'n_new_edges': 0,
                'new_edges': valid_new_edges,
                'new_distances': new_edge_distances,
                'n_removed': removed_count
            }
        
        # Add new edges to the graph more efficiently
        self.timing.start("smooth_graph_with_2hop.add_edges")

        # Build a new graph with all edges combined
        all_edges = np.vstack([edge_arr, valid_new_edges])
        all_distances = np.concatenate([graph_distances, new_edge_distances])
        
        smoothed_graph = self.build_graph_from_edges_memory_efficient(
            all_edges, all_distances, np.ones(len(all_edges), dtype=bool), n
        )
        
        self.timing.end("smooth_graph_with_2hop.add_edges")

        # Update edge data
        self.timing.start("smooth_graph_with_2hop.update_edge_data")
        has_mst_info = 'is_mst_edge' in edge_data
        
        # Get the MST edges for this graph if we have MST info
        if has_mst_info:
            # Use existing MST information from edge_data
            original_mst_mask = edge_data['is_mst_edge']
            original_mst_count = np.sum(original_mst_mask)
            
            # Verify we have the right number of MST edges
            expected_mst_edges = n - 1
            if original_mst_count != expected_mst_edges:
                if self.verbose:
                    print(f"         WARNING: Original MST has {original_mst_count} edges, expected {expected_mst_edges}")
                # If we don't have the right number, recompute
                mst_mask, mst_rows, mst_cols = self.extract_mst_edges_boruvka(graph, n)
                # Create a set for comparison later
                original_mst_edges = set(zip(mst_rows[mst_mask], mst_cols[mst_mask]))
            else:
                # Create a set from the existing MST edges
                original_mst_edges = set()
                for idx, is_mst in enumerate(original_mst_mask):
                    if is_mst:
                        original_mst_edges.add(tuple(edge_arr[idx]))
        
        # Extract edges from smoothed graph directly without COO conversion
        smoothed_edge_arr, smoothed_weights = extract_all_edges_from_csr(smoothed_graph, upper_only=True)
        
        # Determine MST status for smoothed edges if needed
        if has_mst_info:
            smoothed_is_mst_edge = np.array([
                tuple(edge) in original_mst_edges
                for edge in smoothed_edge_arr
            ])
        else:
            smoothed_is_mst_edge = None
        
        smoothed_edge_data = {
            'edge_arr': smoothed_edge_arr,  # Now storing as array
            'graph_distances': smoothed_weights
        }
        
        # Add MST edge information if it was available
        if smoothed_is_mst_edge is not None:
            smoothed_edge_data['is_mst_edge'] = smoothed_is_mst_edge
            
        # Copy over optimization flag
        if 'use_euclidean_as_graph_distance' in edge_data:
            smoothed_edge_data['use_euclidean_as_graph_distance'] = edge_data['use_euclidean_as_graph_distance']
            
        self.timing.end("smooth_graph_with_2hop.update_edge_data")
        
        # Verify the smoothed graph
        self.timing.start("smooth_graph_with_2hop.verify")
        if self.verbose:
            print(f"         Smoothed graph has {len(smoothed_edge_arr)} edges")
            is_smoothed_connected = check_graph_connectivity(smoothed_graph)
            if is_smoothed_connected:
                print(f"         Smoothed graph is connected")
            else:
                n_components, _ = connected_components(smoothed_graph, directed=False)
                print(f"         Smoothed graph has {n_components} connected components")
            
            # Verify MST edge count
            if has_mst_info and smoothed_is_mst_edge is not None:
                mst_edge_count = np.sum(smoothed_edge_data['is_mst_edge'])
                expected_mst_edges = n - 1
                if mst_edge_count != expected_mst_edges:
                    print(f"         WARNING: Smoothed graph has {mst_edge_count} MST edges, expected {expected_mst_edges}")
        self.timing.end("smooth_graph_with_2hop.verify")
        
        self.timing.end("smooth_graph_with_2hop")
        return smoothed_graph, smoothed_edge_data, {
            'n_new_edges': len(valid_new_edges),
            'new_edges': valid_new_edges,
            'new_distances': new_edge_distances,
            'n_removed': removed_count
        }
    
    def iterative_refine_graph(self,
                       graph,
                       edge_data,
                       node_df,
                       premetric_weight_function,
                       n_iterations=2,
                       threshold=None,
                       kneedle_sensitivity=1.0,
                       max_new_edges=None,
                       batch_size=None,
                       max_components=None,
                       use_median_filter=True,
                       preserve_mst=True,
                       smooth_first=False):
        """
        Iteratively refine the graph by pruning and (optionally) smoothing.
    
        If max_new_edges is 0, smoothing is skipped entirely.
        
        NOTE: The smoothing and pruning steps operate independently:
        - Smoothing adds 2-hop edges (filtered by missing values)
        - Pruning removes high-distance edges (preserving MST if specified)
        The order is controlled by smooth_first parameter.
    
        Returns:
        --------
        current_graph : scipy.sparse.csr_matrix
            Final graph after iterations
        current_edge_data : dict
            Final edge data
        history : list of dict
            Summary of each iteration's results
        """
        self.timing.start("iterative_refine_graph")
        
        if self.verbose:
            order = "smoothing then pruning" if smooth_first else "pruning then smoothing"
            print(f"[Iterative] Starting iterative {order} for {n_iterations} iterations")
    
        current_graph = graph.copy()
        current_edge_data = edge_data.copy()
        history = []
        
        # Track edge count for early stopping
        prev_edge_count = len(edge_data['edge_arr'])
    
        for it in range(1, n_iterations + 1):
            self.timing.start(f"iterative_refine_graph.iteration_{it}")
            if self.verbose:
                print(f"         Iteration {it}/{n_iterations}")
    
            # Decide whether to short-circuit smoothing
            do_smoothing = bool(max_new_edges)
    
            if smooth_first:
                # 1a) Smooth (if enabled)
                if do_smoothing:
                    sm_graph, sm_edge_data, sm_res = self.smooth_graph_with_2hop(
                        current_graph, current_edge_data, node_df, premetric_weight_function,
                        max_new_edges=max_new_edges, batch_size=batch_size
                    )
                else:
                    sm_graph, sm_edge_data, sm_res = (
                        current_graph, current_edge_data, {'n_new_edges': 0, 'n_removed': 0}
                    )
    
                # 1b) Prune
                pr_graph, pr_edge_data, pr_res = self.prune_graph_by_threshold(
                    sm_graph, sm_edge_data,
                    threshold=threshold,
                    kneedle_sensitivity=kneedle_sensitivity,
                    max_components=max_components,
                    use_median_filter=use_median_filter,
                    preserve_mst=preserve_mst
                )
    
                next_graph, next_edge_data = pr_graph, pr_edge_data
    
            else:
                # 2a) Prune first
                pr_graph, pr_edge_data, pr_res = self.prune_graph_by_threshold(
                    current_graph, current_edge_data,
                    threshold=threshold,
                    kneedle_sensitivity=kneedle_sensitivity,
                    max_components=max_components,
                    use_median_filter=use_median_filter,
                    preserve_mst=preserve_mst
                )
    
                # 2b) Smooth (if enabled)
                if do_smoothing:
                    sm_graph, sm_edge_data, sm_res = self.smooth_graph_with_2hop(
                        pr_graph, pr_edge_data, node_df, premetric_weight_function,
                        max_new_edges=max_new_edges, batch_size=batch_size
                    )
                else:
                    sm_graph, sm_edge_data, sm_res = (
                        pr_graph, pr_edge_data, {'n_new_edges': 0, 'n_removed': 0}
                    )
    
                next_graph, next_edge_data = sm_graph, sm_edge_data
    
            # Record iteration results
            current_edge_count = len(next_edge_data['edge_arr'])
            edge_change_percent = abs(current_edge_count - prev_edge_count) / max(prev_edge_count, 1)
            
            history.append({
                'iteration': it,
                'smooth_first': smooth_first,
                'n_new_edges': sm_res.get('n_new_edges', 0),
                'n_removed_smoothing': sm_res.get('n_removed', 0),  # Track removed during smoothing
                'prune_threshold': pr_res.get('threshold'),
                'n_pruned_edges': len(pr_res.get('pruned_edges', [])),
                'n_edges': current_edge_count,
                'edge_change_percent': edge_change_percent
            })
            
            # Check for early stopping (minimal change in edge count)
            if edge_change_percent < 0.01:  # Less than 1% change
                if self.verbose:
                    print(f"         Early stopping at iteration {it}: edge count stabilized")
                break
    
            # Prepare for next round
            prev_edge_count = current_edge_count
            current_graph = next_graph
            current_edge_data = next_edge_data
    
            if self.verbose:
                is_iter_connected = check_graph_connectivity(current_graph)
                if is_iter_connected:
                    print(f"         End of iteration: {len(current_edge_data['edge_arr'])} edges, connected graph")
                else:
                    comps, _ = connected_components(current_graph, directed=False)
                    print(f"         End of iteration: {len(current_edge_data['edge_arr'])} edges, {comps} components")
            
            self.timing.end(f"iterative_refine_graph.iteration_{it}")
    
        self.timing.end("iterative_refine_graph")
        return current_graph, current_edge_data, history

    def build_and_refine_graph(self, n_neighbors=30, mst_iterations=3,
                          prune_threshold=None, kneedle_sensitivity=1.0,
                          polish_iterations=2, max_new_edges=None, batch_size=None,
                          max_components=None, use_median_filter=True, 
                          smooth_before_prune=False, preserve_mst=True,
                          log_timing=True, missing_weight=np.inf, use_approximate_nn=False):
        """
        Full pipeline to build and refine the DataGraph object with detailed timing statistics
        
        Parameters:
        -----------
        node_df : pandas.DataFrame
            DataFrame containing node information
        premetric_weight_function : callable
            Function that computes distance between two DataFrame rows
        embedding_function : callable
            Function that takes a DataFrame row and returns embedding coordinates
        n_neighbors : int, default=30
            Number of neighbors for KNN graph
        use_approximate_nn : bool, default=False
            Whether to use approximate nearest neighbor search (nmslib)
        ...
        """
        t0 = time.time()
        # Reset timing stats for this run
        self.timing = TimingStats()
        self.timing.start("build_and_refine_graph")

        node_df = self.node_df
        premetric_weight_function = self.premetric_weight_function
        feature_cols=self.feature_cols
        embedding_function = self.embedding_function
        
        n = len(node_df)
        
        if self.verbose:
            print(f"[Build & Refine] START")
            print(f"Processing {n} nodes")
            print(f"Using custom embedding and distance functions")
            if self.use_euclidean_as_graph_distance:
                print(f"OPTIMIZATION: Using Euclidean distances as graph distances")
            print(f"Missing value indicator: {self.missing_weight}")
        
        # No data preparation needed - we're working with the dataframe directly
        
        # Step 1: KNN+MST
        t1 = time.time()
        self.timing.start("step1_knn_mst")
        mst_graph, edge_data = self.create_knn_graph_with_mst(
            n_neighbors=n_neighbors,
            max_iterations=mst_iterations,
            batch_size=batch_size,
            use_approximate_nn=use_approximate_nn,
            force_mst=preserve_mst  # Only compute MST if we're going to preserve it
        )
        self.timing.end("step1_knn_mst")
        step1_time = time.time()-t1
        
        if log_timing:
            print(f"  • Step 1 (KNN+MST) done in {step1_time:.2f}s")
            if 'n_removed_edges' in edge_data:
                print(f"    - Removed {edge_data['n_removed_edges']} edges with missing values")
            print(self.timing.report_nested_timing("create_knn_graph_with_mst"))
    
        # Step 2: iterative prune/smooth
        t2 = time.time()
        self.timing.start("step2_refinement")
        final_graph, final_edge_data, history = self.iterative_refine_graph(
            mst_graph, edge_data, node_df, premetric_weight_function,
            n_iterations=polish_iterations,
            threshold=prune_threshold,
            kneedle_sensitivity=kneedle_sensitivity,
            max_new_edges=max_new_edges,
            batch_size=batch_size,
            max_components=max_components,
            use_median_filter=use_median_filter,
            preserve_mst=preserve_mst,
            smooth_first=smooth_before_prune
        )
        self.timing.end("step2_refinement")
        step2_time = time.time()-t2
        
        if log_timing:
            print(f"  • Step 2 (refinement) done in {step2_time:.2f}s")
            print(self.timing.report_nested_timing("iterative_refine_graph"))
    
        # Step 3: analyze components
        t3 = time.time()
        self.timing.start("step3_analysis")
        n_components, component_labels = connected_components(final_graph, directed=False)
        component_sizes = np.bincount(component_labels)
        self.timing.end("step3_analysis")
        step3_time = time.time()-t3
        
        if self.verbose:
            print(f"[Final] {len(final_edge_data['edge_arr'])} edges, {n_components} components")
            print(f"Component size distribution: min={component_sizes.min()}, "
                  f"mean={component_sizes.mean():.1f}, max={component_sizes.max()}")
        
        if log_timing:
            print(f"  • Step 3 (analysis) done in {step3_time:.2f}s")
    
        # Create DataGraph object
        graph_obj = DataGraph(
            final_graph, 
            node_df=node_df,
            feature_cols=feature_cols,
            premetric_weight_function=premetric_weight_function,
            embedding_function=embedding_function,
            edge_data=final_edge_data, 
            component_labels=component_labels,
            missing_weight=missing_weight
        )
        
        # End overall timing
        self.timing.end("build_and_refine_graph")
        total_time = time.time()-t0
        
        # Generate timing reports
        if self.verbose or log_timing:
            print(f"[Build & Refine] TOTAL time: {total_time:.2f}s\n")
            
            # Print overall timing summary
            print("\nTiming Summary:")
            print(f"  • Step 1 (KNN+MST): {step1_time:.2f}s ({step1_time/total_time*100:.1f}%)")
            print(f"  • Step 2 (refinement): {step2_time:.2f}s ({step2_time/total_time*100:.1f}%)")
            print(f"  • Step 3 (analysis): {step3_time:.2f}s ({step3_time/total_time*100:.1f}%)")
            print(f"  • Total: {total_time:.2f}s")
            
            # Generate detailed breakdown of time-consuming operations
            print("\nMost Time-Consuming Operations:")
            stats = self.timing.get_stats(as_dict=True)
            sorted_ops = sorted(stats.items(), key=lambda x: x[1]['total'], reverse=True)
            
            # Show top-level operations first
            for i, (op, op_stats) in enumerate(sorted_ops[:10]):  # Top 10 operations
                if '.' not in op:  # Only show top-level operations
                    pct = op_stats['total'] / total_time * 100
                    print(f"  {i+1}. {op}: {op_stats['total']:.2f}s ({pct:.1f}%)")
            
            # Detailed breakdown of computational hotspots
            print("\nComputational Hotspots:")
            
            # Track graph distance calculations specifically
            graph_distance_compute_time = 0
            graph_distance_compute_count = 0
            for op, stats in sorted_ops:
                if "compute_graph_distances" in op:
                    graph_distance_compute_time += stats['total']
                    graph_distance_compute_count += stats['count']
            
            if graph_distance_compute_time > 0:
                pct = graph_distance_compute_time / total_time * 100
                print(f"  • graph distance calculations: {graph_distance_compute_time:.2f}s ({pct:.1f}% of total)")
                print(f"    - {graph_distance_compute_count} batches processed")
                if batch_size:
                    print(f"    - Using batch size: {batch_size}")
                if self.use_euclidean_as_graph_distance:
                    print(f"    - NOTE: Skipped due to Euclidean optimization")
                
            # MST extraction cost
            mst_time = sum(stats['total'] for op, stats in sorted_ops if "extract_mst" in op)
            if mst_time > 0:
                pct = mst_time / total_time * 100
                print(f"  • MST extraction operations: {mst_time:.2f}s ({pct:.1f}% of total)")
                
            # Graph construction cost
            graph_ops = ["build_knn_graph", "build_final_graph"]  
            graph_build_time = sum(stats['total'] for op, stats in sorted_ops 
                                  if any(name == op for name in graph_ops))
            if graph_build_time > 0:
                pct = graph_build_time / total_time * 100
                print(f"  • Graph construction operations: {graph_build_time:.2f}s ({pct:.1f}% of total)")
        
        # Clear caches to free memory
        self._edge_arr_cache.clear()
        self._sorted_edge_cache.clear()
        
        # Return graph object and additional results
        return graph_obj, {
            "initial_mst": mst_graph,
            "initial_edge_data": edge_data,
            "history": history,
            "component_labels": component_labels,
            "n_components": n_components,
            "component_sizes": component_sizes,
            "timing_stats": self.timing.get_stats(as_dict=True),
            # Add summarized timing information for easy access
            "timing_summary": {
                "total": total_time,
                "step1_knn_mst": step1_time,
                "step2_refinement": step2_time, 
                "step3_analysis": step3_time,
                "graph_distance_calculation": graph_distance_compute_time if 'graph_distance_compute_time' in locals() else 0
            }
        }