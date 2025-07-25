"""
DataGraphGenerator - Creates and refines a sparse graph over data according to a semimetric weight function
Updated with missing value handling, MST span checking, and Euclidean optimization
"""
import numpy as np
import time
import matplotlib.pyplot as plt

from scipy import sparse
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from sklearn.neighbors import NearestNeighbors

import pandas as pd

import numpy as np
from numba import njit, prange

from .core_utilities import (
    TimingStats, BatchStats, 
    make_parallel_batcher,
    find_knee_point, 
    find_2hop_neighbors_efficient
)
from .data_graph import DataGraph

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
    
class DataGraphGenerator:
    def __init__(self,
                 node_df,
                 feature_cols,
                 semimetric_weight_function,
                 embedding_function,
                 verbose=True, 
                 use_float32=True,
                 n_jobs=-1, 
                 plot_knee=False,
                 missing_weight=np.inf,  # NEW: configurable missing value
                 use_euclidean_as_graph_distance=False):  # NEW: optimization flag
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
            Value returned by semimetric function to indicate missing/invalid edge
        use_euclidean_as_graph_distance : bool, default=False
            If True, skip graph distance computation and use Euclidean distances directly.
            Set to True when you know the semimetric function returns the same values
            as Euclidean distance in embedding space.
        """
        self.verbose = verbose
        self.use_float32 = use_float32
        self.n_jobs = n_jobs
        self.plot_knee = plot_knee
        self.timing = TimingStats()  # Initialize timing stats
        self.missing_weight = missing_weight
        self.use_euclidean_as_graph_distance = use_euclidean_as_graph_distance

        self.node_df = node_df
        self.feature_cols = feature_cols
        self.semimetric_weight_function = semimetric_weight_function
        self.embedding_function = embedding_function
        
        self._extract_node_features()
        # build (or re-build) a batcher for whichever dist‐fn they just passed
        self._batcher = prewarm_numba_functions(semimetric_weight_function,
                                               feature_dim=len(self.feature_cols or
                                                               node_df.select_dtypes(include='number').columns))

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

    def _check_mst_span(self, mst, graph, n):
        """
        Check if the MST has finite span (all edges have finite weights)
        
        Returns:
        --------
        bool
            True if MST has finite span, False otherwise
        float
            Total span (sum of edge weights) of the MST
        """
        mst_coo = mst.tocoo()
        
        # Check for infinite weights
        if np.any(np.isinf(mst_coo.data)):
            # Find which edges have infinite weight
            inf_edges = [(i, j) for i, j, w in zip(mst_coo.row, mst_coo.col, mst_coo.data) 
                         if np.isinf(w) and i < j]
            
            if self.verbose:
                print(f"         WARNING: MST contains {len(inf_edges)} edges with infinite weight")
                for i, (u, v) in enumerate(inf_edges[:5]):  # Show first 5
                    print(f"           Edge ({u}, {v}) has infinite weight")
                if len(inf_edges) > 5:
                    print(f"           ... and {len(inf_edges) - 5} more")
            
            return False, np.inf
        
        # Calculate total span
        total_span = np.sum(mst_coo.data)
        return True, total_span

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
        
    def extract_mst_edges(self, graph, n):
        """
        Extract MST edges from a graph in a robust manner.
        
        Parameters:
        -----------
        graph : scipy.sparse.csr_matrix
            Graph to extract MST from
        n : int
            Number of nodes in the graph
            
        Returns:
        --------
        set
            Set of MST edges as (min(i,j), max(i,j)) tuples
        """
        # Start timing
        self.timing.start("extract_mst_edges")
        
        # Compute MST using scipy
        # scipy MST will ignore zero edges, and floating point errors will cause small weight to vanish. 
        # Thus we simply add 1 uniformly.
        graph.data += 1
        
        self.timing.start("extract_mst_edges.compute_mst")
        mst = minimum_spanning_tree(graph)
        self.timing.end("extract_mst_edges.compute_mst")
        
        # Check MST span before proceeding
        has_finite_span, total_span = self._check_mst_span(mst, graph, n)
        if not has_finite_span:
            if self.verbose:
                print(f"         ERROR: Failed to find MST with finite span")
                print(f"         The graph may be disconnected or contain invalid edge weights")
        
        # Convert to COO format for edge extraction
        self.timing.start("extract_mst_edges.convert_to_coo")
        mst_coo = mst.tocoo()
        self.timing.end("extract_mst_edges.convert_to_coo")
        
        # Check if the graph is connected
        self.timing.start("extract_mst_edges.check_connectivity")
        n_components, labels = connected_components(graph, directed=False)
        self.timing.end("extract_mst_edges.check_connectivity")
        
        graph.data -= 1
        if n_components > 1 and self.verbose:
            print(f"         WARNING: Graph has {n_components} disconnected components")
            
            # Count nodes in each component
            component_sizes = np.bincount(labels)
            print(f"         Component sizes: {component_sizes}")
        
        # Create a set to store MST edges
        mst_edges = set()
        
        # Extract edges from MST - Include ALL edges
        self.timing.start("extract_mst_edges.extract_edges")
        for i, j in zip(mst_coo.row, mst_coo.col):
            # Only store edges in one direction (smaller index to larger index)
            if i < j:  # Include zero-weight edges too
                mst_edges.add((i, j))
        self.timing.end("extract_mst_edges.extract_edges")
        
        # For a fully connected graph with n nodes, MST should have n-1 edges
        expected_edges = n - n_components  # Adjust for disconnected components
        
        if len(mst_edges) != expected_edges and self.verbose:
            print(f"         WARNING: Found {len(mst_edges)} MST edges, expected {expected_edges}")
            print(f"         This might indicate issues with the MST computation")
            
        # If we don't have enough edges, try adding edges between components
        if len(mst_edges) < expected_edges:
            self.timing.start("extract_mst_edges.add_missing")
            if self.verbose:
                print(f"         Attempting to add missing edges to complete the MST")
            
            # Find representatives for each component
            component_reps = {}
            for node, comp in enumerate(labels):
                if comp not in component_reps:
                    component_reps[comp] = node
            
            # For each pair of components, add an edge between their representatives
            for comp1 in range(n_components):
                for comp2 in range(comp1 + 1, n_components):
                    rep1 = component_reps[comp1]
                    rep2 = component_reps[comp2]
                    edge = (min(rep1, rep2), max(rep1, rep2))
                    if edge not in mst_edges:
                        mst_edges.add(edge)
                        if self.verbose:
                            print(f"         Added edge {edge} to connect components {comp1} and {comp2}")
            self.timing.end("extract_mst_edges.add_missing")
        
        if self.verbose:
            print(f"         Final MST has {len(mst_edges)} edges for {n} nodes in {n_components} components")
            if has_finite_span:
                print(f"         MST total span: {total_span:.4g}")
        
        self.timing.end("extract_mst_edges")    
        return mst_edges

    def create_knn_graph_with_mst(self, 
                                  n_neighbors=30, 
                                  max_iterations=3, 
                                  batch_size=None, 
                                  report_mst_diameters=False):
        """
        Create KNN graph with MST and iteratively refine it using custom distances
        
        Parameters:
        -----------
        """
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
        
        # More efficient: use apply instead of iterrows
        embeddings = np.array(node_df.apply(embedding_function, axis=1).tolist())
        
        if self.use_float32:
            embeddings = embeddings.astype(np.float32)
            
        if self.verbose:
            print(f"         Computed embeddings with shape: {embeddings.shape}")
        
        self.timing.end("create_knn_graph_with_mst.compute_embeddings")
        
        # Step 2: Find nearest neighbors in embedding space
        self.timing.start("create_knn_graph_with_mst.find_nearest_neighbors")
        
        print("Computing raw knn graph on embedding...")
        nbrs = NearestNeighbors(n_neighbors=min(n_neighbors + 1, n), n_jobs=self.n_jobs)
        nbrs.fit(embeddings)
        euclidean_distances, idx = nbrs.kneighbors(embeddings)

        print("Computing self-connection mask...")
        # Explicitly remove self-connections instead of assuming they're first
        # Create masks for non-self connections
        n = len(node_df)
        mask = idx != np.arange(n)[:, None]

        print("Filtering out self-connections...")
        # Filter out self-connections
        filtered_distances = []
        filtered_indices = []
        for i in range(n):
            row_mask = mask[i]
            filtered_distances.append(euclidean_distances[i][row_mask][:n_neighbors])
            filtered_indices.append(idx[i][row_mask][:n_neighbors])

        print("Converting to arrays with padding?...")
        # Convert to arrays, padding if necessary
        max_neighbors = max(len(d) for d in filtered_distances)
        euclidean_distances = np.zeros((n, max_neighbors), dtype=euclidean_distances.dtype)
        idx = np.zeros((n, max_neighbors), dtype=idx.dtype)

        print("Marking missing neighbors?...")
        for i in range(n):
            nd = len(filtered_distances[i])
            euclidean_distances[i, :nd] = filtered_distances[i]
            idx[i, :nd] = filtered_indices[i]
            # Mark missing neighbors with -1
            if nd < max_neighbors:
                idx[i, nd:] = -1
        
        if self.verbose:
            print(f"         Distance stats (in embedding space): min={euclidean_distances.min():.4g}, "
                  f"mean={euclidean_distances.mean():.4g}, max={euclidean_distances.max():.4g}")

        self.timing.end("create_knn_graph_with_mst.find_nearest_neighbors")

        print("Building csr knn graph...")
        self.timing.start("create_knn_graph_with_mst.build_knn_graph")
        # Build KNN graph, handling -1 indices for missing neighbors
        rows = []
        cols = []
        data = []
        
        for i in range(n):
            for j, neighbor_idx in enumerate(idx[i]):
                if neighbor_idx >= 0:  # Valid neighbor
                    rows.append(i)
                    cols.append(neighbor_idx)
                    data.append(euclidean_distances[i, j])
        
        rows = np.array(rows)
        cols = np.array(cols)
        data = np.array(data)
        
        # Find minimum non-zero distance to use as epsilon
        non_zero_distances = data[data > 0]
        if len(non_zero_distances) > 0:
            epsilon = non_zero_distances.min() * 1e-6  # Much smaller than smallest real distance
        else:
            epsilon = 1e-10
        
        # Add epsilon to zero distances to preserve them in sparse matrix
        data = np.where(data == 0, epsilon, data)
        
        G0 = csr_matrix((data, (rows, cols)), shape=(n, n))
        
        # Symmetrize by taking the maximum of (i→j) and (j→i)
        G_knn = G0.maximum(G0.T)
        
        self.timing.end("create_knn_graph_with_mst.build_knn_graph")
        
        # Check if KNN graph is connected
        self.timing.start("create_knn_graph_with_mst.check_connectivity")
        n_components, component_labels_knn = connected_components(G_knn, directed=False)
        self.timing.end("create_knn_graph_with_mst.check_connectivity")
        
        if self.verbose:
            print(f"         KNN graph has {n_components} connected components")

        if True:
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
        if n_components > 1:
            self.timing.start("create_knn_graph_with_mst.connect_components")
            if self.verbose:
                print(f"         Adding edges to connect all components")
            
            # Find representatives for each component
            component_reps = {}
            for i, comp in enumerate(component_labels_knn):
                if comp not in component_reps:
                    component_reps[comp] = i
            
            # Create a fully connected graph between component representatives
            for comp1, rep1 in component_reps.items():
                for comp2, rep2 in component_reps.items():
                    if comp1 < comp2:  # Only add each edge once
                        # Use Euclidean distance as initial weight
                        dist = np.linalg.norm(embeddings[rep1] - embeddings[rep2])
                        G_knn[rep1, rep2] = dist
                        G_knn[rep2, rep1] = dist
                        
            # Verify the graph is now connected
            n_components_after, _ = connected_components(G_knn, directed=False)
            if self.verbose:
                print(f"         After adding edges: {n_components_after} connected components")
                
            if n_components_after > 1:
                print(f"         WARNING: Graph still has disconnected components after attempted repair")
            self.timing.end("create_knn_graph_with_mst.connect_components")
        
        # Initialize edge tracking
        print("Tracking edges for graph distance computation...")
        self.timing.start("create_knn_graph_with_mst.initialize_edge_tracking")
        edge_list = []  # List of (i, j) pairs where i < j
        edge_map = {}   # Maps (i, j) to index in edge_list
        graph_distances = []  # Corresponding graph distances
        euclidean_distances_list = []  # Store Euclidean distances for each edge
        graph_distance_computed = []   # Boolean mask for which edges have graph distances
        is_mst_edge = []   # Flag to indicate if an edge is part of the MST
        is_valid_edge = []  # NEW: Flag to indicate if edge should be kept

        print("Converting csr to COO format...")
        # Convert to COO format for easy edge extraction
        G_knn_coo = G_knn.tocoo()

        print("Adding all KNN edges to tracking list...")
        # First add all KNN edges to our tracking lists
        for i, j, d in zip(G_knn_coo.row, G_knn_coo.col, G_knn_coo.data):
            if i < j:  # Only store one direction
                edge_key = (i, j)
                # Check if this edge is already in the list
                if edge_key not in edge_map:
                    edge_list.append(edge_key)
                    edge_map[edge_key] = len(edge_list) - 1
                    graph_distances.append(0.0)  # Placeholder
                    euclidean_distances_list.append(d)  # Store Euclidean distance
                    graph_distance_computed.append(False)
                    is_mst_edge.append(False)  # Mark as MST edge later
                    is_valid_edge.append(True)  # Initially all edges are valid
                        
        # Convert to numpy arrays for efficiency
        graph_distances = np.array(graph_distances, dtype=np.float32 if self.use_float32 else np.float64)
        euclidean_distances_list = np.array(euclidean_distances_list, dtype=np.float32 if self.use_float32 else np.float64)
        graph_distance_computed = np.array(graph_distance_computed, dtype=bool)
        is_mst_edge = np.array(is_mst_edge, dtype=bool)
        is_valid_edge = np.array(is_valid_edge, dtype=bool)
        self.timing.end("create_knn_graph_with_mst.initialize_edge_tracking")
        
        # Initialize batch statistics trackers
        graph_distance_stats = BatchStats()
        euclidean_stats = BatchStats()
        
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
                graph_distance_stats.update_batch(euclidean_distances_list[valid_mask])
                euclidean_stats.update_batch(euclidean_distances_list[valid_mask])
            
            scale_factor = 1.0  # No scaling needed
            
        else:
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
                
                # Get edges that need graph distances
                new_i = []
                new_j = []
                new_edge_indices = []
                
                for idx in uncomputed_indices:
                    i, j = edge_list[idx]
                    new_i.append(i)
                    new_j.append(j)
                    new_edge_indices.append(idx)
                
                new_i = np.array(new_i, dtype=np.int32)
                new_j = np.array(new_j, dtype=np.int32)
                new_edge_indices = np.array(new_edge_indices, dtype=np.int32)
                
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
                    batch_distances = self._batcher(
                        self.node_features, batch_i, batch_j
                    )
                    
                    # NEW: Check for missing values and mark invalid edges
                    for i, (dist, idx) in enumerate(zip(batch_distances, batch_indices)):
                        if dist == self.missing_weight:
                            is_valid_edge[idx] = False
                            removed_count += 1
                            graph_distances[idx] = np.inf  # Mark as infinite
                        else:
                            graph_distances[idx] = dist
                    
                    # Get the corresponding Euclidean distances for this batch
                    batch_euclidean = euclidean_distances_list[batch_indices]
                    
                    # Update computed flags
                    graph_distance_computed[batch_indices] = True
                    
                    # Find valid pairs for statistics (positive values and not missing)
                    valid_mask = (batch_distances > 0) & (batch_euclidean > 0) & (batch_distances != self.missing_weight)
                    valid_graph = batch_distances[valid_mask]
                    valid_euclidean = batch_euclidean[valid_mask]
                    
                    # Update batch statistics
                    if len(valid_graph) > 0:
                        graph_distance_stats.update_batch(valid_graph)
                        euclidean_stats.update_batch(valid_euclidean)
                
                if removed_count > 0 and self.verbose:
                    print(f"         Removed {removed_count} edges with missing values")
                
                self.timing.end(f"create_knn_graph_with_mst.iteration_{iteration+1}.compute_graph_distances")
        
                # Calculate scaling factor between graph and Euclidean distances using batch statistics
                if graph_distance_stats.count > 0:
                    graph_distance_mean = graph_distance_stats.mean
                    euclidean_mean = euclidean_stats.mean
                    scale_factor = euclidean_mean / graph_distance_mean if graph_distance_mean > 0 else 1.0
                    
                    if self.verbose:
                        print(f"         Scale factor: {scale_factor:.4g} "
                              f"(graph mean: {graph_distance_mean:.4g}, Euclidean mean: {euclidean_mean:.4g})")
                        print(f"         graph stats: min={graph_distance_stats.min_val:.4g}, max={graph_distance_stats.max_val:.4g}, "
                              f"std={graph_distance_stats.get_std():.4g}, count={graph_distance_stats.count}")
                    
                    # Check if we should suggest Euclidean optimization
                    if iteration == 0:  # Only check on first iteration
                        self._suggest_euclidean_optimization(graph_distances[graph_distance_computed], 
                                                           euclidean_distances_list[graph_distance_computed])
                else:
                    scale_factor = 1.0
                    if self.verbose:
                        print(f"         No completed graph distances yet. Using scale factor of 1.0")
                
                # Create hybrid distances for MST computation:
                # - Scaled graph distances when available
                # - Euclidean distances as fallback
                # - Infinite for invalid edges
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
                
                # Create hybrid graph for MST computation
                hybrid_rows = []
                hybrid_cols = []
                hybrid_data = []
                
                for idx, (i, j) in enumerate(edge_list):
                    if is_valid_edge[idx]:  # Only include valid edges
                        dist = hybrid_distances[idx]
                        hybrid_rows.extend([i, j])
                        hybrid_cols.extend([j, i])
                        hybrid_data.extend([dist, dist])
                
                hybrid_graph = csr_matrix(
                    (hybrid_data, (hybrid_rows, hybrid_cols)), 
                    shape=(n, n)
                )
                self.timing.end(f"create_knn_graph_with_mst.iteration_{iteration+1}.build_hybrid_graph")
                
                # Extract MST from hybrid graph
                self.timing.start(f"create_knn_graph_with_mst.iteration_{iteration+1}.extract_mst")
                mst_edges = self.extract_mst_edges(hybrid_graph, n)
                self.timing.end(f"create_knn_graph_with_mst.iteration_{iteration+1}.extract_mst")
                
                # Update MST edge flags
                is_mst_edge[:] = False  # Reset all flags
                for i, j in mst_edges:
                    edge_key = (i, j)
                    if edge_key in edge_map:
                        idx = edge_map[edge_key]
                        is_mst_edge[idx] = True
                
                self.timing.end(f"create_knn_graph_with_mst.iteration_{iteration+1}")
        
        # Report MST diameters if requested (moved outside iteration loop for optimization mode)
        if report_mst_diameters:
            # Calculate MST diameter using unscaled graph distances
            # Create a graph containing only the MST edges with unscaled graph distances
            mst_graph = np.zeros((n, n), dtype=np.float32 if self.use_float32 else np.float64)
            mst_edge_count = 0
            for idx, (i, j) in enumerate(edge_list):
                if is_mst_edge[idx] and graph_distance_computed[idx] and is_valid_edge[idx]:
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
        
        # Create the final graph with graph distances (excluding invalid edges)
        self.timing.start("create_knn_graph_with_mst.build_final_graph")
        rows = []
        cols = []
        data = []
        
        for idx, (i, j) in enumerate(edge_list):
            if is_valid_edge[idx]:  # Only include valid edges
                dist = graph_distances[idx]
                rows.extend([i, j])
                cols.extend([j, i])
                data.extend([dist, dist])
        
        final_graph = csr_matrix(
            (data, (rows, cols)), 
            shape=(n, n)
        )
        self.timing.end("create_knn_graph_with_mst.build_final_graph")
        
        # Final verification that the graph is connected
        self.timing.start("create_knn_graph_with_mst.final_verification")
        n_components_final, _ = connected_components(final_graph, directed=False)
        self.timing.end("create_knn_graph_with_mst.final_verification")
        
        if self.verbose:
            valid_edge_count = np.sum(is_valid_edge)
            mst_edge_count = np.sum(is_mst_edge)
            print(f"         Combined graph has {valid_edge_count} valid edges (of {len(edge_list)} total), "
                  f"including {mst_edge_count} MST edges")
            print(f"         Final graph has {n_components_final} connected components")
            if n_components_final > 1:
                print(f"         WARNING: Final graph is not fully connected!")
        
        self.timing.end("create_knn_graph_with_mst")
        
        # Filter edge data to only include valid edges
        valid_indices = np.where(is_valid_edge)[0]
        filtered_edge_list = [edge_list[i] for i in valid_indices]
        filtered_graph_distances = graph_distances[valid_indices]
        filtered_euclidean_distances = euclidean_distances_list[valid_indices]
        filtered_is_mst_edge = is_mst_edge[valid_indices]
        
        # Rebuild edge map
        filtered_edge_map = {edge: i for i, edge in enumerate(filtered_edge_list)}
        
        return final_graph, {
            'edge_list': filtered_edge_list,
            'graph_distances': filtered_graph_distances,
            'euclidean_distances': filtered_euclidean_distances,
            'edge_map': filtered_edge_map,
            'is_mst_edge': filtered_is_mst_edge,
            'graph_distance_scalar': scale_factor,
            # Add statistics to the returned data
            'graph_distance_stats': graph_distance_stats.get_stats() if graph_distance_stats.count > 0 else None,
            'euclidean_stats': euclidean_stats.get_stats() if euclidean_stats.count > 0 else None,
            'n_removed_edges': len(edge_list) - len(filtered_edge_list),  # Track removed edges
            'use_euclidean_as_graph_distance': self.use_euclidean_as_graph_distance
        }
    
    def prune_graph_by_threshold(self, graph, edge_data, threshold=None, kneedle_sensitivity=1.0, 
                               max_components=None, use_median_filter=True, preserve_mst=True):
        """
        Prune a graph by removing edges with distances above a threshold.
        
        NOTE: Edges that exceed the threshold are already removed during computation,
        so this method primarily handles automatic threshold detection and additional pruning.
        
        Parameters:
        -----------
        graph : scipy.sparse.csr_matrix
            Graph with edge weights
        edge_data : dict
            Dictionary with edge list and distances
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
        edge_list = edge_data['edge_list']
        graph_distances = edge_data['graph_distances']
        edge_map = edge_data.get('edge_map', {})
        
        # OPTIMIZATION: Early return if no pruning needed
        # Case 1: Threshold is infinity
        if threshold == np.inf:
            if self.verbose:
                print(f"[Pruning] No pruning needed: threshold is infinity")
            
            self.timing.end("prune_graph_by_threshold")
            return graph, edge_data, {
                'pruned_edges': [],
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
                'pruned_edges': [],
                'pruned_distances': np.array([]),
                'threshold': threshold
            }
        
        if self.verbose:
            print(f"[Pruning] Pruning graph edges by threshold")
        
        # If we're preserving the MST, use the is_mst_edge flag if available
        self.timing.start("prune_graph_by_threshold.prepare_mst")
        mst_edges = set()
        if preserve_mst:
            if 'is_mst_edge' in edge_data:
                # Use the pre-computed MST edge information
                is_mst_edge_array = edge_data['is_mst_edge']
                mst_indices = np.where(is_mst_edge_array)[0]
                mst_edges = set(edge_list[i] for i in mst_indices)
                
                if self.verbose:
                    print(f"         Using pre-identified MST edges: {len(mst_edges)} edges")
                    expected_mst_edges = n - 1
                    if len(mst_edges) != expected_mst_edges:
                        print(f"         WARNING: MST should have exactly {expected_mst_edges} edges, found {len(mst_edges)}")
                        
                        # If we're missing MST edges, recompute them to ensure we have the right number
                        if len(mst_edges) < expected_mst_edges:
                            if self.verbose:
                                print(f"         Recomputing MST edges to ensure connectivity")
                            mst_edges = self.extract_mst_edges(graph, n)
            else:
                # If we don't have pre-computed MST information, extract it now
                mst_edges = self.extract_mst_edges(graph, n)
            
            if self.verbose:
                print(f"         Preserving {len(mst_edges)} MST edges to maintain connectivity")
        self.timing.end("prune_graph_by_threshold.prepare_mst")
        
        # Sort edges by distance (in descending order)
        self.timing.start("prune_graph_by_threshold.sort_edges")
        sorted_indices = np.argsort(graph_distances)[::-1]
        sorted_edges = [edge_list[idx] for idx in sorted_indices]
        sorted_distances = graph_distances[sorted_indices]
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
        if preserve_mst:
            prunable_mask = np.array([edge not in mst_edges for edge in sorted_edges])
            prunable_edges = [edge for i, edge in enumerate(sorted_edges) if prunable_mask[i]]
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
            edges_to_prune = [edge for idx, edge in enumerate(prunable_edges) 
                             if prunable_distances[idx] > threshold]
            prune_distances = prunable_distances[:len(edges_to_prune)]
            
            if self.verbose:
                print(f"         Pruning {len(edges_to_prune)} edges above threshold {threshold:.4g}")
        
        else:
            if len(prunable_edges) < 2:
                if self.verbose:
                    print(f"         Not enough prunable edges. Skipping pruning.")
                
                self.timing.end("prune_graph_by_threshold.determine_edges")
                self.timing.end("prune_graph_by_threshold")
                return graph, edge_data, {
                    'pruned_edges': [],
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
            # We only need this check if we're not preserving the MST
            # Start with a pruned graph
            test_graph = graph.copy()
            
            # Try removing edges one by one, from highest to lowest distance
            current_components = 1  # Assume we start with a connected graph
            
            for i, (edge_i, edge_j) in enumerate(edges_to_prune):
                # Remove the edge
                test_graph[edge_i, edge_j] = 0
                test_graph[edge_j, edge_i] = 0
                
                # Check if we've reached the component limit (do this every few edges for efficiency)
                if i % 100 == 0 or i == len(edges_to_prune) - 1:
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
                            edges_to_prune = []
                            prune_distances = []
                        
                        if self.verbose:
                            print(f"         Limiting pruning to maintain max {max_components} components")
                            print(f"         Adjusted threshold: {threshold:.4g}, pruning {len(edges_to_prune)} edges")
                        
                        break
                    
                    current_components = n_components
            self.timing.end("prune_graph_by_threshold.check_components")
        
        # Prune the graph by removing the selected edges
        self.timing.start("prune_graph_by_threshold.apply_pruning")
        pruned_graph = graph.copy()
        
        for i, j in edges_to_prune:
            pruned_graph[i, j] = 0
            pruned_graph[j, i] = 0
        
        # Ensure the matrix is still in CSR format and eliminate zeros
        pruned_graph.eliminate_zeros()
        self.timing.end("prune_graph_by_threshold.apply_pruning")
        
        # Create new edge data for the pruned graph more efficiently
        self.timing.start("prune_graph_by_threshold.create_edge_data")
        pruned_edge_list = []
        pruned_edge_map = {}
        pruned_graph_distances = []
        pruned_is_mst_edge = []
        
        # Process edges in a single pass
        has_mst_info = 'is_mst_edge' in edge_data
        pruned_coo = pruned_graph.tocoo()
        processed_edges = set()
        
        for i, j, d in zip(pruned_coo.row, pruned_coo.col, pruned_coo.data):
            if i < j and (i, j) not in processed_edges:  # Only process each edge once
                edge_key = (i, j)
                processed_edges.add(edge_key)
                
                pruned_edge_list.append(edge_key)
                pruned_edge_map[edge_key] = len(pruned_edge_list) - 1
                pruned_graph_distances.append(d)
                
                # Preserve MST edge information if available
                if has_mst_info:
                    # Check if this is an MST edge
                    is_mst = edge_key in mst_edges
                    pruned_is_mst_edge.append(is_mst)
        
        # Create a new BatchStats object to recalculate statistics for the pruned graph
        if 'graph_distance_stats' in edge_data and edge_data['graph_distance_stats'] is not None:
            pruned_graph_distance_stats = BatchStats()
            pruned_graph_distance_stats.update_batch(np.array(pruned_graph_distances))
        else:
            pruned_graph_distance_stats = None
        
        pruned_edge_data = {
            'edge_list': pruned_edge_list,
            'edge_map': pruned_edge_map,
            'graph_distances': np.array(pruned_graph_distances)
        }
        
        # Add MST edge information if it was available
        if has_mst_info or preserve_mst:
            pruned_edge_data['is_mst_edge'] = np.array(pruned_is_mst_edge)
        
        # Add statistics if we calculated them
        if pruned_graph_distance_stats is not None:
            pruned_edge_data['graph_distance_stats'] = pruned_graph_distance_stats.get_stats()
        
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
            n_components, _ = connected_components(pruned_graph, directed=False)
            print(f"         Pruned graph has {len(pruned_edge_list)} edges and {n_components} connected components")
            
            # Verify that all MST edges are still present
            if has_mst_info or preserve_mst:
                mst_edge_count = np.sum(pruned_edge_data['is_mst_edge'])
                expected_mst_edges = n - 1
                if mst_edge_count != expected_mst_edges and preserve_mst:
                    print(f"         WARNING: After pruning, found {mst_edge_count} MST edges, expected {expected_mst_edges}")
                    
                    # Double-check which MST edges are missing
                    if mst_edge_count < expected_mst_edges:
                        missing_count = 0
                        for edge in mst_edges:
                            if edge not in pruned_edge_list:
                                missing_count += 1
                                if missing_count <= 5:  # Only show up to 5 missing edges
                                    print(f"         Missing MST edge: {edge}")
                        if missing_count > 5:
                            print(f"         ... and {missing_count - 5} more missing MST edges")
        self.timing.end("prune_graph_by_threshold.verify")
        self.timing.end("prune_graph_by_threshold")
        
        return pruned_graph, pruned_edge_data, {
            'pruned_edges': edges_to_prune,
            'pruned_distances': prune_distances,
            'threshold': threshold
        }

    def smooth_graph_with_2hop(self, graph, edge_data, node_df, semimetric_weight_function, 
                              max_new_edges=None, batch_size=None):
        """
        Smooth the graph by adding connections between 2-hop neighbors.
        
        NOTE: New edges that have missing values (self.missing_weight) are not added.
        
        Parameters:
        -----------
        graph : scipy.sparse.csr_matrix
            Current graph
        edge_data : dict
            Dictionary with edge list and distances
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
        edge_list = edge_data['edge_list']
        edge_map = edge_data['edge_map']
        graph_distances = edge_data['graph_distances']
        
        # Check if using Euclidean optimization
        use_euclidean = edge_data.get('use_euclidean_as_graph_distance', False)
        
        # Create a set of existing edges for fast lookup
        existing_edges = set(edge_list)
        
        # Find 2-hop neighbors
        self.timing.start("smooth_graph_with_2hop.find_2hop")
        new_edges = find_2hop_neighbors_efficient(graph, existing_edges)
        self.timing.end("smooth_graph_with_2hop.find_2hop")
        
        if max_new_edges is not None and len(new_edges) > max_new_edges:
            if self.verbose:
                print(f"         Found {len(new_edges)} potential 2-hop connections, limiting to {max_new_edges}")
            # Randomly sample a subset of new edges using indices
            random_indices = np.random.choice(len(new_edges), max_new_edges, replace=False)
            new_edges = [new_edges[i] for i in random_indices]
        
        if self.verbose:
            print(f"         Adding up to {len(new_edges)} new 2-hop connections")
        
        # If no new edges, return early
        if len(new_edges) == 0:
            if self.verbose:
                print(f"         No new 2-hop connections to add")
            
            self.timing.end("smooth_graph_with_2hop")
            return graph, edge_data, {
                'n_new_edges': 0,
                'new_edges': [],
                'new_distances': []
            }
        
        # Extract edge endpoints for batch processing
        new_i = np.array([edge[0] for edge in new_edges], dtype=np.int32)
        new_j = np.array([edge[1] for edge in new_edges], dtype=np.int32)
        
        # Compute graph distances for new edges in batches
        new_edge_distances = []
        valid_new_edges = []  # Track which edges are valid
        
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
                    
        n_batches = (len(new_edges) - 1) // batch_size + 1
        if self.verbose and len(new_edges) > batch_size:
            print(f"         Processing in {n_batches} batches of size {batch_size}")
        
        removed_count = 0  # Track edges with missing values
        
        for batch_idx in range(0, len(new_edges), batch_size):
            if self.verbose and len(new_edges) > batch_size and batch_idx % (10 * batch_size) == 0:
                print(f"         Processing batch {batch_idx//batch_size + 1}/{n_batches}")
                
            end_idx = min(batch_idx + batch_size, len(new_edges))
            batch_i = new_i[batch_idx:end_idx]
            batch_j = new_j[batch_idx:end_idx]
            batch_edges = new_edges[batch_idx:end_idx]
            
            if use_euclidean:
                # Compute Euclidean distances directly
                batch_distances = np.array([
                    np.linalg.norm(embeddings[i] - embeddings[j])
                    for i, j in zip(batch_i, batch_j)
                ])
            else:
                # Compute distances for this batch using the custom function
                batch_distances = self._batcher(
                    self.node_features, batch_i, batch_j
                )
            
            # Filter out edges with missing values
            for dist, edge in zip(batch_distances, batch_edges):
                if dist != self.missing_weight:
                    new_edge_distances.append(dist)
                    valid_new_edges.append(edge)
                else:
                    removed_count += 1
                    
        self.timing.end("smooth_graph_with_2hop.compute_distances")
        
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
                'new_edges': [],
                'new_distances': [],
                'n_removed': removed_count
            }
        
        # Add new edges to the graph more efficiently
        self.timing.start("smooth_graph_with_2hop.add_edges")
        new_edge_distances = np.array(new_edge_distances)

        # Modify the original graph in place by adding new edges
        smoothed_graph = graph.copy()  # Still need one copy
        
        # Add new edges directly to the smoothed graph
        for idx, (i, j) in enumerate(valid_new_edges):
            dist = new_edge_distances[idx]
            smoothed_graph[i, j] = dist
            smoothed_graph[j, i] = dist  # Keep the graph symmetric
        self.timing.end("smooth_graph_with_2hop.add_edges")

        # Update edge data
        self.timing.start("smooth_graph_with_2hop.update_edge_data")
        has_mst_info = 'is_mst_edge' in edge_data
        
        # Get the MST edges for this graph if we have MST info
        if has_mst_info:
            # Extract MST edges from original edge data
            original_mst_edges = set(edge_list[i] for i in np.where(edge_data['is_mst_edge'])[0])
            
            # Verify we have the right number of MST edges
            expected_mst_edges = n - 1
            if len(original_mst_edges) != expected_mst_edges:
                if self.verbose:
                    print(f"         WARNING: Original MST has {len(original_mst_edges)} edges, expected {expected_mst_edges}")
                # If we don't have the right number, recompute
                original_mst_edges = self.extract_mst_edges(graph, n)
        
        # Create updated edge data from the smoothed graph
        smoothed_edge_list = []
        smoothed_edge_map = {}
        smoothed_graph_distances = []
        smoothed_is_mst_edge = []
        
        # Extract edges from the smoothed graph
        smoothed_coo = smoothed_graph.tocoo()
        processed_edges = set()
        
        for i, j, d in zip(smoothed_coo.row, smoothed_coo.col, smoothed_coo.data):
            if i < j and (i, j) not in processed_edges:  # Only process each edge once
                edge_key = (i, j)
                processed_edges.add(edge_key)
                
                smoothed_edge_list.append(edge_key)
                smoothed_edge_map[edge_key] = len(smoothed_edge_list) - 1
                smoothed_graph_distances.append(d)
                
                # Determine if this edge is an MST edge
                if has_mst_info:
                    is_mst = edge_key in original_mst_edges
                    smoothed_is_mst_edge.append(is_mst)
        
        smoothed_edge_data = {
            'edge_list': smoothed_edge_list,
            'edge_map': smoothed_edge_map,
            'graph_distances': np.array(smoothed_graph_distances)
        }
        
        # Add MST edge information if it was available
        if has_mst_info:
            smoothed_edge_data['is_mst_edge'] = np.array(smoothed_is_mst_edge)
            
        # Copy over optimization flag
        if 'use_euclidean_as_graph_distance' in edge_data:
            smoothed_edge_data['use_euclidean_as_graph_distance'] = edge_data['use_euclidean_as_graph_distance']
            
        self.timing.end("smooth_graph_with_2hop.update_edge_data")
        
        # Verify the smoothed graph
        self.timing.start("smooth_graph_with_2hop.verify")
        if self.verbose:
            print(f"         Smoothed graph has {len(smoothed_edge_list)} edges")
            n_components, _ = connected_components(smoothed_graph, directed=False)
            print(f"         Smoothed graph has {n_components} connected components")
            
            # Verify MST edge count
            if has_mst_info:
                mst_edge_count = np.sum(smoothed_edge_data['is_mst_edge'])
                expected_mst_edges = n - 1
                if mst_edge_count != expected_mst_edges:
                    print(f"         WARNING: Smoothed graph has {mst_edge_count} MST edges, expected {expected_mst_edges}")
                    
                    # Check if any MST edges are missing
                    if mst_edge_count < expected_mst_edges:
                        missing_count = 0
                        for edge in original_mst_edges:
                            if edge not in smoothed_edge_list:
                                missing_count += 1
                                if missing_count <= 5:  # Only show up to 5 missing edges
                                    print(f"         Missing MST edge: {edge}")
                        if missing_count > 5:
                            print(f"         ... and {missing_count - 5} more missing MST edges")
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
                       semimetric_weight_function,
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
        prev_edge_count = len(edge_data['edge_list'])
    
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
                        current_graph, current_edge_data, node_df, semimetric_weight_function,
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
                        pr_graph, pr_edge_data, node_df, semimetric_weight_function,
                        max_new_edges=max_new_edges, batch_size=batch_size
                    )
                else:
                    sm_graph, sm_edge_data, sm_res = (
                        pr_graph, pr_edge_data, {'n_new_edges': 0, 'n_removed': 0}
                    )
    
                next_graph, next_edge_data = sm_graph, sm_edge_data
    
            # Record iteration results
            current_edge_count = len(next_edge_data['edge_list'])
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
                comps, _ = connected_components(current_graph, directed=False)
                print(f"         End of iteration: "
                      f"{len(current_edge_data['edge_list'])} edges, "
                      f"{comps} components")
            
            self.timing.end(f"iterative_refine_graph.iteration_{it}")
    
        self.timing.end("iterative_refine_graph")
        return current_graph, current_edge_data, history

    def build_and_refine_graph(self, n_neighbors=30, mst_iterations=3,
                          prune_threshold=None, kneedle_sensitivity=1.0,
                          polish_iterations=2, max_new_edges=None, batch_size=None,
                          max_components=None, use_median_filter=True, 
                          smooth_before_prune=False, preserve_mst=True,
                          log_timing=True, missing_weight=np.inf):
        """
        Full pipeline to build and refine the DataGraph object with detailed timing statistics
        
        Parameters:
        -----------
        node_df : pandas.DataFrame
            DataFrame containing node information
        semimetric_weight_function : callable
            Function that computes distance between two DataFrame rows
        embedding_function : callable
            Function that takes a DataFrame row and returns embedding coordinates
        n_neighbors : int, default=30
            Number of neighbors for KNN graph
        ...
        """
        t0 = time.time()
        # Reset timing stats for this run
        self.timing = TimingStats()
        self.timing.start("build_and_refine_graph")

        node_df = self.node_df
        semimetric_weight_function = self.semimetric_weight_function
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
            batch_size=batch_size
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
            mst_graph, edge_data, node_df, semimetric_weight_function,
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
            print(f"[Final] {len(final_edge_data['edge_list'])} edges, {n_components} components")
            print(f"Component size distribution: min={component_sizes.min()}, "
                  f"mean={component_sizes.mean():.1f}, max={component_sizes.max()}")
        
        if log_timing:
            print(f"  • Step 3 (analysis) done in {step3_time:.2f}s")
    
        # Create DataGraph object
        graph_obj = DataGraph(
            final_graph, 
            node_df=node_df,
            feature_cols=feature_cols,
            semimetric_weight_function=semimetric_weight_function,
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
            fr_compute_time = 0
            fr_compute_count = 0
            for op, stats in sorted_ops:
                if "compute_graph_distances" in op:
                    fr_compute_time += stats['total']
                    fr_compute_count += stats['count']
            
            if fr_compute_time > 0:
                pct = fr_compute_time / total_time * 100
                print(f"  • graph distance calculations: {fr_compute_time:.2f}s ({pct:.1f}% of total)")
                print(f"    - {fr_compute_count} batches processed")
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
                "fr_distance_calculation": fr_compute_time if 'fr_compute_time' in locals() else 0
            }
        }