"""
DataGraphAnalyzer - Analysis tools for ManifoldGraph objects.
"""
import numpy as np
from scipy.sparse.csgraph import shortest_path, floyd_warshall
from scipy.stats import gaussian_kde

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve

from .core_utilities import TimingStats


class DataGraphAnalyzer:
    """
    Class for analyzing ManifoldGraph objects.
    """
    
    def __init__(self, verbose=True):
        """
        Initialize the analyzer.
        
        Parameters:
        -----------
        verbose : bool, default=True
            Whether to print progress messages
        """
        self.verbose = verbose
        self.timing = TimingStats()
    def estimate_raw_density(self, graph, method='knn_distance', k=5, dimension=None, return_distances=False):
        """
        Estimate raw density at each node using various methods.
        
        Parameters:
        -----------
        graph : ManifoldGraph or scipy.sparse.csr_matrix
            The graph to analyze
        method : str, default='knn_distance'
            Method to use for density estimation:
            - 'knn_distance': Uses distance to k-th nearest neighbor with proper volume normalization
            - 'avg_knn_distance': Uses average distance to k nearest neighbors
            - 'degree': Uses node degree as a proxy for density
            - 'local_reach': Uses local reachability density (inverse of avg reachability distance)
        k : int, default=5
            Number of neighbors to consider for density estimation
        dimension : int, optional
            Manifold dimension. If None, estimated from the data.
            This is used for scaling the density estimate properly.
        return_distances : bool, default=False
            If True, also return the distances used for density estimation
            
        Returns:
        --------
        numpy.ndarray
            Raw density estimates for each node
        numpy.ndarray or None
            Distances used for estimation (if return_distances=True)
        """
        self.timing.start("estimate_raw_density")
        
        # Extract graph matrix if ManifoldGraph object
        graph_matrix = graph.get_adjacency_matrix() if hasattr(graph, 'get_adjacency_matrix') else graph
        n_nodes = graph_matrix.shape[0]
        
        # Estimate manifold dimension if not provided
        if dimension is None:
            # Simple estimate from average degree and nodes
            avg_degree = graph_matrix.sum() / n_nodes
            # Rough approximation based on geometric random graphs
            dimension = np.ceil(np.log(avg_degree) / np.log(np.log(n_nodes)))
            # Clamp to reasonable range
            dimension = max(2, min(dimension, 10))
            
            if self.verbose:
                print(f"Estimated manifold dimension: {dimension}")
        
        # Initialize arrays for density and distances
        raw_density = np.zeros(n_nodes)
        knn_distances = None if not return_distances else np.zeros((n_nodes, k))
        
        if method == 'knn_distance':
            # For each node, find distance to k-th nearest neighbor
            self.timing.start("estimate_raw_density.knn_distance")
            k_distances = np.zeros(n_nodes)
            
            for i in range(n_nodes):
                # Get neighbors and distances
                start, end = graph_matrix.indptr[i], graph_matrix.indptr[i+1]
                neighbors = graph_matrix.indices[start:end]
                distances = graph_matrix.data[start:end]
                
                # Sort by distance
                sorted_idx = np.argsort(distances)
                neighbors = neighbors[sorted_idx]
                distances = distances[sorted_idx]
                
                # Get k-th distance (or last if fewer than k neighbors)
                if len(distances) >= k:
                    k_distances[i] = distances[k-1]
                    if return_distances:
                        knn_distances[i, :k] = distances[:k]
                else:
                    # If fewer than k neighbors, use the last one
                    if len(distances) > 0:
                        k_distances[i] = distances[-1]
                        if return_distances and len(distances) > 0:
                            knn_distances[i, :len(distances)] = distances
                    else:
                        # No neighbors, use a large value
                        k_distances[i] = 1e6
            
            # Standard KNN density estimator: density ∝ k / (n * volume)
            # where volume ∝ r^dimension and r is the distance to kth neighbor
            epsilon = 1e-10
            
            # Volume of the hypersphere: V = C_d * r^d where C_d is a constant
            # We don't need the exact volume, just the relative densities,
            # so we can omit the dimension-specific constant C_d
            volumes = np.power(k_distances + epsilon, dimension)
            
            # Density = k / (n * volume)
            # The factor n (total points) is constant, so we omit it for relative density
            raw_density = k / volumes
            self.timing.end("estimate_raw_density.knn_distance")
            
        elif method == 'avg_knn_distance':
            # For each node, find average distance to k nearest neighbors
            self.timing.start("estimate_raw_density.avg_knn_distance")
            avg_distances = np.zeros(n_nodes)
            
            for i in range(n_nodes):
                # Get neighbors and distances
                start, end = graph_matrix.indptr[i], graph_matrix.indptr[i+1]
                neighbors = graph_matrix.indices[start:end]
                distances = graph_matrix.data[start:end]
                
                # Sort by distance
                sorted_idx = np.argsort(distances)
                neighbors = neighbors[sorted_idx]
                distances = distances[sorted_idx]
                
                # Calculate average of k nearest (or all if fewer than k)
                if len(distances) > 0:
                    count = min(k, len(distances))
                    avg_distances[i] = np.mean(distances[:count])
                    if return_distances:
                        knn_distances[i, :count] = distances[:count]
                else:
                    avg_distances[i] = 1e6  # No neighbors
            
            # Density is inversely proportional to distance^dimension
            epsilon = 1e-10
            raw_density = 1.0 / np.power(avg_distances + epsilon, dimension)
            self.timing.end("estimate_raw_density.avg_knn_distance")
            
        elif method == 'degree':
            # Use degree as a proxy for density
            self.timing.start("estimate_raw_density.degree")
            degrees = np.diff(graph_matrix.indptr)
            
            # Simple normalization: degree / max_degree
            max_degree = np.max(degrees)
            if max_degree > 0:
                raw_density = degrees / max_degree
            else:
                raw_density = np.zeros(n_nodes)
                
            # No distances to return in this method
            if return_distances:
                knn_distances = None
            self.timing.end("estimate_raw_density.degree")
            
        elif method == 'local_reach':
            # Local reachability density (similar to OPTICS/LOF)
            self.timing.start("estimate_raw_density.local_reach")
            reach_distances = np.zeros(n_nodes)
            
            # First compute core distances (distance to k-th neighbor)
            core_distances = np.zeros(n_nodes)
            for i in range(n_nodes):
                start, end = graph_matrix.indptr[i], graph_matrix.indptr[i+1]
                distances = graph_matrix.data[start:end]
                
                if len(distances) >= k:
                    core_distances[i] = np.sort(distances)[k-1]
                elif len(distances) > 0:
                    core_distances[i] = np.max(distances)
                else:
                    core_distances[i] = 1e6
            
            # Then compute reachability distances
            for i in range(n_nodes):
                start, end = graph_matrix.indptr[i], graph_matrix.indptr[i+1]
                neighbors = graph_matrix.indices[start:end]
                distances = graph_matrix.data[start:end]
                
                # Sort by distance
                sorted_idx = np.argsort(distances)
                neighbors = neighbors[sorted_idx]
                distances = distances[sorted_idx]
                
                # Take at most k neighbors
                neighbors = neighbors[:min(k, len(neighbors))]
                distances = distances[:min(k, len(distances))]
                
                if len(distances) > 0:
                    # Compute reachability distances
                    reach_dists = np.maximum(distances, core_distances[neighbors])
                    reach_distances[i] = np.mean(reach_dists)
                    
                    if return_distances:
                        knn_distances[i, :len(distances)] = reach_dists
                else:
                    reach_distances[i] = 1e6  # No neighbors
            
            # Density is inverse of reachability distance
            epsilon = 1e-10
            raw_density = 1.0 / np.power(reach_distances + epsilon, dimension)
            self.timing.end("estimate_raw_density.local_reach")
        
        else:
            raise ValueError(f"Unknown density estimation method: {method}")
        
        # Normalize to [0, 1] range
        min_density = np.min(raw_density)
        max_density = np.max(raw_density)
        if max_density > min_density:
            raw_density = (raw_density - min_density) / (max_density - min_density)
        
        self.timing.end("estimate_raw_density")
        
        if return_distances:
            return raw_density, knn_distances
        else:
            return raw_density
    
    def smooth_density(self, graph, raw_density, method='neighborhood_avg', 
                      n_iterations=2, alpha=0.5, k=5, reg_lambda=1.0, use_median=False):
        """
        Smooth raw density estimates over the graph.
        
        Parameters:
        -----------
        graph : ManifoldGraph or scipy.sparse.csr_matrix
            The graph to analyze
        raw_density : numpy.ndarray
            Raw density estimates for each node
        method : str, default='neighborhood_avg'
            Method to use for smoothing:
            - 'neighborhood_avg': Simple average over neighborhood
            - 'gaussian': Gaussian kernel weighting by distance
            - 'heat_diffusion': Heat diffusion smoothing (controlled by alpha)
            - 'tikhonov': Tikhonov regularization (graph Laplacian)
        n_iterations : int, default=2
            Number of smoothing iterations
        alpha : float, default=0.5
            Smoothing parameter:
            - For 'neighborhood_avg': Weight of neighbors vs. current node
            - For 'heat_diffusion': Diffusion rate
            - Not used for other methods
        k : int, default=5
            Number of neighbors to consider for neighborhood averaging
        reg_lambda : float, default=1.0
            Regularization parameter for Tikhonov method
        use_median : bool, default=False
            If True, use median instead of mean for neighborhood methods
            (more robust to outliers)
            
        Returns:
        --------
        numpy.ndarray
            Smoothed density estimates
        """
        self.timing.start("smooth_density")
        
        # Extract graph matrix if ManifoldGraph object
        graph_matrix = graph.get_adjacency_matrix() if hasattr(graph, 'get_adjacency_matrix') else graph
        n_nodes = graph_matrix.shape[0]
        
        # Initialize with raw density
        smoothed_density = raw_density.copy()
        
        if method == 'neighborhood_avg':
            # Simple weighted average over neighborhood
            self.timing.start("smooth_density.neighborhood_avg")
            
            for _ in range(n_iterations):
                next_density = smoothed_density.copy()
                
                for i in range(n_nodes):
                    # Get neighbors and distances
                    start, end = graph_matrix.indptr[i], graph_matrix.indptr[i+1]
                    neighbors = graph_matrix.indices[start:end]
                    
                    # Take top k neighbors (if we have that many)
                    if len(neighbors) > k:
                        # Sort by density (to favor high-density regions)
                        sorted_idx = np.argsort(-smoothed_density[neighbors])
                        neighbors = neighbors[sorted_idx[:k]]
                    
                    if len(neighbors) > 0:
                        # Compute weighted average or median
                        if use_median:
                            # Use median (more robust to outliers)
                            neighbor_val = np.median(smoothed_density[neighbors])
                        else:
                            # Use mean
                            neighbor_val = np.mean(smoothed_density[neighbors])
                            
                        # Apply smoothing
                        next_density[i] = (1-alpha) * smoothed_density[i] + alpha * neighbor_val
                
                smoothed_density = next_density
            
            self.timing.end("smooth_density.neighborhood_avg")
        
        elif method == 'gaussian':
            # Gaussian kernel weighting by distance
            self.timing.start("smooth_density.gaussian")
            
            for _ in range(n_iterations):
                next_density = np.zeros_like(smoothed_density)
                
                for i in range(n_nodes):
                    # Get neighbors and distances
                    start, end = graph_matrix.indptr[i], graph_matrix.indptr[i+1]
                    neighbors = graph_matrix.indices[start:end]
                    distances = graph_matrix.data[start:end]
                    
                    if len(neighbors) > 0:
                        # Include self in computation
                        all_neighbors = np.append(neighbors, i)
                        all_distances = np.append(distances, 0.0)
                        all_densities = smoothed_density[all_neighbors]
                        
                        # Calculate distance scale (adaptive bandwidth)
                        # Use median distance for robustness
                        bandwidth = np.median(distances) if len(distances) > 0 else 1.0
                        
                        # Compute Gaussian weights
                        weights = np.exp(-0.5 * (all_distances/bandwidth)**2)
                        weights /= np.sum(weights)
                        
                        # Weighted combination
                        if use_median:
                            # Approximation of weighted median via sampling
                            # Sort by density values
                            sorted_idx = np.argsort(all_densities)
                            sorted_weights = weights[sorted_idx]
                            sorted_densities = all_densities[sorted_idx]
                            
                            # Compute cumulative weights
                            cum_weights = np.cumsum(sorted_weights)
                            
                            # Find median index (first weight >= 0.5)
                            median_idx = np.searchsorted(cum_weights, 0.5)
                            if median_idx >= len(sorted_densities):
                                median_idx = len(sorted_densities) - 1
                                
                            next_density[i] = sorted_densities[median_idx]
                        else:
                            # Weighted average (mean)
                            next_density[i] = np.sum(weights * all_densities)
                    else:
                        # No neighbors, keep original value
                        next_density[i] = smoothed_density[i]
                
                smoothed_density = next_density
            
            self.timing.end("smooth_density.gaussian")
        
        elif method == 'heat_diffusion':
            # Heat diffusion smoothing
            self.timing.start("smooth_density.heat_diffusion")
            
            # Build the graph Laplacian
            # L = D - A where D is degree matrix and A is adjacency
            self.timing.start("smooth_density.heat_diffusion.build_laplacian")
            degrees = np.array(graph_matrix.sum(axis=1)).flatten()
            D = diags(degrees, 0)
            L = D - graph_matrix
            self.timing.end("smooth_density.heat_diffusion.build_laplacian")
            
            # Create identity matrix
            I = diags(np.ones(n_nodes), 0)
            
            # Heat diffusion update: u_{t+1} = (I - alpha*L) u_t
            diffusion_operator = I - alpha * L
            
            for _ in range(n_iterations):
                smoothed_density = diffusion_operator @ smoothed_density
            
            self.timing.end("smooth_density.heat_diffusion")
        
        elif method == 'tikhonov':
            # Tikhonov regularization using graph Laplacian
            self.timing.start("smooth_density.tikhonov")
            
            # Build the graph Laplacian
            self.timing.start("smooth_density.tikhonov.build_laplacian")
            degrees = np.array(graph_matrix.sum(axis=1)).flatten()
            D = diags(degrees, 0)
            L = D - graph_matrix
            self.timing.end("smooth_density.tikhonov.build_laplacian")
            
            # Create identity matrix
            I = diags(np.ones(n_nodes), 0)
            
            # Solve linear system: (I + lambda*L) u = raw_density
            system_matrix = I + reg_lambda * L
            
            try:
                # Solve the system
                smoothed_density = spsolve(system_matrix, raw_density)
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Error in Tikhonov smoothing - {e}")
                    print("Falling back to neighborhood averaging")
                
                # Fall back to neighborhood averaging
                return self.smooth_density(
                    graph, raw_density, method='neighborhood_avg',
                    n_iterations=n_iterations, alpha=alpha, k=k,
                    use_median=use_median
                )
            
            self.timing.end("smooth_density.tikhonov")
        
        else:
            raise ValueError(f"Unknown smoothing method: {method}")
        
        # Normalize to [0, 1] range
        min_density = np.min(smoothed_density)
        max_density = np.max(smoothed_density)
        if max_density > min_density:
            smoothed_density = (smoothed_density - min_density) / (max_density - min_density)
        
        self.timing.end("smooth_density")
        return smoothed_density
    
    def estimate_density(self, graph, method='knn_distance', smoothing='neighborhood_avg',
                        k=5, dimension=None, n_smooth_iterations=2, alpha=0.5, 
                        smoothing_k=None, reg_lambda=1.0, use_median=False):
        """
        Estimate and smooth density on the manifold graph.
        Combines raw density estimation and smoothing in one function.
        
        Parameters:
        -----------
        graph : ManifoldGraph or scipy.sparse.csr_matrix
            The graph to analyze
        method : str, default='knn_distance'
            Method for raw density estimation
        smoothing : str, default='neighborhood_avg'
            Method for density smoothing
        k : int, default=5
            Number of neighbors for density estimation
        dimension : int, optional
            Manifold dimension for scaling density
        n_smooth_iterations : int, default=2
            Number of smoothing iterations
        alpha : float, default=0.5
            Smoothing parameter
        smoothing_k : int, optional
            Number of neighbors for smoothing (defaults to k)
        reg_lambda : float, default=1.0
            Regularization parameter for Tikhonov smoothing
        use_median : bool, default=False
            If True, use median instead of mean for smoothing
            (more robust to outliers)
            
        Returns:
        --------
        dict
            Dictionary containing raw and smoothed density estimates
        """
        self.timing.start("estimate_density")
        
        # Set default smoothing_k if not provided
        if smoothing_k is None:
            smoothing_k = k
        
        # Step 1: Compute raw density
        raw_density = self.estimate_raw_density(
            graph, 
            method=method,
            k=k,
            dimension=dimension
        )
        
        # Step 2: Smooth the density
        smoothed_density = self.smooth_density(
            graph,
            raw_density,
            method=smoothing,
            n_iterations=n_smooth_iterations,
            alpha=alpha,
            k=smoothing_k,
            reg_lambda=reg_lambda,
            use_median=use_median
        )
        
        self.timing.end("estimate_density")
        
        return {
            'raw_density': raw_density,
            'smoothed_density': smoothed_density,
            'params': {
                'estimation_method': method,
                'smoothing_method': smoothing,
                'k': k,
                'dimension': dimension,
                'n_smooth_iterations': n_smooth_iterations,
                'alpha': alpha,
                'smoothing_k': smoothing_k,
                'reg_lambda': reg_lambda,
                'use_median': use_median
            }
        }
    
    def visualize_density(self, graph, density_result, embedding=None, visualizer=None,
                         colormap='plasma', point_size=5, alpha=0.7, show_colorbar=True,
                         title="Density Visualization", colorbar_label="Density",
                         show_raw=False, ax=None, fig_size=(12, 10)):
        """
        Visualize density on the graph.
        
        Parameters:
        -----------
        graph : ManifoldGraph
            Graph to visualize
        density_result : dict
            Density result from estimate_density
        embedding : numpy.ndarray, optional
            Pre-computed embedding for visualization
        visualizer : ManifoldGraphVisualizer, optional
            Visualizer to use for creating the embedding
        colormap : str, default='plasma'
            Colormap for density visualization
        point_size : float, default=5
            Size of scatter points
        alpha : float, default=0.7
            Transparency of points
        show_colorbar : bool, default=True
            Whether to show the colorbar
        title : str, default="Density Visualization"
            Plot title
        colorbar_label : str, default="Density"
            Label for the colorbar
        show_raw : bool, default=False
            If True, show raw density instead of smoothed
        ax : matplotlib.axes.Axes, optional
            Axes to plot on
        fig_size : tuple, default=(12, 10)
            Figure size
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure
        ax : matplotlib.axes.Axes
            The axes
        embedding : numpy.ndarray
            The embedding used for visualization
        """
        self.timing.start("visualize_density")
        
        # Get the density values to visualize
        density_values = density_result['raw_density'] if show_raw else density_result['smoothed_density']
        
        # If no embedding is provided, create one
        if embedding is None:
            if visualizer is None:
                try:
                    from manifold_graph_visualizer import ManifoldGraphVisualizer
                    visualizer = ManifoldGraphVisualizer(verbose=self.verbose)
                except ImportError:
                    raise ImportError("ManifoldGraphVisualizer not found. "
                                     "Please provide an embedding or a visualizer.")
            
            self.timing.start("visualize_density.create_embedding")
            embedding, _ = visualizer.create_umap_embedding(graph)
            self.timing.end("visualize_density.create_embedding")
        
        # Create visualization
        self.timing.start("visualize_density.plot")
        
        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=fig_size)
        else:
            fig = ax.figure
        
        scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=density_values,
            cmap=colormap,
            s=point_size,
            alpha=alpha,
            rasterized=True
        )
        
        ax.set_title(title)
        
        if show_colorbar:
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(colorbar_label)
        
        self.timing.end("visualize_density.plot")
        self.timing.end("visualize_density")
        
        return fig, ax, embedding
    
    
    def compute_centrality(self, graph, measure='degree', normalized=True):
        """
        Compute node centrality.
        
        Parameters:
        -----------
        graph : ManifoldGraph
            The graph to analyze
        measure : str, default='degree'
            Centrality measure to compute: 'degree', 'closeness', or 'betweenness'
        normalized : bool, default=True
            Whether to normalize centrality values
            
        Returns:
        --------
        numpy.ndarray
            Centrality values for each node
        """
        self.timing.start("compute_centrality")
        
        # Extract graph matrix if needed
        graph_matrix = graph.get_adjacency_matrix() if hasattr(graph, 'get_adjacency_matrix') else graph
        n_nodes = graph_matrix.shape[0]
        
        if measure == 'degree':
            # Compute degree centrality (number of neighbors)
            self.timing.start("compute_centrality.degree")
            if hasattr(graph, 'get_all_degrees'):
                centrality = graph.get_all_degrees()
            else:
                # Manual computation from matrix
                centrality = np.diff(graph_matrix.indptr)
            
            # Normalize if requested
            if normalized and n_nodes > 1:
                centrality = centrality / (n_nodes - 1)
            self.timing.end("compute_centrality.degree")
            
        elif measure == 'closeness':
            # Compute closeness centrality (1 / average shortest path length)
            self.timing.start("compute_centrality.closeness")
            
            # Compute shortest paths
            dist_matrix = shortest_path(graph_matrix, directed=False)
            
            # Handle disconnected components
            dist_matrix[np.isinf(dist_matrix)] = 0
            
            # For each node, compute the sum of distances to all other nodes
            total_distances = np.sum(dist_matrix, axis=1)
            
            # To avoid division by zero
            total_distances[total_distances == 0] = np.inf
            
            # Closeness centrality is 1 / average distance
            centrality = (n_nodes - 1) / total_distances
            
            # Normalize if requested
            if normalized and n_nodes > 1:
                centrality = centrality / np.max(centrality)
            self.timing.end("compute_centrality.closeness")
            
        elif measure == 'betweenness':
            # Compute betweenness centrality (fraction of shortest paths passing through a node)
            # This is an approximate implementation for large graphs
            self.timing.start("compute_centrality.betweenness")
            
            # Initialize betweenness values
            centrality = np.zeros(n_nodes)
            
            # Compute shortest paths and predecessors
            try:
                # For smaller graphs, use exact algorithm
                if n_nodes <= 1000:
                    dist_matrix, predecessors = floyd_warshall(
                        graph_matrix, directed=False, return_predecessors=True
                    )
                    
                    # Compute betweenness for each node
                    for s in range(n_nodes):
                        # Count shortest paths from s to all other nodes
                        for t in range(s + 1, n_nodes):
                            if np.isinf(dist_matrix[s, t]):
                                continue  # Skip disconnected pairs
                            
                            # Recursive function to count paths and update centrality
                            def count_paths(v, t, paths, counts):
                                if v == t:
                                    return 1.0
                                if v in paths:
                                    return paths[v]
                                
                                paths[v] = 0
                                for pred in predecessors[v]:
                                    if pred >= 0 and dist_matrix[pred, t] == dist_matrix[v, t] - graph_matrix[pred, v]:
                                        p = count_paths(pred, t, paths, counts)
                                        paths[v] += p
                                        if v != s and v != t:
                                            counts[v] += p
                                
                                return paths[v]
                            
                            paths = {}
                            counts = np.zeros(n_nodes)
                            count_paths(t, s, paths, counts)
                            centrality += counts
                else:
                    # For larger graphs, use approximate algorithm with sampling
                    samples = min(100, n_nodes)
                    sources = np.random.choice(n_nodes, size=samples, replace=False)
                    
                    for source in sources:
                        dist, pred = shortest_path(
                            graph_matrix, directed=False, indices=source, return_predecessors=True
                        )
                        
                        # Initialize dependency values
                        dependency = np.zeros(n_nodes)
                        
                        # Process nodes in order of decreasing distance from source
                        stack = [(i, d) for i, d in enumerate(dist) if not np.isinf(d) and i != source]
                        stack.sort(key=lambda x: -x[1])  # Sort by decreasing distance
                        
                        # Compute dependencies
                        for i, _ in stack:
                            if pred[i] >= 0:
                                dependency[pred[i]] += (1 + dependency[i])
                        
                        # Update centrality
                        centrality += dependency
                    
                    # Scale by the sampling factor
                    centrality *= (n_nodes / samples)
            
            except (MemoryError, RuntimeError):
                if self.verbose:
                    print("Memory error or timeout computing betweenness. Using degree centrality instead.")
                
                # Fall back to degree centrality
                centrality = np.diff(graph_matrix.indptr)
            
            # Normalize if requested
            if normalized and n_nodes > 1:
                max_val = (n_nodes - 1) * (n_nodes - 2) / 2  # Maximum possible value
                centrality = centrality / max_val
            self.timing.end("compute_centrality.betweenness")
            
        else:
            raise ValueError(f"Unknown centrality measure: {measure}. "
                           f"Choose from 'degree', 'closeness', or 'betweenness'")
        
        self.timing.end("compute_centrality")
        return centrality
    
    def compute_graph_diameter(self, graph, component_wise=False):
        """
        Compute the diameter of the graph (longest shortest path).
        
        Parameters:
        -----------
        graph : ManifoldGraph
            The graph to analyze
        component_wise : bool, default=False
            If True, compute diameter for each connected component separately
            
        Returns:
        --------
        float or dict
            Diameter of the graph, or dictionary of diameters by component
        """
        self.timing.start("compute_graph_diameter")
        
        # Extract graph matrix and component info if needed
        graph_matrix = graph.get_adjacency_matrix() if hasattr(graph, 'get_adjacency_matrix') else graph
        
        if component_wise:
            if hasattr(graph, 'get_component_labels'):
                component_labels = graph.get_component_labels()
                n_components = graph.n_components
            else:
                # Compute components
                from scipy.sparse.csgraph import connected_components
                n_components, component_labels = connected_components(graph_matrix, directed=False)
            
            # Compute diameter for each component
            diameters = {}
            
            for comp_idx in range(n_components):
                # Get nodes in this component
                comp_nodes = np.where(component_labels == comp_idx)[0]
                
                if len(comp_nodes) <= 1:
                    diameters[comp_idx] = 0.0
                    continue
                
                # Extract subgraph for this component
                subgraph = graph_matrix[comp_nodes][:, comp_nodes]
                
                # Compute shortest paths within component
                dist_matrix = shortest_path(subgraph, directed=False)
                
                # Find maximum finite distance
                finite_dists = dist_matrix[np.isfinite(dist_matrix)]
                if len(finite_dists) > 0:
                    diameters[comp_idx] = np.max(finite_dists)
                else:
                    diameters[comp_idx] = 0.0
            
            self.timing.end("compute_graph_diameter")
            return diameters
            
        else:
            # Compute shortest paths for the entire graph
            dist_matrix = shortest_path(graph_matrix, directed=False)
            
            # Find maximum finite distance
            finite_dists = dist_matrix[np.isfinite(dist_matrix)]
            if len(finite_dists) > 0:
                diameter = np.max(finite_dists)
            else:
                diameter = 0.0
            
            self.timing.end("compute_graph_diameter")
            return diameter
    
    def find_central_nodes(self, graph, k=5, measure='degree'):
        """
        Find the k most central nodes in the graph.
        
        Parameters:
        -----------
        graph : ManifoldGraph
            The graph to analyze
        k : int, default=5
            Number of central nodes to return
        measure : str, default='degree'
            Centrality measure to use
            
        Returns:
        --------
        tuple
            (indices of central nodes, centrality values)
        """
        self.timing.start("find_central_nodes")
        
        # Compute centrality
        centrality = self.compute_centrality(graph, measure=measure)
        
        # Find the k most central nodes
        if k >= len(centrality):
            indices = np.argsort(centrality)[::-1]
        else:
            indices = np.argpartition(centrality, -k)[-k:]
            indices = indices[np.argsort(centrality[indices])[::-1]]
        
        self.timing.end("find_central_nodes")
        return indices, centrality[indices]
    
    def compute_shortest_paths(self, graph, source, targets=None):
        """
        Compute shortest paths from a source node to target nodes.
        
        Parameters:
        -----------
        graph : ManifoldGraph
            The graph to analyze
        source : int
            Index of the source node
        targets : list or numpy.ndarray, optional
            Indices of target nodes. If None, compute paths to all nodes.
            
        Returns:
        --------
        dict
            Dictionary mapping target indices to (path, distance) tuples
        """
        self.timing.start("compute_shortest_paths")
        
        # Extract graph matrix if needed
        graph_matrix = graph.get_adjacency_matrix() if hasattr(graph, 'get_adjacency_matrix') else graph
        
        # Set default targets if not provided
        if targets is None:
            targets = np.arange(graph_matrix.shape[0])
            targets = targets[targets != source]  # Exclude source
        
        # Compute shortest paths and predecessors
        dist, pred = shortest_path(
            graph_matrix, directed=False, indices=source, return_predecessors=True
        )
        
        # Reconstruct paths
        paths = {}
        
        for target in targets:
            if target == source:
                paths[target] = ([source], 0.0)
                continue
                
            if np.isinf(dist[target]):
                paths[target] = (None, np.inf)  # No path exists
                continue
            
            # Reconstruct the path
            path = [target]
            current = target
            
            while current != source:
                current = pred[current]
                if current < 0:  # No path exists (should not happen given the distance check above)
                    path = None
                    break
                path.append(current)
            
            if path is not None:
                path.reverse()
                paths[target] = (path, dist[target])
            else:
                paths[target] = (None, np.inf)
        
        self.timing.end("compute_shortest_paths")
        return paths
    
    def estimate_density(self, graph, embedding, bandwidth=None, normalize=True):
        """
        Estimate node density in the embedding space.
        
        Parameters:
        -----------
        graph : ManifoldGraph
            The graph to analyze
        embedding : numpy.ndarray
            2D or 3D embedding of the graph
        bandwidth : float, optional
            Bandwidth for kernel density estimation. If None, estimated automatically.
        normalize : bool, default=True
            Whether to normalize density values to [0, 1]
            
        Returns:
        --------
        numpy.ndarray
            Density values for each node
        """
        self.timing.start("estimate_density")
        
        if embedding.shape[1] not in [2, 3]:
            raise ValueError(f"Embedding must be 2D or 3D, got {embedding.shape[1]}D")
        
        # Use Gaussian KDE to estimate density
        kde = gaussian_kde(embedding.T, bw_method=bandwidth)
        density = kde(embedding.T)
        
        # Normalize if requested
        if normalize and len(density) > 0:
            min_val, max_val = np.min(density), np.max(density)
            if max_val > min_val:
                density = (density - min_val) / (max_val - min_val)
        
        self.timing.end("estimate_density")
        return density
    
    def find_structural_bridges(self, graph, k=5):
        """
        Find structural bridges - edges whose removal would increase the number of components.
        
        Parameters:
        -----------
        graph : ManifoldGraph
            The graph to analyze
        k : int, default=5
            Number of bridge edges to return (or all if there are fewer)
            
        Returns:
        --------
        list
            List of (i, j, importance) tuples for bridge edges
        """
        self.timing.start("find_structural_bridges")
        
        # Extract graph matrix and component info
        graph_matrix = graph.get_adjacency_matrix() if hasattr(graph, 'get_adjacency_matrix') else graph
        
        if hasattr(graph, 'n_components'):
            initial_components = graph.n_components
        else:
            # Compute initial number of components
            from scipy.sparse.csgraph import connected_components
            initial_components, _ = connected_components(graph_matrix, directed=False)
        
        # Extract edges
        edges = []
        if hasattr(graph, 'get_edge_list'):
            edges = [(i, j, w) for i, j, w in graph.get_edge_list()]
        else:
            # Extract from matrix
            coo = graph_matrix.tocoo()
            for i, j, w in zip(coo.row, coo.col, coo.data):
                if i < j:  # Only include each edge once
                    edges.append((i, j, w))
        
        # Test each edge to see if it's a bridge
        bridge_edges = []
        
        for i, j, weight in edges:
            # Create a copy of the graph without this edge
            test_matrix = graph_matrix.copy()
            test_matrix[i, j] = 0
            test_matrix[j, i] = 0
            
            # Check if removing the edge increases the number of components
            from scipy.sparse.csgraph import connected_components
            n_components, _ = connected_components(test_matrix, directed=False)
            
            if n_components > initial_components:
                # This is a bridge edge - compute its importance
                importance = n_components - initial_components
                bridge_edges.append((i, j, importance))
        
        # Sort by importance and return top k
        bridge_edges.sort(key=lambda x: x[2], reverse=True)
        
        if self.verbose and len(bridge_edges) > 0:
            print(f"Found {len(bridge_edges)} bridge edges that would create new components if removed")
        
        self.timing.end("find_structural_bridges")
        return bridge_edges[:k]
    
    def compute_edge_betweenness(self, graph, normalized=True, sample_size=None):
        """
        Compute edge betweenness centrality - the fraction of shortest paths passing through each edge.
        
        Parameters:
        -----------
        graph : ManifoldGraph
            The graph to analyze
        normalized : bool, default=True
            Whether to normalize betweenness values
        sample_size : int, optional
            Number of source nodes to sample for approximation. If None, use all nodes.
            
        Returns:
        --------
        dict
            Dictionary mapping edge tuples (i, j) to betweenness values
        """
        self.timing.start("compute_edge_betweenness")
        
        # Extract graph matrix
        graph_matrix = graph.get_adjacency_matrix() if hasattr(graph, 'get_adjacency_matrix') else graph
        n_nodes = graph_matrix.shape[0]
        
        # Extract edges
        edges = {}
        if hasattr(graph, 'get_edge_list'):
            for i, j, _ in graph.get_edge_list():
                edges[(min(i, j), max(i, j))] = 0.0
        else:
            # Extract from matrix
            coo = graph_matrix.tocoo()
            for i, j in zip(coo.row, coo.col):
                if i < j:  # Only include each edge once
                    edges[(i, j)] = 0.0
        
        # Use sampling for large graphs
        if sample_size is None or sample_size >= n_nodes:
            sources = range(n_nodes)
            scale_factor = 1.0
        else:
            sources = np.random.choice(n_nodes, size=sample_size, replace=False)
            scale_factor = n_nodes / sample_size
        
        # For each source node, compute shortest paths
        for source in sources:
            # Compute shortest paths and predecessors
            dist, pred = shortest_path(
                graph_matrix, directed=False, indices=source, return_predecessors=True
            )
            
            # Count paths passing through each edge
            for target in range(n_nodes):
                if target == source or np.isinf(dist[target]):
                    continue
                
                # Reconstruct the path
                path = []
                current = target
                
                while current != source:
                    prev = pred[current]
                    if prev < 0:  # No path exists (should not happen given the distance check)
                        break
                    
                    # Add edge to path
                    edge = (min(prev, current), max(prev, current))
                    path.append(edge)
                    
                    current = prev
                
                # Increment betweenness for each edge in the path
                for edge in path:
                    if edge in edges:
                        edges[edge] += 1.0
        
        # Scale by sampling factor
        if scale_factor != 1.0:
            for edge in edges:
                edges[edge] *= scale_factor
        
        # Normalize if requested
        if normalized and n_nodes > 1:
            # Maximum possible value is the number of pairs of nodes
            max_val = n_nodes * (n_nodes - 1) / 2
            for edge in edges:
                edges[edge] /= max_val
        
        self.timing.end("compute_edge_betweenness")
        return edges
    
    def compute_component_statistics(self, graph):
        """
        Compute statistics for each component in the graph.
        
        Parameters:
        -----------
        graph : ManifoldGraph
            The graph to analyze
            
        Returns:
        --------
        dict
            Dictionary of component statistics
        """
        self.timing.start("compute_component_statistics")
        
        # Get component information
        if hasattr(graph, 'get_component_labels'):
            component_labels = graph.get_component_labels()
            n_components = graph.n_components
            component_sizes = graph.component_sizes
        else:
            # Compute components
            from scipy.sparse.csgraph import connected_components
            graph_matrix = graph if not hasattr(graph, 'get_adjacency_matrix') else graph.get_adjacency_matrix()
            n_components, component_labels = connected_components(graph_matrix, directed=False)
            component_sizes = np.bincount(component_labels)
        
        # Compute diameters
        diameters = self.compute_graph_diameter(graph, component_wise=True)
        
        # Compute degree statistics for each component
        degree_stats = {}
        graph_matrix = graph if not hasattr(graph, 'get_adjacency_matrix') else graph.get_adjacency_matrix()
        degrees = np.diff(graph_matrix.indptr)
        
        for comp_idx in range(n_components):
            comp_nodes = np.where(component_labels == comp_idx)[0]
            comp_degrees = degrees[comp_nodes]
            
            degree_stats[comp_idx] = {
                'min': comp_degrees.min(),
                'max': comp_degrees.max(),
                'mean': comp_degrees.mean(),
                'median': np.median(comp_degrees),
                'std': comp_degrees.std()
            }
        
        # Assemble results
        results = {
            'n_components': n_components,
            'component_sizes': {i: component_sizes[i] for i in range(n_components)},
            'diameters': diameters,
            'degree_stats': degree_stats
        }
        
        self.timing.end("compute_component_statistics")
        return results
    
    def find_articulation_points(self, graph, k=5):
        """
        Find articulation points - nodes whose removal would increase the number of components.
        
        Parameters:
        -----------
        graph : ManifoldGraph
            The graph to analyze
        k : int, default=5
            Number of articulation points to return (or all if there are fewer)
            
        Returns:
        --------
        list
            List of (node_idx, importance) tuples for articulation points
        """
        self.timing.start("find_articulation_points")
        
        # Extract graph matrix and component info
        graph_matrix = graph.get_adjacency_matrix() if hasattr(graph, 'get_adjacency_matrix') else graph
        n_nodes = graph_matrix.shape[0]
        
        if hasattr(graph, 'n_components'):
            initial_components = graph.n_components
        else:
            # Compute initial number of components
            from scipy.sparse.csgraph import connected_components
            initial_components, _ = connected_components(graph_matrix, directed=False)
        
        # Try a more efficient algorithm for smaller graphs
        try:
            import networkx as nx
            
            # Convert to NetworkX graph
            G = nx.from_scipy_sparse_matrix(graph_matrix)
            
            # Find articulation points
            articulation_points = list(nx.articulation_points(G))
            
            # Compute importance by actually removing each node
            results = []
            for node in articulation_points:
                # Create a copy of the graph without this node
                test_matrix = graph_matrix.copy()
                test_matrix[node, :] = 0
                test_matrix[:, node] = 0
                
                # Check how many new components are created
                from scipy.sparse.csgraph import connected_components
                n_components, _ = connected_components(test_matrix, directed=False)
                
                importance = n_components - initial_components
                results.append((node, importance))
            
            # Sort by importance
            results.sort(key=lambda x: x[1], reverse=True)
            
            if self.verbose:
                print(f"Found {len(results)} articulation points using NetworkX")
            
            self.timing.end("find_articulation_points")
            return results[:k]
            
        except ImportError:
            # NetworkX not available, use our own implementation
            pass
        
        # Test each node to see if it's an articulation point
        articulation_points = []
        
        for node in range(n_nodes):
            # Create a copy of the graph without this node
            test_matrix = graph_matrix.copy()
            test_matrix[node, :] = 0
            test_matrix[:, node] = 0
            
            # Check if removing the node increases the number of components
            from scipy.sparse.csgraph import connected_components
            n_components, _ = connected_components(test_matrix, directed=False)
            
            if n_components > initial_components:
                # This is an articulation point - compute its importance
                importance = n_components - initial_components
                articulation_points.append((node, importance))
        
        # Sort by importance and return top k
        articulation_points.sort(key=lambda x: x[1], reverse=True)
        
        if self.verbose and len(articulation_points) > 0:
            print(f"Found {len(articulation_points)} articulation points that would create new components if removed")
        
        self.timing.end("find_articulation_points")
        return articulation_points[:k]

    def estimate_raw_density(self, graph, method='knn_distance', k=5, dimension=None, 
                        return_distances=False, boundary_correction='k-1'):
        """
        Estimate raw density at each node using various methods.
        
        Parameters:
        -----------
        graph : ManifoldGraph or scipy.sparse.csr_matrix
            The graph to analyze
        method : str, default='knn_distance'
            Method to use for density estimation:
            - 'knn_distance': Uses distance to k-th nearest neighbor with proper volume normalization
            - 'avg_knn_distance': Uses average distance to k nearest neighbors
            - 'degree': Uses node degree as a proxy for density
            - 'local_reach': Uses local reachability density (inverse of avg reachability distance)
        k : int, default=5
            Number of neighbors to consider for density estimation
        dimension : int, optional
            Manifold dimension. If None, estimated from the data.
            This is used for scaling the density estimate properly.
        return_distances : bool, default=False
            If True, also return the distances used for density estimation
        boundary_correction : str, default='k-1'
            How to handle the boundary point in KNN density estimation:
            - 'none': Use k points (often overestimates density)
            - 'k-1': Use k-1 points (standard unbiased estimator)
            - 'k-1/2': Use k-1/2 points (counts boundary point as half)
            
        Returns:
        --------
        numpy.ndarray
            Raw density estimates for each node
        numpy.ndarray or None
            Distances used for estimation (if return_distances=True)
        """
        self.timing.start("estimate_raw_density")
        
        # Extract graph matrix if ManifoldGraph object
        graph_matrix = graph.get_adjacency_matrix() if hasattr(graph, 'get_adjacency_matrix') else graph
        n_nodes = graph_matrix.shape[0]
        
        # Estimate manifold dimension if not provided
        if dimension is None:
            # Simple estimate from average degree and nodes
            avg_degree = graph_matrix.sum() / n_nodes
            # Rough approximation based on geometric random graphs
            dimension = np.ceil(np.log(avg_degree) / np.log(np.log(n_nodes)))
            # Clamp to reasonable range
            dimension = max(2, min(dimension, 10))
            
            if self.verbose:
                print(f"Estimated manifold dimension: {dimension}")
        
        # Initialize arrays for density and distances
        raw_density = np.zeros(n_nodes)
        knn_distances = None if not return_distances else np.zeros((n_nodes, k))
        
        if method == 'knn_distance':
            # For each node, find distance to k-th nearest neighbor
            self.timing.start("estimate_raw_density.knn_distance")
            k_distances = np.zeros(n_nodes)
            
            for i in range(n_nodes):
                # Get neighbors and distances
                start, end = graph_matrix.indptr[i], graph_matrix.indptr[i+1]
                neighbors = graph_matrix.indices[start:end]
                distances = graph_matrix.data[start:end]
                
                # Sort by distance
                sorted_idx = np.argsort(distances)
                neighbors = neighbors[sorted_idx]
                distances = distances[sorted_idx]
                
                # Get k-th distance (or last if fewer than k neighbors)
                if len(distances) >= k:
                    k_distances[i] = distances[k-1]
                    if return_distances:
                        knn_distances[i, :k] = distances[:k]
                else:
                    # If fewer than k neighbors, use the last one
                    if len(distances) > 0:
                        k_distances[i] = distances[-1]
                        if return_distances and len(distances) > 0:
                            knn_distances[i, :len(distances)] = distances
                    else:
                        # No neighbors, use a large value
                        k_distances[i] = 1e6
            
            # Standard KNN density estimator: density ∝ (k - correction) / (n * volume)
            # where volume ∝ r^dimension and r is the distance to kth neighbor
            epsilon = 1e-10
            
            # Apply the appropriate boundary correction
            if boundary_correction == 'none':
                # No correction, use all k points
                numerator = k
            elif boundary_correction == 'k-1':
                # Standard correction, exclude the boundary point
                numerator = k - 1
            elif boundary_correction == 'k-1/2':
                # Count boundary point as half in, half out
                numerator = k - 0.5
            else:
                raise ValueError(f"Unknown boundary correction: {boundary_correction}. "
                                f"Use 'none', 'k-1', or 'k-1/2'.")
            
            # Volume of the hypersphere: V = C_d * r^d where C_d is a constant
            # We don't need the exact volume, just the relative densities,
            # so we can omit the dimension-specific constant C_d
            volumes = np.power(k_distances + epsilon, dimension)
            
            # Density = (k - correction) / (n * volume)
            # The factor n (total points) is constant, so we omit it for relative density
            raw_density = numerator / volumes
            self.timing.end("estimate_raw_density.knn_distance")
            
        elif method == 'avg_knn_distance':
            # For each node, find average distance to k nearest neighbors
            self.timing.start("estimate_raw_density.avg_knn_distance")
            avg_distances = np.zeros(n_nodes)
            
            for i in range(n_nodes):
                # Get neighbors and distances
                start, end = graph_matrix.indptr[i], graph_matrix.indptr[i+1]
                neighbors = graph_matrix.indices[start:end]
                distances = graph_matrix.data[start:end]
                
                # Sort by distance
                sorted_idx = np.argsort(distances)
                neighbors = neighbors[sorted_idx]
                distances = distances[sorted_idx]
                
                # Calculate average of k nearest (or all if fewer than k)
                if len(distances) > 0:
                    count = min(k, len(distances))
                    avg_distances[i] = np.mean(distances[:count])
                    if return_distances:
                        knn_distances[i, :count] = distances[:count]
                else:
                    avg_distances[i] = 1e6  # No neighbors
            
            # Density is inversely proportional to distance^dimension
            epsilon = 1e-10
            raw_density = 1.0 / np.power(avg_distances + epsilon, dimension)
            self.timing.end("estimate_raw_density.avg_knn_distance")
            
        elif method == 'degree':
            # Use degree as a proxy for density
            self.timing.start("estimate_raw_density.degree")
            degrees = np.diff(graph_matrix.indptr)
            
            # Simple normalization: degree / max_degree
            max_degree = np.max(degrees)
            if max_degree > 0:
                raw_density = degrees / max_degree
            else:
                raw_density = np.zeros(n_nodes)
                
            # No distances to return in this method
            if return_distances:
                knn_distances = None
            self.timing.end("estimate_raw_density.degree")
            
        elif method == 'local_reach':
            # Local reachability density (similar to OPTICS/LOF)
            self.timing.start("estimate_raw_density.local_reach")
            reach_distances = np.zeros(n_nodes)
            
            # First compute core distances (distance to k-th neighbor)
            core_distances = np.zeros(n_nodes)
            for i in range(n_nodes):
                start, end = graph_matrix.indptr[i], graph_matrix.indptr[i+1]
                distances = graph_matrix.data[start:end]
                
                if len(distances) >= k:
                    core_distances[i] = np.sort(distances)[k-1]
                elif len(distances) > 0:
                    core_distances[i] = np.max(distances)
                else:
                    core_distances[i] = 1e6
            
            # Then compute reachability distances
            for i in range(n_nodes):
                start, end = graph_matrix.indptr[i], graph_matrix.indptr[i+1]
                neighbors = graph_matrix.indices[start:end]
                distances = graph_matrix.data[start:end]
                
                # Sort by distance
                sorted_idx = np.argsort(distances)
                neighbors = neighbors[sorted_idx]
                distances = distances[sorted_idx]
                
                # Take at most k neighbors
                neighbors = neighbors[:min(k, len(neighbors))]
                distances = distances[:min(k, len(distances))]
                
                if len(distances) > 0:
                    # Compute reachability distances
                    reach_dists = np.maximum(distances, core_distances[neighbors])
                    reach_distances[i] = np.mean(reach_dists)
                    
                    if return_distances:
                        knn_distances[i, :len(distances)] = reach_dists
                else:
                    reach_distances[i] = 1e6  # No neighbors
            
            # Density is inverse of reachability distance
            epsilon = 1e-10
            raw_density = 1.0 / np.power(reach_distances + epsilon, dimension)
            self.timing.end("estimate_raw_density.local_reach")
        
        else:
            raise ValueError(f"Unknown density estimation method: {method}")
        
        # Normalize to [0, 1] range
        min_density = np.min(raw_density)
        max_density = np.max(raw_density)
        if max_density > min_density:
            raw_density = (raw_density - min_density) / (max_density - min_density)
        
        self.timing.end("estimate_raw_density")
        
        if return_distances:
            return raw_density, knn_distances
        else:
            return raw_density
    
    def smooth_density(self, graph, raw_density, method='neighborhood_avg', 
                      n_iterations=2, alpha=0.5, k=5, reg_lambda=1.0, use_median=False):
        """
        Smooth raw density estimates over the graph.
        
        Parameters:
        -----------
        graph : ManifoldGraph or scipy.sparse.csr_matrix
            The graph to analyze
        raw_density : numpy.ndarray
            Raw density estimates for each node
        method : str, default='neighborhood_avg'
            Method to use for smoothing:
            - 'neighborhood_avg': Simple average over neighborhood
            - 'gaussian': Gaussian kernel weighting by distance
            - 'heat_diffusion': Heat diffusion smoothing (controlled by alpha)
            - 'tikhonov': Tikhonov regularization (graph Laplacian)
        n_iterations : int, default=2
            Number of smoothing iterations
        alpha : float, default=0.5
            Smoothing parameter:
            - For 'neighborhood_avg': Weight of neighbors vs. current node
            - For 'heat_diffusion': Diffusion rate
            - Not used for other methods
        k : int, default=5
            Number of neighbors to consider for neighborhood averaging
        reg_lambda : float, default=1.0
            Regularization parameter for Tikhonov method
        use_median : bool, default=False
            If True, use median instead of mean for neighborhood methods
            (more robust to outliers)
            
        Returns:
        --------
        numpy.ndarray
            Smoothed density estimates
        """
        self.timing.start("smooth_density")
        
        # Extract graph matrix if ManifoldGraph object
        graph_matrix = graph.get_adjacency_matrix() if hasattr(graph, 'get_adjacency_matrix') else graph
        n_nodes = graph_matrix.shape[0]
        
        # Initialize with raw density
        smoothed_density = raw_density.copy()
        
        if method == 'neighborhood_avg':
            # Simple weighted average over neighborhood
            self.timing.start("smooth_density.neighborhood_avg")
            
            for _ in range(n_iterations):
                next_density = smoothed_density.copy()
                
                for i in range(n_nodes):
                    # Get neighbors and distances
                    start, end = graph_matrix.indptr[i], graph_matrix.indptr[i+1]
                    neighbors = graph_matrix.indices[start:end]
                    
                    # Take top k neighbors (if we have that many)
                    if len(neighbors) > k:
                        # Sort by density (to favor high-density regions)
                        sorted_idx = np.argsort(-smoothed_density[neighbors])
                        neighbors = neighbors[sorted_idx[:k]]
                    
                    if len(neighbors) > 0:
                        # Compute weighted average or median
                        if use_median:
                            # Use median (more robust to outliers)
                            neighbor_val = np.median(smoothed_density[neighbors])
                        else:
                            # Use mean
                            neighbor_val = np.mean(smoothed_density[neighbors])
                            
                        # Apply smoothing
                        next_density[i] = (1-alpha) * smoothed_density[i] + alpha * neighbor_val
                
                smoothed_density = next_density
            
            self.timing.end("smooth_density.neighborhood_avg")
        
        elif method == 'gaussian':
            # Gaussian kernel weighting by distance
            self.timing.start("smooth_density.gaussian")
            
            for _ in range(n_iterations):
                next_density = np.zeros_like(smoothed_density)
                
                for i in range(n_nodes):
                    # Get neighbors and distances
                    start, end = graph_matrix.indptr[i], graph_matrix.indptr[i+1]
                    neighbors = graph_matrix.indices[start:end]
                    distances = graph_matrix.data[start:end]
                    
                    if len(neighbors) > 0:
                        # Include self in computation
                        all_neighbors = np.append(neighbors, i)
                        all_distances = np.append(distances, 0.0)
                        all_densities = smoothed_density[all_neighbors]
                        
                        # Calculate distance scale (adaptive bandwidth)
                        # Use median distance for robustness
                        bandwidth = np.median(distances) if len(distances) > 0 else 1.0
                        
                        # Compute Gaussian weights
                        weights = np.exp(-0.5 * (all_distances/bandwidth)**2)
                        weights /= np.sum(weights)
                        
                        # Weighted combination
                        if use_median:
                            # Approximation of weighted median via sampling
                            # Sort by density values
                            sorted_idx = np.argsort(all_densities)
                            sorted_weights = weights[sorted_idx]
                            sorted_densities = all_densities[sorted_idx]
                            
                            # Compute cumulative weights
                            cum_weights = np.cumsum(sorted_weights)
                            
                            # Find median index (first weight >= 0.5)
                            median_idx = np.searchsorted(cum_weights, 0.5)
                            if median_idx >= len(sorted_densities):
                                median_idx = len(sorted_densities) - 1
                                
                            next_density[i] = sorted_densities[median_idx]
                        else:
                            # Weighted average (mean)
                            next_density[i] = np.sum(weights * all_densities)
                    else:
                        # No neighbors, keep original value
                        next_density[i] = smoothed_density[i]
                
                smoothed_density = next_density
            
            self.timing.end("smooth_density.gaussian")
        
        elif method == 'heat_diffusion':
            # Heat diffusion smoothing
            self.timing.start("smooth_density.heat_diffusion")
            
            # Build the graph Laplacian
            # L = D - A where D is degree matrix and A is adjacency
            self.timing.start("smooth_density.heat_diffusion.build_laplacian")
            degrees = np.array(graph_matrix.sum(axis=1)).flatten()
            D = diags(degrees, 0)
            L = D - graph_matrix
            self.timing.end("smooth_density.heat_diffusion.build_laplacian")
            
            # Create identity matrix
            I = diags(np.ones(n_nodes), 0)
            
            # Heat diffusion update: u_{t+1} = (I - alpha*L) u_t
            diffusion_operator = I - alpha * L
            
            for _ in range(n_iterations):
                smoothed_density = diffusion_operator @ smoothed_density
            
            self.timing.end("smooth_density.heat_diffusion")
        
        elif method == 'tikhonov':
            # Tikhonov regularization using graph Laplacian
            self.timing.start("smooth_density.tikhonov")
            
            # Build the graph Laplacian
            self.timing.start("smooth_density.tikhonov.build_laplacian")
            degrees = np.array(graph_matrix.sum(axis=1)).flatten()
            D = diags(degrees, 0)
            L = D - graph_matrix
            self.timing.end("smooth_density.tikhonov.build_laplacian")
            
            # Create identity matrix
            I = diags(np.ones(n_nodes), 0)
            
            # Solve linear system: (I + lambda*L) u = raw_density
            system_matrix = I + reg_lambda * L
            
            try:
                # Solve the system
                smoothed_density = spsolve(system_matrix, raw_density)
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Error in Tikhonov smoothing - {e}")
                    print("Falling back to neighborhood averaging")
                
                # Fall back to neighborhood averaging
                return self.smooth_density(
                    graph, raw_density, method='neighborhood_avg',
                    n_iterations=n_iterations, alpha=alpha, k=k,
                    use_median=use_median
                )
            
            self.timing.end("smooth_density.tikhonov")
        
        else:
            raise ValueError(f"Unknown smoothing method: {method}")
        
        # Normalize to [0, 1] range
        min_density = np.min(smoothed_density)
        max_density = np.max(smoothed_density)
        if max_density > min_density:
            smoothed_density = (smoothed_density - min_density) / (max_density - min_density)
        
        self.timing.end("smooth_density")
        return smoothed_density
    
    def estimate_density(self, graph, method='knn_distance', smoothing='neighborhood_avg',
                        k=5, dimension=None, n_smooth_iterations=2, alpha=0.5, 
                        smoothing_k=None, reg_lambda=1.0, use_median=False,
                        boundary_correction='k-1'):
        """
        Estimate and smooth density on the manifold graph.
        Combines raw density estimation and smoothing in one function.
        
        Parameters:
        -----------
        graph : ManifoldGraph or scipy.sparse.csr_matrix
            The graph to analyze
        method : str, default='knn_distance'
            Method for raw density estimation
        smoothing : str, default='neighborhood_avg'
            Method for density smoothing
        k : int, default=5
            Number of neighbors for density estimation
        dimension : int, optional
            Manifold dimension for scaling density
        n_smooth_iterations : int, default=2
            Number of smoothing iterations
        alpha : float, default=0.5
            Smoothing parameter
        smoothing_k : int, optional
            Number of neighbors for smoothing (defaults to k)
        reg_lambda : float, default=1.0
            Regularization parameter for Tikhonov smoothing
        use_median : bool, default=False
            If True, use median instead of mean for smoothing
            (more robust to outliers)
        boundary_correction : str, default='k-1'
            How to handle the boundary point in KNN density estimation:
            - 'none': Use k points (often overestimates density)
            - 'k-1': Use k-1 points (standard unbiased estimator)
            - 'k-1/2': Use k-1/2 points (counts boundary point as half)
            
        Returns:
        --------
        dict
            Dictionary containing raw and smoothed density estimates
        """
        self.timing.start("estimate_density")
        
        # Set default smoothing_k if not provided
        if smoothing_k is None:
            smoothing_k = k
        
        # Step 1: Compute raw density
        raw_density = self.estimate_raw_density(
            graph, 
            method=method,
            k=k,
            dimension=dimension,
            boundary_correction=boundary_correction
        )
        
        # Step 2: Smooth the density
        smoothed_density = self.smooth_density(
            graph,
            raw_density,
            method=smoothing,
            n_iterations=n_smooth_iterations,
            alpha=alpha,
            k=smoothing_k,
            reg_lambda=reg_lambda,
            use_median=use_median
        )
        
        self.timing.end("estimate_density")
        
        return {
            'raw_density': raw_density,
            'smoothed_density': smoothed_density,
            'params': {
                'estimation_method': method,
                'smoothing_method': smoothing,
                'k': k,
                'dimension': dimension,
                'n_smooth_iterations': n_smooth_iterations,
                'alpha': alpha,
                'smoothing_k': smoothing_k,
                'reg_lambda': reg_lambda,
                'use_median': use_median,
                'boundary_correction': boundary_correction
            }
        }
    
    def visualize_density(self, graph, density_result, embedding=None, visualizer=None,
                         colormap='plasma', point_size=5, alpha=0.7, show_colorbar=True,
                         title="Density Visualization", colorbar_label="Density",
                         show_raw=False, ax=None, fig_size=(12, 10)):
        """
        Visualize density on the graph.
        
        Parameters:
        -----------
        graph : ManifoldGraph
            Graph to visualize
        density_result : dict
            Density result from estimate_density
        embedding : numpy.ndarray, optional
            Pre-computed embedding for visualization
        visualizer : ManifoldGraphVisualizer, optional
            Visualizer to use for creating the embedding
        colormap : str, default='plasma'
            Colormap for density visualization
        point_size : float, default=5
            Size of scatter points
        alpha : float, default=0.7
            Transparency of points
        show_colorbar : bool, default=True
            Whether to show the colorbar
        title : str, default="Density Visualization"
            Plot title
        colorbar_label : str, default="Density"
            Label for the colorbar
        show_raw : bool, default=False
            If True, show raw density instead of smoothed
        ax : matplotlib.axes.Axes, optional
            Axes to plot on
        fig_size : tuple, default=(12, 10)
            Figure size
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure
        ax : matplotlib.axes.Axes
            The axes
        embedding : numpy.ndarray
            The embedding used for visualization
        """
        self.timing.start("visualize_density")
        
        # Get the density values to visualize
        density_values = density_result['raw_density'] if show_raw else density_result['smoothed_density']
        
        # If no embedding is provided, create one
        if embedding is None:
            if visualizer is None:
                try:
                    from manifold_graph_visualizer import ManifoldGraphVisualizer
                    visualizer = ManifoldGraphVisualizer(verbose=self.verbose)
                except ImportError:
                    raise ImportError("ManifoldGraphVisualizer not found. "
                                     "Please provide an embedding or a visualizer.")
            
            self.timing.start("visualize_density.create_embedding")
            embedding, _ = visualizer.create_umap_embedding(graph)
            self.timing.end("visualize_density.create_embedding")
        
        # Create visualization
        self.timing.start("visualize_density.plot")
        
        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=fig_size)
        else:
            fig = ax.figure
        
        scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=density_values,
            cmap=colormap,
            s=point_size,
            alpha=alpha,
            rasterized=True
        )
        
        ax.set_title(title)
        
        if show_colorbar:
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(colorbar_label)
        
        self.timing.end("visualize_density.plot")
        self.timing.end("visualize_density")
        
        return fig, ax, embedding
"""
# Median-based smoothing example
density_results['knn_median'] = analyzer.estimate_density(
    graph,
    method='knn_distance',
    smoothing='neighborhood_avg',
    k=5,
    n_smooth_iterations=2,
    alpha=0.5,
    use_median=True  # Use median instead of mean
)
"""