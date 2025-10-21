"""
DataGraphVisualizer - Visualization tools for DataGraph objects.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from .core_utilities import TimingStats

# --- Lazy loaders for heavy/optional deps -------------------------------------
import contextlib
import importlib
import builtins

@contextlib.contextmanager
def _block_tensorflow_import():
    """Temporarily raise ImportError for any tensorflow* import.

    This lets `umap` import without pulling in its parametric (TF) module.
    """
    real_import = builtins.__import__
    def _hook(name, *a, **kw):
        if name.startswith("tensorflow"):
            raise ImportError("TF intentionally blocked during UMAP import")
        return real_import(name, *a, **kw)
    builtins.__import__ = _hook
    try:
        yield
    finally:
        builtins.__import__ = real_import

def _load_umap_primitives():
    """Import UMAP primitives without initializing TensorFlow."""
    with _block_tensorflow_import():
        umap_umap = importlib.import_module("umap.umap_")
    # Return what we need (no top-level `import umap`)
    return (
        umap_umap.UMAP,
        umap_umap.fuzzy_simplicial_set,
        umap_umap.simplicial_set_embedding,
        umap_umap.find_ab_params,
    )

def _load_matplotlib():
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    return plt, LineCollection

def _load_plotly():
    import plotly.graph_objs as go
    return go

def _row_medians_from_csr(A):
    A = A.tocsr()
    n = A.shape[0]
    med = np.zeros(n, dtype=np.float64)
    for i in range(n):
        s, e = A.indptr[i], A.indptr[i+1]
        if e > s:
            med[i] = np.median(A.data[s:e])
        else:
            med[i] = 1.0
    return med

def distances_to_similarities(A, mode="exp"):
    """Convert distance-like CSR to similarity CSR, row-wise scaled; returns row-normalized P."""
    A = A.tocsr().copy()
    if mode == "exp":
        tau = _row_medians_from_csr(A) + 1e-9
        for i in range(A.shape[0]):
            s, e = A.indptr[i], A.indptr[i+1]
            if e > s:
                A.data[s:e] = np.exp(-A.data[s:e] / tau[i])
    elif mode == "recip":
        A.data = 1.0 / (A.data + 1e-9)
    else:
        raise ValueError("Unknown mode")

    # zero diagonal if any
    A.setdiag(0.0); A.eliminate_zeros()
    # Row normalize
    row_sums = np.array(A.sum(axis=1)).ravel() + 1e-12
    Dinv = sp.diags(1.0 / row_sums)
    P = Dinv @ A
    return P  # row-stochastic

def sample_landmarks_knn_blue_noise(A, quota, min_hops=2, degree_bias=0.5, seed=42, per_component_labels=None):
    """
    Blue-noise-ish landmark sampling with δ-hop blocking on a KNN graph.
    quota: int or array per-component; if int, interpreted as global count.
    per_component_labels: optional array of component ids (len n)
    """
    rng = np.random.default_rng(seed)
    A = A.tocsr()
    n = A.shape[0]
    deg = np.diff(A.indptr)

    # handle per-component quotas
    if per_component_labels is None:
        comps = np.zeros(n, dtype=np.int32)
        comp_ids = [0]
    else:
        comps = per_component_labels.astype(np.int32)
        comp_ids = np.unique(comps)

    selected = []
    blocked = np.zeros(n, dtype=bool)

    for c in comp_ids:
        idx = np.where(comps == c)[0]
        if idx.size == 0:
            continue
        if np.isscalar(quota):
            s_c = max(int(np.ceil(quota * idx.size)) if quota < 1.0 else min(quota, idx.size), 1)
        else:
            s_c = int(quota[c])

        # seed: highest-degree node in the component
        i0 = idx[np.argmax(deg[idx])]
        selected.append(i0); blocked[i0] = True

        # block δ hops around a node
        def block_hops(src):
            frontier = [src]
            vis = set([src])
            for _ in range(min_hops):
                nxt = []
                for u in frontier:
                    s, e = A.indptr[u], A.indptr[u+1]
                    for v in A.indices[s:e]:
                        if v not in vis:
                            vis.add(v); nxt.append(v)
                frontier = nxt
                if not frontier: break
            blocked[list(vis)] = True

        block_hops(i0)

        # candidate weights (degree^bias) within component
        cand = idx.copy()
        w = deg[cand] ** degree_bias
        w = w / (w.sum() + 1e-12)

        while len([j for j in selected if comps[j]==c]) < s_c:
            # resample until we hit an unblocked candidate
            for _ in range(1000):
                cand_j = rng.choice(cand, p=w)
                if not blocked[cand_j]:
                    break
            selected.append(cand_j)
            block_hops(cand_j)

    return np.array(selected, dtype=np.int64)

def harmonic_interpolate(P, landmarks, Y_land, max_iter=200, tol=1e-3, alpha=1.0, init=None):
    """
    Jacobi-style harmonic extension on row-stochastic P.
    landmarks: array of landmark indices
    Y_land: (s, d) embedding of landmarks
    Returns Y (n, d) with Y[landmarks]=Y_land.
    """
    n = P.shape[0]
    d = Y_land.shape[1]
    Y = np.zeros((n, d), dtype=np.float32) if init is None else init.astype(np.float32, copy=True)

    # set landmarks
    Y[landmarks] = Y_land

    land_mask = np.zeros(n, dtype=bool)
    land_mask[landmarks] = True
    U = np.where(~land_mask)[0]

    # one sensible init: one-pass barycentric using P restricted to landmarks
    if init is None:
        # P_L: keep only columns of landmarks
        P_L = P[:, landmarks]
        denom = (np.array(P_L.sum(axis=1)).ravel() + 1e-12)
        Y = (P_L @ Y_land) / denom[:, None]
        Y[landmarks] = Y_land

    # iterate
    for _ in range(max_iter):
        Y_old = Y.copy()
        # only update unlabeled rows
        Y[U] = (1.0 - alpha) * Y[U] + alpha * (P[U, :] @ Y)
        # keep landmarks pinned
        Y[landmarks] = Y_land
        rel = np.linalg.norm(Y - Y_old) / (np.linalg.norm(Y_old) + 1e-9)
        if rel < tol:
            break
    return Y

def umap_via_landmarks_and_interpolation(graph_csr, n_landmarks=30000, ratio=None,
                                         min_hops=2, n_neighbors_umap=15, min_dist=0.1,
                                         sim_mode="exp", interp="harmonic",
                                         max_iter=150, tol=1e-3, random_state=42,
                                         n_components=2, visualizer=None):
    """
    End-to-end: sample landmarks, run UMAP on landmark subgraph, interpolate rest.
    """
    A = graph_csr.tocsr()
    n = A.shape[0]

    # components (for quotas)
    from scipy.sparse.csgraph import connected_components
    n_comp, comp_labels = connected_components(A, directed=False)

    # decide count
    if ratio is not None:
        nL = max(int(ratio * n), 10)
    else:
        nL = min(n_landmarks, n)

    # per-component proportional quota
    sizes = np.bincount(comp_labels)
    quota = np.maximum((sizes / sizes.sum()) * nL, np.minimum(10, nL)).astype(int)

    landmarks = sample_landmarks_knn_blue_noise(A, quota=quota, min_hops=min_hops,
                                                degree_bias=0.5, seed=random_state,
                                                per_component_labels=comp_labels)

    # induced landmark subgraph (optionally densify)
    rows, cols = [], []
    data = []
    A = A.tocsr()
    land_set = set(landmarks.tolist())
    for i in landmarks:
        s, e = A.indptr[i], A.indptr[i+1]
        nbr = A.indices[s:e]; w = A.data[s:e]
        mask = np.isin(nbr, landmarks)
        rows.extend([np.where(landmarks==i)[0][0]] * int(mask.sum()))
        cols.extend(np.searchsorted(landmarks, np.sort(landmarks[np.searchsorted(landmarks, nbr[mask])])))
        data.extend(w[mask])
    A_L = csr_matrix((np.array(data), (np.array(rows), np.array(cols))),
                     shape=(len(landmarks), len(landmarks)))
    # light densification: add top-κ mutual neighbors among landmarks if too sparse (optional)

    # run UMAP on landmarks
    if visualizer is None:
        UMAP, fss, sse, find_ab = _load_umap_primitives()
        emb_L, info = DataGraphVisualizer(verbose=False).create_umap_embedding(
            A_L, n_neighbors=n_neighbors_umap, min_dist=min_dist,
            n_components=n_components, random_state=random_state
        )
    else:
        emb_L, _ = visualizer.create_umap_embedding(
            A_L, n_neighbors=n_neighbors_umap, min_dist=min_dist,
            n_components=n_components, random_state=random_state
        )

    # build row-stochastic similarity on full graph
    P = distances_to_similarities(graph_csr, mode=sim_mode)

    # interpolate
    if interp == "barycentric":
        P_L = P[:, landmarks]
        denom = (np.array(P_L.sum(axis=1)).ravel() + 1e-12)
        Y = (P_L @ emb_L) / denom[:, None]
        Y[landmarks] = emb_L
    else:
        Y = harmonic_interpolate(P, landmarks, emb_L, max_iter=max_iter, tol=tol, alpha=1.0)

    return Y, {"landmarks": landmarks, "embedding_landmarks": emb_L}


class DataGraphVisualizer:
    """
    Class for visualizing ManifoldGraph objects with UMAP and other techniques.
    Supports both 2D and 3D visualizations with static and interactive options.
    """
    
    def __init__(self, verbose=True):
        """
        Initialize the visualizer.
        
        Parameters:
        -----------
        verbose : bool, default=True
            Whether to print progress messages
        """
        self.verbose = verbose
        self.timing = TimingStats()
        
    def prepare_graph_for_umap(self, graph, n_neighbors=15):
        """
        Create optimized dummy clique with minimal connections and zero weights.
        
        Parameters:
        -----------
        graph : scipy.sparse.csr_matrix
            The graph to prepare
        n_neighbors : int, default=15
            Target minimum number of neighbors for each node
            
        Returns:
        --------
        scipy.sparse.csr_matrix
            Prepared graph with dummy nodes if needed
        dict
            Information about dummy nodes
        """
        self.timing.start("prepare_graph_for_umap")
        
        n = graph.shape[0]
        degrees = np.diff(graph.indptr)
        low = np.where(degrees < n_neighbors)[0]
        
        if low.size == 0:
            self.timing.end("prepare_graph_for_umap")
            return graph, {
                'has_dummies': False,
                'dummy_indices': np.array([], dtype=np.int32),
                'original_to_dummies': {},
                'n_original': n,
                'n_dummies': 0
            }
        
        # Calculate optimal dummy count
        deficits = n_neighbors - degrees[low]
        max_deficit = int(deficits.max())
        dummy_for_dummies = max(0, n_neighbors - low.size + 1) 
        optimal_dummy_count = max(max_deficit, dummy_for_dummies)
        
        if self.verbose:
            print(f"  {len(low)} low-degree nodes with max deficit {max_deficit}")
            print(f"  Creating optimal clique of {optimal_dummy_count} dummy nodes")
        
        # Create indices
        dummy_indices = np.arange(n, n + optimal_dummy_count, dtype=np.int32)
        
        # Create dummy clique (all-to-all connections except self)
        self.timing.start("prepare_graph_for_umap.create_dummy_clique")
        d_row, d_col = np.meshgrid(dummy_indices, dummy_indices)
        mask = d_row != d_col  # Exclude self-connections
        clique_rows = d_row[mask]
        clique_cols = d_col[mask]
        clique_data = np.zeros_like(clique_rows, dtype=graph.dtype)  # Zero weights
        self.timing.end("prepare_graph_for_umap.create_dummy_clique")
        
        # Create connections between low-degree nodes and ONLY the needed dummy nodes
        self.timing.start("prepare_graph_for_umap.connect_low_degree")
        cross_rows = []
        cross_cols = []
        original_to_dummies = {}
        
        for i, node in enumerate(low):
            # Only connect to as many dummies as needed
            needed = deficits[i]
            if needed > 0:
                # Take just the needed number of dummies (first ones for simplicity)
                node_dummies = dummy_indices[:needed]
                original_to_dummies[int(node)] = node_dummies
                
                # Add these connections
                cross_rows.extend([node] * len(node_dummies))
                cross_cols.extend(node_dummies)
        
        cross_data = np.zeros(len(cross_rows), dtype=graph.dtype)  # Zero weights
        
        # Get reciprocal connections (dummy → node)
        recip_rows = cross_cols
        recip_cols = cross_rows
        recip_data = np.zeros_like(recip_rows, dtype=graph.dtype)  # Zero weights
        self.timing.end("prepare_graph_for_umap.connect_low_degree")
        
        # Combine all connections for efficiency
        self.timing.start("prepare_graph_for_umap.build_result_matrix")
        all_rows = np.concatenate([graph.tocoo().row, clique_rows, cross_rows, recip_rows])
        all_cols = np.concatenate([graph.tocoo().col, clique_cols, cross_cols, recip_cols])
        all_data = np.concatenate([graph.tocoo().data, clique_data, cross_data, recip_data])
        
        # Create the result matrix in one operation
        result = csr_matrix(
            (all_data, (all_rows, all_cols)), 
            shape=(n + optimal_dummy_count, n + optimal_dummy_count)
        )
        self.timing.end("prepare_graph_for_umap.build_result_matrix")
        
        # Create dummy information
        dummy_info = {
            'has_dummies': True,
            'dummy_indices': dummy_indices,
            'original_to_dummies': original_to_dummies,
            'n_original': n,
            'n_dummies': optimal_dummy_count
        }
        
        self.timing.end("prepare_graph_for_umap")
        return result, dummy_info
        
    def create_umap_embedding(self, graph, means=None, sigmas=None, n_neighbors=15, min_dist=0.1, 
                             random_state=42, n_components=2, init='spectral', update_node_df=False):
        """
        Create a UMAP embedding of the graph.
        
        Parameters:
        -----------
        graph : ManifoldGraph or scipy.sparse.csr_matrix
            The graph to embed
        means : numpy.ndarray, optional
            Mean vectors for initialization (not used for distance computation)
        sigmas : numpy.ndarray, optional
            Standard deviation vectors (not used in this method)
        n_neighbors : int, default=15
            Number of neighbors for UMAP
        min_dist : float, default=0.1
            Minimum distance parameter for UMAP
        random_state : int, default=42
            Random seed for reproducibility
        n_components : int, default=2
            Number of dimensions for the embedding (set to 3 for 3D visualizations)
        init : str, default='spectral'
            Initialization method for UMAP
        update_node_df : bool, default=False
            Whether to update the graph's node_df with UMAP coordinates
            
        Returns:
        --------
        numpy.ndarray
            UMAP embedding
        dict
            Information about the embedding process including dummy nodes if added
        """
        self.timing.start("create_umap_embedding")
        if self.verbose:
            print(f"[UMAP] Loading umap primitives...")
        UMAP, fuzzy_simplicial_set, simplicial_set_embedding, find_ab_params = _load_umap_primitives()
        
        if self.verbose:
            print(f"[UMAP] Creating {n_components}D UMAP embedding")
        
        # Extract graph matrix if a ManifoldGraph object is provided
        if hasattr(graph, 'get_adjacency_matrix'):
            graph_matrix = graph.get_adjacency_matrix()
        else:
            graph_matrix = graph
            
        # Ensure graph is in CSR format
        graph_matrix = graph_matrix.tocsr()
        
        # 1) Pad low-degree nodes and get dummy info
        degrees = np.diff(graph_matrix.indptr)
        has_low_degree = degrees.min() < n_neighbors
        
        if has_low_degree:
            self.timing.start("create_umap_embedding.prepare_graph")
            if self.verbose:
                print(f"  Padding {np.sum(degrees < n_neighbors)} low-degree nodes → {n_neighbors} neighbors")
            graph_matrix, dummy_info = self.prepare_graph_for_umap(graph_matrix, n_neighbors)
            self.timing.end("create_umap_embedding.prepare_graph")
        else:
            # No dummy nodes needed
            dummy_info = {
                'has_dummies': False,
                'dummy_indices': np.array([], dtype=np.int32),
                'original_to_dummies': {},
                'n_original': graph_matrix.shape[0],
                'n_dummies': 0
            }
        
        n_padded = graph_matrix.shape[0]
        n_orig = dummy_info['n_original']
        
        if self.verbose:
            print(f"  Graph shape after padding: {n_padded} nodes "
                  f"({dummy_info['n_dummies']} dummy nodes)")
        
        # 2) Build true k-NN lists from the sparse graph
        self.timing.start("create_umap_embedding.build_knn_lists")
        knn_indices = np.zeros((n_padded, n_neighbors), dtype=np.int32)
        knn_dists = np.zeros((n_padded, n_neighbors), dtype=np.float64)
        
        for i in range(n_padded):
            start, end = graph_matrix.indptr[i], graph_matrix.indptr[i+1]
            nbrs = list(zip(graph_matrix.indices[start:end], graph_matrix.data[start:end]))
            nbrs.sort(key=lambda x: x[1])
            
            if len(nbrs) < n_neighbors:
                pad_val = nbrs[-1][1] * 2 if nbrs else 1.0
                nbrs += [(i, pad_val)] * (n_neighbors - len(nbrs))
            
            top = nbrs[:n_neighbors]
            knn_indices[i] = [j for j, _ in top]
            knn_dists[i] = [w for _, w in top]
        self.timing.end("create_umap_embedding.build_knn_lists")
        
        # 3) Instantiate UMAP to grab hyperparams (but don't fit)
        um = UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            spread=1.0,
            metric='euclidean',
            init=init,
            random_state=random_state,
            n_components=n_components,
            verbose=False
        )
        
        # 4) Compute a,b from spread & min_dist
        a, b = find_ab_params(um.spread, min_dist)
        
        # 5) Build fuzzy simplicial set
        self.timing.start("create_umap_embedding.fuzzy_simplicial_set")
        # Create a dummy matrix for X - we're not using it for distances
        if means is not None and means.shape[0] >= n_orig:
            # If means are available, use them to pad the dummy nodes
            dummy_X = np.zeros((n_padded, means.shape[1]))
            dummy_X[:n_orig] = means[:n_orig]
        else:
            # Otherwise just use an empty matrix of the right shape
            dummy_X = np.zeros((n_padded, n_components))
            
        graph_simp, sigmas_out, rhos_out = fuzzy_simplicial_set(
            X=dummy_X,
            n_neighbors=n_neighbors,
            random_state=um.random_state,
            metric=um.metric,
            metric_kwds=getattr(um, 'metric_kwds', {}),
            knn_indices=knn_indices,
            knn_dists=knn_dists,
            set_op_mix_ratio=getattr(um, 'set_op_mix_ratio', 1.0),
            local_connectivity=getattr(um, 'local_connectivity', 1.0)
        )
        self.timing.end("create_umap_embedding.fuzzy_simplicial_set")
        
        # 6) Run the embedding with floats for a,b
        rng = np.random.RandomState(um.random_state)
        
        # Setup initialization
        self.timing.start("create_umap_embedding.init")
        if init == 'spectral' and means is not None and n_components <= means.shape[1]:
            # Use the first n_components of means as initialization
            embedding = np.zeros((n_padded, n_components))
            embedding[:n_orig] = means[:n_orig, :n_components]
            
            # Initialize dummy nodes with random positions near the center
            if dummy_info['has_dummies']:
                center = np.mean(embedding[:n_orig], axis=0)
                spread = np.std(embedding[:n_orig], axis=0) * 0.1
                embedding[n_orig:] = center + rng.normal(0, spread, size=(n_padded - n_orig, n_components))
        else:
            # Let UMAP handle initialization
            embedding = None
        self.timing.end("create_umap_embedding.init")
        
        # Run the embedding
        self.timing.start("create_umap_embedding.simplicial_set_embedding")
        result = simplicial_set_embedding(
            data=None,
            graph=graph_simp,
            n_components=n_components,
            initial_alpha=um.learning_rate,
            a=a,
            b=b,
            gamma=um.repulsion_strength,
            negative_sample_rate=um.negative_sample_rate,
            n_epochs=um.n_epochs,
            init=embedding if embedding is not None else init,
            random_state=rng,
            metric=um.metric,
            metric_kwds=getattr(um, 'metric_kwds', {}),
            densmap=um.densmap,
            densmap_kwds=getattr(um, 'densmap_kwds', {}),
            output_dens=um.output_dens,
            verbose=False
        )
        self.timing.end("create_umap_embedding.simplicial_set_embedding")
        
        # Unpack in case it's a tuple
        embedding = result[0] if isinstance(result, tuple) else result
        
        # Create results dictionary
        results = {
            'full_embedding': embedding,
            'original_embedding': embedding[:n_orig],
            'dummy_info': dummy_info,
            'parameters': {
                'n_neighbors': n_neighbors,
                'min_dist': min_dist,
                'n_components': n_components,
                'a': a,
                'b': b
            }
        }
        
        if self.verbose:
            print(f"  UMAP embedding created with shape {embedding.shape}")
            
        # Update the graph's node_df if requested and if it's a proper ManifoldGraph
        if update_node_df and hasattr(graph, 'node_df'):
            self.add_umap_to_graph(graph, embedding[:n_orig], n_components)
            if self.verbose:
                print(f"  Added UMAP_{n_components}D coordinates to graph's node_df")
                
        self.timing.end("create_umap_embedding")
        return embedding[:n_orig], results
    
    def add_umap_to_graph(self, graph, embedding, n_components=None):
        """
        Add UMAP coordinates to the graph's node dataframe.
        
        Parameters:
        -----------
        graph : ManifoldGraph
            The graph to update
        embedding : numpy.ndarray
            UMAP embedding array (n_nodes x n_dimensions)
        n_components : int, optional
            Number of dimensions in the embedding.
            If None, it will be inferred from the embedding shape.
            
        Returns:
        --------
        ManifoldGraph
            The updated graph instance
        """
        if not hasattr(graph, 'node_df'):
            if self.verbose:
                print("  Warning: Graph does not have a node_df attribute. UMAP coordinates not added.")
            return graph
        
        if n_components is None:
            n_components = embedding.shape[1]
            
        # Add prefix based on dimensionality
        prefix = f"UMAP_{n_components}D"
        
        # Add coordinates to node_df
        for i in range(min(n_components, embedding.shape[1])):
            col_name = f"{prefix}_{i+1}"
            graph.node_df[col_name] = embedding[:, i]
            
        return graph
        
    def plot_embedding(self, embedding, color_by=None, edge_list=None, edge_weights=None,
                      ax=None, point_size=5, alpha=0.7, edge_alpha=0.3, max_edges=1000, 
                      edge_weight_scale=0.5, cmap=None, title=None, show_colorbar=True,
                      colorbar_label=None, fig_size=(12, 10), edge_color='gray',
                      include_dummy_nodes=False, dummy_info=None, dummy_alpha=0.2, 
                      dummy_size=2, dummy_color='gray', highlight_indices=None,
                      highlight_color='red', highlight_size=10, return_fig=True):
        """
        Plot a 2D embedding with customization options.
        
        Parameters:
        -----------
        embedding : numpy.ndarray
            The embedding to plot (shape should be (n_samples, 2))
        color_by : array-like, optional
            Values to color nodes by (defaults to component labels)
        edge_list : list of tuples, optional
            List of (i, j) edge tuples to draw
        edge_weights : array-like, optional
            Weights for edges (used for line thickness)
        ax : matplotlib.axes.Axes, optional
            Axes to plot on, if None, create a new figure
        point_size : float, default=5
            Size of scatter points
        alpha : float, default=0.7
            Transparency of points
        edge_alpha : float, default=0.3
            Transparency of edges
        max_edges : int, default=1000
            Maximum number of edges to draw (to avoid overcrowding)
        edge_weight_scale : float, default=0.5
            Scale factor for edge widths
        cmap : str or matplotlib.colors.Colormap, optional
            Colormap to use
        title : str, optional
            Plot title
        show_colorbar : bool, default=True
            Whether to show a colorbar
        colorbar_label : str, optional
            Label for the colorbar
        fig_size : tuple, default=(12, 10)
            Figure size if creating a new figure
        edge_color : str or tuple, default='gray'
            Color for edges
        include_dummy_nodes : bool, default=False
            Whether to include dummy nodes in the plot
        dummy_info : dict, optional
            Information about dummy nodes
        dummy_alpha : float, default=0.2
            Transparency for dummy nodes
        dummy_size : float, default=2
            Size for dummy nodes
        dummy_color : str or tuple, default='gray'
            Color for dummy nodes
        highlight_indices : array-like, optional
            Indices of points to highlight
        highlight_color : str or tuple, default='red'
            Color for highlighted points
        highlight_size : float, default=10
            Size for highlighted points
        return_fig : bool, default=True
            Whether to return the figure object
            
        Returns:
        --------
        fig : matplotlib.figure.Figure or None
            The figure if return_fig is True
        ax : matplotlib.axes.Axes
            The axes object
        """
        self.timing.start("plot_embedding")
        plt, LineCollection = _load_matplotlib()
        
        # Validate embedding dimensions
        if embedding.shape[1] != 2:
            raise ValueError(f"Expected 2D embedding, got shape {embedding.shape}. For 3D embeddings, use plot_3d_embedding.")
        
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=fig_size)
        else:
            fig = ax.figure
        
        # Set default title
        if title is None:
            title = "Embedding Visualization"
        
        # Extract full embedding if dummy info is provided and we want to include dummies
        full_embedding = embedding
        n_orig = embedding.shape[0]
        
        if include_dummy_nodes and dummy_info is not None and dummy_info['has_dummies']:
            if 'full_embedding' in dummy_info and dummy_info['full_embedding'] is not None:
                full_embedding = dummy_info['full_embedding']
                n_orig = dummy_info['n_original']
            else:
                # Just use the provided embedding
                include_dummy_nodes = False
        
        # Determine coloring
        if color_by is None:
            if hasattr(embedding, 'get_component_labels'):
                # If embedding is a ManifoldGraph object
                color_values = embedding.get_component_labels()
            else:
                # Create a single color for all points
                color_values = np.zeros(n_orig)
        else:
            color_values = color_by
        
        # Choose colormap based on data type
        if cmap is None:
            vals = np.unique(color_values)
            if len(vals) <= 20 and np.issubdtype(color_values.dtype, np.integer):
                # Discrete data - use categorical colormap
                cmap = plt.cm.tab10 if len(vals) <= 10 else plt.cm.tab20
            else:
                # Continuous data - use sequential colormap
                cmap = plt.cm.viridis
            
        if colorbar_label is None:
            if len(np.unique(color_values)) <= 20 and np.issubdtype(color_values.dtype, np.integer):
                colorbar_label = "Component"
            else:
                colorbar_label = "Value"
        
        # Plot original nodes
        scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=color_values,
            cmap=cmap,
            s=point_size,
            alpha=alpha,
            rasterized=True
        )
        
        # Plot edges if provided
        if edge_list is not None:
            self.timing.start("plot_embedding.draw_edges")
            
            # Limit number of edges to draw
            if len(edge_list) > max_edges:
                # If edge weights are provided, select the strongest edges
                if edge_weights is not None:
                    strongest_idx = np.argsort(edge_weights)[-max_edges:]
                    edge_list = [edge_list[i] for i in strongest_idx]
                    edge_weights = edge_weights[strongest_idx]
                else:
                    # Otherwise randomly sample edges
                    idx = np.random.choice(len(edge_list), max_edges, replace=False)
                    edge_list = [edge_list[i] for i in idx]
            
            # Prepare line segments
            segments = []
            for i, j in edge_list:
                if i < n_orig and j < n_orig:  # Only include edges between original nodes
                    segments.append([(embedding[i, 0], embedding[i, 1]), 
                                    (embedding[j, 0], embedding[j, 1])])
            
            # If edge weights are provided, scale line widths
            if edge_weights is not None and len(edge_weights) == len(segments):
                # Normalize weights for line width
                min_width, max_width = 0.1, 2.0
                if len(edge_weights) > 0:
                    norm_weights = edge_weight_scale * (edge_weights - np.min(edge_weights)) / \
                                (np.max(edge_weights) - np.min(edge_weights) + 1e-10)
                    line_widths = min_width + norm_weights * (max_width - min_width)
                else:
                    line_widths = np.ones(len(segments)) * min_width
            else:
                line_widths = np.ones(len(segments)) * 0.5
            
            # Create line collection
            edge_collection = LineCollection(
                segments, 
                linewidths=line_widths,
                colors=edge_color,
                alpha=edge_alpha,
                zorder=1  # Draw edges below points
            )
            ax.add_collection(edge_collection)
            self.timing.end("plot_embedding.draw_edges")
        
        # Optionally plot dummy nodes
        if include_dummy_nodes and dummy_info is not None and dummy_info['has_dummies']:
            self.timing.start("plot_embedding.draw_dummies")
            dummy_indices = dummy_info['dummy_indices']
            if len(dummy_indices) > 0 and full_embedding.shape[0] > n_orig:
                dummy_positions = full_embedding[dummy_indices]
                ax.scatter(
                    dummy_positions[:, 0],
                    dummy_positions[:, 1],
                    c=dummy_color,
                    s=dummy_size,
                    alpha=dummy_alpha,
                    marker='.',
                    rasterized=True
                )
            self.timing.end("plot_embedding.draw_dummies")
        
        # Highlight specific points if requested
        if highlight_indices is not None and len(highlight_indices) > 0:
            self.timing.start("plot_embedding.highlight_points")
            highlight_indices = np.asarray(highlight_indices)
            highlight_indices = highlight_indices[highlight_indices < n_orig]  # Ensure indices are valid
            
            if len(highlight_indices) > 0:
                ax.scatter(
                    embedding[highlight_indices, 0],
                    embedding[highlight_indices, 1],
                    c=highlight_color,
                    s=highlight_size,
                    marker='o',
                    edgecolors='k',
                    linewidths=1,
                    zorder=10  # Draw highlighted points on top
                )
            self.timing.end("plot_embedding.highlight_points")
        
        # Set title and axes
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        
        # Add colorbar if requested
        if show_colorbar:
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(colorbar_label)
            
            # For discrete component data, adjust tick labels
            vals = np.unique(color_values)
            if len(vals) <= 20 and np.issubdtype(color_values.dtype, np.integer):
                cbar.set_ticks(np.arange(len(vals)))
                if len(vals) <= 10:  # Only show labels for a reasonable number of components
                    cbar.set_ticklabels([f"{i}" for i in range(len(vals))])
        
        # Ensure tight layout
        plt.tight_layout()
        
        self.timing.end("plot_embedding")
        
        if return_fig:
            return fig, ax
        else:
            return ax
            
    def plot_3d_embedding(self, embedding, color_by=None, edge_list=None, edge_weights=None,
                         point_size=5, opacity=0.7, edge_opacity=0.3, max_edges=1000,
                         edge_width=1, colorscale='Viridis', title="3D Embedding Visualization",
                         show_colorbar=True, colorbar_label=None, width=1000, height=800,
                         include_dummy_nodes=False, dummy_info=None, dummy_opacity=0.2,
                         dummy_size=2, dummy_color='gray', highlight_indices=None,
                         highlight_color='red', highlight_size=10, hover_data=None,
                         hover_template=None, filename=None):
        """
        Create an interactive 3D embedding visualization using Plotly.
        
        Parameters:
        -----------
        embedding : numpy.ndarray
            The embedding to plot (shape should be (n_samples, 3))
        color_by : array-like, optional
            Values to color nodes by (defaults to component labels)
        edge_list : list of tuples, optional
            List of (i, j) edge tuples to draw
        edge_weights : array-like, optional
            Weights for edges (used for line thickness)
        point_size : float, default=5
            Size of scatter points
        opacity : float, default=0.7
            Transparency of points
        edge_opacity : float, default=0.3
            Transparency of edges
        max_edges : int, default=1000
            Maximum number of edges to draw (to avoid overcrowding)
        edge_width : float, default=1
            Width for edge lines
        colorscale : str, default='Viridis'
            Plotly colorscale to use
        title : str, default="3D Embedding Visualization"
            Plot title
        show_colorbar : bool, default=True
            Whether to show a colorbar
        colorbar_label : str, optional
            Label for the colorbar
        width : int, default=1000
            Width of the plot in pixels
        height : int, default=800
            Height of the plot in pixels
        include_dummy_nodes : bool, default=False
            Whether to include dummy nodes in the plot
        dummy_info : dict, optional
            Information about dummy nodes
        dummy_opacity : float, default=0.2
            Transparency for dummy nodes
        dummy_size : float, default=2
            Size for dummy nodes
        dummy_color : str, default='gray'
            Color for dummy nodes
        highlight_indices : array-like, optional
            Indices of points to highlight
        highlight_color : str, default='red'
            Color for highlighted points
        highlight_size : float, default=10
            Size for highlighted points
        hover_data : dict or list, optional
            Additional data for hover information
        hover_template : str, optional
            Custom hover template
        filename : str, optional
            If provided, save the plot to this file
            
        Returns:
        --------
        plotly.graph_objs.Figure
            The interactive figure object
        """
        self.timing.start("plot_3d_embedding")
        go = _load_plotly()
        
        # Validate embedding dimensions
        if embedding.shape[1] != 3:
            raise ValueError(f"Expected 3D embedding, got shape {embedding.shape}. For 2D embeddings, use plot_embedding.")
        
        # Extract full embedding if dummy info is provided and we want to include dummies
        full_embedding = embedding
        n_orig = embedding.shape[0]
        
        if include_dummy_nodes and dummy_info is not None and dummy_info['has_dummies']:
            if 'full_embedding' in dummy_info and dummy_info['full_embedding'] is not None:
                full_embedding = dummy_info['full_embedding']
                n_orig = dummy_info['n_original']
            else:
                # Just use the provided embedding
                include_dummy_nodes = False
        
        # Determine coloring
        if color_by is None:
            if hasattr(embedding, 'get_component_labels'):
                # If embedding is a ManifoldGraph object
                color_values = embedding.get_component_labels()
            else:
                # Create a single color for all points
                color_values = np.zeros(n_orig)
        else:
            color_values = color_by
            
        if colorbar_label is None:
            if len(np.unique(color_values)) <= 20 and np.issubdtype(color_values.dtype, np.integer):
                colorbar_label = "Component"
            else:
                colorbar_label = "Value"
        
        # Create figure
        fig = go.Figure()
        
        # Create hover text
        if hover_data is None:
            if hover_template is None:
                # Default hover template
                hover_template = (
                    f"{colorbar_label}: %{{marker.color}}<br>"
                    "x: %{x}<br>"
                    "y: %{y}<br>"
                    "z: %{z}<extra></extra>"
                )
            hover_text = color_values
        else:
            hover_text = [f"{k}: {v}" for k, v in hover_data.items()]
        
        # Add scatter trace for original nodes
        scatter = go.Scatter3d(
            x=embedding[:, 0],
            y=embedding[:, 1],
            z=embedding[:, 2],
            mode='markers',
            marker=dict(
                size=point_size,
                color=color_values,
                colorscale=colorscale,
                opacity=opacity,
                colorbar=dict(
                    title=colorbar_label,
                    titleside='right'
                ),
                showscale=show_colorbar
            ),
            text=hover_text,
            hovertemplate=hover_template
        )
        fig.add_trace(scatter)
        
        # Add edges if provided
        if edge_list is not None:
            self.timing.start("plot_3d_embedding.draw_edges")
            
            # Limit number of edges to draw
            if len(edge_list) > max_edges:
                # If edge weights are provided, select the strongest edges
                if edge_weights is not None:
                    strongest_idx = np.argsort(edge_weights)[-max_edges:]
                    edge_list = [edge_list[i] for i in strongest_idx]
                    edge_weights = edge_weights[strongest_idx]
                else:
                    # Otherwise randomly sample edges
                    idx = np.random.choice(len(edge_list), max_edges, replace=False)
                    edge_list = [edge_list[i] for i in idx]
            
            # Draw edges as scatter lines
            edge_x = []
            edge_y = []
            edge_z = []
            
            for i, j in edge_list:
                if i < n_orig and j < n_orig:  # Only include edges between original nodes
                    # Add each line as a separate segment with None values between
                    edge_x.extend([embedding[i, 0], embedding[j, 0], None])
                    edge_y.extend([embedding[i, 1], embedding[j, 1], None])
                    edge_z.extend([embedding[i, 2], embedding[j, 2], None])
            
            # Add edges trace
            edges_trace = go.Scatter3d(
                x=edge_x,
                y=edge_y,
                z=edge_z,
                mode='lines',
                line=dict(
                    color='gray',
                    width=edge_width
                ),
                opacity=edge_opacity,
                hoverinfo='none'
            )
            fig.add_trace(edges_trace)
            self.timing.end("plot_3d_embedding.draw_edges")
        
        # Add dummy nodes if requested
        if include_dummy_nodes and dummy_info is not None and dummy_info['has_dummies']:
            self.timing.start("plot_3d_embedding.draw_dummies")
            dummy_indices = dummy_info['dummy_indices']
            if len(dummy_indices) > 0 and full_embedding.shape[0] > n_orig:
                dummy_positions = full_embedding[dummy_indices]
                
                dummy_trace = go.Scatter3d(
                    x=dummy_positions[:, 0],
                    y=dummy_positions[:, 1],
                    z=dummy_positions[:, 2],
                    mode='markers',
                    marker=dict(
                        size=dummy_size,
                        color=dummy_color,
                        opacity=dummy_opacity
                    ),
                    hoverinfo='none',
                    showlegend=False
                )
                fig.add_trace(dummy_trace)
            self.timing.end("plot_3d_embedding.draw_dummies")
        
        # Highlight specific points if requested
        if highlight_indices is not None and len(highlight_indices) > 0:
            self.timing.start("plot_3d_embedding.highlight_points")
            highlight_indices = np.asarray(highlight_indices)
            highlight_indices = highlight_indices[highlight_indices < n_orig]  # Ensure indices are valid
            
            if len(highlight_indices) > 0:
                highlight_trace = go.Scatter3d(
                    x=embedding[highlight_indices, 0],
                    y=embedding[highlight_indices, 1],
                    z=embedding[highlight_indices, 2],
                    mode='markers',
                    marker=dict(
                        size=highlight_size,
                        color=highlight_color,
                        line=dict(
                            width=1,
                            color='black'
                        )
                    ),
                    name='Highlighted',
                    showlegend=True
                )
                fig.add_trace(highlight_trace)
            self.timing.end("plot_3d_embedding.highlight_points")
        
        # Update layout
        fig.update_layout(
            title=title,
            width=width,
            height=height,
            scene=dict(
                xaxis_title="UMAP 1",
                yaxis_title="UMAP 2",
                zaxis_title="UMAP 3",
                aspectmode='cube'
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            hovermode='closest'
        )
        
        # Save to file if requested
        if filename is not None:
            fig.write_html(filename)
            if self.verbose:
                print(f"Interactive 3D plot saved to: {filename}")
        
        self.timing.end("plot_3d_embedding")
        return fig
        
    def plot_graph(self, graph, means=None, sigmas=None, n_neighbors=15, min_dist=0.1,
                  color_by=None, embedding=None, umap_results=None, point_size=5, 
                  alpha=0.7, edge_alpha=0.3, max_edges=1000, title=None, 
                  show_colorbar=True, fig_size=(12, 10), show_dummy_nodes=False,
                  dummy_alpha=0.2, dummy_size=2, show_edges=True, 
                  highlight_indices=None, highlight_color='red', highlight_size=10,
                  n_components=2, update_node_df=False):
        """
        Create a complete visualization of a graph, including UMAP embedding if not provided.
        
        Parameters:
        -----------
        graph : ManifoldGraph
            The graph to visualize
        means : numpy.ndarray, optional
            Mean vectors for initialization (not used for distance computation)
        sigmas : numpy.ndarray, optional
            Standard deviation vectors (not used in this method)
        n_neighbors : int, default=15
            Number of neighbors for UMAP
        min_dist : float, default=0.1
            Minimum distance parameter for UMAP
        color_by : array-like, optional
            Values to color nodes by (defaults to component labels)
        embedding : numpy.ndarray, optional
            Pre-computed embedding to use instead of generating one
        umap_results : dict, optional
            Results from a previous call to create_umap_embedding
        point_size : float, default=5
            Size of scatter points
        alpha : float, default=0.7
            Transparency of points
        edge_alpha : float, default=0.3
            Transparency of edges
        max_edges : int, default=1000
            Maximum number of edges to draw
        title : str, optional
            Plot title
        show_colorbar : bool, default=True
            Whether to show a colorbar
        fig_size : tuple, default=(12, 10)
            Figure size
        show_dummy_nodes : bool, default=False
            Whether to show dummy nodes in the plot
        dummy_alpha : float, default=0.2
            Transparency for dummy nodes
        dummy_size : float, default=2
            Size for dummy nodes
        show_edges : bool, default=True
            Whether to show edges in the plot
        highlight_indices : array-like, optional
            Indices of points to highlight
        highlight_color : str, default='red'
            Color for highlighted points
        highlight_size : float, default=10
            Size for highlighted points
        n_components : int, default=2
            Dimensionality of the embedding (2 or 3)
            
        Returns:
        --------
        fig : matplotlib.figure.Figure or plotly.graph_objs.Figure
            The figure object
        ax : matplotlib.axes.Axes or None
            The axes object (for 2D plots only)
        dict
            Results including the embedding if generated
        """
        self.timing.start("plot_graph")
        
        # Validate dimensionality
        if n_components not in [2, 3]:
            raise ValueError("n_components must be either 2 or 3")
        
        # Extract matrix and information if ManifoldGraph object
        graph_matrix = graph.get_adjacency_matrix() if hasattr(graph, 'get_adjacency_matrix') else graph
        
        # Get component labels for coloring if not provided
        if color_by is None and hasattr(graph, 'get_component_labels'):
            color_by = graph.get_component_labels()
        
        # Generate an embedding if not provided
        if embedding is None and umap_results is None:
            self.timing.start("plot_graph.create_embedding")
            embedding, umap_results = self.create_umap_embedding(
                graph, means, sigmas, n_neighbors, min_dist, n_components=n_components,
                update_node_df=update_node_df
            )
            self.timing.end("plot_graph.create_embedding")
        elif umap_results is not None and embedding is None:
            # Extract embedding from results
            embedding = umap_results['original_embedding']
        
        # Extract edges for visualization
        edge_list = None
        edge_weights = None
        
        if show_edges:
            self.timing.start("plot_graph.extract_edges")
            if hasattr(graph, 'get_edge_list'):
                # Use get_edge_list method if available
                edges_with_weights = graph.get_edge_list()
                edge_list = [(i, j) for i, j, w in edges_with_weights]
                edge_weights = np.array([w for i, j, w in edges_with_weights])
            else:
                # Extract from matrix
                coo = graph_matrix.tocoo()
                edges = {}
                
                for i, j, w in zip(coo.row, coo.col, coo.data):
                    if i < j:  # Only include each edge once
                        edges[(i, j)] = w
                
                edge_list = list(edges.keys())
                edge_weights = np.array(list(edges.values()))
            self.timing.end("plot_graph.extract_edges")
        
        # Determine title if not provided
        if title is None:
            if hasattr(graph, 'n_nodes') and hasattr(graph, 'n_components'):
                title = f"Graph with {graph.n_nodes} nodes, {len(edge_list) if edge_list else 0} edges, {graph.n_components} components"
            else:
                title = f"{n_components}D Graph Visualization"
        
        # Plot the embedding based on dimensionality
        self.timing.start("plot_graph.plot_embedding")
        
        if n_components == 2:
            # 2D visualization with matplotlib
            fig, ax = self.plot_embedding(
                embedding=embedding,
                color_by=color_by,
                edge_list=edge_list,
                edge_weights=edge_weights,
                point_size=point_size,
                alpha=alpha,
                edge_alpha=edge_alpha,
                max_edges=max_edges,
                title=title,
                show_colorbar=show_colorbar,
                fig_size=fig_size,
                include_dummy_nodes=show_dummy_nodes,
                dummy_info=umap_results['dummy_info'] if umap_results else None,
                dummy_alpha=dummy_alpha,
                dummy_size=dummy_size,
                highlight_indices=highlight_indices,
                highlight_color=highlight_color,
                highlight_size=highlight_size
            )
        else:
            # 3D interactive visualization with plotly
            fig = self.plot_3d_embedding(
                embedding=embedding,
                color_by=color_by,
                edge_list=edge_list,
                edge_weights=edge_weights,
                point_size=point_size,
                opacity=alpha,
                edge_opacity=edge_alpha,
                max_edges=max_edges,
                title=title,
                show_colorbar=show_colorbar,
                include_dummy_nodes=show_dummy_nodes,
                dummy_info=umap_results['dummy_info'] if umap_results else None,
                dummy_opacity=dummy_alpha,
                dummy_size=dummy_size,
                highlight_indices=highlight_indices,
                highlight_color=highlight_color,
                highlight_size=highlight_size
            )
            ax = None
            
        self.timing.end("plot_graph.plot_embedding")
        
        self.timing.end("plot_graph")
        
        # Return results
        return fig, ax, {
            'embedding': embedding,
            'umap_results': umap_results
        }
    
    def plot_3d_graph(self, graph, means=None, sigmas=None, n_neighbors=15, min_dist=0.1,
                     color_by=None, embedding=None, umap_results=None, point_size=5,
                     opacity=0.7, edge_opacity=0.3, max_edges=1000, title=None,
                     show_colorbar=True, width=1000, height=800, show_dummy_nodes=False,
                     dummy_opacity=0.2, dummy_size=2, show_edges=True,
                     highlight_indices=None, highlight_color='red', highlight_size=10,
                     hover_data=None, hover_template=None, filename=None, update_node_df=False):
        """
        Create a complete 3D interactive visualization of a graph.
        
        Parameters:
        -----------
        graph : ManifoldGraph
            The graph to visualize
        means : numpy.ndarray, optional
            Mean vectors for initialization (not used for distance computation)
        sigmas : numpy.ndarray, optional
            Standard deviation vectors (not used in this method)
        n_neighbors : int, default=15
            Number of neighbors for UMAP
        min_dist : float, default=0.1
            Minimum distance parameter for UMAP
        color_by : array-like, optional
            Values to color nodes by (defaults to component labels)
        embedding : numpy.ndarray, optional
            Pre-computed embedding to use instead of generating one
        umap_results : dict, optional
            Results from a previous call to create_umap_embedding
        point_size : float, default=5
            Size of scatter points
        opacity : float, default=0.7
            Transparency of points
        edge_opacity : float, default=0.3
            Transparency of edges
        max_edges : int, default=1000
            Maximum number of edges to draw
        title : str, optional
            Plot title
        show_colorbar : bool, default=True
            Whether to show a colorbar
        width : int, default=1000
            Width of the plot in pixels
        height : int, default=800
            Height of the plot in pixels
        show_dummy_nodes : bool, default=False
            Whether to show dummy nodes in the plot
        dummy_opacity : float, default=0.2
            Transparency for dummy nodes
        dummy_size : float, default=2
            Size for dummy nodes
        show_edges : bool, default=True
            Whether to show edges in the plot
        highlight_indices : array-like, optional
            Indices of points to highlight
        highlight_color : str, default='red'
            Color for highlighted points
        highlight_size : float, default=10
            Size for highlighted points
        hover_data : dict or list, optional
            Additional data for hover information
        hover_template : str, optional
            Custom hover template
        filename : str, optional
            If provided, save the plot to this file
            
        Returns:
        --------
        plotly.graph_objs.Figure
            The interactive figure object
        dict
            Results including the embedding if generated
        """
        # This is essentially a wrapper around plot_graph with n_components=3
        fig, _, results = self.plot_graph(
            graph=graph,
            means=means,
            sigmas=sigmas,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            color_by=color_by,
            embedding=embedding,
            umap_results=umap_results,
            point_size=point_size,
            alpha=opacity,
            edge_alpha=edge_opacity,
            max_edges=max_edges,
            title=title,
            show_colorbar=show_colorbar,
            show_dummy_nodes=show_dummy_nodes,
            dummy_alpha=dummy_opacity,
            dummy_size=dummy_size,
            show_edges=show_edges,
            highlight_indices=highlight_indices,
            highlight_color=highlight_color,
            highlight_size=highlight_size,
            n_components=3,
            update_node_df=update_node_df
        )
        
        # Set specific 3D properties and save if requested
        if filename is not None:
            fig.write_html(filename)
            if self.verbose:
                print(f"Interactive 3D graph visualization saved to: {filename}")
        
        return fig, results
    
    def create_embeddings_2d_and_3d(self, graph, means=None, sigmas=None, n_neighbors=15, min_dist=0.1,
                              random_state=42, init='spectral', update_node_df=True):
        """
        Create both 2D and 3D UMAP embeddings for a graph and optionally update its node_df.
        
        Parameters:
        -----------
        graph : ManifoldGraph
            The graph to embed
        means : numpy.ndarray, optional
            Mean vectors for initialization (not used for distance computation)
        sigmas : numpy.ndarray, optional
            Standard deviation vectors (not used in this method)
        n_neighbors : int, default=15
            Number of neighbors for UMAP
        min_dist : float, default=0.1
            Minimum distance parameter for UMAP
        random_state : int, default=42
            Random seed for reproducibility
        init : str, default='spectral'
            Initialization method for UMAP
        update_node_df : bool, default=True
            Whether to update the graph's node_df with both 2D and 3D UMAP coordinates
            
        Returns:
        --------
        tuple
            (embedding_2d, results_2d, embedding_3d, results_3d)
        """
        self.timing.start("create_embeddings_2d_and_3d")
        
        if self.verbose:
            print("[UMAP] Creating both 2D and 3D embeddings")
        
        # Create 2D embedding
        embedding_2d, results_2d = self.create_umap_embedding(
            graph, means, sigmas, n_neighbors, min_dist, 
            random_state=random_state, n_components=2, init=init,
            update_node_df=update_node_df
        )
        
        # Create 3D embedding
        embedding_3d, results_3d = self.create_umap_embedding(
            graph, means, sigmas, n_neighbors, min_dist, 
            random_state=random_state, n_components=3, init=init,
            update_node_df=update_node_df
        )
        
        self.timing.end("create_embeddings_2d_and_3d")
        return embedding_2d, results_2d, embedding_3d, results_3d
    
    def visualize_components(self, graph, embedding=None, umap_results=None, 
                            max_components=9, fig_size=(15, 12), component_indices=None,
                            alpha=0.7, highlight_alpha=1.0, point_size=5, highlight_size=8,
                            show_all=True, title=None, n_components=2):
        """
        Create a grid of plots showing each component highlighted.
        
        Parameters:
        -----------
        graph : ManifoldGraph
            The graph to visualize
        embedding : numpy.ndarray, optional
            Pre-computed embedding to use
        umap_results : dict, optional
            Results from a previous call to create_umap_embedding
        max_components : int, default=9
            Maximum number of components to show
        fig_size : tuple, default=(15, 12)
            Figure size
        component_indices : list, optional
            Specific component indices to show (overrides max_components)
        alpha : float, default=0.7
            Transparency of background points
        highlight_alpha : float, default=1.0
            Transparency of highlighted points
        point_size : float, default=5
            Size of background points
        highlight_size : float, default=8
            Size of highlighted points
        show_all : bool, default=True
            Whether to include a plot showing all components
        title : str, optional
            Main figure title
        n_components : int, default=2
            Dimensionality of the embedding (2 or 3)
            
        Returns:
        --------
        fig : matplotlib.figure.Figure or list of plotly.graph_objs.Figure
            The figure object(s)
        """
        self.timing.start("visualize_components")
        
        # Validate dimensionality
        if n_components not in [2, 3]:
            raise ValueError("n_components must be either 2 or 3")
        
        # Get component information
        if hasattr(graph, 'get_component_labels'):
            component_labels = graph.get_component_labels()
            n_components_graph = graph.n_components
            component_sizes = graph.component_sizes
        else:
            # Compute components if not available
            n_components_graph, component_labels = connected_components(graph, directed=False)
            component_sizes = np.bincount(component_labels)
        
        # Generate an embedding if not provided
        if embedding is None and umap_results is None:
            self.timing.start("visualize_components.create_embedding")
            embedding, umap_results = self.create_umap_embedding(
                graph, n_components=n_components
            )
            self.timing.end("visualize_components.create_embedding")
        elif umap_results is not None and embedding is None:
            # Extract embedding from results
            embedding = umap_results['original_embedding']
        
        # Determine which components to show
        if component_indices is None:
            # Show the largest components
            sorted_indices = np.argsort(component_sizes)[::-1]
            component_indices = sorted_indices[:max_components]
        else:
            max_components = len(component_indices)
        
        # For 2D embeddings, use matplotlib grid
        if n_components == 2:
            # Calculate grid dimensions
            if show_all:
                n_plots = max_components + 1
            else:
                n_plots = max_components
                
            n_cols = min(3, n_plots)
            n_rows = (n_plots + n_cols - 1) // n_cols
            
            # Create figure
            fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size)
            if n_rows == 1 and n_cols == 1:
                axes = np.array([axes])
            axes = axes.flatten()
            
            # Set title
            if title is None:
                title = f"Component Visualization ({n_components_graph} components)"
            fig.suptitle(title, fontsize=16)
            
            # Plot overview if requested
            plot_idx = 0
            if show_all:
                self.timing.start("visualize_components.plot_all")
                self.plot_embedding(
                    embedding=embedding,
                    color_by=component_labels,
                    ax=axes[plot_idx],
                    point_size=point_size,
                    alpha=alpha,
                    title="All Components",
                    show_colorbar=True,
                    return_fig=False
                )
                plot_idx += 1
                self.timing.end("visualize_components.plot_all")
            
            # Plot individual components
            self.timing.start("visualize_components.plot_individual")
            for i, comp_idx in enumerate(component_indices):
                if plot_idx >= len(axes):
                    break
                    
                # Get nodes in this component
                comp_mask = component_labels == comp_idx
                comp_nodes = np.where(comp_mask)[0]
                
                # Create a mask for coloring
                color_mask = np.zeros_like(component_labels)
                color_mask[comp_nodes] = 1
                    
                # Plot with highlighting
                self.plot_embedding(
                    embedding=embedding,
                    color_by=None,  # Use a single color for all points
                    ax=axes[plot_idx],
                    point_size=point_size,
                    alpha=0.2,  # Low alpha for background
                    title=f"Component {comp_idx} ({len(comp_nodes)} nodes)",
                    show_colorbar=False,
                    return_fig=False,
                    highlight_indices=comp_nodes,
                    highlight_color='tab:blue',
                    highlight_size=highlight_size
                )
                
                plot_idx += 1
            self.timing.end("visualize_components.plot_individual")
            
            # Hide any unused axes
            for i in range(plot_idx, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)  # Make room for the title
            
            self.timing.end("visualize_components")
            return fig
            
        else:
            # For 3D embeddings, create separate interactive Plotly figures
            figures = []
            
            # Create "all components" plot if requested
            if show_all:
                self.timing.start("visualize_components.plot_all_3d")
                all_fig = self.plot_3d_embedding(
                    embedding=embedding,
                    color_by=component_labels,
                    point_size=point_size,
                    opacity=alpha,
                    title="All Components",
                    show_colorbar=True
                )
                figures.append(all_fig)
                self.timing.end("visualize_components.plot_all_3d")
            
            # Plot individual components
            self.timing.start("visualize_components.plot_individual_3d")
            for comp_idx in component_indices:
                # Get nodes in this component
                comp_mask = component_labels == comp_idx
                comp_nodes = np.where(comp_mask)[0]
                
                # Create a figure highlighting this component
                comp_fig = go.Figure()
                
                # Add background points (all nodes)
                background = go.Scatter3d(
                    x=embedding[:, 0],
                    y=embedding[:, 1],
                    z=embedding[:, 2],
                    mode='markers',
                    marker=dict(
                        size=point_size,
                        color='lightgray',
                        opacity=0.2
                    ),
                    hoverinfo='none',
                    showlegend=False
                )
                comp_fig.add_trace(background)
                
                # Add highlighted component points
                highlight = go.Scatter3d(
                    x=embedding[comp_nodes, 0],
                    y=embedding[comp_nodes, 1],
                    z=embedding[comp_nodes, 2],
                    mode='markers',
                    marker=dict(
                        size=highlight_size,
                        color='blue',
                        opacity=highlight_alpha
                    ),
                    name=f"Component {comp_idx}",
                    hovertemplate="Component: %{meta}<br>x: %{x}<br>y: %{y}<br>z: %{z}<extra></extra>",
                    meta=[comp_idx] * len(comp_nodes)
                )
                comp_fig.add_trace(highlight)
                
                # Update layout
                comp_fig.update_layout(
                    title=f"Component {comp_idx} ({len(comp_nodes)} nodes)",
                    scene=dict(
                        xaxis_title="UMAP 1",
                        yaxis_title="UMAP 2",
                        zaxis_title="UMAP 3",
                        aspectmode='cube'
                    ),
                    margin=dict(l=0, r=0, b=0, t=40)
                )
                
                figures.append(comp_fig)
            self.timing.end("visualize_components.plot_individual_3d")
            
            self.timing.end("visualize_components")
            return figures
        
    def plot_mst(self, graph, embedding=None, umap_results=None, 
                ax=None, fig_size=(12, 10), point_size=5, alpha=0.7,
                mst_color='red', mst_width=1.5, mst_alpha=0.8,
                color_by=None, show_colorbar=True, title=None,
                n_components=2, width=1000, height=800, filename=None):
        """
        Plot the minimum spanning tree of the graph.
        
        Parameters:
        -----------
        graph : ManifoldGraph
            The graph to visualize
        embedding : numpy.ndarray, optional
            Pre-computed embedding to use
        umap_results : dict, optional
            Results from a previous call to create_umap_embedding
        ax : matplotlib.axes.Axes, optional
            Axes to plot on (for 2D only)
        fig_size : tuple, default=(12, 10)
            Figure size if creating a new figure (for 2D only)
        point_size : float, default=5
            Size of points
        alpha : float, default=0.7
            Transparency of points
        mst_color : str or tuple, default='red'
            Color for MST edges
        mst_width : float, default=1.5
            Width for MST edges
        mst_alpha : float, default=0.8
            Transparency for MST edges
        color_by : array-like, optional
            Values to color nodes by
        show_colorbar : bool, default=True
            Whether to show a colorbar
        title : str, optional
            Plot title
        n_components : int, default=2
            Dimensionality of the embedding (2 or 3)
        width : int, default=1000
            Width of the 3D plot in pixels
        height : int, default=800
            Height of the 3D plot in pixels
        filename : str, optional
            If provided, save the 3D plot to this file
            
        Returns:
        --------
        fig : matplotlib.figure.Figure or plotly.graph_objs.Figure
            The figure object
        ax : matplotlib.axes.Axes or None
            The axes object (for 2D plots only)
        """
        self.timing.start("plot_mst")
        
        # Validate dimensionality
        if n_components not in [2, 3]:
            raise ValueError("n_components must be either 2 or 3")
        
        # Extract edge data to find MST edges
        edge_list = []
        is_mst_edge = []
        
        if hasattr(graph, 'edge_data') and 'is_mst_edge' in graph.edge_data:
            # Get MST edges from graph's edge_data
            edge_list = graph.edge_data['edge_list']
            is_mst_edge = graph.edge_data['is_mst_edge']
        elif hasattr(graph, 'get_edge_list'):
            # Extract from edge list and recompute MST
            from scipy.sparse.csgraph import minimum_spanning_tree
            
            # Get all edges
            edges_with_weights = graph.get_edge_list()
            edge_list = [(i, j) for i, j, w in edges_with_weights]
            
            # Create a matrix and compute MST
            matrix = graph.get_adjacency_matrix()
            mst = minimum_spanning_tree(matrix)
            mst_coo = mst.tocoo()
            
            # Find which edges are in the MST
            mst_edges = set()
            for i, j in zip(mst_coo.row, mst_coo.col):
                if i < j:  # Only store one direction
                    mst_edges.add((i, j))
            
            # Mark MST edges
            is_mst_edge = np.array([edge in mst_edges for edge in edge_list])
        else:
            # Try to extract from the matrix directly
            matrix = graph if not hasattr(graph, 'get_adjacency_matrix') else graph.get_adjacency_matrix()
            
            # Compute MST
            from scipy.sparse.csgraph import minimum_spanning_tree
            mst = minimum_spanning_tree(matrix)
            mst_coo = mst.tocoo()
            
            # Extract MST edges
            for i, j, w in zip(mst_coo.row, mst_coo.col, mst_coo.data):
                if i < j and w > 0:  # Only include non-zero edges once
                    edge_list.append((i, j))
                    is_mst_edge.append(True)
        
        # Keep only MST edges
        mst_edge_list = [edge for edge, is_mst in zip(edge_list, is_mst_edge) if is_mst]
        
        # Generate an embedding if not provided
        if embedding is None and umap_results is None:
            self.timing.start("plot_mst.create_embedding")
            embedding, umap_results = self.create_umap_embedding(
                graph, n_components=n_components
            )
            self.timing.end("plot_mst.create_embedding")
        elif umap_results is not None and embedding is None:
            # Extract embedding from results
            embedding = umap_results['original_embedding']
        
        # Set default title
        if title is None:
            title = f"Minimum Spanning Tree ({len(mst_edge_list)} edges)"
            
        # Determine coloring
        if color_by is None and hasattr(graph, 'get_component_labels'):
            color_by = graph.get_component_labels()
            
        # For 2D visualizations, use matplotlib
        if n_components == 2:
            # Create axes if needed
            if ax is None:
                fig, ax = plt.subplots(figsize=fig_size)
            else:
                fig = None
            
            # Plot nodes
            self.timing.start("plot_mst.plot_nodes")
            scatter = ax.scatter(
                embedding[:, 0],
                embedding[:, 1],
                c=color_by,
                s=point_size,
                alpha=alpha,
                rasterized=True
            )
            self.timing.end("plot_mst.plot_nodes")
            
            # Plot MST edges
            self.timing.start("plot_mst.plot_edges")
            segments = []
            for i, j in mst_edge_list:
                segments.append([(embedding[i, 0], embedding[i, 1]), 
                              (embedding[j, 0], embedding[j, 1])])
            
            edge_collection = LineCollection(
                segments, 
                linewidths=mst_width,
                colors=mst_color,
                alpha=mst_alpha,
                zorder=1.5  # Draw above regular edges but below points
            )
            ax.add_collection(edge_collection)
            self.timing.end("plot_mst.plot_edges")
            
            # Set title and axes
            ax.set_title(title)
            ax.set_aspect('equal')
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            
            # Add colorbar if requested
            if show_colorbar and color_by is not None:
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label("Component" if color_by is None else "Value")
            
            plt.tight_layout()
            
            self.timing.end("plot_mst")
            return fig, ax
            
        else:
            # For 3D visualizations, use plotly
            fig = go.Figure()
            
            # Add scatter trace for nodes
            scatter = go.Scatter3d(
                x=embedding[:, 0],
                y=embedding[:, 1],
                z=embedding[:, 2],
                mode='markers',
                marker=dict(
                    size=point_size,
                    color=color_by,
                    colorscale='Viridis',
                    opacity=alpha,
                    colorbar=dict(
                        title="Component" if color_by is None else "Value",
                        titleside='right'
                    ),
                    showscale=show_colorbar
                ),
                hovertemplate="x: %{x}<br>y: %{y}<br>z: %{z}<extra></extra>"
            )
            fig.add_trace(scatter)
            
            # Add MST edges
            mst_x = []
            mst_y = []
            mst_z = []
            
            for i, j in mst_edge_list:
                # Add each line as a separate segment with None values between
                mst_x.extend([embedding[i, 0], embedding[j, 0], None])
                mst_y.extend([embedding[i, 1], embedding[j, 1], None])
                mst_z.extend([embedding[i, 2], embedding[j, 2], None])
            
            # Add MST trace
            mst_trace = go.Scatter3d(
                x=mst_x,
                y=mst_y,
                z=mst_z,
                mode='lines',
                line=dict(
                    color=mst_color,
                    width=mst_width
                ),
                opacity=mst_alpha,
                hoverinfo='none',
                name="MST Edges"
            )
            fig.add_trace(mst_trace)
            
            # Update layout
            fig.update_layout(
                title=title,
                width=width,
                height=height,
                scene=dict(
                    xaxis_title="UMAP 1",
                    yaxis_title="UMAP 2",
                    zaxis_title="UMAP 3",
                    aspectmode='cube'
                ),
                margin=dict(l=0, r=0, b=0, t=40)
            )
            
            # Save to file if requested
            if filename is not None:
                fig.write_html(filename)
                if self.verbose:
                    print(f"Interactive 3D MST plot saved to: {filename}")
            
            self.timing.end("plot_mst")
            return fig, None
            

    def export_interactive_3d_plot(self, embedding, color_by=None, color_column_name=None,
                              filename='umap_3d_plot.html', point_size=5, opacity=0.7,
                              title="3D UMAP Embedding", max_points=10000, width=1200, height=800,
                              colorscale='Viridis', sample_method='random'):
        """
        Export a standalone interactive 3D plot using Plotly with sampling for large datasets.
        
        Parameters:
        -----------
        embedding : numpy.ndarray
            The 3D embedding to plot (shape should be (n_samples, 3))
        color_by : array-like, optional
            Values to color nodes by
        color_column_name : str, optional
            Name of the color column for hover information
        filename : str, default='umap_3d_plot.html'
            Filename to save the HTML plot
        point_size : float, default=5
            Size of scatter points
        opacity : float, default=0.7
            Transparency of points
        title : str, default="3D UMAP Embedding"
            Plot title
        max_points : int, default=10000
            Maximum number of points to plot (will sample if necessary)
        width : int, default=1200
            Width of the plot in pixels
        height : int, default=800
            Height of the plot in pixels
        colorscale : str, default='Viridis'
            Plotly colorscale to use
        sample_method : str, default='random'
            Method for sampling points: 'random' or 'stride'
            
        Returns:
        --------
        plotly.graph_objs.Figure
            The interactive figure object
        """
        go = _load_plotly()
        print(f"Original embedding shape: {embedding.shape}")
        
        # Sample data if needed
        n_points = embedding.shape[0]
        sampled_indices = None
        
        if n_points > max_points:
            print(f"Sampling {max_points} points from {n_points} total points")
            if sample_method == 'random':
                # Random sampling
                sampled_indices = np.random.choice(n_points, max_points, replace=False)
                embedding = embedding[sampled_indices]
            else:
                # Stride sampling (take every Nth point)
                stride = n_points // max_points
                sampled_indices = np.arange(0, n_points, stride)[:max_points]
                embedding = embedding[sampled_indices]
            
            # Also sample the color values if provided
            if color_by is not None:
                color_by = color_by[sampled_indices]
        
        print(f"Sampled embedding shape: {embedding.shape}")
        
        # Validate embedding dimensions
        if embedding.shape[1] != 3:
            raise ValueError(f"Expected 3D embedding, got shape {embedding.shape}")
        
        # Check for invalid values
        if np.isnan(embedding).any() or np.isinf(embedding).any():
            print("Warning: Replacing NaN/Inf values with zeros")
            embedding = np.nan_to_num(embedding)
        
        # Set default color values if not provided
        if color_by is None:
            color_by = np.zeros(embedding.shape[0])
            if color_column_name is None:
                color_column_name = "Cluster"
        elif color_column_name is None:
            color_column_name = "Value"
        
        # Check color_by for issues
        if isinstance(color_by, np.ndarray):
            if np.isnan(color_by).any() or np.isinf(color_by).any():
                print("Warning: Replacing NaN/Inf values in color data with zeros")
                color_by = np.nan_to_num(color_by)
        
        # Calculate data ranges with a small buffer (10%)
        x_min, y_min, z_min = embedding.min(axis=0)
        x_max, y_max, z_max = embedding.max(axis=0)
        
        # Add 10% buffer to each end
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min
        
        x_buffer = x_range * 0.1
        y_buffer = y_range * 0.1
        z_buffer = z_range * 0.1
        
        # Create hover template
        hover_template = (
            f"{color_column_name}: %{{marker.color}}<br>"
            "UMAP 1: %{x}<br>"
            "UMAP 2: %{y}<br>"
            "UMAP 3: %{z}<extra></extra>"
        )
        
        # Create figure with explicit marker settings
        fig = go.Figure(data=[
            go.Scatter3d(
                x=embedding[:, 0],
                y=embedding[:, 1],
                z=embedding[:, 2],
                mode='markers',
                marker=dict(
                    size=point_size,
                    color=color_by,
                    colorscale=colorscale,
                    opacity=opacity,
                    showscale=True,
                    symbol='circle',
                    line=dict(width=0)  # No marker border
                ),
                text=color_by,
                hovertemplate=hover_template
            )
        ])
        
        # Set explicit layout with calculated ranges
        fig.update_layout(
            title=f"{title} (Sampled: {embedding.shape[0]} points)",
            scene=dict(
                xaxis=dict(
                    title="UMAP 1",
                    range=[x_min - x_buffer, x_max + x_buffer],
                ),
                yaxis=dict(
                    title="UMAP 2",
                    range=[y_min - y_buffer, y_max + y_buffer],
                ),
                zaxis=dict(
                    title="UMAP 3",
                    range=[z_min - z_buffer, z_max + z_buffer],
                ),
                aspectmode='data'  # Use 'data' instead of 'cube'
            ),
            width=width,
            height=height,
            margin=dict(l=0, r=0, b=0, t=40)  # Tighter margins
        )
        
        # Print the calculated ranges for debugging
        print(f"X range: [{x_min - x_buffer:.2f}, {x_max + x_buffer:.2f}]")
        print(f"Y range: [{y_min - y_buffer:.2f}, {y_max + y_buffer:.2f}]")
        print(f"Z range: [{z_min - z_buffer:.2f}, {z_max + z_buffer:.2f}]")
        
        # Save the plot with CDN plotly.js
        fig.write_html(
            filename, 
            include_plotlyjs='cdn',  # Use CDN version
            full_html=True,
            config={
                'displayModeBar': True,
                'responsive': True
            }
        )
        
        print(f"Interactive 3D plot saved to: {filename}")
        return fig