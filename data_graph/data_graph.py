"""
DataGraph - Core data structure for graphs on the manifold of Gaussian distributions.
"""
import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.csgraph import connected_components, shortest_path

import numpy as np
from numba import njit, prange

class DataGraph:
    """
    Core data structure representing a data graph.
    Encapsulates the graph and its properties.
    """
    
    def __init__(self, graph_matrix=None, 
                 node_df=None,
                 feature_cols=None,
                 semimetric_weight_function=None, 
                 embedding_function=None, 
                 edge_data=None, 
                 component_labels=None,
                 self_weight=0.0,
                 missing_weight=float('inf')):
        """
        Initialize a DataGraph.
        
        Parameters:
        -----------
        graph_matrix : scipy.sparse.csr_matrix, optional
            The sparse matrix representation of the graph. If None, an empty graph is created.
        node_df : pandas.DataFrame, optional
            DataFrame containing node attributes. If None, an empty DataFrame is created.
        feature_cols : list or None, optional
            List of column names to use as features. If None, all numeric columns are used.
        semimetric_weight_function : callable, optional
            Function to compute weights between nodes. If None, a default Euclidean distance is used.
        embedding_function : callable, optional
            Function to compute embeddings. If None, a default identity function is used.
        edge_data : dict, optional
            Dictionary containing edge information
        component_labels : array-like, optional
            Labels indicating connected components
        self_weight : float, optional
            Weight for self-connections
        missing_weight : float, optional
            Weight for missing connections
        """
        # Set default weight function if not provided
        if semimetric_weight_function is None:
            self.semimetric_weight_function = self._default_weight_function
        else:
            self.semimetric_weight_function = semimetric_weight_function
        
        # Set default embedding function if not provided
        if embedding_function is None:
            self.embedding_function = self._default_embedding_function
        else:
            self.embedding_function = embedding_function

        self.missing_weight = missing_weight
        self.self_weight = self_weight
        
        # Handle empty initialization (for loading later)
        if graph_matrix is None:
            if node_df is not None:
                raise ValueError("If graph_matrix is None, node_df must also be None")
            # Initialize with placeholder values
            self.node_df = None
            self.feature_cols = None
            self.graph = None
            self.n_nodes = 0
            self.edge_data = {}
            self.component_labels = None
            self.n_components = 0
            self.component_sizes = None
            self.node_features = None
            return
            
        # Normal initialization with provided values
        self.node_df = node_df
        self.feature_cols = feature_cols
        self.graph = graph_matrix.tocsr()  # Ensure CSR format
        self.n_nodes = graph_matrix.shape[0]
        
        # Store edge data
        self.edge_data = edge_data if edge_data is not None else {}
        
        # Compute component information if not provided
        if component_labels is None:
            self.compute_components()
        else:
            self.component_labels = component_labels
            self.n_components = len(np.unique(component_labels))
            self.component_sizes = np.bincount(component_labels)

        self._extract_node_features()

        self.coarsened_graph = None
    
    @staticmethod
    def _default_weight_function(features, i, j):
        """Default weight function using Euclidean distance."""
        return np.sqrt(np.sum((features[i] - features[j])**2))
    
    @staticmethod
    def _default_embedding_function(features, graph):
        """Default embedding function (identity)."""
        return features
    
    def _extract_node_features(self):
        """Extract numeric features from node_df."""
        if self.node_df is None:
            self.node_features = None
            return
            
        # pick your columns
        if self.feature_cols is None:
            df = self.node_df.select_dtypes(include='number')
        else:
            df = self.node_df[self.feature_cols]
    
        # convert to float32 and then force C-order
        arr = df.to_numpy(dtype=np.float32)          # no `order=` here
        arr = np.ascontiguousarray(arr)              # ensure C-contiguous layout
    
        self.node_features = arr
        
    def compute_graph_distance(self, i, j):
        return self.semimetric_weight_function(self.node_features, i, j)
        
    def compute_components(self):
        """Compute connected components of the graph"""
        self.n_components, self.component_labels = connected_components(self.graph, directed=False)
        self.component_sizes = np.bincount(self.component_labels)
        return self.component_labels

    def get_node_data(self, node_idx):
        """Get all data for a specific node from the original dataframe."""
        if self.node_df is None:
            raise ValueError("No node dataframe available")
        
        if node_idx < 0 or node_idx >= len(self.node_df):
            raise ValueError(f"Node index {node_idx} out of range [0, {len(self.node_df)-1}]")
            
        return self.node_df.iloc[node_idx]
    
    def get_nodes_data(self, node_indices):
        """Get data for multiple nodes from the original dataframe."""
        if self.node_df is None:
            raise ValueError("No node dataframe available")
            
        return self.node_df.iloc[node_indices]
    
    def get_component_data(self, component_idx):
        """Get data for all nodes in a specific component."""
        if self.node_df is None:
            raise ValueError("No node dataframe available")
            
        component_nodes = self.get_component(component_idx)
        return self.node_df.iloc[component_nodes]
        
    def get_neighbors(self, node_idx):
        """
        Get neighbors of a node.
        
        Parameters:
        -----------
        node_idx : int
            Index of the node
            
        Returns:
        --------
        neighbors : list
            List of neighbor indices
        weights : list
            Corresponding edge weights
        """
        if node_idx < 0 or node_idx >= self.n_nodes:
            raise ValueError(f"Node index {node_idx} out of range [0, {self.n_nodes-1}]")
            
        start, end = self.graph.indptr[node_idx], self.graph.indptr[node_idx+1]
        neighbors = self.graph.indices[start:end]
        weights = self.graph.data[start:end]
        
        return neighbors, weights
    
    def get_edge_weight(self, i, j):
        if i==j:
            return self.self_weight
        start, end = self.graph.indptr[i], self.graph.indptr[i+1]
        row_cols = self.graph.indices[start:end]
        # binary‐search if row_cols is sorted (it usually is):
        import bisect
        idx = bisect.bisect_left(row_cols, j)
        if idx < (end-start) and row_cols[idx] == j:
            return self.graph.data[start + idx]
        else:
            return self.missing_weight
    
    def get_component_labels(self):
        """Get the component labels for all nodes"""
        return self.component_labels
    
    def get_component(self, component_idx):
        """
        Get nodes in a specific component.
        
        Parameters:
        -----------
        component_idx : int
            Index of the component
            
        Returns:
        --------
        nodes : numpy.ndarray
            Indices of nodes in the component
        """
        if component_idx < 0 or component_idx >= self.n_components:
            raise ValueError(f"Component index {component_idx} out of range [0, {self.n_components-1}]")
            
        return np.where(self.component_labels == component_idx)[0]
    
    def get_shortest_path(self, source, target):
        """
        Find the shortest path between two nodes.
        
        Parameters:
        -----------
        source, target : int
            Indices of the source and target nodes
            
        Returns:
        --------
        path : list
            List of node indices in the path
        distance : float
            Total distance of the path
        """
        if self.component_labels[source] != self.component_labels[target]:
            return None, float('inf')  # No path exists
            
        # Compute shortest path
        dist_matrix, predecessors = shortest_path(
            self.graph, 
            directed=False, 
            indices=source,
            return_predecessors=True
        )
        
        if np.isinf(dist_matrix[target]):
            return None, float('inf')  # No path exists
            
        # Reconstruct the path
        path = [target]
        current = target
        while current != source:
            current = predecessors[current]
            path.append(current)
        
        path.reverse()
        return path, dist_matrix[target]
    
    def get_degree(self, node_idx):
        """Get the degree of a node"""
        return self.graph.indptr[node_idx+1] - self.graph.indptr[node_idx]
    
    def get_all_degrees(self):
        """Get the degrees of all nodes"""
        return np.diff(self.graph.indptr)
    
    def get_edge_list(self):
        """
        Get a list of all edges in the graph.
        
        Returns:
        --------
        edges : list of tuples
            List of (i, j, weight) tuples
        """
        coo = self.graph.tocoo()
        edges = []
        
        for i, j, w in zip(coo.row, coo.col, coo.data):
            if i < j:  # Only include each edge once
                edges.append((i, j, w))
                
        return edges
    
    def get_adjacency_matrix(self):
        """Get the adjacency matrix representation of the graph"""
        return self.graph
    
    def __str__(self):
        """String representation of the graph"""
        if self.graph is None:
            return "Empty DataGraph (not loaded)"
        
        return (f"DataGraph with {self.n_nodes} nodes, "
                f"{self.graph.nnz//2} edges, {self.n_components} components")
    
    def __repr__(self):
        return self.__str__()

    def save(self, output_dir="saved_data_graph", embedding=None, embedding_params=None,
             compress=True, function_references=None):
        """
        Save this DataGraph instance and associated data to disk.
        
        Parameters:
            output_dir: Directory to save files
            embedding: Optional embedding array (e.g., from UMAP or other dimensionality reduction)
            embedding_params: Optional dictionary of embedding parameters
            compress: Whether to use compression for large arrays
            function_references: Dict with function references as 
                               {'semimetric_weight_function': 'module.submodule:function_name', 
                                'embedding_function': 'module.submodule:function_name'}
                               If None, functions will be pickled (less portable but simpler)
        """
        import os
        import json
        import pickle
        import numpy as np
        import scipy.sparse as sp
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Save the node DataFrame
        print(f"Saving node dataframe with {len(self.node_df)} rows...")
        self.node_df.to_parquet(f"{output_dir}/node_df.parquet", compression="gzip" if compress else None)
        
        # 2. Save the graph structure (adjacency matrix)
        print("Saving graph structure...")
        sp.save_npz(f"{output_dir}/graph_adjacency.npz", self.graph, compressed=compress)
        
        # 3. Save the embedding if provided
        if embedding is not None:
            print(f"Saving embedding with shape {embedding.shape}...")
            np.save(f"{output_dir}/embedding.npy", embedding, allow_pickle=False)
        
        # 4. Save node features
        print("Saving node features...")
        np.save(f"{output_dir}/node_features.npy", self.node_features, allow_pickle=False)
        
        # 5. Save edge data using pickle (contains complex structures)
        print("Saving edge data...")
        with open(f"{output_dir}/edge_data.pkl", 'wb') as f:
            pickle.dump(self.edge_data, f)
        
        # 6. Save functions - either as references or pickled
        if function_references:
            print("Saving function references...")
            with open(f"{output_dir}/function_references.json", 'w') as f:
                json.dump(function_references, f, indent=2)
        else:
            print("Saving weight and embedding functions (pickled)...")
            with open(f"{output_dir}/functions.pkl", 'wb') as f:
                # Check if we're using default functions
                using_default_weight = self.semimetric_weight_function == self._default_weight_function
                using_default_embedding = self.embedding_function == self._default_embedding_function
                
                pickle.dump({
                    'semimetric_weight_function': self.semimetric_weight_function,
                    'embedding_function': self.embedding_function,
                    'using_default_weight': using_default_weight,
                    'using_default_embedding': using_default_embedding
                }, f)
        
        # 7. Save metadata and parameters as JSON
        metadata = {
            "feature_cols": self.feature_cols,
            "n_nodes": self.n_nodes,
            "n_components": self.n_components,
            "self_weight": float(self.self_weight),  # Ensure JSON serialization
            "missing_weight": float(self.missing_weight) if not np.isinf(self.missing_weight) else "inf",
            "component_sizes": self.component_sizes.tolist() if hasattr(self, 'component_sizes') else None,
            "has_embedding": embedding is not None,
            "using_function_references": function_references is not None
        }
        
        # Add embedding parameters if provided
        if embedding_params:
            metadata["embedding_parameters"] = embedding_params
        
        print("Saving metadata...")
        with open(f"{output_dir}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # 8. Save component labels
        print("Saving component labels...")
        np.save(f"{output_dir}/component_labels.npy", self.component_labels)
        
        print(f"All data successfully saved to {output_dir}/")

    @classmethod
    def load(cls, input_dir="saved_data_graph", load_full_data=True):
        """
        Load a saved DataGraph from disk.
        
        Parameters:
            input_dir: Directory containing saved files
            load_full_data: Whether to load all data including large arrays
            
        Returns:
            graph: DataGraph instance
            embedding: Optional embedding array (if it was saved)
            embedding_params: Optional dictionary of embedding parameters (if saved)
        """
        import os
        import json
        import pickle
        import numpy as np
        import pandas as pd
        import scipy.sparse as sp
        
        print(f"Loading data from {input_dir}/...")
        
        # 1. Load metadata first to understand what we have
        with open(f"{input_dir}/metadata.json", 'r') as f:
            metadata = json.load(f)
        
        # 2. Load node DataFrame
        print("Loading node dataframe...")
        node_df = pd.read_parquet(f"{input_dir}/node_df.parquet")
        
        # 3. Load adjacency matrix
        print("Loading graph structure...")
        adjacency = sp.load_npz(f"{input_dir}/graph_adjacency.npz")
        
        # 4. Load component labels
        component_labels = np.load(f"{input_dir}/component_labels.npy")
        
        # 5. Load edge data
        print("Loading edge data...")
        with open(f"{input_dir}/edge_data.pkl", 'rb') as f:
            edge_data = pickle.load(f)
        
        # 6. Load functions - either from references or pickled
        if metadata.get("using_function_references", False):
            print("Loading function references...")
            with open(f"{input_dir}/function_references.json", 'r') as f:
                function_refs = json.load(f)
                
            # Import the functions based on references
            semimetric_weight_function = cls._import_function(function_refs['semimetric_weight_function'])
            embedding_function = cls._import_function(function_refs['embedding_function'])
        else:
            print("Loading weight and embedding functions (pickled)...")
            try:
                with open(f"{input_dir}/functions.pkl", 'rb') as f:
                    functions = pickle.load(f)
                    
                # Check if we stored info about default functions
                if 'using_default_weight' in functions and functions['using_default_weight']:
                    semimetric_weight_function = cls._default_weight_function
                else:
                    semimetric_weight_function = functions['semimetric_weight_function']
                    
                if 'using_default_embedding' in functions and functions['using_default_embedding']:
                    embedding_function = cls._default_embedding_function
                else:
                    embedding_function = functions['embedding_function']
            except Exception as e:
                print(f"Warning: Could not load functions properly: {e}")
                print("Using default functions instead")
                semimetric_weight_function = cls._default_weight_function
                embedding_function = cls._default_embedding_function
        
        # 7. Parse special values
        missing_weight = metadata["missing_weight"]
        if missing_weight == "inf":
            missing_weight = float('inf')
        else:
            missing_weight = float(missing_weight)
        
        # 8. Reconstruct the DataGraph
        print("Reconstructing DataGraph...")
        data_graph = cls(
            graph_matrix=adjacency,
            node_df=node_df,
            feature_cols=metadata["feature_cols"],
            semimetric_weight_function=semimetric_weight_function,
            embedding_function=embedding_function,
            edge_data=edge_data,
            component_labels=component_labels,
            self_weight=float(metadata["self_weight"]),
            missing_weight=missing_weight
        )
        
        # 9. Load the embedding if it was saved
        embedding = None
        if metadata["has_embedding"] and load_full_data:
            print("Loading embedding...")
            embedding = np.load(f"{input_dir}/embedding.npy")
        
        # 10. Extract embedding parameters
        embedding_params = metadata.get("embedding_parameters", None)
        
        print("Data loading complete!")
        return data_graph, embedding, embedding_params

    @staticmethod
    def _import_function(function_ref):
        """Helper to import a function from a string reference."""
        module_name, function_name = function_ref.split(':')
        module = __import__(module_name, fromlist=[''])
        return getattr(module, function_name)
    
    def load_from(self, input_dir="saved_data_graph", load_full_data=True):
        """
        Load a saved DataGraph from disk into this instance.
        
        Parameters:
            input_dir: Directory containing saved files
            load_full_data: Whether to load all data including large arrays
            
        Returns:
            embedding: Optional embedding array (if it was saved)
            embedding_params: Optional dictionary of embedding parameters (if saved)
        """
        import os
        import json
        import pickle
        import numpy as np
        import pandas as pd
        import scipy.sparse as sp
        
        print(f"Loading data from {input_dir}/...")
        
        # 1. Load metadata first to understand what we have
        with open(f"{input_dir}/metadata.json", 'r') as f:
            metadata = json.load(f)
        
        # 2. Load node DataFrame
        print("Loading node dataframe...")
        self.node_df = pd.read_parquet(f"{input_dir}/node_df.parquet")
        
        # 3. Load adjacency matrix
        print("Loading graph structure...")
        self.graph = sp.load_npz(f"{input_dir}/graph_adjacency.npz")
        
        # 4. Load component labels
        self.component_labels = np.load(f"{input_dir}/component_labels.npy")
        
        # 5. Load edge data
        print("Loading edge data...")
        with open(f"{input_dir}/edge_data.pkl", 'rb') as f:
            self.edge_data = pickle.load(f)
        
        # 6. Load functions - either from references or pickled
        if metadata.get("using_function_references", False):
            print("Loading function references...")
            with open(f"{input_dir}/function_references.json", 'r') as f:
                function_refs = json.load(f)
                
            # Import the functions based on references
            self.semimetric_weight_function = self._import_function(function_refs['semimetric_weight_function'])
            self.embedding_function = self._import_function(function_refs['embedding_function'])
        else:
            print("Loading weight and embedding functions (pickled)...")
            try:
                with open(f"{input_dir}/functions.pkl", 'rb') as f:
                    functions = pickle.load(f)
                    
                # Check if we stored info about default functions
                if 'using_default_weight' in functions and functions['using_default_weight']:
                    self.semimetric_weight_function = self._default_weight_function
                else:
                    self.semimetric_weight_function = functions['semimetric_weight_function']
                    
                if 'using_default_embedding' in functions and functions['using_default_embedding']:
                    self.embedding_function = self._default_embedding_function
                else:
                    self.embedding_function = functions['embedding_function']
            except Exception as e:
                print(f"Warning: Could not load functions properly: {e}")
                print("Using default functions instead")
                self.semimetric_weight_function = self._default_weight_function
                self.embedding_function = self._default_embedding_function
        
        # 7. Parse special values
        missing_weight = metadata["missing_weight"]
        if missing_weight == "inf":
            self.missing_weight = float('inf')
        else:
            self.missing_weight = float(missing_weight)
            
        self.self_weight = float(metadata["self_weight"])
        
        # 8. Set other attributes
        self.feature_cols = metadata["feature_cols"]
        self.n_nodes = metadata["n_nodes"]
        self.n_components = metadata["n_components"]
        if "component_sizes" in metadata and metadata["component_sizes"] is not None:
            self.component_sizes = np.array(metadata["component_sizes"])
        
        # 9. Extract features from node_df
        self._extract_node_features()
        
        # 10. Load the embedding if it was saved
        embedding = None
        if metadata["has_embedding"] and load_full_data:
            print("Loading embedding...")
            embedding = np.load(f"{input_dir}/embedding.npy")
        
        # 11. Extract embedding parameters
        embedding_params = metadata.get("embedding_parameters", None)
        
        print("Data loading complete!")
        return embedding, embedding_params