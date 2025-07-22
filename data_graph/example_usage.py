import numpy as np
import pandas as pd
from numba import njit
from data_graph_generator import DataGraphGenerator

# 1. Create sample data with cluster structure
np.random.seed(42)
n_samples = 200
n_dims = 4

# Create 3 distinct clusters
cluster_centers = np.array([
    [1.0, 1.0, 1.0, 1.0],
    [-1.0, -1.0, -1.0, -1.0],
    [1.0, -1.0, 1.0, -1.0]
])
n_per_cluster = n_samples // len(cluster_centers)
points = np.vstack([
    np.random.normal(center, 0.3, size=(n_per_cluster, n_dims))
    for center in cluster_centers
])

# Create DataFrame
df = pd.DataFrame(
    points, 
    columns=[f'dim_{i}' for i in range(n_dims)]
)

# 2. Define the semimetric weight function
@njit
def euclidean_semimetric(features, i, j):
    """Compute Euclidean distance between points at indices i and j"""
    return np.sqrt(np.sum((features[i] - features[j])**2))

# 3. Define embedding function (for initial approximation)
def identity_embedding(row):
    """Return point coordinates as embedding"""
    return np.array([row[f'dim_{i}'] for i in range(n_dims)])

# 4. Create the generator
generator = DataGraphGenerator(
    node_df=df,
    feature_cols=[f'dim_{i}' for i in range(n_dims)],
    semimetric_weight_function=euclidean_semimetric,
    embedding_function=identity_embedding,
    verbose=True
)

# 5. Build and refine the graph
graph_obj, results = generator.build_and_refine_graph(
    n_neighbors=15,
    mst_iterations=2,
    smooth_iterations=2,
    max_new_edges=1000,
    preserve_mst=True
)

# 6. Print graph information
print(f"\nGraph summary:")
print(f"- {len(results['component_labels'])} nodes")
print(f"- {len(results['initial_edge_data']['edge_list'])} edges")
print(f"- {results['n_components']} components")
print(f"- Component sizes: {results['component_sizes']}")

# 7. Basic visualization (optional)
try:
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    
    # Get 2D embedding for visualization
    tsne = TSNE(n_components=2, random_state=42)
    embedding = tsne.fit_transform(df.values)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(
        embedding[:, 0], 
        embedding[:, 1], 
        c=results['component_labels'], 
        cmap='tab10', 
        s=40, 
        alpha=0.7
    )
    
    # Plot subset of edges
    edges = results['initial_edge_data']['edge_list']
    for idx, (i, j) in enumerate(edges):
        if idx > 500:  # Limit for clarity
            break
        plt.plot(
            [embedding[i, 0], embedding[j, 0]], 
            [embedding[i, 1], embedding[j, 1]], 
            'k-', 
            alpha=0.2, 
            linewidth=0.5
        )
    
    plt.title('Graph Visualization (t-SNE embedding)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('graph_test.png', dpi=300)
    print("\nVisualization saved to 'graph_test.png'")
except ImportError:
    print("\nVisualization skipped (requires matplotlib and sklearn)")