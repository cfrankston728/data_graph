# First, import the necessary modules using the package-style import
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add the parent directory to sys.path (contains the manifold_graph package)
#sys.path.append('/home/groups/CEDAR/franksto/00_PROJECT_HUB/version_1/projects/HiC_VAE/version-1.0/src/algorithmic_scripts')

# # Import from the manifold_graph package
# from manifold_graph import ManifoldGraph, ManifoldGraphGenerator, DiagCovGaussianManifoldGraph, DiagCovGaussianManifoldGraphGenerator
# from manifold_graph import ManifoldGraphVisualizer, ManifoldGraphAnalyzer

from data_graph import DataGraphGenerator
from numba import njit, prange

# 4. Create the generator
n_dims=4

@njit
def euclidean_semimetric(f, i, j):
    diff = f[i] - f[j]
    return np.sqrt(np.dot(diff, diff))


from numba import njit
import numpy as np

@njit
def fisher_rao_distance(f, i, j):
    eps      = 1e-8
    safe_eps = 1e-30

    # f is shape (n_nodes, 2*n_dims)
    means_i  = f[i,  :4] + 0.0   # first 4 dims
    sigma_i  = f[i,  4:8] + eps  # next 4 dims
    means_j  = f[j,  :4] + 0.0
    sigma_j  = f[j,  4:8] + eps

    # variance‐like term
    var_term = (sigma_i**2 + sigma_j**2)/2 + eps

    # log‐term, clamped away from zero
    ratio    = var_term / (sigma_i * sigma_j)
    log_term = np.log(np.maximum(ratio, 1e-16))

    diff     = means_i - means_j
    d2       = 2 * np.sum(diff**2/var_term + log_term)

    # final Fisher–Rao distance
    return np.sqrt(max(safe_eps, d2))


import numpy as np

# assume n_dims is defined in your scope
def logspace_embedding(row):
    # 1) grab means and sigmas as 1D arrays
    n_dims=4
    means  = np.array([row[f"latent_{i+1}"] for i in range(n_dims)], dtype=float)
    sigmas = np.array([row[f"sig_{i+1}"]    for i in range(n_dims)], dtype=float)
    
    # 2) prepare output vector of length 2*n_dims
    emb = np.zeros(2 * n_dims, dtype=float)
    
    # 3) fill: even entries = mu/sigma, odd entries = sqrt(2)*ln(sigma)
    emb[0::2] = means / sigmas
    emb[1::2] = np.sqrt(2) * np.log(sigmas)
    
    return emb

    
generator = DataGraphGenerator(
    node_df=model_df.iloc[:2000],
    feature_cols=[f'latent_{i+1}' for i in range(n_dims)] + [f'sig_{i+1}' for i in range(n_dims)],
    semimetric_weight_function=fisher_rao_distance,
    embedding_function=logspace_embedding,
    verbose=True
)

# 5. Build and refine the graph
graph_obj, results = generator.build_and_refine_graph(
    n_neighbors=200,
    mst_iterations=2,
    smooth_iterations=0,
    #max_new_edges=1000,
    preserve_mst=True
)

# 6. Print graph information
print(f"\nGraph summary:")
print(f"- {len(results['component_labels'])} nodes")
print(f"- {len(results['initial_edge_data']['edge_list'])} edges")
print(f"- {results['n_components']} components")
print(f"- Component sizes: {results['component_sizes']}")
    