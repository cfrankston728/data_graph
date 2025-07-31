# data\_graph

# DataGraphGenerator: Custom Distance Graph Library

A Python library for creating and refining sparse graph representations of data according to custom **premetric** weight functions.

## Overview

`DataGraphGenerator` builds data-driven graphs where edges represent meaningful relationships between data points. It leverages user-defined premetric weight functions and applies graph refinement techniques to produce high-quality, sparse graph representations.

## Key Features

* **Custom premetric** weight functions for flexible distance definitions
* Embedding-based KNN for initial neighbor discovery
* MST-based connectivity to guarantee a fully connected backbone
* Automatic knee-point detection for intelligent graph pruning
* Graph polishing by connecting 2-hop neighbors
* Iterative refine-and-polish cycles controlled by `polish_iterations`
* Comprehensive timing statistics and optimization diagnostics
* Memory-efficient batch processing for large datasets

## Usage Example

```python
import pandas as pd
import numpy as np
from numba import njit
from data_graph import DataGraphGenerator

# Define custom premetric function (Numba-compatible)
@njit(parallel=True)
def my_premetric(features, i_indices, j_indices):
    """Compute custom premetric distances between data point pairs."""
    result = np.zeros(len(i_indices), dtype=np.float32)
    for idx in range(len(i_indices)):
        i, j = i_indices[idx], j_indices[idx]
        # Custom distance calculation
        result[idx] = np.sum(np.abs(features[i] - features[j]))
    return result

# Define embedding function for initial neighbor discovery
def my_embedding(row):
    """Convert a data row to coordinates for KNN search."""
    return np.array([row['feature1'], row['feature2']])

# Create graph generator
generator = DataGraphGenerator(
    node_df=df,
    feature_cols=['feature1', 'feature2', 'feature3'],
    premetric_weight_function=my_premetric,
    embedding_function=my_embedding,
    verbose=True
)

# Build and refine graph
graph_obj, results = generator.build_and_refine_graph(
    n_neighbors=30,
    mst_iterations=3,
    polish_iterations=2,    # renamed from smooth_iterations
    prune_threshold=None,
    kneedle_sensitivity=1.0,
    max_new_edges=None,
    preserve_mst=True
)

# Access the graph
sparse_matrix = graph_obj.graph          # scipy.sparse.csr_matrix
component_labels = graph_obj.component_labels
```

## Core Components

### `DataGraphGenerator`

Main class that orchestrates graph creation and refinement:

```python
generator = DataGraphGenerator(
    node_df,                       # pandas.DataFrame with node data
    feature_cols,                  # list of columns to compute premetric
    premetric_weight_function,     # custom premetric weight function
    embedding_function,            # function mapping row to embedding
    verbose=True,                  # print progress messages
    use_float32=True,              # use float32 arrays
    n_jobs=-1,                     # parallel jobs for KNN
    plot_knee=False                # visualize knee-point detection
)
```

**Parameters**

* `n_neighbors` (int): number of neighbors in initial KNN graph
* `mst_iterations` (int): number of refine-MST passes
* `polish_iterations` (int): number of prune→polish loops
* `prune_threshold` (float or None): distance cutoff (None = auto)
* `kneedle_sensitivity` (float): controls threshold detection
* `max_new_edges` (int or None): limit on 2-hop polishing edges
* `preserve_mst` (bool): whether to keep MST edges through pruning

## Graph Building Pipeline

1. **KNN + MST**: Create initial graph using k-nearest neighbors in embedding space and ensure connectivity with a minimum spanning tree
2. **Prune**: Remove edges with distances above a threshold (determined automatically by knee-point detection or user-specified)
3. **Polish**: Add edges between 2-hop neighbors to improve graph structure
4. **Iterative Refinement**: Alternate between Prune and Polish for `polish_iterations`

## Optimization Considerations

The library includes multiple performance optimizations:

* Numba-accelerated premetric computations
* Batch processing for memory efficiency
* Efficient graph algorithms (MST, connected components)
* Memory usage estimation for optimal batch sizing
* Tracking MST edges to maintain connectivity
* Early stopping when graph structure stabilizes

## Dependencies

* `numpy`
* `scipy`
* `pandas`
* `scikit-learn`
* `numba`
* `matplotlib` (optional)

## Installation

```bash
pip install data-graph
```

## Advanced Usage

### Custom Premetric Functions

Your premetric weight function must be Numba-compatible with signature:

```python
@njit(parallel=True)
def distance_fn(features: np.ndarray, i_indices: np.ndarray, j_indices: np.ndarray) -> np.ndarray:
    # returns array of distances
    ...
```

### Controlling Graph Density

```python
# Sparser graph
graph_obj, results = generator.build_and_refine_graph(
    n_neighbors=10,
    prune_threshold=0.3,
    kneedle_sensitivity=2.0,
    max_new_edges=100
)

# Denser graph
graph_obj, results = generator.build_and_refine_graph(
    n_neighbors=50,
    prune_threshold=None,
    kneedle_sensitivity=0.5,
    max_new_edges=None
)
```
