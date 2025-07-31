# DataGraphGenerator: Custom Distance Graph Library

A Python library for creating and refining sparse graph representations of data according to custom **premetric** weighting functions.

## Overview

`DataGraphGenerator` builds data-driven graphs where edges represent meaningful relationships between data points. It leverages user-defined premetric weight functions and applies graph refinement techniques to produce high-quality, sparse graph representations.

## Key Features

* **Custom premetric** weight functions for flexible distance definitions
* Embedding-based KNN for initial neighbor discovery
* MST-based connectivity to guarantee a fully connected backbone
* Automatic knee-point detection for intelligent graph pruning
* Graph polishing via 2-hop neighbor connections
* Iterative refine-and-polish cycles controlled by `polish_iterations`
* Comprehensive timing statistics and optimization diagnostics
* Memory-efficient batch processing for large datasets

## Usage Example

```python
import pandas as pd
import numpy as np
from numba import njit
from data_graph_generator import DataGraphGenerator

# 1. Define a Numba-compatible premetric function
@njit(parallel=True)
def my_premetric(features, i_indices, j_indices):
    """Compute custom premetric distances between data point pairs."""
    result = np.zeros(len(i_indices), dtype=np.float32)
    for idx in range(len(i_indices)):
        i, j = i_indices[idx], j_indices[idx]
        # Example: Manhattan distance
        result[idx] = np.sum(np.abs(features[i] - features[j]))
    return result

# 2. Define an embedding function for KNN search
def my_embedding(row):
    """Map DataFrame row to numeric vector for neighbor search."""
    return np.array([row['x'], row['y']])

# 3. Create the graph generator
generator = DataGraphGenerator(
    node_df=df,
    feature_cols=['x', 'y', 'z'],
    premetric_weight_function=my_premetric,
    embedding_function=my_embedding,
    verbose=True
)

# 4. Build and refine the graph
graph_obj, results = generator.build_and_refine_graph(
    n_neighbors=30,
    mst_iterations=3,
    polish_iterations=2,    # renamed from smooth_iterations
    prune_threshold=None,
    kneedle_sensitivity=1.0,
    max_new_edges=500,      # limit on 2-hop additions
    preserve_mst=True
)

# 5. Inspect results
sparse_graph = graph_obj.graph          # scipy.sparse.csr_matrix
labels = graph_obj.component_labels     # numpy array of component IDs
stats = results['timing_summary']      # timing breakdown
```

## Core Components

### `DataGraphGenerator`

Main class that orchestrates the pipeline:

```python
generator = DataGraphGenerator(
    node_df,                       # pandas.DataFrame of nodes
    feature_cols,                  # list of columns to compute premetric
    premetric_weight_function,     # custom premetric weight function
    embedding_function,            # function mapping row to embedding
    verbose=True,                  # show console output
    use_float32=True,              # use float32 arrays
    n_jobs=-1,                     # parallel jobs for sklearn KNN
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

1. **KNN + MST**: Build KNN with `n_neighbors` in embedding space, then refine via MST scans
2. **Prune**: Remove edges above `prune_threshold` (auto-detected via kneedle)
3. **Polish**: Add edges between 2-hop neighbors, up to `max_new_edges`
4. **Iterate**: Alternate steps 2 & 3, for `polish_iterations`

## Output

* **`DataGraph` object**: holds `graph` (CSR), `component_labels`, and metadata
* **`results` dict**: includes original edge data, component sizes, history, and timing stats

## Advanced Usage

### Custom Premetrics

Your premetric function signature must be:

```python
@njit(parallel=True)
def distance_fn(features: np.ndarray, i_indices: np.ndarray, j_indices: np.ndarray) -> np.ndarray:
    # returns array of distances
    ...
```

### Controlling Density

```python
# Sparser graph
graph_obj, results = generator.build_and_refine_graph(
    n_neighbors=10,
    prune_threshold=0.3,
    kneedle_sensitivity=2.0,
    max_new_edges=200
)

# Denser graph
graph_obj, results = generator.build_and_refine_graph(
    n_neighbors=50,
    prune_threshold=None,
    kneedle_sensitivity=0.5,
    max_new_edges=None
)
```

## Installation

```bash
pip install data-graph
```

## Dependencies

* `numpy`
* `scipy`
* `pandas`
* `scikit-learn`
* `numba`
* `matplotlib` (optional)
