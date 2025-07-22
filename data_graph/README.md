# DataGraphGenerator: Custom Distance Graph Library

A Python library for creating and refining sparse graph representations of data according to custom semimetric weighting functions.

## Overview

DataGraphGenerator builds data-driven graphs where edges represent meaningful relationships between data points. It uses custom weight edges and applies graph refinement techniques to produce high-quality sparse graph representations.

## Key Features

- Custom distance metrics through user-defined semimetric weight functions
- Flexible embedding functions for initial neighbor discovery
- Robust graph construction with MST to ensure connectivity
- Intelligent graph pruning with automatic threshold detection
- Graph smoothing by connecting 2-hop neighbors
- Iterative refinement for optimal graph structure
- Comprehensive timing statistics and optimization diagnostics
- Memory-efficient batch processing for large datasets

## Usage Example

```python
import pandas as pd
import numpy as np
from data_graph_generator import DataGraphGenerator

# Define custom distance function (Numba-compatible)
@njit(parallel=True)
def my_distance(features, i_indices, j_indices):
    """Compute custom distances between pairs of data points"""
    result = np.zeros(len(i_indices), dtype=np.float32)
    for idx in range(len(i_indices)):
        i, j = i_indices[idx], j_indices[idx]
        # Custom distance calculation
        result[idx] = np.sum(np.abs(features[i] - features[j]))
    return result

# Define embedding function for initial neighbor discovery
def my_embedding(row):
    """Convert a data row to coordinates for KNN search"""
    return np.array([row['feature1'], row['feature2']])

# Create graph generator
generator = DataGraphGenerator(
    node_df=df,
    feature_cols=['feature1', 'feature2', 'feature3'],
    semimetric_weight_function=my_distance,
    embedding_function=my_embedding
)

# Build and refine graph
graph_obj, results = generator.build_and_refine_graph(
    n_neighbors=30,
    mst_iterations=3,
    smooth_iterations=2,
    preserve_mst=True
)

# Access the graph
sparse_matrix = graph_obj.graph
component_labels = graph_obj.component_labels
```

## Core Components

### DataGraphGenerator

The main class that orchestrates graph creation and refinement:

```python
generator = DataGraphGenerator(
    node_df,                      # DataFrame with node data
    feature_cols,                 # Columns to use for distance calculation
    semimetric_weight_function,   # Custom distance function
    embedding_function,           # Function to create coordinates for KNN
    verbose=True,                 # Print progress messages
    use_float32=True,             # Use float32 for memory efficiency
    n_jobs=-1,                    # Parallel processing jobs
    plot_knee=False               # Plot knee point detection
)
```

### Graph Building Pipeline

1. **KNN + MST Graph**: Creates initial graph using k-nearest neighbors in embedding space and ensures connectivity with a minimum spanning tree
2. **Pruning**: Removes edges with distances above a threshold (determined automatically by knee-point detection or user-specified)
3. **Smoothing**: Adds edges between 2-hop neighbors to improve graph structure
4. **Iterative Refinement**: Alternates between pruning and smoothing to optimize graph structure

### Output

- **DataGraph object**: Contains the sparse graph matrix and associated metadata
- **Results dictionary**: Contains detailed statistics, component information, and timing data

## Optimization Considerations

The code includes several performance optimizations:

- Numba-accelerated distance computations
- Batch processing for memory efficiency
- Efficient graph algorithms (MST, connected components)
- Memory usage estimation for optimal batch sizing
- Tracking of MST edges to maintain connectivity
- Early stopping when graph structure stabilizes

## Dependencies

- numpy
- scipy
- pandas
- scikit-learn
- numba
- matplotlib (for optional visualization)

## Installation

```bash
pip install data-graph
```

## Advanced Usage

### Custom Distance Functions

The semimetric weight function must be Numba-compatible and have the signature:

```python
@njit(parallel=True)
def distance_function(features, i_indices, j_indices):
    # features: np.array of shape (n_samples, n_features)
    # i_indices, j_indices: arrays of node indices to compute distances between
    # return: array of distances with same length as i_indices
    ...
```

### Controlling Graph Density

Adjust these parameters to control graph density:

```python
# Sparser graph
graph_obj, results = generator.build_and_refine_graph(
    n_neighbors=15,              # Fewer initial neighbors
    prune_threshold=0.5,         # Lower threshold removes more edges
    kneedle_sensitivity=1.5,     # Higher sensitivity finds knee point earlier
    max_new_edges=1000           # Limit number of 2-hop connections
)

# Denser graph
graph_obj, results = generator.build_and_refine_graph(
    n_neighbors=50,              # More initial neighbors
    prune_threshold=None,        # Auto-detect threshold
    kneedle_sensitivity=0.5,     # Lower sensitivity finds knee point later
    max_new_edges=None           # No limit on 2-hop connections
)
```