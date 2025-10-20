# DataGraphGenerator — Custom Distance-Graph Builder

*Fast FAISS-KNN with optional pre-KNN dimensionality reduction*

A Python library for creating and refining sparse graphs from data using custom **premetric** (semi-metric) functions. The pipeline includes a **FAISS** backend for high-performance KNN search and an optional **dimensionality reduction** stage **before** neighbor search to boost speed and neighbor recall on high-D data.

---

## Table of Contents

* [Overview](#overview)
* [What’s New](#whats-new)
* [Installation](#installation)
* [Quick Start](#quick-start)
* [Core Concepts](#core-concepts)
* [API](#api)

  * [`DataGraphGenerator`](#datagraphgenerator)
  * [`build_and_refine_graph`](#build_and_refine_graph)
  * [`KNNReductionConfig`](#knnreductionconfig)
* [Pipeline Stages](#pipeline-stages)
* [Controlling Density](#controlling-density)
* [Performance Tips](#performance-tips)
* [Dependencies](#dependencies)
* [License](#license)

---

## Overview

`data_graph` builds data-driven graphs where edges capture meaningful relationships between points. You provide a premetric (distance or similarity) and, optionally, an embedding for neighbor discovery. The library refines an initial KNN graph with MST connectivity, knee-point pruning, and targeted 2-hop “polishing,” producing a high-quality, sparse graph for downstream clustering or community analysis.

---

## What’s New

* **FAISS KNN backend** (CPU/GPU) with automatic fallback to scikit-learn KNN.
* **Pre-KNN dimensionality reduction** (Incremental PCA / PCA / Randomized SVD) to reduce latency and memory while preserving neighbor recall on high-D data.
* **Configurable weight source:** compute final edge weights from the **original** high-D features or from the **embedding / reduced** representation (`knn_weights_from="original"|"embedding"`).
* **Cleaner configuration surface:** `knn_backend="faiss"|"sklearn"`, `knn_backend_opts={...}`, and `knn_reduction=KNNReductionConfig(...)`.
* Numerous micro-optimizations: float32/int32 arrays, batch computations, and optional Numba-accelerated premetrics.

---

## Installation

```bash
# Base library
pip install data-graph
```

Optional FAISS (choose one):

```bash
# CPU build
pip install faiss-cpu

# GPU build (requires CUDA toolkit)
pip install faiss-gpu
```

> If FAISS is not installed, the library falls back to scikit-learn’s KNN.

---

## Quick Start

```python
import numpy as np
import pandas as pd
from numba import njit

from data_graph import DataGraphGenerator
from data_graph.knn_dim_reduction import KNNReductionConfig

# Example data
df = pd.DataFrame({
    "x": np.random.randn(1000),
    "y": np.random.randn(1000),
    "z": np.random.randn(1000),
})

# 1) Define a Numba-compatible premetric (scalar form)
@njit
def manhattan_semimetric(features, i, j):
    return np.abs(features[i] - features[j]).sum()

# 2) Map a row → embedding (used for neighbor search)
def embed_row(row):
    return np.array([row["x"], row["y"]], dtype=np.float32)

# 3) Create the generator
generator = DataGraphGenerator(
    node_df=df,
    feature_cols=["x", "y", "z"],
    premetric_weight_function=manhattan_semimetric,
    embedding_function=embed_row,
    verbose=True,
    knn_backend="faiss",              # "faiss" or "sklearn"
    knn_backend_opts=dict(            # forwarded to FAISS if used
        index_factory="HNSW32,Flat",  # examples: "Flat", "IVF4096,Flat", "HNSW32,Flat"
        metric="l2",
        nprobe=32                     # used by IVF; ignored by HNSW/Flat
    ),
    knn_reduction=KNNReductionConfig( # optional pre-KNN dimensionality reduction
        method="ipca",                # "ipca"|"pca"|"rsvd"
        var=0.98,                     # retain 98% variance
        random_state=0,
        cast_float32=True
    ),
    knn_weights_from="original"       # compute final edge weights in high-D space
)

# 4) Build and refine the graph
graph_obj, results = generator.build_and_refine_graph(
    n_neighbors=50,
    mst_iterations=2,
    polish_iterations=1,
    prune_threshold=None,         # auto via knee-point detection
    kneedle_sensitivity=1.0,
    max_new_edges=500,
    preserve_mst=True
)

# 5) Inspect results
print(graph_obj.graph.shape, "CSR adjacency")
print("Components:", results["n_components"])
print("Timing:", results["timing_summary"])
```

---

## Core Concepts

* **Premetric (semi-metric):** Your custom distance/score between two nodes. Smaller is “closer” by default.
* **Embedding vs. Weight Source:** Use a (possibly reduced) embedding to **find neighbors**, but compute final edge **weights** using either the original features or the embedding.
* **MST Backbone:** A minimum-spanning-tree pass ensures global connectivity.
* **Knee-Point Pruning:** Automatic thresholding (Kneedle) removes spurious long edges.
* **Polishing:** Targeted addition of 2-hop neighbors to recover local structure lost during pruning.

---

## API

### `DataGraphGenerator`

```python
DataGraphGenerator(
    node_df: pd.DataFrame,
    feature_cols: list[str],
    premetric_weight_function,           # see signatures below
    embedding_function=None,             # row -> np.ndarray
    *,
    knn_backend: str = "faiss",          # "faiss" | "sklearn"
    knn_backend_opts: dict | None = None,
    knn_reduction: "KNNReductionConfig" | None = None,
    knn_weights_from: str = "original",  # "original" | "embedding"
    use_float32: bool = True,
    n_jobs: int = -1,                    # for sklearn backend
    plot_knee: bool = False,
    verbose: bool = True,
    random_state: int | None = 42
)
```

**Premetric signatures (both supported):**

* **Scalar:** `fn(features: np.ndarray, i: int, j: int) -> float`
* **Batched:** `fn(features: np.ndarray, i_idx: np.ndarray, j_idx: np.ndarray) -> np.ndarray`

> `features` has shape `(n_nodes, len(feature_cols))`. Use `@njit` (Numba) for speed.

**Important options:**

* `knn_backend_opts` (FAISS examples)

  * `index_factory`: `"Flat"`, `"IVF4096,Flat"`, `"HNSW32,Flat"`, etc.
  * `metric`: `"l2"` (default) or `"ip"` (inner product)
  * `nprobe`: search breadth for IVF indexes (e.g., 16–256)
  * `efSearch`/`efConstruction`: HNSW tuning knobs (if applicable)

* `knn_reduction`: apply dimensionality reduction **before** KNN. See [`KNNReductionConfig`](#knnreductionconfig).

* `knn_weights_from`: decide whether final edge weights are computed in the **original** high-D space or in the **embedding**/reduced space.

---

### `build_and_refine_graph`

```python
graph_obj, results = generator.build_and_refine_graph(
    n_neighbors: int = 30,
    mst_iterations: int = 1,
    polish_iterations: int = 1,
    prune_threshold: float | None = None,    # None → auto via knee-point
    kneedle_sensitivity: float = 1.0,
    max_new_edges: int | None = 1000,
    preserve_mst: bool = True
)
```

**Returns**

* `graph_obj`: a lightweight container with

  * `graph`: `scipy.sparse.csr_matrix` (symmetric, unweighted/weighted depending on configuration)
  * `component_labels`: `np.ndarray[int32]` of connected components
  * `metadata`: dict with build settings

* `results`: dict with

  * `initial_edge_data`: raw KNN edges and distances
  * `component_sizes`, `n_components`
  * `history`: pruning / polishing decisions
  * `timing_summary`: per-stage timings

---

### `KNNReductionConfig`

```python
KNNReductionConfig(
    method: str = "ipca",     # "ipca" | "pca" | "rsvd"
    var: float = 0.98,        # target explained variance (0–1)
    n_components: int | None = None,  # override 'var' if set
    random_state: int | None = 0,
    cast_float32: bool = True
)
```

* **`ipca` (Incremental PCA):** streaming & memory-safe for very large datasets.
* **`pca`:** full PCA (dense) for moderate-sized data.
* **`rsvd`:** randomized SVD for fast low-rank approximations.

---

## Pipeline Stages

1. **(Optional) Pre-KNN Reduction**
   Reduce dimensionality to speed up neighbor search and lower memory.

2. **KNN Construction (FAISS / sklearn)**
   Build a k-NN graph in embedding space with `n_neighbors`.

3. **MST Connectivity**
   Ensure a connected backbone with a lightweight MST pass.

4. **Knee-Point Pruning**
   Auto-detect a distance cutoff via Kneedle to remove weak edges.

5. **Polishing (2-Hop Neighbors)**
   Add a limited number of 2-hop edges (up to `max_new_edges`) to restore local structure.

6. **Iterate**
   Repeat prune → polish for `polish_iterations`. Optionally preserve MST edges.

---

## Controlling Density

```python
# Sparser graph
graph_obj, results = generator.build_and_refine_graph(
    n_neighbors=15,
    prune_threshold=0.25,
    kneedle_sensitivity=2.0,
    max_new_edges=200,
    preserve_mst=True
)

# Denser graph
graph_obj, results = generator.build_and_refine_graph(
    n_neighbors=80,
    prune_threshold=None,      # let knee-point decide
    kneedle_sensitivity=0.7,
    max_new_edges=None
)
```

---

## Performance Tips

* **Prefer float32/int32** for features and indices (`use_float32=True`).
* **Choose an appropriate FAISS index:**

  * *Flat:* exact but slower (small/medium datasets).
  * *IVF*, tune `nlist` (via `index_factory`) and `nprobe` for speed/recall.
  * *HNSW:* high recall with low memory; tune `efSearch` and `M`.
* **Use `ipca`** when `n_samples` or `n_features` is large to keep memory stable.
* **Warm up Numba**: the first call to a JIT premetric compiles; subsequent calls are fast.
* **Set `random_state`** for reproducibility of reduction and approximate search.
* **Batch carefully** if you implement custom pair generation; keep arrays C-contiguous.

---

## End-to-End Example (Latent Means + Scales with Fisher–Rao)

```python
from numba import njit
import numpy as np
from data_graph import DataGraphGenerator
from data_graph.knn_dim_reduction import KNNReductionConfig

n_dims = 8  # latent dimensionality (example)

@njit
def fisher_rao_distance(f, i, j):
    eps      = 1e-8
    safe_eps = 1e-30

    means_i  = f[i,  :n_dims] + 0.0
    sigma_i  = f[i,  n_dims:2*n_dims] + eps
    means_j  = f[j,  :n_dims] + 0.0
    sigma_j  = f[j,  n_dims:2*n_dims] + eps

    var_term = (sigma_i**2 + sigma_j**2) / 2 + eps
    ratio    = var_term / (sigma_i * sigma_j)
    log_term = np.log(np.maximum(ratio, 1e-16))

    diff = means_i - means_j
    d2   = 2 * np.sum(diff**2 / var_term + log_term)
    return np.sqrt(max(safe_eps, d2))

def logspace_embedding_with_noise(row, noise_scale=1e-6):
    means  = np.array([row[f"latent_{k+1}"] for k in range(n_dims)], dtype=np.float32)
    sigmas = np.array([row[f"sig_{k+1}"]    for k in range(n_dims)], dtype=np.float32)

    emb = np.zeros(2 * n_dims, dtype=np.float32)
    emb[0::2] = means / sigmas
    emb[1::2] = np.sqrt(2, dtype=np.float32) * np.log(np.maximum(sigmas, 1e-12))
    emb += np.random.randn(emb.size).astype(np.float32) * noise_scale
    return emb

generator = DataGraphGenerator(
    node_df=model_df,  # your dataframe with latent_* and sig_* columns
    feature_cols=[f"latent_{k+1}" for k in range(n_dims)] +
                 [f"sig_{k+1}"    for k in range(n_dims)],
    premetric_weight_function=fisher_rao_distance,
    embedding_function=logspace_embedding_with_noise,
    verbose=True,
    knn_reduction=KNNReductionConfig(
        method="ipca", var=0.98, random_state=0, cast_float32=True
    ),
    knn_weights_from="original",
    knn_backend="faiss",
    knn_backend_opts=dict(index_factory="HNSW32,Flat", metric="l2")
)

graph_obj, results = generator.build_and_refine_graph(
    n_neighbors=100,
    mst_iterations=1,
    polish_iterations=1,
    preserve_mst=True
)
```

---

## Dependencies

* `numpy`
* `scipy`
* `pandas`
* `scikit-learn`
* `numba`
* `faiss-cpu` or `faiss-gpu` *(optional; recommended for speed)*
* `matplotlib` *(optional; for knee-point plots)*

---

## License

MIT (see `LICENSE` file).
