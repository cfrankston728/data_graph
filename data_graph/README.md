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
* [Benchmarking & Timing](#benchmarking--timing)

  * [End-to-End Pipeline (2M scMicroC models)](#endtoend-pipeline-2m-scmicroc-models)
  * [Example: Large-Scale Run Log](#example-largescale-run-log)
  * [Timing Summary (parsed)](#timing-summary-parsed)
  * [Interpreting the Numbers](#interpreting-the-numbers)
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

## Benchmarking & Timing

### End-to-End Pipeline (2M scMicroC models)

Rough wall-clock times observed on a ~2M-item scMicroC dataset (10kbp scHiC + scRNA), provided for planning/ballparking:

* **Data prep**

  * Imputations (10kbp scHiC + scRNA): ~**6 hours**
  * Submatrix extractions (independent of imputation): ~**1 hour**
* **Modeling**

  * VAE training (~50 epochs, batch 512): ~**5–6 hours**
  * Full dataset encodings: ~**1–2 hours** *(can be faster with larger batch size)*
* **Graph**

  * Build data graph: ~**10–30 minutes** *(~10 min with no QC/polishing)*
  * Save data graph: ~**3 minutes**
  * Load data graph: ~**30 seconds**
  * Coarsen/sparsify (for fast Louvain/Leiden): ~**10 minutes** *(one-time; could be amortized in graph gen)*
  * Leiden per resolution (post-coarsen/sparsify): ~**10–15 minutes**

    * Warm-starts across successive resolutions are theoretically feasible but would require adding support in csr-native implementations (see discussion: [https://github.com/sknetwork-team/scikit-network/discussions/588#discussioncomment-13761800](https://github.com/sknetwork-team/scikit-network/discussions/588#discussioncomment-13761800)).

> Actual times depend on hardware (CPU cores/threads, RAM, storage throughput) and FAISS index settings.

---

### Example: Large-Scale Run Log

Below is an excerpt from running the interface-detection pipeline on a graph with **2,056,772 nodes** and **330,372,002 edges** (persistent pool of 32 threads):

```
Starting optimized pipeline...
Processing 2 resolutions: [1.5, 1.75]

Loading graph...
  Nodes: 2,056,772
  Edges: 330,372,002
  [Graph loading] completed in 30.60 seconds

Initializing analyzer...
Analyzer initialized (lazy mode)
  Nodes: 2,056,772
  Edges: 330,372,002

Pre-building CSR for interface detection...
Loading CSR graph structure from cache...
  [Analyzer initialization] completed in 22.63 seconds

Processing resolutions sequentially...
Preparing graph…
  Pre-coarsening sparsification step
Sparsifying pre-coarsened graph (k=80)…
  Sparsified pre-coarsened graph to 324,666,490 edges
  → Coarsening step
Applying 2 levels of coarsening…
  Spawning a persistent pool of 32 threads
  Level 1: 2056772 nodes
    → 1769564 meta‐nodes (ratio: 0.860)
    Building meta-edges of coarsened graph...
    Computing target batching for edge aggregation...
    Organizing edge batches...
    Aggregating edge batches in parallel...
    Combining aggregated edges...
    → Aggregated to 228602194 edges; distances tracked
Sparsifying post-coarsened graph (k=68)…
  Sparsified to 237,643,288 edges
  Level 2: 1769564 nodes
    → 1689972 meta‐nodes (ratio: 0.955)
    Stopping early

--- Running CSR-native Leiden with resolution=1.5 ---
Found 26 communities
Calculating cluster statistics...
Applying standardized Yeo-Johnson transform to 'RNA_stat'...
Created transformed column 'RNA_stat_transformed'

Pruning small clusters:
- Size threshold: 50 (knee point at index 25)
- Kept 26 clusters, pruned 0 clusters
Identifying interface edges...
Found 123052888 interface edges:
  - 123052888 cross-community edges
  - 0 edges in pruned communities
Calculating community statistics for run 'experiment1_res1.5'...
  [Process resolution 1.5] completed in 1456.39 seconds

--- Running CSR-native Leiden with resolution=1.75 ---
Found 30 communities
Calculating cluster statistics...
Applying standardized Yeo-Johnson transform to 'RNA_stat'...
Created transformed column 'RNA_stat_transformed'

Pruning small clusters:
- Size threshold: 50 (knee point at index 29)
- Kept 30 clusters, pruned 0 clusters
Identifying interface edges...
Found 130296504 interface edges:
  - 130296504 cross-community edges
  - 0 edges in pruned communities
Calculating community statistics for run 'experiment1_res1.75'...
  [Process resolution 1.75] completed in 1065.85 seconds

Pipeline complete!
Processed 2 resolutions: [1.5, 1.75]
Results saved to: /home/groups/CEDAR/franksto/00_PROJECT_HUB/version_1/projects/HiC_VAE/version-1.0/data/2025-10-20/2025-10-18_scMicroC_32D_QC_data_graph_YES_GC_3/leiden_csr_graph_interfaces

======== TIMING SUMMARY ========
Total execution time: 2575.47 seconds

Breakdown by operation:
  Process resolution 1.5            1456.39s ( 56.55%)  |  1 calls, avg 1456.3926s per call
  Process resolution 1.75           1065.85s ( 41.38%)  |  1 calls, avg 1065.8484s per call
  Leiden clustering (res=1.75)       773.58s ( 30.04%)  |  1 calls, avg 773.5765s per call
  Leiden clustering (res=1.5)        743.20s ( 28.86%)  |  1 calls, avg 743.1970s per call
  Identify interface edges           369.77s ( 14.36%)  |  2 calls, avg 184.8865s per call
  Save interface edges               145.95s (  5.67%)  |  2 calls, avg 72.9734s per call
  Pre-coarsened graph sparsification      59.19s (  2.30%)  |  1 calls, avg 59.1920s per call
  Graph sparsification                43.42s (  1.69%)  |  1 calls, avg 43.4177s per call
  Graph loading                       30.60s (  1.19%)  |  1 calls, avg 30.5975s per call
  Analyzer initialization             22.63s (  0.88%)  |  1 calls, avg 22.6323s per call
  Load edge arrays                    20.74s (  0.81%)  |  1 calls, avg 20.7416s per call
  Calculate community statistics (run=experiment1_res1.5)      19.00s (  0.74%)  |  1 calls, avg 18.9978s per call
  Calculate community statistics (run=experiment1_res1.75)      17.93s (  0.70%)  |  1 calls, avg 17.9313s per call
  Load CSR cache                       8.10s (  0.31%)  |  1 calls, avg 8.0975s per call
  Process cluster statistics           1.80s (  0.07%)  |  2 calls, avg 0.8990s per call
  Load node dataframe                  1.75s (  0.07%)  |  1 calls, avg 1.7535s per call
  Process cluster labels               1.23s (  0.05%)  |  2 calls, avg 0.6157s per call
  Store run results                    0.00s (  0.00%)  |  2 calls, avg 0.0000s per call
================================

Closing analyzer pool...
Coarsening pool closed.
```

---

### Timing Summary (parsed)

| Operation                            | Time (s) |  Share | Calls | Avg/Call (s) |
| ------------------------------------ | -------: | -----: | ----: | -----------: |
| Process resolution **1.5**           |  1456.39 | 56.55% |     1 |      1456.39 |
| Process resolution **1.75**          |  1065.85 | 41.38% |     1 |      1065.85 |
| Leiden clustering (res=1.75)         |   773.58 | 30.04% |     1 |       773.58 |
| Leiden clustering (res=1.5)          |   743.20 | 28.86% |     1 |       743.20 |
| Identify interface edges (both runs) |   369.77 | 14.36% |     2 |       184.89 |
| Save interface edges                 |   145.95 |  5.67% |     2 |        72.97 |
| Pre-coarsened graph sparsification   |    59.19 |  2.30% |     1 |        59.19 |
| Graph sparsification (post-coarsen)  |    43.42 |  1.69% |     1 |        43.42 |
| Graph loading                        |    30.60 |  1.19% |     1 |        30.60 |
| Analyzer initialization              |    22.63 |  0.88% |     1 |        22.63 |
| Load edge arrays                     |    20.74 |  0.81% |     1 |        20.74 |

**Total:** 2575.47 s (~42.9 min)

---

### Interpreting the Numbers

* **Leiden dominates** when run on very large graphs; consider:

  * Running **coarsening + sparsification** (already ~10 min one-time) to shrink the problem.
  * Using **fewer / smarter interface-edge checks** if you only need cross-community boundaries on a subset.
  * Implementing **warm-starts** across nearby resolutions (requires support in csr-native backends; see discussion linked above).
* **I/O is small** relative to compute on SSD/NVMe (~3 min save, ~30 s load).
* **Pre-KNN reduction** and **FAISS index choice** can cut graph build time significantly without hurting neighbor recall when tuned properly.

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
