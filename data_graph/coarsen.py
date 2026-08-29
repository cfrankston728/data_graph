"""Configuration containers for planned graph coarsening workflows.

This module currently defines lightweight dataclasses only. The active
community-detection pipeline performs reciprocal-nearest-neighbor coarsening in
``generate_community_interface_subgraphs.py``; these containers document the
intended package-level API surface for a future persistent/amortized coarsening
path without changing current runtime behavior.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Literal
import numpy as np
import scipy.sparse as sp

@dataclass
class SparsifyConfig:
    """Options for reducing graph edge density before or after coarsening.

    The fields are intentionally declarative: callers can describe top-k,
    threshold, percentile, epsilon, or KNN-style pruning while leaving the exact
    implementation to the coarsening pipeline that consumes the config.
    """
    strategy: Literal["topk", "threshold", "percentile", "eps", "knn"] = "topk"
    param: float = 8.0                 # e.g., k for topk, threshold value, etc.
    per_node: bool = True
    symmetrize: Literal["max", "mean", "sum", "or"] = "max"
    keep_mst: bool = True              # keep MST edges after sparsify
    include_self_loops: bool = False

@dataclass
class CoarsenConfig:
    """High-level request for one or more graph coarsening levels.

    ``method`` names the clustering/matching strategy that should create
    supernodes. ``pre_sparsify`` and ``post_sparsify`` allow callers to record
    density-control choices separately from the graph-construction settings.
    """
    method: Literal["leiden", "louvain", "labelprop", "custom"] = "leiden"
    resolution: float = 1.0
    random_state: int = 0
    max_levels: int = 1                # how many times to cascade coarsening
    pre_sparsify: Optional[SparsifyConfig] = None
    post_sparsify: Optional[SparsifyConfig] = SparsifyConfig(strategy="topk", param=16)

@dataclass
class CoarsenedLevel:
    """Materialized state for one level in a coarsening hierarchy.

    ``membership`` maps parent-level nodes to supernodes in ``A``. ``sizes`` and
    ``params`` preserve enough context to project labels back to finer levels
    and audit how the coarse adjacency was produced.
    """
    level: int
    membership: np.ndarray             # shape (n_nodes_at_parent,), values in [0..C-1]
    sizes: np.ndarray                  # size per supernode, shape (C,)
    A: sp.csr_matrix                   # coarse adjacency (C x C), typically weighted
    params: Dict[str, Any]             # coarsen + sparsify params, timestamps, versions
    parent_level: Optional[int]        # None for original graph, else previous level
