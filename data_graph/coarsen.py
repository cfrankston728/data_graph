# data_graph/coarsen.py (new)
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Literal
import numpy as np
import scipy.sparse as sp

@dataclass
class SparsifyConfig:
    strategy: Literal["topk", "threshold", "percentile", "eps", "knn"] = "topk"
    param: float = 8.0                 # e.g., k for topk, threshold value, etc.
    per_node: bool = True
    symmetrize: Literal["max", "mean", "sum", "or"] = "max"
    keep_mst: bool = True              # keep MST edges after sparsify
    include_self_loops: bool = False

@dataclass
class CoarsenConfig:
    method: Literal["leiden", "louvain", "labelprop", "custom"] = "leiden"
    resolution: float = 1.0
    random_state: int = 0
    max_levels: int = 1                # how many times to cascade coarsening
    pre_sparsify: Optional[SparsifyConfig] = None
    post_sparsify: Optional[SparsifyConfig] = SparsifyConfig(strategy="topk", param=16)

@dataclass
class CoarsenedLevel:
    level: int
    membership: np.ndarray             # shape (n_nodes_at_parent,), values in [0..C-1]
    sizes: np.ndarray                  # size per supernode, shape (C,)
    A: sp.csr_matrix                   # coarse adjacency (C x C), typically weighted
    params: Dict[str, Any]             # coarsen + sparsify params, timestamps, versions
    parent_level: Optional[int]        # None for original graph, else previous level
