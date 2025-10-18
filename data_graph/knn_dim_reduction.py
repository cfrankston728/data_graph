# data_graph/knn_dim_reduction.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, Any
import numpy as np

from sklearn.decomposition import PCA, IncrementalPCA, TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection

@dataclass
class KNNReductionConfig:
    """
    Settings for dimensionality reduction used *only* to choose KNN neighbors.
    """
    method: str = "none"        # "none" | "pca" | "ipca" | "svd" | "rp" | "srp"
    n_components: Optional[int] = None  # target dim (overrides var/eps)
    var: Optional[float] = None # target explained variance for pca/ipca (e.g., 0.97)
    eps: float = 0.15           # JL distortion for RP/SRP ("auto" n_components)
    batch_size: Optional[int] = None    # ipca batch size
    random_state: Optional[int] = 0
    cast_float32: bool = True

def _auto_ipca_batch_size(n: int, d: int) -> int:
    # ~200MB cap in float32
    target_bytes = 200_000_000
    per_row = d * 4
    bs = max(1024, min(n, target_bytes // max(per_row, 1)))
    # snap near a power-of-two
    return int(max(512, 2**int(np.round(np.log2(bs)))))

def _estimate_n_for_var(Xs: np.ndarray, var: float, random_state: int) -> int:
    k_try = min(Xs.shape[1], max(8, int(0.6 * Xs.shape[1])))
    pca = PCA(n_components=k_try, svd_solver="randomized", random_state=random_state)
    pca.fit(Xs)
    cum = np.cumsum(pca.explained_variance_ratio_)
    return int(np.searchsorted(cum, var) + 1)

def reduce_for_knn(
    X: np.ndarray,
    cfg: Optional[KNNReductionConfig],
    *,
    verbose: bool = False,
    sample_for_var: int = 100_000
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Return (X_reduced, info) for KNN neighbor *selection only*.
    """
    if cfg is None or cfg.method == "none":
        Y = X.astype(np.float32) if (cfg and cfg.cast_float32) else X
        return Y, {"method": "none", "n_components": Y.shape[1]}

    n, d = X.shape
    info: Dict[str, Any] = {"input_shape": (n, d), **asdict(cfg)}
    method = cfg.method.lower()
    n_components = cfg.n_components

    if method in ("pca", "ipca") and n_components is None and cfg.var is not None:
        s = min(n, sample_for_var)
        Xs = X if s == n else X[np.random.default_rng(cfg.random_state).choice(n, s, replace=False)]
        n_components = min(d, max(2, _estimate_n_for_var(Xs, cfg.var, cfg.random_state)))
        if verbose:
            print(f"[KNN-DR] chose n_components={n_components} for var≈{cfg.var}")

    if method == "pca":
        n_components = n_components or min(d, max(16, int(0.5 * d)))
        Y = PCA(n_components=n_components, svd_solver="randomized",
                random_state=cfg.random_state).fit_transform(X)

    elif method == "ipca":
        n_components = n_components or min(d, max(16, int(0.5 * d)))
        bs = cfg.batch_size or _auto_ipca_batch_size(n, d)
        ipca = IncrementalPCA(n_components=n_components, batch_size=bs)
        for s in range(0, n, bs):
            ipca.partial_fit(X[s:s+bs])
        Y = np.empty((n, n_components), dtype=np.float32)
        for s in range(0, n, bs):
            Y[s:s+bs] = ipca.transform(X[s:s+bs])

    elif method == "svd":
        n_components = n_components or min(d-1, max(16, int(0.5 * d)))
        Y = TruncatedSVD(n_components=n_components, random_state=cfg.random_state).fit_transform(X)

    elif method in ("rp", "srp"):
        proj_cls = GaussianRandomProjection if method == "rp" else SparseRandomProjection
        proj = proj_cls(n_components="auto", eps=cfg.eps, random_state=cfg.random_state)
        Y = proj.fit_transform(X)
        n_components = Y.shape[1]

    else:
        raise ValueError(f"Unknown reduction method: {cfg.method}")

    if cfg.cast_float32 and Y.dtype != np.float32:
        Y = Y.astype(np.float32, copy=False)

    info.update({"method": method, "n_components": int(n_components), "output_shape": tuple(Y.shape)})
    return Y, info
