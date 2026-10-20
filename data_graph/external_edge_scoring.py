"""Out-of-core scoring of candidate KNN edges using an external distance batcher.

The callback owns the metric (CPU, GPU, learned decoder, etc.). This module
only provides deterministic edge iteration, within-shard deduplication,
shape validation, and a race-safe single-writer output artifact.

Candidate retrieval remains external. Scoring a KNN topology under a second
metric does not establish nearest-neighbor recall under it.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable
from uuid import uuid4

import numpy as np


class ExternalEdgeScoreError(ValueError):
    """Invalid graph or distance scorer output."""


def score_knn_row_shards(
    node_features: np.ndarray,
    neighbor_ids: np.ndarray,
    batcher: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    *,
    rows_per_shard: int = 2048,
    output_path: str | os.PathLike[str] | None = None,
    missing_neighbor: int = -1,
    progress: Callable[[dict], None] | None = None,
) -> np.ndarray:
    """Score directed KNN positions in bounded, deduplicated row shards.

    `batcher(features, source_ids, target_ids)` receives sorted undirected
    pairs, each once within a row shard. Distances must be nonnegative and
    finite. The callback receives a complete shard to optimize node decoding.

    Missing neighbors receive NaN; self-edges receive exactly zero. The result
    is float32 and keeps the input KNN order. When output_path is given,
    scores stream to a temporary .npy memmap and publish atomically. A
    single-writer .lock blocks concurrent writers. A hard-killed worker can
    leave a lock requiring manual inspection.

    Deduplication is within each row shard, not globally across shards.
    This method does not select nearest neighbors, symmetrize graph topology,
    convert distances to affinities, or handle the batcher's GPU cache.
    """
    features = np.asarray(node_features)
    neighbors = np.asarray(neighbor_ids)
    if features.ndim != 2 or not np.issubdtype(features.dtype, np.number):
        raise ExternalEdgeScoreError("node_features must be a numeric 2D array")
    if neighbors.ndim != 2 or not np.issubdtype(neighbors.dtype, np.integer):
        raise ExternalEdgeScoreError("neighbor_ids must be an integer 2D array")
    n, k = neighbors.shape
    if not n or not k or features.shape[0] != n:
        raise ExternalEdgeScoreError("nonempty candidate array must match feature rows")
    if rows_per_shard < 1:
        raise ExternalEdgeScoreError("rows_per_shard must be positive")
    if 0 <= missing_neighbor < n:
        raise ExternalEdgeScoreError("missing_neighbor overlaps valid node identifiers")
    if not callable(batcher):
        raise ExternalEdgeScoreError("batcher must be callable")

    final = Path(output_path) if output_path is not None else None
    pending = None
    lock = None
    lock_acquired = False
    scores = None
    try:
        if final is not None:
            if final.suffix != ".npy":
                raise ExternalEdgeScoreError("output_path must have .npy extension")
            if not final.parent.is_dir():
                raise FileNotFoundError(f"Output directory does not exist: {final.parent}")
            lock = final.with_name(final.name + ".lock")
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            lock_acquired = True
            with os.fdopen(fd, "w") as f:
                f.write(f"pid={os.getpid()}\n")
            pending = final.with_name(final.name + "." + uuid4().hex + ".pending.npy")
            scores = np.lib.format.open_memmap(
                pending, mode="w+", dtype=np.float32, shape=(n, k)
            )
        else:
            scores = np.empty((n, k), dtype=np.float32)

        for lo in range(0, n, rows_per_shard):
            hi = min(n, lo + rows_per_shard)
            slab = np.asarray(neighbors[lo:hi], dtype=np.int64)
            valid = slab != missing_neighbor
            chosen = slab[valid]
            if chosen.size and (np.min(chosen) < 0 or np.max(chosen) >= n):
                raise ExternalEdgeScoreError(f"out-of-range candidate in rows [{lo},{hi})")
            shard_scores = np.full(slab.shape, np.nan, dtype=np.float32)
            pair_count = 0
            if chosen.size:
                sources = np.broadcast_to(
                    np.arange(lo, hi, dtype=np.int64)[:, None], slab.shape
                )[valid]
                pairs = np.stack(
                    [np.minimum(sources, chosen), np.maximum(sources, chosen)],
                    axis=1,
                )
                unique_pairs, inverse = np.unique(pairs, axis=0, return_inverse=True)
                pair_count = len(unique_pairs)
                unique_scores = np.zeros(pair_count, dtype=np.float32)
                nonself = unique_pairs[:, 0] != unique_pairs[:, 1]
                if np.any(nonself):
                    raw = np.asarray(
                        batcher(
                            features,
                            unique_pairs[nonself, 0],
                            unique_pairs[nonself, 1],
                        )
                    )
                    if raw.shape != (int(nonself.sum()),):
                        raise ExternalEdgeScoreError(
                            f"batcher returned shape {raw.shape}, expected {(int(nonself.sum()),)}"
                        )
                    if not np.all(np.isfinite(raw)) or np.any(raw < 0):
                        raise ExternalEdgeScoreError("batcher returned nonfinite or negative distances")
                    unique_scores[nonself] = raw.astype(np.float32)
                    if not np.all(np.isfinite(unique_scores)):
                        raise ExternalEdgeScoreError("distance overflow in float32 output")
                shard_scores[valid] = unique_scores[inverse]
            scores[lo:hi] = shard_scores
            if progress is not None:
                progress({
                    "row_start": lo,
                    "row_stop": hi,
                    "candidate_positions": int(chosen.size),
                    "unique_undirected_pairs": int(pair_count),
                })

        if pending is not None:
            scores.flush()
            scores = None
            os.replace(pending, final)
            pending = None
            return np.load(final, mmap_mode="r", allow_pickle=False)
        return scores
    finally:
        if scores is not None and isinstance(scores, np.memmap):
            scores.flush()
        if pending is not None:
            pending.unlink(missing_ok=True)
        if lock_acquired:
            lock.unlink(missing_ok=True)
