"""Out-of-core full-population FAISS KNN construction.

Extracted from CYC26 V13/V14 (K200 FAISS IndexIVFFlat, IPCA,
chunked queries, and .npy memory-mapped topology). These candidates
are nearest in reduced latent embedding, NOT decoder Fisher-Rao space.
"""
from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import numpy as np


def fit_full_ipca(embedded, *, n_components=9, io_block=65536,
                  output_path=None):
    """Incremental PCA in two bounded passes, as in CYC26 V13."""
    from sklearn.decomposition import IncrementalPCA

    data = np.asarray(embedded)
    if data.ndim != 2 or not 0 < n_components <= data.shape[1]:
        raise ValueError("invalid embedding shape or n_components")
    if io_block < n_components:
        raise ValueError("io_block must accommodate n_components")
    ipca = IncrementalPCA(n_components=n_components, batch_size=io_block)
    # Join a trailing undersized block to the previous fitting block.
    starts = list(range(0, len(data), io_block))
    if len(starts) > 1 and len(data) - starts[-1] < n_components:
        starts.pop()
    for t, lo in enumerate(starts):
        hi = starts[t + 1] if t + 1 < len(starts) else len(data)
        chunk = np.asarray(data[lo:hi], dtype=np.float32)
        if len(chunk) < n_components or not np.isfinite(chunk).all():
            raise ValueError("invalid IPCA fit batch")
        ipca.partial_fit(chunk)
    if output_path is None:
        reduced = np.empty((len(data), n_components), dtype=np.float32)
    else:
        reduced = np.lib.format.open_memmap(
            output_path, mode="w+", dtype=np.float32,
            shape=(len(data), n_components),
        )
    for lo in range(0, len(data), io_block):
        hi = min(lo + io_block, len(data))
        block = np.asarray(ipca.transform(data[lo:hi]), dtype=np.float32)
        if not np.isfinite(block).all():
            raise ValueError("nonfinite IPCA projection")
        reduced[lo:hi] = block
    if isinstance(reduced, np.memmap):
        reduced.flush()
    return reduced, dict(
        nodes=len(data),
        n_components=n_components,
        explained_variance_ratio_sum=float(ipca.explained_variance_ratio_.sum()),
    )


def build_ivf_flat_index(reduced, *, nlist=2048, nprobe=32,
                         train_rows=180000, io_block=65536,
                         seed=291, threads=8):
    """Fit IVF on a seeded subsample, then add the entire population."""
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("FAISS is required for full-population IVF") from exc
    data = np.asarray(reduced)
    if (data.ndim != 2 or len(data) < nlist or
            min(nlist, nprobe, train_rows, io_block) < 1):
        raise ValueError("invalid IVF parameters or feature matrix")
    faiss.omp_set_num_threads(max(1, min(threads, os.cpu_count() or 1)))
    n, d = data.shape
    rng = np.random.default_rng(seed)
    sampled = np.sort(rng.choice(n, min(n, train_rows), replace=False))
    training = np.ascontiguousarray(data[sampled], dtype=np.float32)
    if not np.isfinite(training).all():
        raise ValueError("nonfinite FAISS training rows")
    index = faiss.IndexIVFFlat(faiss.IndexFlatL2(d), d, nlist, faiss.METRIC_L2)
    index.train(training)
    if not index.is_trained:
        raise RuntimeError("FAISS IVF training failed")
    for lo in range(0, n, io_block):
        block = np.ascontiguousarray(data[lo:lo + io_block], dtype=np.float32)
        if not np.isfinite(block).all():
            raise ValueError("nonfinite FAISS population rows")
        index.add(block)
    index.nprobe = nprobe
    if index.ntotal != n:
        raise RuntimeError("FAISS did not index the full population")
    return index, dict(
        nodes=n, nlist=nlist, nprobe=nprobe,
        train_rows=len(sampled), seed=seed,
    )


def select_ranked_nonself_neighbors(candidate_ids, row_ids, *, k, n_nodes):
    """Preserve FAISS rank after removing self, requiring unique complete KNN."""
    ids = np.asarray(candidate_ids, dtype=np.int64)
    rows = np.asarray(row_ids, dtype=np.int64).reshape(-1)
    if ids.ndim != 2 or ids.shape[0] != len(rows) or not 0 < k < n_nodes:
        raise ValueError("invalid KNN candidate array")
    if np.any(ids < -1) or np.any(ids >= n_nodes):
        raise ValueError("invalid FAISS index")
    valid = (ids >= 0) & (ids != rows[:, None])
    ranks = np.cumsum(valid, axis=1, dtype=np.int32)
    if np.any(valid.sum(axis=1) < k):
        raise ValueError("FAISS did not return k distinct candidates")
    selected = ids[valid & (ranks <= k)].reshape(len(rows), k)
    if np.any(selected == rows[:, None]):
        raise ValueError("self-edge survived filtering")
    if k > 1 and np.any(np.diff(np.sort(selected, axis=1), axis=1) == 0):
        raise ValueError("duplicate FAISS candidate")
    return selected.astype(np.uint32 if n_nodes < 2**32 else np.int64)


def iter_full_knn_shards(index, reduced, *, k=200,
                         rows_per_shard=512, extra_candidates=16):
    """Stream KNN row shards from an index covering *all* nodes."""
    data = np.asarray(reduced)
    if (data.ndim != 2 or not 0 < k < len(data) or
            rows_per_shard < 1 or extra_candidates < 0):
        raise ValueError("invalid KNN query configuration")
    if int(index.ntotal) != len(data):
        raise ValueError("FAISS index does not cover the full population")
    for lo in range(0, len(data), rows_per_shard):
        hi = min(lo + rows_per_shard, len(data))
        rows = np.arange(lo, hi, dtype=np.int64)
        queries = np.ascontiguousarray(data[lo:hi], dtype=np.float32)
        if not np.isfinite(queries).all():
            raise ValueError("nonfinite KNN queries")
        _, ids = index.search(
            queries, min(len(data), k + 1 + extra_candidates),
        )
        yield rows, select_ranked_nonself_neighbors(
            ids, rows, k=k, n_nodes=len(data),
        )


def build_full_knn_memmap(index, reduced, output_path, *, k=200,
                          rows_per_shard=512, extra_candidates=16,
                          progress=None):
    """Atomically persist a complete [N,K] candidate matrix, one writer."""
    path = Path(output_path)
    if path.suffix != ".npy" or not path.parent.is_dir():
        raise ValueError("output must be .npy in an existing directory")
    n = len(reduced)
    dtype = np.uint32 if n < 2**32 else np.int64
    lock = path.with_name(path.name + ".lock")
    pending = path.with_name(path.stem + "." + uuid4().hex + ".pending.npy")
    owned = False
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        owned = True
        with os.fdopen(fd, "w") as stream:
            stream.write(f"pid={os.getpid()}\n")
        result = np.lib.format.open_memmap(
            pending, mode="w+", dtype=dtype, shape=(n, k),
        )
        for rows, ids in iter_full_knn_shards(
            index, reduced, k=k, rows_per_shard=rows_per_shard,
            extra_candidates=extra_candidates,
        ):
            result[rows] = ids
            if progress is not None:
                progress(dict(
                    row_start=int(rows[0]), row_stop=int(rows[-1]) + 1, k=k,
                ))
        result.flush()
        del result
        os.replace(pending, path)
        return np.load(path, mmap_mode="r", allow_pickle=False)
    finally:
        pending.unlink(missing_ok=True)
        if owned:
            lock.unlink(missing_ok=True)
