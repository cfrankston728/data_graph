from __future__ import annotations

import numpy as np
from numba import njit
from scipy import sparse


@njit(
    cache=False,
    nogil=True,
)
def _same_label_roots(
    indptr,
    indices,
    labels,
):
    n = labels.size

    parent = np.arange(
        n,
        dtype=np.int32,
    )

    rank = np.zeros(
        n,
        dtype=np.uint8,
    )

    for u in range(n):
        label_u = labels[u]

        for p in range(
            indptr[u],
            indptr[u + 1],
        ):
            v = indices[p]

            # Undirected graph: process each pair once.
            if v <= u:
                continue

            if labels[v] != label_u:
                continue

            x = u

            while parent[x] != x:
                parent[x] = parent[
                    parent[x]
                ]
                x = parent[x]

            root_u = x

            x = v

            while parent[x] != x:
                parent[x] = parent[
                    parent[x]
                ]
                x = parent[x]

            root_v = x

            if root_u == root_v:
                continue

            if (
                rank[root_u]
                < rank[root_v]
            ):
                parent[root_u] = root_v

            elif (
                rank[root_u]
                > rank[root_v]
            ):
                parent[root_v] = root_u

            else:
                parent[root_v] = root_u
                rank[root_u] += 1

    roots = np.empty(
        n,
        dtype=np.int32,
    )

    for u in range(n):
        x = u

        while parent[x] != x:
            parent[x] = parent[
                parent[x]
            ]
            x = parent[x]

        roots[u] = x

    return roots


def _compact(labels):
    labels = np.asarray(
        labels
    )

    if labels.ndim != 1:
        raise ValueError(
            "membership labels must be one-dimensional"
        )

    if not np.issubdtype(
        labels.dtype,
        np.integer,
    ):
        raise TypeError(
            "membership labels must be integer-valued"
        )

    if (
        labels.size
        and np.any(
            labels < 0
        )
    ):
        raise ValueError(
            "membership labels must be non-negative"
        )

    if labels.size == 0:
        return np.empty(
            0,
            dtype=np.int32,
        )

    return np.unique(
        labels,
        return_inverse=True,
    )[1].astype(
        np.int32,
        copy=False,
    )


def canonicalize_connected_membership(
    adjacency,
    labels,
):
    """Split every supplied community into graph-connected pieces.

    This operation only refines the supplied partition. It never
    merges distinct input communities.

    Intended execution point:
        after Leiden CSR preprocessing and before level-0 local moving.

    Complexity:
        O(n + nnz) graph traversal,
        O(n) auxiliary arrays,
        plus deterministic compact-label canonicalization.
    """

    if not sparse.isspmatrix_csr(
        adjacency
    ):
        raise TypeError(
            "adjacency must be a scipy CSR matrix"
        )

    if (
        adjacency.shape[0]
        != adjacency.shape[1]
    ):
        raise ValueError(
            "adjacency must be square"
        )

    if (
        adjacency.indices.dtype
        != np.int32
        or adjacency.indptr.dtype
        != np.int32
    ):
        raise TypeError(
            "connected warm-start canonicalization "
            "requires int32 CSR indices/indptr"
        )

    labels = _compact(
        labels
    )

    n = int(
        adjacency.shape[0]
    )

    if labels.shape != (n,):
        raise ValueError(
            "membership length does not match graph node count"
        )

    if n == 0:
        return labels

    roots = _same_label_roots(
        adjacency.indptr,
        adjacency.indices,
        labels,
    )

    # Pair each old community id with its within-community
    # connected-component root. np.unique sorts deterministically.
    encoded = (
        labels.astype(
            np.int64
        )
        * np.int64(
            n + 1
        )
        + roots.astype(
            np.int64
        )
    )

    return np.unique(
        encoded,
        return_inverse=True,
    )[1].astype(
        np.int32,
        copy=False,
    )
