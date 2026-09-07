from __future__ import annotations

import numpy as np
from scipy import sparse

from .community_backends import (
    run_community_backend,
)
from .connected_warm_start import (
    canonicalize_connected_membership,
)


def gaussian_similarity(
    distances,
    scale="adaptive",
):
    distances = np.asarray(
        distances,
        dtype=np.float32,
    )

    if (
        scale is None
        or scale == "adaptive"
    ):
        scale = (
            1.0
            if distances.size == 0
            else float(
                np.median(
                    distances
                )
            )
        )

    elif isinstance(
        scale,
        str,
    ):
        raise ValueError(
            "unknown similarity scale: "
            + repr(scale)
        )

    else:
        scale = float(
            scale
        )

    if (
        not np.isfinite(
            scale
        )
        or scale <= 0.0
    ):
        raise ValueError(
            "similarity scale must be finite and positive"
        )

    out = distances.copy()

    scale32 = np.float32(
        scale
    )

    np.divide(
        out,
        scale32,
        out=out,
    )

    np.square(
        out,
        out=out,
    )

    np.multiply(
        out,
        np.float32(
            -0.5
        ),
        out=out,
    )

    np.exp(
        out,
        out=out,
    )

    return out


def distance_csr_to_similarity_csr(
    distance_csr,
    *,
    scale="adaptive",
    similarity_function=None,
):
    csr = (
        distance_csr
        .copy()
        .tocsr()
    )

    csr.sum_duplicates()
    csr.sort_indices()

    if similarity_function is None:
        weights = gaussian_similarity(
            csr.data,
            scale=scale,
        )
    else:
        weights = np.asarray(
            similarity_function(
                csr.data,
                scale,
            ),
            dtype=np.float32,
        )

    if (
        weights.shape
        != csr.data.shape
    ):
        raise ValueError(
            "similarity function returned wrong shape"
        )

    csr.data = np.ascontiguousarray(
        weights,
        dtype=np.float32,
    )

    return csr


def distance_csr_from_edge_arrays(
    sources,
    targets,
    distances,
    *,
    n_nodes,
):
    """Fallback only when a loader cannot expose adjacency directly.

    Accept either:
    - unique one-direction undirected edges, or
    - an already symmetric bidirectional edge representation.

    Ambiguous asymmetric bidirectional values are rejected.
    """

    sources = np.asarray(
        sources,
        dtype=np.int32,
    )

    targets = np.asarray(
        targets,
        dtype=np.int32,
    )

    distances = np.asarray(
        distances,
        dtype=np.float32,
    )

    if not (
        sources.shape
        == targets.shape
        == distances.shape
    ):
        raise ValueError(
            "edge arrays must have identical shapes"
        )

    if sources.ndim != 1:
        raise ValueError(
            "edge arrays must be one-dimensional"
        )

    if np.any(
        sources == targets
    ):
        raise ValueError(
            "self-edges are not supported by this fallback"
        )

    csr = sparse.csr_matrix(
        (
            distances,
            (
                sources,
                targets,
            ),
        ),
        shape=(
            int(
                n_nodes
            ),
            int(
                n_nodes
            ),
        ),
        dtype=np.float32,
    )

    csr.sum_duplicates()
    csr.sort_indices()

    all_upper = bool(
        np.all(
            sources
            < targets
        )
    )

    all_lower = bool(
        np.all(
            sources
            > targets
        )
    )

    if (
        all_upper
        or all_lower
    ):
        csr = (
            csr
            + csr.T
        ).tocsr()

        csr.sum_duplicates()
        csr.sort_indices()

        return csr

    diff = (
        csr
        - csr.T
    ).tocsr()

    diff.eliminate_zeros()

    if diff.nnz:
        max_abs = float(
            np.max(
                np.abs(
                    diff.data
                )
            )
        )

        if max_abs > 1e-6:
            raise ValueError(
                "bidirectional edge arrays are asymmetric "
                f"(max abs difference={max_abs})"
            )

    return csr


def distance_csr_from_loader(
    loader,
):
    try:
        adjacency = (
            loader.adjacency
        )
    except (
        AttributeError,
        NotImplementedError,
    ):
        adjacency = None

    if adjacency is not None:
        csr = (
            adjacency
            .copy()
            .tocsr()
        )

        csr.sum_duplicates()
        csr.sort_indices()

        return csr

    sources, targets, distances = (
        loader.edge_arrays
    )

    return (
        distance_csr_from_edge_arrays(
            sources,
            targets,
            distances,
            n_nodes=int(
                loader.n_nodes
            ),
        )
    )


def run_distance_graph_communities(
    distance_csr,
    *,
    resolution: float,
    scale="adaptive",
    backend="cpu_lean_leiden",
    initial_membership=None,
    canonicalize_warm_start=True,
    similarity_function=None,
    backend_kwargs=None,
):
    weighted_csr = (
        distance_csr_to_similarity_csr(
            distance_csr,
            scale=scale,
            similarity_function=
                similarity_function,
        )
    )

    warm = initial_membership

    if (
        warm is not None
        and canonicalize_warm_start
    ):
        warm = (
            canonicalize_connected_membership(
                weighted_csr,
                warm,
            )
        )

    labels = run_community_backend(
        weighted_csr,
        backend=backend,
        resolution=float(
            resolution
        ),
        initial_membership=warm,
        backend_kwargs=
            backend_kwargs,
    )

    return (
        labels,
        weighted_csr,
    )
