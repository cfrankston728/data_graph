
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from scipy import sparse

from .community_backends import (
    community_backend_status,
    run_community_backend,
)
from .sample_projection import (
    project_sample_labels_to_full,
)
from .connected_warm_start import (
    canonicalize_connected_membership,
)
from .sampled_graph_orchestrator import (
    deterministic_sample_indices,
)


@dataclass(frozen=True)
class CommunityResult:
    labels: np.ndarray
    backend: str
    resolution: float


@dataclass(frozen=True)
class SampledCommunityResult:
    final_labels: np.ndarray
    warm_start_labels: np.ndarray
    projected_labels: np.ndarray
    sample_labels: np.ndarray
    sampled_indices: np.ndarray
    projection: Any
    sample_backend: str
    full_backend: str
    resolution: float
    sample_fraction: float
    sample_seed: int


def _as_csr(matrix):
    if not sparse.isspmatrix_csr(matrix):
        matrix = sparse.csr_matrix(matrix)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            "community adjacency must be a square matrix"
        )

    return matrix


def _labels_from_result(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        labels = value

    elif isinstance(value, (tuple, list)) and len(value):
        labels = value[0]

    else:
        labels = None

        for name in (
            "labels",
            "membership",
            "communities",
            "final_labels",
        ):
            if hasattr(value, name):
                labels = getattr(value, name)
                break

        if labels is None:
            labels = value

    labels = np.asarray(labels)

    if labels.ndim != 1:
        raise TypeError(
            "community backend did not return a 1D label vector"
        )

    return labels.astype(
        np.int64,
        copy=False,
    )


def _projection_labels(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return _labels_from_result(value)

    for name in (
        "labels",
        "full_labels",
        "projected_labels",
        "membership",
    ):
        if hasattr(value, name):
            return _labels_from_result(
                getattr(value, name)
            )

    if isinstance(value, (tuple, list)) and len(value):
        return _labels_from_result(value[0])

    raise TypeError(
        "projection result does not expose full labels"
    )


def _connected_labels(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return _labels_from_result(value)

    if isinstance(value, (tuple, list)) and len(value):
        return _labels_from_result(value[0])

    for name in (
        "labels",
        "membership",
        "canonical_labels",
    ):
        if hasattr(value, name):
            return _labels_from_result(
                getattr(value, name)
            )

    return _labels_from_result(value)


def _invoke_backend(
    weighted_csr,
    *,
    resolution: float,
    backend: str,
    initial_membership=None,
    backend_kwargs: Mapping[str, Any] | None = None,
):
    """Adapt the registered backend runner without exposing its internals."""
    fn = run_community_backend
    sig = inspect.signature(fn)
    params = sig.parameters

    kwargs = {}
    args = []

    graph_names = (
        "weighted_csr",
        "input_matrix",
        "adjacency",
        "graph",
        "csr",
        "matrix",
    )

    graph_name = next(
        (
            name
            for name in graph_names
            if name in params
        ),
        None,
    )

    if graph_name is not None:
        kwargs[graph_name] = weighted_csr
    else:
        args.append(weighted_csr)

    if "resolution" in params:
        kwargs["resolution"] = float(resolution)

    backend_name = next(
        (
            name
            for name in (
                "backend",
                "backend_name",
                "community_backend",
            )
            if name in params
        ),
        None,
    )

    if backend_name is None:
        raise TypeError(
            "registered backend runner has no backend selector"
        )

    kwargs[backend_name] = backend

    if initial_membership is not None:
        warm_name = next(
            (
                name
                for name in (
                    "previous_labels",
                    "initial_membership",
                    "initial_labels",
                    "warm_start_labels",
                )
                if name in params
            ),
            None,
        )

        if warm_name is None:
            raise TypeError(
                f"backend runner does not expose warm-start labels "
                f"for backend {backend!r}"
            )

        kwargs[warm_name] = np.asarray(
            initial_membership,
            dtype=np.int64,
        )

    extra = dict(
        backend_kwargs or {}
    )

    if "backend_kwargs" in params:
        kwargs["backend_kwargs"] = extra

    elif any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in params.values()
    ):
        kwargs.update(extra)

    elif extra:
        unknown = sorted(
            set(extra)
            - set(params)
        )

        if unknown:
            raise TypeError(
                "unsupported backend kwargs: "
                + repr(unknown)
            )

        kwargs.update(extra)

    return _labels_from_result(
        fn(
            *args,
            **kwargs,
        )
    )


def run_communities(
    weighted_csr,
    *,
    resolution: float = 1.0,
    backend: str = "cpu_lean_leiden",
    initial_membership=None,
    backend_kwargs: Mapping[str, Any] | None = None,
) -> CommunityResult:
    """Run one registered community backend on an already-weighted graph."""
    weighted_csr = _as_csr(
        weighted_csr
    )

    labels = _invoke_backend(
        weighted_csr,
        resolution=resolution,
        backend=backend,
        initial_membership=initial_membership,
        backend_kwargs=backend_kwargs,
    )

    if labels.shape[0] != weighted_csr.shape[0]:
        raise ValueError(
            "backend label count does not match graph node count"
        )

    return CommunityResult(
        labels=labels,
        backend=str(backend),
        resolution=float(resolution),
    )


def run_sampled_communities(
    weighted_full_csr,
    *,
    resolution: float = 1.0,
    sample_fraction: float = 0.085,
    sample_seed: int = 0,
    sample_backend: str = "cpu_lean_leiden",
    full_backend: str = "cpu_lean_leiden",
    projection_max_rounds: int = 15,
    sample_backend_kwargs: Mapping[str, Any] | None = None,
    full_backend_kwargs: Mapping[str, Any] | None = None,
) -> SampledCommunityResult:
    """Cold sample -> weighted projection -> connected warm full refinement."""
    weighted_full_csr = _as_csr(
        weighted_full_csr
    )

    n = weighted_full_csr.shape[0]

    sampled_indices = np.asarray(
        deterministic_sample_indices(
            n,
            fraction=float(sample_fraction),
            seed=int(sample_seed),
        ),
        dtype=np.int64,
    )

    if sampled_indices.size == 0:
        raise ValueError(
            "sampling produced zero nodes"
        )

    sampled_csr = (
        weighted_full_csr[
            sampled_indices
        ][
            :,
            sampled_indices
        ]
        .tocsr()
    )

    sample_result = run_communities(
        sampled_csr,
        resolution=resolution,
        backend=sample_backend,
        backend_kwargs=sample_backend_kwargs,
    )

    projection = project_sample_labels_to_full(
        weighted_full_csr,
        tuple(
            int(i)
            for i in sampled_indices
        ),
        sample_result.labels,
        max_rounds=int(
            projection_max_rounds
        ),
    )

    projected_labels = _projection_labels(
        projection
    )

    if projected_labels.shape[0] != n:
        raise ValueError(
            "projection did not return one label per full node"
        )

    warm_start_labels = _connected_labels(
        canonicalize_connected_membership(
            weighted_full_csr,
            projected_labels,
        )
    )

    if warm_start_labels.shape[0] != n:
        raise ValueError(
            "connected warm start has wrong node count"
        )

    final_result = run_communities(
        weighted_full_csr,
        resolution=resolution,
        backend=full_backend,
        initial_membership=warm_start_labels,
        backend_kwargs=full_backend_kwargs,
    )

    return SampledCommunityResult(
        final_labels=final_result.labels,
        warm_start_labels=warm_start_labels,
        projected_labels=projected_labels,
        sample_labels=sample_result.labels,
        sampled_indices=sampled_indices,
        projection=projection,
        sample_backend=str(sample_backend),
        full_backend=str(full_backend),
        resolution=float(resolution),
        sample_fraction=float(sample_fraction),
        sample_seed=int(sample_seed),
    )


__all__ = [
    "CommunityResult",
    "SampledCommunityResult",
    "community_backend_status",
    "run_communities",
    "run_sampled_communities",
]
