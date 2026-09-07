from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Hashable, Mapping, Sequence

import numpy as np

from .community_backends import (
    get_community_backend,
    run_community_backend,
)
from .community_runtime import (
    distance_csr_to_similarity_csr,
)
from .connected_warm_start import (
    canonicalize_connected_membership,
)
from .sample_projection import (
    project_sample_labels_to_full,
)


@dataclasses.dataclass
class SampledToFullCommunityResult:
    manifest_id: str
    sampled_indices: tuple[int, ...]
    sampled_result: Any
    sampled_labels_by_resolution: dict[Hashable, np.ndarray]
    projected_labels_by_resolution: dict[Hashable, np.ndarray]
    warm_labels_by_resolution: dict[Hashable, np.ndarray]
    full_labels_by_resolution: dict[Hashable, np.ndarray]
    projection_diagnostics_by_resolution: dict[Hashable, dict[str, Any]]
    full_community_backend: str
    similarity_scale: Any
    full_node_count: int
    weighted_full_nnz: int
    artifact_path: str | None


def _label_descriptor(
    labels,
):
    labels = np.asarray(
        labels,
        dtype=np.int32,
    )

    return {
        "length":
            int(
                labels.size
            ),

        "communities":
            int(
                np.unique(
                    labels
                ).size
            ),

        "sha256":
            hashlib.sha256(
                np.ascontiguousarray(
                    labels
                ).tobytes()
            ).hexdigest(),
    }


def _validate_labels(
    labels,
    *,
    expected_count,
    name,
):
    labels = np.asarray(
        labels
    )

    if (
        labels.ndim != 1
        or labels.shape[0]
        != int(
            expected_count
        )
    ):
        raise ValueError(
            f"{name} must have shape ({expected_count},), "
            f"observed {labels.shape}"
        )

    if not np.issubdtype(
        labels.dtype,
        np.integer,
    ):
        raise TypeError(
            f"{name} must be integer-valued"
        )

    if np.any(
        labels < 0
    ):
        raise ValueError(
            f"{name} contains negative labels"
        )

    return labels.astype(
        np.int32,
        copy=False,
    )


def _compact_labels(
    labels,
):
    labels = _validate_labels(
        labels,
        expected_count=len(
            labels
        ),
        name="sample labels",
    )

    return np.unique(
        labels,
        return_inverse=True,
    )[1].astype(
        np.int32,
        copy=False,
    )


def _parse_runtime_config(
    inputs,
    *,
    full_community_backend,
):
    kwargs = dict(
        inputs.leiden_kwargs
    )

    algorithm = str(
        kwargs.pop(
            "algorithm",
            "leiden_csr",
        )
    ).lower()

    if algorithm == "leiden":
        algorithm = (
            "leiden_csr"
        )

    if algorithm != "leiden_csr":
        raise ValueError(
            "sample-to-full warm refinement currently requires "
            "the consolidated Leiden backend family; "
            f"received algorithm={algorithm!r}"
        )

    scale = kwargs.pop(
        "scale",
        "adaptive",
    )

    sampled_backend = kwargs.pop(
        "community_backend",
        None,
    )

    backend_kwargs = kwargs.pop(
        "community_backend_kwargs",
        None,
    )

    # These keys govern sampled analyzer presentation/control, not the
    # mathematical full-refinement operation.
    kwargs.pop(
        "run_name",
        None,
    )
    kwargs.pop(
        "min_cluster_size",
        None,
    )
    kwargs.pop(
        "rank_stat_col",
        None,
    )
    kwargs.pop(
        "warm_start",
        None,
    )

    if kwargs:
        raise TypeError(
            "unsupported Leiden configuration for sampled-to-full "
            "completion: "
            + repr(
                sorted(
                    kwargs
                )
            )
        )

    selected = (
        full_community_backend
        if full_community_backend
        is not None
        else (
            sampled_backend
            if sampled_backend
            is not None
            else "cpu_lean_leiden"
        )
    )

    backend = get_community_backend(
        str(
            selected
        )
    )

    if not (
        backend
        .capabilities
        .supports_warm_start
    ):
        raise ValueError(
            "selected full community backend does not support "
            "warm-start refinement: "
            + backend.capabilities.name
        )

    if not backend.is_available():
        raise RuntimeError(
            "selected full community backend is unavailable: "
            + backend.capabilities.name
        )

    return (
        algorithm,
        scale,
        backend.capabilities.name,
        backend_kwargs,
    )


def _projection_diagnostics(
    projection,
):
    return {
        "direct_orphans":
            int(
                projection.direct_orphans
            ),

        "rounds":
            [
                dict(
                    row
                )
                for row
                in projection.rounds
            ],

        "remaining_final":
            int(
                projection.remaining_final
            ),
    }


def _write_artifact(
    path,
    *,
    result,
):
    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "schema":
            "sampled-to-full-community-provenance-v1",

        "manifest_id":
            result.manifest_id,

        "sampled_indices_count":
            len(
                result.sampled_indices
            ),

        "full_node_count":
            result.full_node_count,

        "full_community_backend":
            result.full_community_backend,

        "similarity_scale":
            result.similarity_scale,

        "weighted_full_nnz":
            result.weighted_full_nnz,

        "sampled_labels_by_resolution": {
            repr(
                key
            ):
                _label_descriptor(
                    labels
                )

            for key, labels
            in result.sampled_labels_by_resolution.items()
        },

        "projected_labels_by_resolution": {
            repr(
                key
            ):
                _label_descriptor(
                    labels
                )

            for key, labels
            in result.projected_labels_by_resolution.items()
        },

        "warm_labels_by_resolution": {
            repr(
                key
            ):
                _label_descriptor(
                    labels
                )

            for key, labels
            in result.warm_labels_by_resolution.items()
        },

        "full_labels_by_resolution": {
            repr(
                key
            ):
                _label_descriptor(
                    labels
                )

            for key, labels
            in result.full_labels_by_resolution.items()
        },

        "projection_diagnostics_by_resolution": {
            repr(
                key
            ):
                value

            for key, value
            in result.projection_diagnostics_by_resolution.items()
        },
    }

    tmp = path.with_name(
        path.name
        + ".tmp."
        + str(
            os.getpid()
        )
        + "."
        + str(
            threading.get_ident()
        )
    )

    tmp.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    os.replace(
        tmp,
        path,
    )


def complete_sampled_to_full_communities(
    handle,
    *,
    inputs,
    resolutions: Sequence[Hashable] | None = None,
    timeout: float | None = None,
    projection_max_rounds: int = 15,
    require_fully_resolved: bool = True,
    full_community_backend: str | None = None,
    completion_artifact_path=None,
) -> SampledToFullCommunityResult:
    """Complete the sampled workflow after the full CSR has been built.

    Execution order:

      sampled worker join
        -> one full distance->affinity CSR transform
        -> sampled partition projection for every resolution
        -> connected warm-start canonicalization
        -> selected full-graph community backend refinement

    The weighted full CSR is shared by projection and all requested
    full-resolution refinements. OptimizedCommunityAnalyzer is deliberately
    not instantiated on the full graph.
    """
    if resolutions is None:
        resolutions = tuple(
            handle.manifest.resolutions
        )
    else:
        resolutions = tuple(
            resolutions
        )

    manifest_resolutions = tuple(
        handle.manifest.resolutions
    )

    if resolutions != manifest_resolutions:
        raise ValueError(
            "completion resolutions must exactly match the sampled "
            "manifest order; "
            f"manifest={manifest_resolutions!r}, "
            f"requested={resolutions!r}"
        )

    if (
        isinstance(
            projection_max_rounds,
            bool,
        )
        or int(
            projection_max_rounds
        )
        < 0
    ):
        raise ValueError(
            "projection_max_rounds must be a nonnegative integer"
        )

    full_graph = (
        handle.full_graph
    )

    if not hasattr(
        full_graph,
        "graph",
    ):
        raise TypeError(
            "full DataGraph does not expose .graph CSR"
        )

    full_distance_csr = (
        full_graph
        .graph
        .tocsr(
            copy=False
        )
    )

    full_distance_csr.sum_duplicates()
    full_distance_csr.sort_indices()

    expected_n = int(
        handle
        .manifest
        .full_node_count
    )

    if (
        full_distance_csr.shape
        != (
            expected_n,
            expected_n,
        )
    ):
        raise ValueError(
            "full CSR shape does not match manifest node count: "
            f"{full_distance_csr.shape} vs ({expected_n}, {expected_n})"
        )

    sampled_result = (
        handle.join_after_full_csr(
            full_distance_csr,
            timeout=timeout,
        )
    )

    if (
        sampled_result.manifest_id
        != handle.manifest.manifest_id
    ):
        raise RuntimeError(
            "sampled result manifest mismatch after join"
        )

    if (
        tuple(
            sampled_result.sampled_indices
        )
        != tuple(
            handle.manifest.sampled_indices
        )
    ):
        raise RuntimeError(
            "sampled result indices differ from manifest"
        )

    if set(
        sampled_result.labels_by_resolution
    ) != set(
        resolutions
    ):
        raise ValueError(
            "sampled label resolution keys differ from requested resolutions"
        )

    (
        _algorithm,
        scale,
        backend_name,
        backend_kwargs,
    ) = _parse_runtime_config(
        inputs,
        full_community_backend=
            full_community_backend,
    )

    full_backend_kwargs = dict(
        backend_kwargs
        or {}
    )

    if backend_name == "cpu_lean_leiden":
        full_backend_kwargs.setdefault(
            "warm_level0_scheduler",
            "exact_dependency_skip",
        )

    analyzer_kwargs = dict(
        inputs.analyzer_kwargs
    )

    similarity_function = (
        analyzer_kwargs.get(
            "similarity_function"
        )
    )

    weighted_full_csr = (
        distance_csr_to_similarity_csr(
            full_distance_csr,
            scale=scale,
            similarity_function=
                similarity_function,
        )
    )

    if (
        weighted_full_csr.shape
        != full_distance_csr.shape
        or weighted_full_csr.nnz
        != full_distance_csr.nnz
    ):
        raise RuntimeError(
            "full distance->affinity transform changed graph topology"
        )

    sampled_labels = {}
    projected_labels = {}
    warm_labels = {}
    full_labels = {}
    projection_diagnostics = {}

    sampled_indices = np.asarray(
        sampled_result.sampled_indices,
        dtype=np.int64,
    )

    for resolution in resolutions:
        sample = _validate_labels(
            sampled_result.labels_by_resolution[
                resolution
            ],
            expected_count=len(
                sampled_indices
            ),
            name=(
                "sample labels at resolution "
                + repr(
                    resolution
                )
            ),
        )

        projection = (
            project_sample_labels_to_full(
                weighted_full_csr,
                sampled_indices,
                sample,
                max_rounds=int(
                    projection_max_rounds
                ),
            )
        )

        projected = np.asarray(
            projection.labels,
            dtype=np.int32,
        )

        diagnostics = (
            _projection_diagnostics(
                projection
            )
        )

        if (
            require_fully_resolved
            and diagnostics[
                "remaining_final"
            ]
            != 0
        ):
            raise RuntimeError(
                "sample-to-full projection left unresolved nodes at "
                f"resolution={resolution!r}: "
                f"{diagnostics['remaining_final']}"
            )

        projected = _validate_labels(
            projected,
            expected_count=expected_n,
            name=(
                "projected full labels at resolution "
                + repr(
                    resolution
                )
            ),
        )

        compact_sample = (
            _compact_labels(
                sample
            )
        )

        if not np.array_equal(
            projected[
                sampled_indices
            ],
            compact_sample,
        ):
            raise RuntimeError(
                "projection changed sampled anchor memberships at "
                f"resolution={resolution!r}"
            )

        warm = np.asarray(
            canonicalize_connected_membership(
                weighted_full_csr,
                projected,
            ),
            dtype=np.int32,
        )

        warm = _validate_labels(
            warm,
            expected_count=expected_n,
            name=(
                "canonical full warm labels at resolution "
                + repr(
                    resolution
                )
            ),
        )

        final = np.asarray(
            run_community_backend(
                weighted_full_csr,
                backend=
                    backend_name,
                resolution=float(
                    resolution
                ),
                initial_membership=
                    warm,
                backend_kwargs=
                    full_backend_kwargs,
            ),
            dtype=np.int32,
        )

        final = _validate_labels(
            final,
            expected_count=expected_n,
            name=(
                "full refined labels at resolution "
                + repr(
                    resolution
                )
            ),
        )

        sampled_labels[
            resolution
        ] = sample

        projected_labels[
            resolution
        ] = projected

        warm_labels[
            resolution
        ] = warm

        full_labels[
            resolution
        ] = final

        projection_diagnostics[
            resolution
        ] = diagnostics

    result = SampledToFullCommunityResult(
        manifest_id=
            sampled_result.manifest_id,

        sampled_indices=tuple(
            int(
                value
            )
            for value
            in sampled_result.sampled_indices
        ),

        sampled_result=
            sampled_result,

        sampled_labels_by_resolution=
            sampled_labels,

        projected_labels_by_resolution=
            projected_labels,

        warm_labels_by_resolution=
            warm_labels,

        full_labels_by_resolution=
            full_labels,

        projection_diagnostics_by_resolution=
            projection_diagnostics,

        full_community_backend=
            backend_name,

        similarity_scale=
            scale,

        full_node_count=
            expected_n,

        weighted_full_nnz=
            int(
                weighted_full_csr.nnz
            ),

        artifact_path=(
            str(
                completion_artifact_path
            )
            if completion_artifact_path
            is not None
            else None
        ),
    )

    if completion_artifact_path is not None:
        _write_artifact(
            completion_artifact_path,
            result=result,
        )

    return result


__all__ = [
    "SampledToFullCommunityResult",
    "complete_sampled_to_full_communities",
]
