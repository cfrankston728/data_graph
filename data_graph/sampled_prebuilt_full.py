from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Hashable, Mapping, Sequence

import numpy as np
from scipy import sparse

from .sampled_graph_orchestrator import (
    DEFAULT_SAMPLE_FRACTION,
    GraphInputs,
    SampleManifest,
    SampledWorkerResult,
    deterministic_sample_indices,
    make_sample_manifest,
    resolve_sampled_graph_build_kwargs,
    _validate_label_vector,
    _write_provenance_artifact,
)

from .sampled_graph_live_adapters import (
    live_graph_build_call,
    live_graph_factory,
)
from .sampled_community_strategy import (
    live_prepared_sampled_community_call,
)

from .sampled_full_refinement import (
    complete_sampled_to_full_communities,
)


@dataclasses.dataclass
class _PrebuiltFullGraphSurface:
    graph: Any


class PrebuiltFullSampledHandle:
    """Minimal R180-compatible handle around an already-built full CSR."""

    def __init__(
        self,
        *,
        full_csr,
        sampled_result: SampledWorkerResult,
        manifest: SampleManifest,
        full_build_info: Mapping[str, Any] | None = None,
    ):
        self._full_csr = full_csr
        self._sampled_result = sampled_result

        self.manifest = manifest

        self.full_generator = None

        self.full_build_result = (
            _PrebuiltFullGraphSurface(
                graph=full_csr,
            ),
            dict(
                full_build_info
                or {
                    "prebuilt_full_graph":
                        True,
                }
            ),
        )

        self._joined = False
        self._closed = False

    @property
    def sampled_future_done(
        self,
    ) -> bool:
        # The prebuilt-full adapter intentionally completes the sampled stage
        # synchronously; there is no full build left to overlap with it.
        return True

    @property
    def joined(
        self,
    ) -> bool:
        return self._joined

    @property
    def full_graph(
        self,
    ):
        return self.full_build_result[
            0
        ]

    @property
    def full_build_info(
        self,
    ):
        return self.full_build_result[
            1
        ]

    def join_after_full_csr(
        self,
        full_csr,
        *,
        timeout: float | None = None,
    ) -> SampledWorkerResult:
        del timeout

        if self._closed:
            raise RuntimeError(
                "prebuilt-full sampled handle is closed"
            )

        if self._joined:
            raise RuntimeError(
                "sampled result has already been joined"
            )

        if full_csr is None:
            raise ValueError(
                "full_csr must already exist before joining sampled result"
            )

        if (
            tuple(
                full_csr.shape
            )
            != tuple(
                self._full_csr.shape
            )
            or int(
                full_csr.nnz
            )
            != int(
                self._full_csr.nnz
            )
        ):
            raise RuntimeError(
                "join full CSR does not match bound prebuilt full CSR"
            )

        if (
            self._sampled_result.manifest_id
            != self.manifest.manifest_id
        ):
            raise RuntimeError(
                "sampled result provenance mismatch"
            )

        self._joined = True
        self._closed = True

        return self._sampled_result

    def close_without_assignment(
        self,
    ) -> None:
        self._closed = True


def _take_rows(
    obj,
    indices,
):
    if hasattr(
        obj,
        "iloc",
    ):
        return obj.iloc[
            list(
                indices
            )
        ].copy()

    array = np.asarray(
        obj
    )

    return array[
        np.asarray(
            indices,
            dtype=np.int64,
        )
    ].copy()


def _validate_prebuilt_full_csr(
    full_csr,
    *,
    expected_nodes: int,
):
    if not sparse.isspmatrix_csr(
        full_csr
    ):
        raise TypeError(
            "prebuilt_full_csr must be scipy CSR; "
            "load/convert it explicitly before entering this adapter"
        )

    if (
        full_csr.shape
        != (
            expected_nodes,
            expected_nodes,
        )
    ):
        raise ValueError(
            "prebuilt full CSR shape mismatch: "
            f"{full_csr.shape} vs "
            f"({expected_nodes}, {expected_nodes})"
        )

    if full_csr.nnz < 0:
        raise ValueError(
            "invalid prebuilt full CSR nnz"
        )

    # Do not normalize/sort/deduplicate here: the full graph is treated as
    # a frozen qualified artifact. Production preflight may require these
    # properties, but this adapter must not mutate the CSR.
    return full_csr


def build_sampled_handle_for_prebuilt_full(
    *,
    inputs: GraphInputs,
    prebuilt_full_csr,
    resolutions: Sequence[Hashable],
    sample_fraction: float = DEFAULT_SAMPLE_FRACTION,
    sample_seed: int = 0,
    sampled_n_neighbors: int | None = None,
    provenance_artifact_path=None,
    full_build_info: Mapping[str, Any] | None = None,
) -> PrebuiltFullSampledHandle:
    """Build only the sampled graph while binding an existing full CSR."""

    resolutions = tuple(
        resolutions
    )

    if not resolutions:
        raise ValueError(
            "at least one community resolution is required"
        )

    if inputs.embedding_values is None:
        raise ValueError(
            "embedding_values must already be materialized"
        )

    if (
        len(
            inputs.embedding_values
        )
        != len(
            inputs.node_df
        )
    ):
        raise ValueError(
            "embedding_values row count must match node_df"
        )

    full_csr = _validate_prebuilt_full_csr(
        prebuilt_full_csr,
        expected_nodes=len(
            inputs.node_df
        ),
    )

    sampled_indices = (
        deterministic_sample_indices(
            len(
                inputs.node_df
            ),
            fraction=float(
                sample_fraction
            ),
            seed=int(
                sample_seed
            ),
        )
    )

    sampled_graph_build_kwargs = (
        resolve_sampled_graph_build_kwargs(
            inputs.graph_build_kwargs,
            sample_fraction=float(
                sample_fraction
            ),
            sampled_node_count=len(
                sampled_indices
            ),
            sampled_n_neighbors=
                sampled_n_neighbors,
        )
    )

    manifest = make_sample_manifest(
        inputs,
        sampled_indices=
            sampled_indices,
        sample_fraction=float(
            sample_fraction
        ),
        sample_seed=int(
            sample_seed
        ),
        resolutions=
            resolutions,
        sampled_graph_build_kwargs=
            sampled_graph_build_kwargs,
    )

    sample_inputs = dataclasses.replace(
        inputs,

        node_df=_take_rows(
            inputs.node_df,
            sampled_indices,
        ),

        embedding_values=_take_rows(
            inputs.embedding_values,
            sampled_indices,
        ),

        graph_build_kwargs=dict(
            sampled_graph_build_kwargs
        ),

        source_provenance={
            **dict(
                inputs.source_provenance
            ),

            "parent_sample_manifest_id":
                manifest.manifest_id,
        },
    )

    sampled_generator = (
        live_graph_factory(
            sample_inputs
        )
    )

    sampled_build_result = (
        live_graph_build_call(
            sampled_generator,
            sample_inputs.graph_build_kwargs,
        )
    )

    if (
        not isinstance(
            sampled_build_result,
            tuple,
        )
        or len(
            sampled_build_result
        )
        < 2
    ):
        raise TypeError(
            "sample graph build must return "
            "(DataGraph, build_info)"
        )

    labels_by_resolution = dict(
        live_prepared_sampled_community_call(
            sample_inputs,
            sampled_build_result,
            resolutions,
        )
    )

    if (
        set(
            labels_by_resolution
        )
        != set(
            resolutions
        )
    ):
        raise ValueError(
            "sample community result resolution keys "
            "do not match requested resolutions"
        )

    for resolution in resolutions:
        _validate_label_vector(
            labels_by_resolution[
                resolution
            ],
            expected_count=len(
                sampled_indices
            ),
            resolution=
                resolution,
        )

    artifact_path = (
        Path(
            provenance_artifact_path
        )
        if provenance_artifact_path
        is not None
        else None
    )

    if artifact_path is not None:
        _write_provenance_artifact(
            artifact_path,
            manifest=manifest,
            labels_by_resolution=
                labels_by_resolution,
        )

    sampled_result = (
        SampledWorkerResult(
            manifest_id=
                manifest.manifest_id,

            sampled_indices=
                manifest.sampled_indices,

            sampled_generator=
                sampled_generator,

            sampled_graph=
                sampled_build_result[
                    0
                ],

            sampled_build_info=
                sampled_build_result[
                    1
                ],

            labels_by_resolution=
                labels_by_resolution,

            artifact_path=(
                str(
                    artifact_path
                )
                if artifact_path
                is not None
                else None
            ),
        )
    )

    return PrebuiltFullSampledHandle(
        full_csr=full_csr,
        sampled_result=
            sampled_result,
        manifest=manifest,
        full_build_info=
            full_build_info,
    )


def complete_communities_with_prebuilt_full(
    *,
    inputs: GraphInputs,
    prebuilt_full_csr,
    resolutions: Sequence[Hashable],
    sample_fraction: float = DEFAULT_SAMPLE_FRACTION,
    sample_seed: int = 0,
    sampled_n_neighbors: int | None = None,
    sampled_provenance_artifact_path=None,
    completion_artifact_path=None,
    projection_max_rounds: int = 15,
    require_fully_resolved: bool = True,
    full_community_backend: str | None = None,
    full_build_info: Mapping[str, Any] | None = None,
):
    """Sample -> project -> full warm refinement using an existing full CSR."""

    handle = (
        build_sampled_handle_for_prebuilt_full(
            inputs=inputs,
            prebuilt_full_csr=
                prebuilt_full_csr,
            resolutions=
                resolutions,
            sample_fraction=
                sample_fraction,
            sample_seed=
                sample_seed,
            sampled_n_neighbors=
                sampled_n_neighbors,
            provenance_artifact_path=
                sampled_provenance_artifact_path,
            full_build_info=
                full_build_info,
        )
    )

    return (
        complete_sampled_to_full_communities(
            handle,
            inputs=inputs,
            resolutions=
                resolutions,
            projection_max_rounds=
                projection_max_rounds,
            require_fully_resolved=
                require_fully_resolved,
            full_community_backend=
                full_community_backend,
            completion_artifact_path=
                completion_artifact_path,
        )
    )


__all__ = [
    "PrebuiltFullSampledHandle",
    "build_sampled_handle_for_prebuilt_full",
    "complete_communities_with_prebuilt_full",
]
