from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Hashable, Mapping, Sequence

import numpy as np
from numba import njit
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
    live_sampled_community_call,
)

from .sampled_prebuilt_full import (
    PrebuiltFullSampledHandle,
)

from .sampled_full_refinement import (
    complete_sampled_to_full_communities,
)


HISTORICAL_FISHER_AST_SHA256 = (
    "90a55d78a5fb30e3366044c9b8fba597cef3bfe60f1134aede414b98963c3079"
)

HISTORICAL_EMBEDDING_AST_SHA256 = (
    "61f8af22d9a725b92d3f608ff575b853f28d378315bbcf1b48db90dd13c9d47f"
)

HISTORICAL_CALLABLE_PAIR_AST_SHA256 = (
    "d2e318c0df9d84044f9a32bfab510665951c82e5818a2cd89a43862e86bf4e96"
)

HISTORICAL_FEATURE_COLUMNS = tuple(
    [
        f"latent_{i}"
        for i in range(1, 7)
    ]
    +
    [
        f"sig_{i}"
        for i in range(1, 7)
    ]
)


@njit(cache=True)
def fisher_rao_distance_6d(
    f,
    i,
    j,
):
    """Fixed-6D implementation of the R184-resolved historical formula."""
    eps = 1e-8
    safe_eps = 1e-30

    means_i = (
        f[
            i,
            :6,
        ]
        + 0.0
    )

    sigma_i = (
        f[
            i,
            6:12,
        ]
        + eps
    )

    means_j = (
        f[
            j,
            :6,
        ]
        + 0.0
    )

    sigma_j = (
        f[
            j,
            6:12,
        ]
        + eps
    )

    var_term = (
        (
            sigma_i ** 2
            + sigma_j ** 2
        )
        / 2
        + eps
    )

    ratio = (
        var_term
        / (
            sigma_i
            * sigma_j
        )
    )

    log_term = np.log(
        np.maximum(
            ratio,
            1e-16,
        )
    )

    diff = (
        means_i
        - means_j
    )

    d2 = (
        2
        * np.sum(
            diff ** 2
            / var_term
            + log_term
        )
    )

    return np.sqrt(
        max(
            safe_eps,
            d2,
        )
    )


def materialize_historical_logspace_sample_embedding(
    sampled_raw_features,
    *,
    noise_scale: float = 1e-6,
    random_seed: int = 0,
):
    """Vectorized exact row-order equivalent of the historical transform.

    For a fixed input row sequence and seed, this uses the same MT19937
    Gaussian stream as:

        np.random.seed(seed)
        for row in rows:
            logspace_embedding_with_noise(row, noise_scale)

    The result remains float64, matching the historical Python function.
    """
    raw = np.asarray(
        sampled_raw_features
    )

    if (
        raw.ndim != 2
        or raw.shape[1] != 12
    ):
        raise ValueError(
            "sampled_raw_features must have shape (n, 12); "
            f"observed {raw.shape}"
        )

    if not np.isfinite(
        raw
    ).all():
        raise ValueError(
            "sampled raw Fisher-Rao features contain non-finite values"
        )

    means = np.asarray(
        raw[
            :,
            :6,
        ],
        dtype=float,
    )

    sigmas = np.asarray(
        raw[
            :,
            6:12,
        ],
        dtype=float,
    )

    sigmas = np.maximum(
        sigmas,
        1e-8,
    )

    emb = np.zeros(
        (
            raw.shape[0],
            12,
        ),
        dtype=float,
    )

    emb[
        :,
        0::2,
    ] = (
        means
        / sigmas
    )

    emb[
        :,
        1::2,
    ] = (
        np.sqrt(
            2.0
        )
        * np.log(
            sigmas
        )
    )

    if noise_scale:
        rng = np.random.RandomState(
            int(
                random_seed
            )
        )

        emb += (
            rng.randn(
                *emb.shape
            )
            * float(
                noise_scale
            )
        )

    return emb


def _array_sha256(
    value,
):
    array = np.ascontiguousarray(
        np.asarray(
            value
        )
    )

    return hashlib.sha256(
        array.tobytes()
    ).hexdigest()


def _sample_index_sha256(
    sampled_indices,
):
    array = np.ascontiguousarray(
        np.asarray(
            sampled_indices,
            dtype=np.int64,
        )
    )

    return hashlib.sha256(
        array.tobytes()
    ).hexdigest()


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

    return np.asarray(
        obj
    )[
        np.asarray(
            indices,
            dtype=np.int64,
        )
    ].copy()


def _validate_prebuilt_full_csr(
    full_csr,
    *,
    expected_nodes,
):
    if not sparse.isspmatrix_csr(
        full_csr
    ):
        raise TypeError(
            "prebuilt_full_csr must already be scipy CSR"
        )

    if (
        full_csr.shape
        != (
            int(
                expected_nodes
            ),
            int(
                expected_nodes
            ),
        )
    ):
        raise ValueError(
            "prebuilt full CSR shape mismatch"
        )

    return full_csr


def _validate_raw_feature_surface(
    raw_feature_values,
    *,
    expected_rows,
):
    raw = np.asarray(
        raw_feature_values
    )

    if (
        raw.ndim != 2
        or raw.shape
        != (
            int(
                expected_rows
            ),
            12,
        )
    ):
        raise ValueError(
            "raw feature surface must have shape "
            f"({expected_rows}, 12); observed {raw.shape}"
        )

    if not np.issubdtype(
        raw.dtype,
        np.floating,
    ):
        raise TypeError(
            "raw Fisher-Rao features must be floating point"
        )

    return raw_feature_values


def _write_embedding_provenance(
    path,
    payload,
):
    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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


def _make_sample_only_manifest(
    inputs: GraphInputs,
    *,
    sampled_indices,
    sampled_embedding_values,
    sample_fraction: float,
    sample_seed: int,
    resolutions,
    sampled_graph_build_kwargs,
    source_provenance,
) -> SampleManifest:
    """Manifest with a full node fingerprint and sampled embedding fingerprint."""
    from . import sampled_graph_orchestrator as _o

    sampled_indices = tuple(int(i) for i in sampled_indices)

    if len(sampled_embedding_values) != len(sampled_indices):
        raise ValueError(
            "sample embedding row count must equal sampled index count"
        )

    if not source_provenance:
        raise ValueError(
            "source_provenance must be non-empty"
        )

    stable = _o._stable_descriptor

    scoped_source = {
        **dict(source_provenance),
        "sample_manifest_contract": {
            "schema": "sample-only-manifest-v1",
            "node_content_scope": "full_node_df",
            "embedding_content_scope": "sampled_indices_only",
        },
    }

    node_hash = _o.content_fingerprint(
        inputs.node_df
    )

    embedding_hash = _o.content_fingerprint(
        sampled_embedding_values
    )

    payload = {
        "sample_fraction": float(sample_fraction),
        "sample_seed": int(sample_seed),
        "full_node_count": len(inputs.node_df),
        "sampled_indices": list(sampled_indices),
        "feature_columns": list(inputs.feature_columns),
        "premetric": stable(inputs.premetric_weight_function),
        "embedding_function": stable(inputs.embedding_function),
        "generator_kwargs": stable(dict(inputs.generator_kwargs)),
        "graph_build_kwargs": stable(dict(inputs.graph_build_kwargs)),
        "sampled_graph_build_kwargs":
            stable(dict(sampled_graph_build_kwargs)),
        "analyzer_kwargs": stable(dict(inputs.analyzer_kwargs)),
        "leiden_kwargs": stable(dict(inputs.leiden_kwargs)),
        "resolutions": [
            stable(value)
            for value in resolutions
        ],
        "source_provenance": stable(scoped_source),
        "node_content_order_sha256": node_hash,
        "embedding_content_sha256": embedding_hash,
    }

    manifest_id = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()

    return _o.SampleManifest(
        manifest_id=manifest_id,
        sample_fraction=float(sample_fraction),
        sample_seed=int(sample_seed),
        full_node_count=len(inputs.node_df),
        sampled_indices=sampled_indices,
        feature_columns=tuple(inputs.feature_columns),
        premetric_descriptor=
            stable(inputs.premetric_weight_function),
        embedding_function_descriptor=
            stable(inputs.embedding_function),
        generator_kwargs=
            stable(dict(inputs.generator_kwargs)),
        graph_build_kwargs=
            stable(dict(inputs.graph_build_kwargs)),
        sampled_graph_build_kwargs=
            stable(dict(sampled_graph_build_kwargs)),
        analyzer_kwargs=
            stable(dict(inputs.analyzer_kwargs)),
        leiden_kwargs=
            stable(dict(inputs.leiden_kwargs)),
        resolutions=tuple(resolutions),
        source_provenance=
            stable(scoped_source),
        node_content_order_sha256=node_hash,
        embedding_content_sha256=embedding_hash,
    )

def build_sampled_handle_for_prebuilt_full_sample_embedding(
    *,
    inputs: GraphInputs,
    raw_feature_values,
    prebuilt_full_csr,
    resolutions: Sequence[Hashable],
    sample_fraction: float = DEFAULT_SAMPLE_FRACTION,
    sample_seed: int = 0,
    sampled_n_neighbors: int | None = None,
    embedding_noise_scale: float = 1e-6,
    embedding_random_seed: int = 0,
    embedding_source_provenance: Mapping[str, Any] | None = None,
    sampled_provenance_artifact_path=None,
    sample_embedding_provenance_artifact_path=None,
    full_build_info: Mapping[str, Any] | None = None,
) -> PrebuiltFullSampledHandle:
    """Build only sampled embeddings + sampled graph around a frozen full CSR.

    Crucially, the deterministic sampled indices are chosen BEFORE any
    logspace embedding is materialized.
    """
    resolutions = tuple(
        resolutions
    )

    if not resolutions:
        raise ValueError(
            "at least one resolution is required"
        )

    if inputs.embedding_values is not None:
        raise ValueError(
            "sample-only embedding path requires inputs.embedding_values=None"
        )

    if tuple(
        inputs.feature_columns
    ) != HISTORICAL_FEATURE_COLUMNS:
        raise ValueError(
            "sample-only historical embedding requires feature columns "
            "[latent_1..latent_6, sig_1..sig_6] in exact order"
        )

    full_node_count = len(
        inputs.node_df
    )

    raw_feature_values = (
        _validate_raw_feature_surface(
            raw_feature_values,
            expected_rows=
                full_node_count,
        )
    )

    full_csr = (
        _validate_prebuilt_full_csr(
            prebuilt_full_csr,
            expected_nodes=
                full_node_count,
        )
    )

    # --------------------------------------------------------------
    # Ordering contract: sample first, embed second.
    # --------------------------------------------------------------

    sampled_indices = (
        deterministic_sample_indices(
            full_node_count,
            fraction=float(
                sample_fraction
            ),
            seed=int(
                sample_seed
            ),
        )
    )

    sampled_raw_features = np.asarray(
        raw_feature_values[
            np.asarray(
                sampled_indices,
                dtype=np.int64,
            )
        ]
    )

    sampled_embedding_values = (
        materialize_historical_logspace_sample_embedding(
            sampled_raw_features,
            noise_scale=
                float(
                    embedding_noise_scale
                ),
            random_seed=
                int(
                    embedding_random_seed
                ),
        )
    )

    if (
        sampled_embedding_values.shape
        != (
            len(
                sampled_indices
            ),
            12,
        )
    ):
        raise RuntimeError(
            "sample-only embedding returned unexpected shape"
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

    embedding_contract = {
        "schema":
            "sample-only-historical-logspace-embedding-v1",

        "historical_fisher_ast_sha256":
            HISTORICAL_FISHER_AST_SHA256,

        "historical_embedding_ast_sha256":
            HISTORICAL_EMBEDDING_AST_SHA256,

        "historical_callable_pair_ast_sha256":
            HISTORICAL_CALLABLE_PAIR_AST_SHA256,

        "feature_columns":
            list(
                HISTORICAL_FEATURE_COLUMNS
            ),

        "sample_fraction":
            float(
                sample_fraction
            ),

        "sample_seed":
            int(
                sample_seed
            ),

        "sampled_node_count":
            int(
                len(
                    sampled_indices
                )
            ),

        "sampled_indices_sha256":
            _sample_index_sha256(
                sampled_indices
            ),

        "embedding_noise_scale":
            float(
                embedding_noise_scale
            ),

        "embedding_random_seed":
            int(
                embedding_random_seed
            ),

        "raw_sample_shape":
            list(
                sampled_raw_features.shape
            ),

        "raw_sample_dtype":
            str(
                sampled_raw_features.dtype
            ),

        "raw_sample_sha256":
            _array_sha256(
                sampled_raw_features
            ),

        "embedding_shape":
            list(
                sampled_embedding_values.shape
            ),

        "embedding_dtype":
            str(
                sampled_embedding_values.dtype
            ),

        "embedding_sha256":
            _array_sha256(
                sampled_embedding_values
            ),

        "source":
            dict(
                embedding_source_provenance
                or {}
            ),
    }

    source_provenance = {
        **dict(
            inputs.source_provenance
        ),

        "sample_only_embedding_materialization":
            embedding_contract,
    }

    # Sample-only execution has its own truthful manifest scope.
    manifest = _make_sample_only_manifest(
        inputs,
        sampled_indices=
            sampled_indices,
        sampled_embedding_values=
            sampled_embedding_values,
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
        source_provenance=
            source_provenance,
    )

    sample_inputs = dataclasses.replace(
        inputs,

        node_df=_take_rows(
            inputs.node_df,
            sampled_indices,
        ),

        premetric_weight_function=
            fisher_rao_distance_6d,

        embedding_function=None,

        embedding_values=
            sampled_embedding_values,

        graph_build_kwargs=dict(
            sampled_graph_build_kwargs
        ),

        source_provenance={
            **source_provenance,

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
        live_sampled_community_call(
            sample_inputs,
            sampled_build_result,
            resolutions,
        )
    )

    if set(
        labels_by_resolution
    ) != set(
        resolutions
    ):
        raise ValueError(
            "sampled resolution keys differ from requested resolutions"
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

    sampled_artifact_path = (
        Path(
            sampled_provenance_artifact_path
        )
        if sampled_provenance_artifact_path
        is not None
        else None
    )

    if sampled_artifact_path is not None:
        _write_provenance_artifact(
            sampled_artifact_path,
            manifest=manifest,
            labels_by_resolution=
                labels_by_resolution,
        )

    embedding_artifact_path = (
        Path(
            sample_embedding_provenance_artifact_path
        )
        if sample_embedding_provenance_artifact_path
        is not None
        else None
    )

    if embedding_artifact_path is not None:
        _write_embedding_provenance(
            embedding_artifact_path,
            {
                **embedding_contract,

                "manifest_id":
                    manifest.manifest_id,
            },
        )

    sampled_result = SampledWorkerResult(
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
                sampled_artifact_path
            )
            if sampled_artifact_path
            is not None
            else None
        ),
    )

    return PrebuiltFullSampledHandle(
        full_csr=full_csr,
        sampled_result=
            sampled_result,
        manifest=manifest,
        full_build_info={
            **dict(
                full_build_info
                or {}
            ),

            "prebuilt_full_graph":
                True,

            "sample_only_embedding":
                embedding_contract,
        },
    )


def complete_communities_with_prebuilt_full_sample_embedding(
    *,
    inputs: GraphInputs,
    raw_feature_values,
    prebuilt_full_csr,
    resolutions: Sequence[Hashable],
    sample_fraction: float = DEFAULT_SAMPLE_FRACTION,
    sample_seed: int = 0,
    sampled_n_neighbors: int | None = None,
    embedding_noise_scale: float = 1e-6,
    embedding_random_seed: int = 0,
    embedding_source_provenance: Mapping[str, Any] | None = None,
    sampled_provenance_artifact_path=None,
    sample_embedding_provenance_artifact_path=None,
    completion_artifact_path=None,
    projection_max_rounds: int = 15,
    require_fully_resolved: bool = True,
    full_community_backend: str | None = None,
    full_build_info: Mapping[str, Any] | None = None,
):
    handle = (
        build_sampled_handle_for_prebuilt_full_sample_embedding(
            inputs=inputs,
            raw_feature_values=
                raw_feature_values,
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
            embedding_noise_scale=
                embedding_noise_scale,
            embedding_random_seed=
                embedding_random_seed,
            embedding_source_provenance=
                embedding_source_provenance,
            sampled_provenance_artifact_path=
                sampled_provenance_artifact_path,
            sample_embedding_provenance_artifact_path=
                sample_embedding_provenance_artifact_path,
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
    "HISTORICAL_FISHER_AST_SHA256",
    "HISTORICAL_EMBEDDING_AST_SHA256",
    "HISTORICAL_CALLABLE_PAIR_AST_SHA256",
    "HISTORICAL_FEATURE_COLUMNS",
    "fisher_rao_distance_6d",
    "materialize_historical_logspace_sample_embedding",
    "build_sampled_handle_for_prebuilt_full_sample_embedding",
    "complete_communities_with_prebuilt_full_sample_embedding",
]
