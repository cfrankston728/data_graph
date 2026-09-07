from __future__ import annotations

import dataclasses
from typing import Any, Mapping, Sequence

import numpy as np

from .sampled_graph_orchestrator import (
    DEFAULT_SAMPLE_FRACTION,
    GraphInputs,
    launch_sampled_worker_then_build_full,
)
from .sampled_graph_live_adapters import (
    live_graph_build_call,
    live_graph_factory,
    live_sampled_community_call,
)

from .sampled_community_strategy import live_prepared_sampled_community_call

SAMPLE_FRACTION = DEFAULT_SAMPLE_FRACTION


def materialize_embedding_values(node_df, embedding_function):
    values = np.asarray(
        node_df.apply(embedding_function, axis=1).tolist()
    )
    if values.ndim != 2:
        raise ValueError(
            "materialized embedding must be a two-dimensional [n_nodes, d] array; shape="
            + repr(values.shape)
        )
    if values.shape[0] != len(node_df):
        raise ValueError("materialized embedding row count does not match node_df")
    return values


def make_graph_inputs(
    *,
    node_df,
    feature_columns,
    premetric_weight_function,
    embedding_function,
    embedding_values=None,
    generator_kwargs: Mapping[str, Any] | None = None,
    graph_build_kwargs: Mapping[str, Any] | None = None,
    analyzer_kwargs: Mapping[str, Any] | None = None,
    leiden_kwargs: Mapping[str, Any] | None = None,
    source_provenance: Mapping[str, Any],
):
    if embedding_values is None:
        embedding_values = materialize_embedding_values(node_df, embedding_function)

    return GraphInputs(
        node_df=node_df,
        feature_columns=tuple(feature_columns),
        premetric_weight_function=premetric_weight_function,
        embedding_function=embedding_function,
        embedding_values=embedding_values,
        generator_kwargs=dict(generator_kwargs or {}),
        graph_build_kwargs=dict(graph_build_kwargs or {}),
        analyzer_kwargs=dict(analyzer_kwargs or {}),
        leiden_kwargs=dict(leiden_kwargs or {}),
        source_provenance=dict(source_provenance),
    )


def build_full_with_concurrent_sampled_communities(
    *,
    inputs: GraphInputs,
    resolutions: Sequence[Any],
    sample_seed: int = 0,
    sampled_n_neighbors: int | None = None,
    provenance_artifact_path=None,
    executor=None,
    worker_start_timeout_s: float = 10.0,
):
    if inputs.embedding_values is None:
        inputs = dataclasses.replace(
            inputs,
            embedding_values=materialize_embedding_values(
                inputs.node_df,
                inputs.embedding_function,
            ),
        )

    return launch_sampled_worker_then_build_full(
        inputs=inputs,
        graph_factory=live_graph_factory,
        graph_build_call=live_graph_build_call,
        community_call=live_prepared_sampled_community_call,
        resolutions=resolutions,
        sample_fraction=SAMPLE_FRACTION,
        sample_seed=sample_seed,
        sampled_n_neighbors=sampled_n_neighbors,
        provenance_artifact_path=provenance_artifact_path,
        executor=executor,
        worker_start_timeout_s=worker_start_timeout_s,
    )


__all__ = [
    "GraphInputs",
    "SAMPLE_FRACTION",
    "make_graph_inputs",
    "materialize_embedding_values",
    "build_full_with_concurrent_sampled_communities",
]
# Additive sampled->full completion surface.
from .sampled_full_refinement import (
    SampledToFullCommunityResult,
    complete_sampled_to_full_communities,
)

for _sampled_full_export in (
    "SampledToFullCommunityResult",
    "complete_sampled_to_full_communities",
):
    if _sampled_full_export not in __all__:
        __all__.append(_sampled_full_export)

del _sampled_full_export
from .sampled_prebuilt_full import (
    PrebuiltFullSampledHandle,
    build_sampled_handle_for_prebuilt_full,
    complete_communities_with_prebuilt_full,
)

for _prebuilt_full_export in (
    "PrebuiltFullSampledHandle",
    "build_sampled_handle_for_prebuilt_full",
    "complete_communities_with_prebuilt_full",
):
    if _prebuilt_full_export not in __all__:
        __all__.append(_prebuilt_full_export)

del _prebuilt_full_export
from .sampled_sample_only_embedding import (
    HISTORICAL_FISHER_AST_SHA256,
    HISTORICAL_EMBEDDING_AST_SHA256,
    HISTORICAL_CALLABLE_PAIR_AST_SHA256,
    HISTORICAL_FEATURE_COLUMNS,
    fisher_rao_distance_6d,
    materialize_historical_logspace_sample_embedding,
    build_sampled_handle_for_prebuilt_full_sample_embedding,
    complete_communities_with_prebuilt_full_sample_embedding,
)

for _sample_only_embedding_export in (
    "HISTORICAL_FISHER_AST_SHA256",
    "HISTORICAL_EMBEDDING_AST_SHA256",
    "HISTORICAL_CALLABLE_PAIR_AST_SHA256",
    "HISTORICAL_FEATURE_COLUMNS",
    "fisher_rao_distance_6d",
    "materialize_historical_logspace_sample_embedding",
    "build_sampled_handle_for_prebuilt_full_sample_embedding",
    "complete_communities_with_prebuilt_full_sample_embedding",
):
    if _sample_only_embedding_export not in __all__:
        __all__.append(_sample_only_embedding_export)

del _sample_only_embedding_export
