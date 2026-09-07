from __future__ import annotations

import inspect
from typing import Any, Mapping, Sequence

import numpy as np

from .community_api import run_communities
from .community_runtime import distance_csr_to_similarity_csr
from .generate_community_interface_subgraphs import (
    OptimizedCommunityAnalyzer,
    _csr_from_undirected_edges,
)
from .sampled_graph_live_adapters import InMemoryGraphLoader


SAMPLED_PREPARE_COARSEN = True
SAMPLED_PREPARE_COARSEN_LEVELS = 1
SAMPLED_PREPARE_SPARSIFY = False


def _candidate_analyzer_kwargs(
    raw: Mapping[str, Any] | None,
) -> dict[str, Any]:
    kwargs = dict(raw or {})

    # The accelerated sampled strategy has an explicit preparation contract.
    # Do not inherit the old notebook's three-level/sparsified behavior.
    kwargs["coarsen"] = SAMPLED_PREPARE_COARSEN
    kwargs["coarsen_levels"] = SAMPLED_PREPARE_COARSEN_LEVELS
    kwargs["sparsify"] = SAMPLED_PREPARE_SPARSIFY

    # In-memory sampled analysis must not create persistent prepared caches.
    signature = inspect.signature(OptimizedCommunityAnalyzer)

    for key in (
        "prepared_graph_cache",
        "prepared_graph_cache_read",
        "prepared_graph_cache_write",
        "coarsening_stack_cache",
        "coarsening_stack_cache_write",
    ):
        if key in signature.parameters:
            kwargs[key] = False

    accepted = {
        name
        for name, parameter in signature.parameters.items()
        if name != "graph_loader"
        and parameter.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }

    unknown = sorted(
        set(kwargs) - accepted
    )

    if unknown:
        raise TypeError(
            "unsupported sampled analyzer configuration keys: "
            + repr(unknown)
        )

    return kwargs


def _parse_leiden_kwargs(
    raw: Mapping[str, Any] | None,
) -> dict[str, Any]:
    kwargs = dict(raw or {})

    run_name = str(
        kwargs.pop(
            "run_name",
            "sampled_prepared",
        )
    )

    scale = kwargs.pop(
        "scale",
        "adaptive",
    )

    algorithm = str(
        kwargs.pop(
            "algorithm",
            "leiden_csr",
        )
    )

    warm_start = bool(
        kwargs.pop(
            "warm_start",
            True,
        )
    )

    backend = str(
        kwargs.pop(
            "community_backend",
            "cpu_lean_leiden",
        )
    )

    backend_kwargs = kwargs.pop(
        "community_backend_kwargs",
        None,
    )

    # Sample communities are landmarks for projection. Final min-cluster
    # cleaning belongs on the full-node result, not on sampled labels.
    kwargs.pop(
        "min_cluster_size",
        None,
    )
    kwargs.pop(
        "rank_stat_col",
        None,
    )

    if kwargs:
        raise TypeError(
            "unsupported sampled Leiden configuration keys: "
            + repr(sorted(kwargs))
        )

    if algorithm not in {
        "leiden_csr",
        "leiden",
    }:
        raise ValueError(
            "prepared sampled strategy currently supports only "
            "Leiden-compatible community backends"
        )

    if not (
        isinstance(scale, str)
        and scale.lower() == "adaptive"
    ):
        raise ValueError(
            "prepared sampled strategy is currently qualified only "
            "for scale='adaptive'"
        )

    return {
        "run_name": run_name,
        "scale": scale,
        "warm_start": warm_start,
        "backend": backend,
        "backend_kwargs": backend_kwargs,
    }


def _backend_call(
    csr,
    *,
    resolution: float,
    backend: str,
    backend_kwargs,
    initial_membership=None,
):
    kwargs = {
        "resolution": float(resolution),
        "backend": str(backend),
    }

    if initial_membership is not None:
        kwargs["initial_membership"] = np.asarray(
            initial_membership,
            dtype=np.int64,
        )

    if backend_kwargs is not None:
        kwargs["backend_kwargs"] = dict(
            backend_kwargs
        )

    return run_communities(
        csr,
        **kwargs,
    )


def live_prepared_sampled_community_call(
    inputs,
    sampled_build_result,
    resolutions: Sequence[Any],
):
    if (
        not isinstance(
            sampled_build_result,
            tuple,
        )
        or len(sampled_build_result) < 2
    ):
        raise TypeError(
            "sampled graph build must return "
            "(DataGraph, build_info)"
        )

    if not resolutions:
        raise ValueError(
            "at least one Leiden resolution is required"
        )

    sampled_graph = sampled_build_result[0]

    loader = InMemoryGraphLoader(
        sampled_graph,
        inputs.embedding_values,
    )

    config = _parse_leiden_kwargs(
        inputs.leiden_kwargs
    )

    analyzer_kwargs = (
        _candidate_analyzer_kwargs(
            inputs.analyzer_kwargs
        )
    )

    weighted_sample_csr = (
        distance_csr_to_similarity_csr(
            loader.adjacency,
            scale=config["scale"],
        )
        .tocsr(copy=False)
    )

    weighted_sample_csr.sum_duplicates()
    weighted_sample_csr.sort_indices()

    analyzer = OptimizedCommunityAnalyzer(
        loader,
        **analyzer_kwargs,
    )

    labels_by_resolution = {}
    previous_labels = None

    try:
        analyzer._ensure_prepared()

        sources = np.asarray(
            analyzer.coarsened_sources,
            dtype=np.int32,
        )

        targets = np.asarray(
            analyzer.coarsened_targets,
            dtype=np.int32,
        )

        weights = np.asarray(
            analyzer.coarsened_weights,
            dtype=np.float32,
        )

        n_prepared = int(
            analyzer.n_nodes_final
        )

        prepared_csr = (
            _csr_from_undirected_edges(
                sources,
                targets,
                weights,
                n_prepared,
            )
        )

        prepared_csr = prepared_csr.tocsr(
            copy=False
        )
        prepared_csr.sum_duplicates()
        prepared_csr.sort_indices()

        for resolution in resolutions:
            if (
                previous_labels is not None
                and config["warm_start"]
            ):
                initializer = previous_labels

            else:
                coarse_result = _backend_call(
                    prepared_csr,
                    resolution=float(
                        resolution
                    ),
                    backend=config["backend"],
                    backend_kwargs=config[
                        "backend_kwargs"
                    ],
                    initial_membership=None,
                )

                coarse_labels = np.asarray(
                    coarse_result.labels,
                    dtype=np.int64,
                )

                if coarse_labels.shape != (
                    n_prepared,
                ):
                    raise RuntimeError(
                        "prepared backend returned "
                        "incorrect label count"
                    )

                if bool(
                    analyzer.coarsened
                ):
                    meta_id = np.asarray(
                        analyzer.meta_id,
                        dtype=np.int64,
                    )

                    if meta_id.shape != (
                        loader.n_nodes,
                    ):
                        raise RuntimeError(
                            "coarsening meta_id does not "
                            "match sampled node count"
                        )

                    initializer = (
                        coarse_labels[
                            meta_id
                        ]
                    )
                else:
                    initializer = (
                        coarse_labels.copy()
                    )

            full_result = _backend_call(
                weighted_sample_csr,
                resolution=float(
                    resolution
                ),
                backend=config["backend"],
                backend_kwargs=config[
                    "backend_kwargs"
                ],
                initial_membership=initializer,
            )

            labels = np.asarray(
                full_result.labels,
                dtype=np.int64,
            )

            if labels.shape != (
                loader.n_nodes,
            ):
                raise RuntimeError(
                    "sample-full warm refinement did "
                    "not return one label per sampled node"
                )

            labels_by_resolution[
                resolution
            ] = labels

            previous_labels = labels

    finally:
        analyzer.close()

    return labels_by_resolution


__all__ = [
    "SAMPLED_PREPARE_COARSEN",
    "SAMPLED_PREPARE_COARSEN_LEVELS",
    "SAMPLED_PREPARE_SPARSIFY",
    "live_prepared_sampled_community_call",
]
