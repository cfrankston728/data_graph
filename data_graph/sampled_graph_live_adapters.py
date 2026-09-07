from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from . import DataGraphGenerator
from .generate_community_interface_subgraphs import (
    OptimizedCommunityAnalyzer,
    process_single_resolution,
)


_RESERVED_GENERATOR_KWARGS = {
    "node_df",
    "feature_cols",
    "premetric_weight_function",
    "embedding_function",
    "embedding_vectorizer",
}

_RESERVED_ANALYZER_KWARGS = {"graph_loader"}

_RESERVED_LEIDEN_KWARGS = {"resolution", "initial_membership"}


def _frozen_embedding_vectorizer(values):
    frozen = np.asarray(values)

    def vectorizer(node_df):
        if len(node_df) != len(frozen):
            raise ValueError(
                "frozen embedding row count does not match generator node_df: "
                + str(len(frozen))
                + " != "
                + str(len(node_df))
            )
        return frozen

    return vectorizer


def live_graph_factory(inputs):
    if inputs.embedding_values is None:
        raise ValueError("embedding_values must be materialized before live graph construction")

    generator_kwargs = dict(inputs.generator_kwargs)
    conflicting = sorted(_RESERVED_GENERATOR_KWARGS.intersection(generator_kwargs))
    if conflicting:
        raise TypeError("generator_kwargs contains pipeline-owned keys: " + repr(conflicting))

    return DataGraphGenerator(
        node_df=inputs.node_df,
        feature_cols=list(inputs.feature_columns),
        premetric_weight_function=inputs.premetric_weight_function,
        embedding_function=inputs.embedding_function,
        embedding_vectorizer=_frozen_embedding_vectorizer(inputs.embedding_values),
        **generator_kwargs,
    )


def live_graph_build_call(generator, graph_build_kwargs: Mapping[str, Any]):
    return generator.build_and_refine_graph(**dict(graph_build_kwargs))


class InMemoryGraphLoader:
    def __init__(self, graph, embedding_values):
        adjacency = graph.graph.tocsr()
        adjacency.sum_duplicates()
        adjacency.sort_indices()

        self.input_dir = None
        self._node_df = graph.node_df
        self._component_labels = graph.component_labels
        self._embedding = np.asarray(embedding_values)
        self._full_embedding = self._embedding
        self._adjacency = adjacency
        self._metadata = {}
        self._means = None
        self._sigmas = None

        coo = adjacency.tocoo()
        self._edge_arrays = (
            np.ascontiguousarray(coo.row, dtype=np.int64),
            np.ascontiguousarray(coo.col, dtype=np.int64),
            np.ascontiguousarray(coo.data, dtype=np.float32),
        )
        self.csr_offsets = np.asarray(adjacency.indptr)
        self.csr_indices = np.asarray(adjacency.indices)

    @property
    def metadata(self):
        return self._metadata

    @property
    def node_df(self):
        return self._node_df

    @property
    def component_labels(self):
        return self._component_labels

    @property
    def embedding(self):
        return self._embedding

    @property
    def full_embedding(self):
        return self._full_embedding

    @property
    def means(self):
        return self._means

    @property
    def sigmas(self):
        return self._sigmas

    @property
    def adjacency(self):
        return self._adjacency

    @property
    def edge_arrays(self):
        return self._edge_arrays

    @property
    def n_nodes(self):
        return len(self._node_df)

    @property
    def n_edges(self):
        return len(self._edge_arrays[0])

    def build_graph_wrapper(self, include_embedding: bool = True):
        loader = self

        class GraphWrapper:
            def __init__(self):
                self.node_df = loader.node_df.copy()
                if include_embedding and loader.embedding is not None:
                    emb = np.asarray(loader.embedding)
                    if emb.ndim == 2 and emb.shape[1] >= 2:
                        if "UMAP1" not in self.node_df:
                            self.node_df["UMAP1"] = emb[:, 0]
                            self.node_df["UMAP2"] = emb[:, 1]
                self.graph = type(
                    "GraphObj",
                    (),
                    {
                        "n_nodes": loader.n_nodes,
                        "get_edge_list": lambda _self: [
                            (int(u), int(v), float(d))
                            for u, v, d in zip(*loader.edge_arrays)
                        ],
                    },
                )()

        return GraphWrapper()


def live_sampled_community_call(inputs, sampled_build_result, resolutions: Sequence[Any]):
    if not isinstance(sampled_build_result, tuple) or len(sampled_build_result) < 2:
        raise TypeError("sampled graph build must return (DataGraph, build_info)")

    sampled_graph = sampled_build_result[0]
    loader = InMemoryGraphLoader(sampled_graph, inputs.embedding_values)

    analyzer_kwargs = dict(inputs.analyzer_kwargs)
    conflicting = sorted(_RESERVED_ANALYZER_KWARGS.intersection(analyzer_kwargs))
    if conflicting:
        raise TypeError("analyzer_kwargs contains pipeline-owned keys: " + repr(conflicting))

    # In-memory sampled analysis must never read/write disk prepared caches.
    analyzer_kwargs["prepared_graph_cache"] = False
    analyzer_kwargs["prepared_graph_cache_read"] = False
    analyzer_kwargs["prepared_graph_cache_write"] = False
    analyzer_kwargs["coarsening_stack_cache"] = False
    analyzer_kwargs["coarsening_stack_cache_write"] = False

    leiden_kwargs = dict(inputs.leiden_kwargs)
    forbidden = {"resolution", "analyzer", "output_dir", "prev_labels", "save_outputs"}
    conflicting = sorted(forbidden.intersection(leiden_kwargs))
    if conflicting:
        raise TypeError("leiden_kwargs contains pipeline-owned keys: " + repr(conflicting))

    run_name = str(leiden_kwargs.pop("run_name", "sampled"))
    scale = leiden_kwargs.pop("scale", "adaptive")
    min_cluster_size = leiden_kwargs.pop("min_cluster_size", None)
    rank_stat_col = leiden_kwargs.pop("rank_stat_col", None)
    warm_start = bool(leiden_kwargs.pop("warm_start", True))
    algorithm = str(leiden_kwargs.pop("algorithm", "leiden_csr"))
    community_backend = leiden_kwargs.pop("community_backend", None)

    if leiden_kwargs:
        raise TypeError("unsupported sampled Leiden configuration keys: " + repr(sorted(leiden_kwargs)))

    analyzer = OptimizedCommunityAnalyzer(loader, **analyzer_kwargs)
    labels_by_resolution = {}
    previous_labels = None

    try:
        for resolution in resolutions:
            labels = process_single_resolution(
                resolution=float(resolution),
                analyzer=analyzer,
                output_dir="",
                run_name=run_name,
                scale=scale,
                min_cluster_size=min_cluster_size,
                rank_stat_col=rank_stat_col,
                prev_labels=previous_labels,
                warm_start=warm_start,
                save_outputs=False,
                algorithm=algorithm,
                community_backend=community_backend,
            )
            labels_by_resolution[resolution] = labels
            previous_labels = labels
    finally:
        analyzer.close()

    return labels_by_resolution
