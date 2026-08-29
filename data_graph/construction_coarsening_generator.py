
from __future__ import annotations

import numpy as np

from . import data_graph_generator as _generator_module

from .data_graph_generator import DataGraphGenerator
from .first_coarsening_core import (
    aggregate_undirected_edges_with_dist_chunked,
    mutual_nn_coarsening_directed,
    sparsify_knn_undirected,
)

from .first_coarsening_core import (
    BoundedFirstCoarseningAccumulator,
)


def _level0_weights(distances):
    distances = np.asarray(
        distances,
        dtype=np.float32,
    )

    if distances.size == 0:
        return np.asarray(
            [],
            dtype=np.float32,
        )

    scale = float(
        np.median(
            distances
        )
    )

    if (
        not np.isfinite(scale)
        or scale <= 0.0
    ):
        scale = 1.0

    return np.exp(
        -(
            distances
            / scale
        ) ** 2
        / 2.0
    ).astype(
        np.float32,
        copy=False,
    )


def build_first_coarsening_seed(
    accumulator,
    *,
    pre_k,
):
    """
    Convert the bounded construction-time accumulator into the exact
    pre-final-sparsification hierarchy prefix:

        level_000 -> one coarsening round -> level_001

    The canonical full graph is not modified.
    """
    (
        level0_sources,
        level0_targets,
        level0_distances,
    ) = accumulator.materialize_level0()

    level0_sources = np.ascontiguousarray(
        level0_sources,
        dtype=np.int64,
    )
    level0_targets = np.ascontiguousarray(
        level0_targets,
        dtype=np.int64,
    )
    level0_distances = np.ascontiguousarray(
        level0_distances,
        dtype=np.float32,
    )

    n_nodes = int(
        accumulator.n_nodes
    )

    level0_weights = np.ascontiguousarray(
        _level0_weights(
            level0_distances
        ),
        dtype=np.float32,
    )

    level0_mapping = np.arange(
        n_nodes,
        dtype=np.int64,
    )

    (
        local_meta_id,
        n_meta,
    ) = mutual_nn_coarsening_directed(
        level0_sources,
        level0_targets,
        level0_weights,
        n_nodes,
    )

    local_meta_id = np.asarray(
        local_meta_id,
        dtype=np.int64,
    )

    ratio = (
        float(n_meta)
        / float(n_nodes)
        if n_nodes
        else 1.0
    )

    mapped_sources = (
        local_meta_id[
            level0_sources
        ]
    )

    mapped_targets = (
        local_meta_id[
            level0_targets
        ]
    )

    keep = (
        mapped_sources
        != mapped_targets
    )

    mapped_sources = (
        mapped_sources[keep]
    )
    mapped_targets = (
        mapped_targets[keep]
    )
    mapped_weights = (
        level0_weights[keep]
    )
    mapped_distances = (
        level0_distances[keep]
    )

    (
        level1_sources,
        level1_targets,
        level1_weights,
        level1_distances,
    ) = aggregate_undirected_edges_with_dist_chunked(
        mapped_sources,
        mapped_targets,
        mapped_weights,
        mapped_distances,
        chunk=8_000_000,
    )

    k_level = max(
        1,
        int(
            round(
                int(pre_k)
                * ratio
            )
        ),
    )

    (
        level1_sources,
        level1_targets,
        level1_weights,
        level1_distances,
    ) = sparsify_knn_undirected(
        level1_sources,
        level1_targets,
        level1_weights,
        level1_distances,
        int(n_meta),
        int(k_level),
    )

    level1_mapping = (
        local_meta_id[
            level0_mapping
        ]
    )

    return {
        "schema_version": 1,
        "source": "construction_time_zero_polish",
        "pre_k": int(pre_k),
        "completed_rounds": 1,
        "level_000": {
            "sources": np.asarray(
                level0_sources,
                dtype=np.int64,
            ),
            "targets": np.asarray(
                level0_targets,
                dtype=np.int64,
            ),
            "distances": np.asarray(
                level0_distances,
                dtype=np.float32,
            ),
            "weights": np.asarray(
                level0_weights,
                dtype=np.float32,
            ),
            "cumulative_mapping": np.asarray(
                level0_mapping,
                dtype=np.int64,
            ),
            "local_mapping": None,
            "n_nodes": int(n_nodes),
            "n_edges": int(
                len(
                    level0_sources
                )
            ),
            "k_prev": int(pre_k),
            "reduction_ratio": 1.0,
        },
        "level_001": {
            "sources": np.asarray(
                level1_sources,
                dtype=np.int64,
            ),
            "targets": np.asarray(
                level1_targets,
                dtype=np.int64,
            ),
            "distances": np.asarray(
                level1_distances,
                dtype=np.float32,
            ),
            "weights": np.asarray(
                level1_weights,
                dtype=np.float32,
            ),
            "cumulative_mapping": np.asarray(
                level1_mapping,
                dtype=np.int64,
            ),
            "local_mapping": np.asarray(
                local_meta_id,
                dtype=np.int64,
            ),
            "n_nodes": int(n_meta),
            "n_edges": int(
                len(
                    level1_sources
                )
            ),
            "k_prev": int(k_level),
            "reduction_ratio": float(
                ratio
            ),
        },
    }


class ConstructionCoarseningDataGraphGenerator(
    DataGraphGenerator
):
    """
    Scratch integration candidate.

    Adds an optional construction-time first-coarsening seed without changing
    the canonical full-resolution graph.

    Current guarded contract:
      - opt-in only
      - finalized only for polish_iterations == 0
      - Euclidean graph-distance path feeds compact extraction directly
      - custom graph-distance path consumes existing distance batches
      - no persistent cache is written here
    """

    def build_and_refine_graph(
        self,
        *args,
        construction_coarsening_seed=False,
        construction_coarsening_pre_k=10,
        **kwargs,
    ):
        polish_iterations = int(
            kwargs.get(
                "polish_iterations",
                0,
            )
        )

        requested = bool(
            construction_coarsening_seed
        )

        enabled = (
            requested
            and polish_iterations == 0
        )

        accumulator = None

        if enabled:
            accumulator = (
                BoundedFirstCoarseningAccumulator(
                    n_nodes=len(
                        self.node_df
                    ),
                    k=int(
                        construction_coarsening_pre_k
                    ),
                )
            )

        original_extract = (
            _generator_module
            .extract_all_edges_from_csr
        )

        old_batcher_exists = hasattr(
            self,
            "_batcher",
        )

        old_batcher = getattr(
            self,
            "_batcher",
            None,
        )

        old_batcher_initialized_exists = hasattr(
            self,
            "_batcher_initialized",
        )

        old_batcher_initialized = getattr(
            self,
            "_batcher_initialized",
            None,
        )

        fed_euclidean = {
            "done": False,
        }

        try:
            if (
                enabled
                and self.use_euclidean_as_graph_distance
            ):
                def capture_extract(
                    graph,
                    upper_only=False,
                ):
                    (
                        edge_arr,
                        distances,
                    ) = original_extract(
                        graph,
                        upper_only=upper_only,
                    )

                    if (
                        upper_only
                        and not fed_euclidean[
                            "done"
                        ]
                    ):
                        accumulator.add_compact_arrays(
                            edge_arr,
                            distances,
                            missing_weight=(
                                self.missing_weight
                            ),
                        )

                        fed_euclidean[
                            "done"
                        ] = True

                    return (
                        edge_arr,
                        distances,
                    )

                _generator_module.extract_all_edges_from_csr = (
                    capture_extract
                )

            elif enabled:
                base_batcher = (
                    self._get_batcher()
                )

                def accumulating_batcher(
                    feats,
                    idx_i,
                    idx_j,
                ):
                    distances = np.asarray(
                        base_batcher(
                            feats,
                            idx_i,
                            idx_j,
                        )
                    )

                    accumulator.add_batch(
                        idx_i,
                        idx_j,
                        distances,
                        missing_weight=(
                            self.missing_weight
                        ),
                    )

                    return distances

                self._batcher = (
                    accumulating_batcher
                )
                self._batcher_initialized = True

            (
                graph_obj,
                build_info,
            ) = super().build_and_refine_graph(
                *args,
                **kwargs,
            )

        finally:
            _generator_module.extract_all_edges_from_csr = (
                original_extract
            )

            if old_batcher_exists:
                self._batcher = (
                    old_batcher
                )
            elif hasattr(
                self,
                "_batcher",
            ):
                delattr(
                    self,
                    "_batcher",
                )

            if old_batcher_initialized_exists:
                self._batcher_initialized = (
                    old_batcher_initialized
                )
            elif hasattr(
                self,
                "_batcher_initialized",
            ):
                delattr(
                    self,
                    "_batcher_initialized",
                )

        build_info = dict(
            build_info
        )

        build_info[
            "construction_coarsening_requested"
        ] = requested

        build_info[
            "construction_coarsening_finalized"
        ] = enabled

        if requested and not enabled:
            build_info[
                "construction_coarsening_deferred_reason"
            ] = (
                "polish_iterations > 0"
            )

            build_info[
                "construction_coarsening_seed"
            ] = None

        elif enabled:
            build_info[
                "construction_coarsening_seed"
            ] = build_first_coarsening_seed(
                accumulator,
                pre_k=int(
                    construction_coarsening_pre_k
                ),
            )

        else:
            build_info[
                "construction_coarsening_seed"
            ] = None

        return (
            graph_obj,
            build_info,
        )
