from __future__ import annotations

import ctypes
import time
from pathlib import Path
from typing import Optional

import numpy as np
from numba import njit
from scipy import sparse
from sknetwork.clustering import Leiden

from . import empty_aware_fast_local_move as _fine_move
from . import reactive_after_first_fast_local_move as _warm_reactive_move
from . import exact_dependency_fast_local_move as _exact_dependency_move
from . import empty_aware_selfloop_fast_local_move as _coarse_move


@njit(cache=False, nogil=True)
def _modularity_core(
    indptr,
    indices,
    data,
    labels,
    n_communities,
    resolution,
):
    total = 0.0
    internal = 0.0

    volume = np.zeros(
        n_communities,
        dtype=np.float64,
    )

    for u in range(labels.size):
        cu = labels[u]
        strength = 0.0

        for p in range(indptr[u], indptr[u + 1]):
            v = indices[p]
            w = float(data[p])

            total += w
            strength += w

            if cu == labels[v]:
                internal += w

        volume[cu] += strength

    if total <= 0.0:
        return 0.0

    null = 0.0

    for c in range(n_communities):
        x = volume[c] / total
        null += x * x

    return internal / total - resolution * null


def _compact(labels):
    labels = np.asarray(labels)

    if labels.ndim != 1:
        raise ValueError(
            "membership labels must be one-dimensional"
        )

    if labels.size == 0:
        return np.empty(0, dtype=np.int32)

    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError(
            "membership labels must be integer-valued"
        )

    if np.any(labels < 0):
        raise ValueError(
            "membership labels must be non-negative"
        )

    return np.unique(
        labels,
        return_inverse=True,
    )[1].astype(
        np.int32,
        copy=False,
    )


def _exact_q(adjacency, labels, resolution):
    labels = _compact(labels)

    if labels.size == 0:
        return 0.0

    return float(
        _modularity_core(
            adjacency.indptr,
            adjacency.indices,
            adjacency.data,
            labels,
            int(labels.max()) + 1,
            float(resolution),
        )
    )


class _CanonicalRefiner:
    def __init__(self):
        so = (
            Path(__file__).resolve().parent
            / "libcanonical_refine_core.so"
        )

        if not so.is_file():
            raise RuntimeError(
                "LeanLeiden native refinement library missing: "
                + str(so)
            )

        self._lib = ctypes.CDLL(str(so))

        self._i32p = ctypes.POINTER(ctypes.c_int32)
        self._f32p = ctypes.POINTER(ctypes.c_float)
        self._i64p = ctypes.POINTER(ctypes.c_int64)
        self._f64p = ctypes.POINTER(ctypes.c_double)

        fn = self._lib.canonical_refine_newman_undirected

        fn.argtypes = [
            self._i32p,
            self._i32p,
            self._i32p,
            self._f32p,
            self._f32p,
            ctypes.c_int32,
            ctypes.c_float,
            ctypes.c_double,
            ctypes.c_uint64,
            self._i32p,
            self._i64p,
            self._f64p,
        ]

        fn.restype = ctypes.c_int
        self._fn = fn

    def run(
        self,
        adjacency,
        out_weights,
        parent,
        *,
        resolution,
        theta,
        seed,
    ):
        if (
            adjacency.indices.dtype != np.int32
            or adjacency.indptr.dtype != np.int32
        ):
            raise TypeError(
                "LeanLeiden refinement requires int32 CSR indices/indptr"
            )

        parent32 = np.ascontiguousarray(
            parent,
            dtype=np.int32,
        )

        data32 = np.ascontiguousarray(
            adjacency.data,
            dtype=np.float32,
        )

        out32 = np.ascontiguousarray(
            out_weights,
            dtype=np.float32,
        )

        labels = np.empty(
            parent32.size,
            dtype=np.int32,
        )

        stats_i64 = np.zeros(
            8,
            dtype=np.int64,
        )

        stats_f64 = np.zeros(
            3,
            dtype=np.float64,
        )

        rc = self._fn(
            parent32.ctypes.data_as(self._i32p),
            adjacency.indices.ctypes.data_as(self._i32p),
            adjacency.indptr.ctypes.data_as(self._i32p),
            data32.ctypes.data_as(self._f32p),
            out32.ctypes.data_as(self._f32p),
            ctypes.c_int32(parent32.size),
            ctypes.c_float(float(resolution)),
            ctypes.c_double(float(theta)),
            ctypes.c_uint64(int(seed)),
            labels.ctypes.data_as(self._i32p),
            stats_i64.ctypes.data_as(self._i64p),
            stats_f64.ctypes.data_as(self._f64p),
        )

        if rc != 0:
            raise RuntimeError(
                "canonical refinement failed with rc="
                + str(rc)
            )

        return _compact(labels)


class LeanLeiden:
    """Lean project-owned multilevel Leiden.

    Fine level:
        corrected empty-aware R154 mover

    Refinement:
        qualified R143 canonical/well-connected refiner

    Aggregation:
        scikit-network Leiden refined aggregation

    Coarse levels:
        self-loop-aware R155 mover

    Execution:
        streaming state only; no graph-history retention;
        stop when local moving reports zero moves.
    """

    def __init__(
        self,
        *,
        resolution=1.0,
        modularity="newman",
        n_aggregations=50,
        refinement_theta=0.01,
        refinement_seed=155143,
        warm_level0_scheduler=None,
        verbose=False,
    ):
        if str(modularity).lower() != "newman":
            raise ValueError(
                "LeanLeiden currently supports modularity='newman' only"
            )

        if int(n_aggregations) < 1:
            raise ValueError(
                "n_aggregations must be positive"
            )

        self.resolution = float(resolution)
        self.modularity = "newman"
        self.n_aggregations = int(n_aggregations)
        self.refinement_theta = float(refinement_theta)
        self.refinement_seed = int(refinement_seed)
        self.verbose = bool(verbose)

        if warm_level0_scheduler is None:
            warm_level0_scheduler = "canonical"

        warm_level0_scheduler = str(
            warm_level0_scheduler
        ).lower()

        if warm_level0_scheduler not in {
            "canonical",
            "reactive_after_first",
            "exact_dependency_skip",
        }:
            raise ValueError(
                "warm_level0_scheduler must be one of "
                "{'canonical', 'reactive_after_first', "
                "'exact_dependency_skip'}"
            )

        self.warm_level0_scheduler = (
            warm_level0_scheduler
        )

        self._engine = Leiden(
            resolution=self.resolution,
            modularity="newman",
            return_probs=False,
            return_aggregate=False,
            tol_optimization=1e-3,
            tol_aggregation=1e-3,
            n_aggregations=self.n_aggregations,
            shuffle_nodes=False,
            random_state=42,
            verbose=False,
        )

        self._refiner = _CanonicalRefiner()

        self.labels_ = None
        self.initial_objective_ = None
        self.objective_ = None
        self.terminal_objective_ = None
        self.terminal_representation_error_ = None
        self.level_reports_ = None

    def _preprocess(self, input_matrix):
        if not sparse.isspmatrix_csr(input_matrix):
            input_matrix = sparse.csr_matrix(input_matrix)

        # scikit-network 0.33.0 directed2undirected() branches on
        # ``adjacency.data.dtype == float``. That test is false for
        # float32 and causes one copy of the weighted adjacency to be
        # cast to integer before A + A.T, distorting sub-unit weights.
        # Promote explicitly so symmetric weighted graphs retain their
        # relative edge weights through engine preprocessing.
        input_matrix = input_matrix.astype(
            np.float64,
            copy=False,
        )

        (
            adjacency,
            out_weights,
            in_weights,
            _membership,
            _index,
        ) = self._engine._pre_processing(
            input_matrix,
            False,
        )

        if (
            adjacency.indices.dtype != np.int32
            or adjacency.indptr.dtype != np.int32
        ):
            raise TypeError(
                "LeanLeiden requires int32 CSR indices/indptr"
            )

        if (
            adjacency.data.size
            and float(adjacency.data.min()) < 0.0
        ):
            raise ValueError(
                "LeanLeiden requires non-negative weights"
            )

        return adjacency, out_weights, in_weights

    def score_partition(self, input_matrix, labels):
        adjacency, _, _ = self._preprocess(input_matrix)
        labels = _compact(labels)

        if labels.shape[0] != adjacency.shape[0]:
            raise ValueError(
                "partition length does not match graph node count"
            )

        return _exact_q(
            adjacency,
            labels,
            self.resolution,
        )

    def fit_predict(
        self,
        input_matrix,
        initial_membership: Optional[np.ndarray] = None,
    ):
        (
            original_adj,
            original_out,
            original_in,
        ) = self._preprocess(input_matrix)

        n = int(original_adj.shape[0])

        # Qualified R154 fine-level contract.
        if np.any(
            np.asarray(original_adj.diagonal()) != 0
        ):
            raise ValueError(
                "LeanLeiden fine-level input must not contain self-loops"
            )

        warm_start_supplied = (
            initial_membership is not None
        )

        if initial_membership is None:
            current_labels = np.arange(
                n,
                dtype=np.int32,
            )
        else:
            current_labels = _compact(initial_membership)

            if current_labels.shape[0] != n:
                raise ValueError(
                    "initial_membership length must match node count"
                )

        initial_q = _exact_q(
            original_adj,
            current_labels,
            self.resolution,
        )

        current_adj = original_adj
        current_out = original_out
        current_in = original_in

        original_to_current = np.arange(
            n,
            dtype=np.int32,
        )

        reports = []
        final_labels = None
        terminal_q = None

        for level in range(self.n_aggregations):
            q_before = _exact_q(
                current_adj,
                current_labels,
                self.resolution,
            )

            if level == 0:
                if (
                    warm_start_supplied
                    and self.warm_level0_scheduler
                    == "exact_dependency_skip"
                    and int(np.max(current_labels)) < 64
                ):
                    moved = (
                        _exact_dependency_move.
                        optimize_exact_dependency_skip(
                            current_adj.indptr,
                            current_adj.indices,
                            current_adj.data,
                            current_labels,
                            self.resolution,
                        )
                    )
                elif (
                    warm_start_supplied
                    and self.warm_level0_scheduler
                    == "reactive_after_first"
                ):
                    reactive = (
                        _warm_reactive_move.
                        optimize_empty_aware_newman_reactive_after_first(
                            current_adj.indptr,
                            current_adj.indices,
                            current_adj.data,
                            current_labels,
                            self.resolution,
                        )
                    )

                    completed = (
                        _fine_move.optimize_empty_aware_newman(
                            current_adj.indptr,
                            current_adj.indices,
                            current_adj.data,
                            reactive["labels"],
                            self.resolution,
                        )
                    )

                    moved = {
                        "labels":
                            completed["labels"],
                        "increase":
                            float(reactive["increase"])
                            + float(completed["increase"]),
                        "passes":
                            int(reactive["passes"])
                            + int(completed["passes"]),
                        "moves":
                            int(reactive["moves"])
                            + int(completed["moves"]),
                        "empty_community_moves":
                            int(
                                reactive[
                                    "empty_community_moves"
                                ]
                            )
                            + int(
                                completed[
                                    "empty_community_moves"
                                ]
                            ),
                    }
                else:
                    moved = _fine_move.optimize_empty_aware_newman(
                        current_adj.indptr,
                        current_adj.indices,
                        current_adj.data,
                        current_labels,
                        self.resolution,
                    )
            else:
                moved = _coarse_move.optimize(
                    current_adj.indptr,
                    current_adj.indices,
                    current_adj.data,
                    current_labels,
                    self.resolution,
                )

            parent = _compact(moved["labels"])

            moves = int(moved["moves"])
            empty_moves = int(
                moved["empty_community_moves"]
            )

            reported_increase = float(
                moved["increase"]
            )

            q_after = _exact_q(
                current_adj,
                parent,
                self.resolution,
            )

            delta = float(q_after - q_before)
            gain_error = abs(
                delta - reported_increase
            )

            if delta < -1e-9:
                raise RuntimeError(
                    f"local moving decreased Q at level {level}"
                )

            if gain_error > 5e-8:
                raise RuntimeError(
                    f"mover gain mismatch at level {level}: "
                    f"{gain_error}"
                )

            report = {
                "level": int(level),
                "n": int(current_adj.shape[0]),
                "nnz": int(current_adj.nnz),
                "communities_before":
                    int(np.unique(current_labels).size),
                "communities_after":
                    int(np.unique(parent).size),
                "q_before": float(q_before),
                "q_after": float(q_after),
                "delta": float(delta),
                "moves": moves,
                "empty_moves": empty_moves,
                "gain_error": float(gain_error),
            }

            if moves == 0:
                final_labels = _compact(
                    parent[original_to_current]
                )

                terminal_q = float(q_after)

                report["terminal"] = True
                reports.append(report)
                break

            t0 = time.perf_counter()

            refined = self._refiner.run(
                current_adj,
                current_out,
                parent,
                resolution=self.resolution,
                theta=self.refinement_theta,
                seed=self.refinement_seed,
            )

            refinement_seconds = float(
                time.perf_counter() - t0
            )

            t0 = time.perf_counter()

            (
                aggregate_initial,
                aggregate_adj,
                aggregate_out,
                aggregate_in,
            ) = self._engine._aggregate_refine(
                parent,
                refined,
                current_adj,
                current_out,
                current_in,
            )

            aggregation_seconds = float(
                time.perf_counter() - t0
            )

            aggregate_initial = _compact(
                np.asarray(
                    aggregate_initial,
                    dtype=np.int32,
                )
            )

            aggregate_q = _exact_q(
                aggregate_adj,
                aggregate_initial,
                self.resolution,
            )

            representation_error = abs(
                aggregate_q - q_after
            )

            if representation_error > 5e-8:
                raise RuntimeError(
                    f"aggregate representation mismatch "
                    f"at level {level}: "
                    f"{representation_error}"
                )

            report.update({
                "refined_communities":
                    int(np.unique(refined).size),

                "refinement_seconds":
                    refinement_seconds,

                "aggregate_n":
                    int(aggregate_adj.shape[0]),

                "aggregate_nnz":
                    int(aggregate_adj.nnz),

                "aggregate_q":
                    float(aggregate_q),

                "representation_error":
                    float(representation_error),

                "aggregation_seconds":
                    aggregation_seconds,

                "terminal":
                    False,
            })

            reports.append(report)

            original_to_current = (
                refined[original_to_current]
            ).astype(
                np.int32,
                copy=False,
            )

            current_adj = aggregate_adj
            current_out = np.asarray(aggregate_out)
            current_in = np.asarray(aggregate_in)
            current_labels = aggregate_initial

        else:
            raise RuntimeError(
                "LeanLeiden exceeded n_aggregations without "
                "zero-move termination"
            )

        if final_labels is None:
            raise RuntimeError(
                "LeanLeiden produced no terminal partition"
            )

        final_q = _exact_q(
            original_adj,
            final_labels,
            self.resolution,
        )

        terminal_error = abs(
            final_q - float(terminal_q)
        )

        if terminal_error > 5e-8:
            raise RuntimeError(
                "flattened partition does not reproduce terminal Q: "
                + str(terminal_error)
            )

        if final_q < initial_q - 1e-9:
            raise RuntimeError(
                "final objective is below initial objective"
            )

        self.labels_ = final_labels
        self.initial_objective_ = float(initial_q)
        self.objective_ = float(final_q)
        self.terminal_objective_ = float(terminal_q)
        self.terminal_representation_error_ = float(
            terminal_error
        )
        self.level_reports_ = reports

        return self.labels_
