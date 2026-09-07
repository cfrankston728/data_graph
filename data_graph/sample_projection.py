from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit


@njit(cache=False, nogil=True)
def _direct_projection(
    indptr,
    indices,
    data,
    old_to_sample,
    sample_labels,
    n_communities,
):
    n = old_to_sample.size

    assigned = np.full(
        n,
        -1,
        dtype=np.int32,
    )

    for u in range(n):
        s = old_to_sample[u]

        if s >= 0:
            assigned[u] = (
                sample_labels[s]
            )

    seen = np.full(
        n_communities,
        -1,
        dtype=np.int32,
    )

    weight_to = np.zeros(
        n_communities,
        dtype=np.float64,
    )

    touched = np.empty(
        n_communities,
        dtype=np.int32,
    )

    unresolved = 0

    for u in range(n):
        if old_to_sample[u] >= 0:
            continue

        nt = 0
        stamp = u

        for p in range(
            indptr[u],
            indptr[u + 1],
        ):
            v = indices[p]
            s = old_to_sample[v]

            if s < 0:
                continue

            c = sample_labels[s]

            if seen[c] != stamp:
                seen[c] = stamp
                weight_to[c] = 0.0
                touched[nt] = c
                nt += 1

            weight_to[c] += float(
                data[p]
            )

        if nt == 0:
            unresolved += 1
            continue

        best_c = touched[0]
        best_w = weight_to[
            best_c
        ]

        for j in range(
            1,
            nt,
        ):
            c = touched[j]
            w = weight_to[c]

            if (
                w > best_w
                or (
                    w == best_w
                    and c < best_c
                )
            ):
                best_c = c
                best_w = w

        assigned[u] = best_c

    return (
        assigned,
        unresolved,
    )


@njit(cache=False, nogil=True)
def _propagation_round(
    indptr,
    indices,
    data,
    assigned,
    n_communities,
):
    n = assigned.size

    # Exact R166/R166-R1 semantic:
    # use a synchronous snapshot each round.
    next_assigned = (
        assigned.copy()
    )

    seen = np.full(
        n_communities,
        -1,
        dtype=np.int32,
    )

    weight_to = np.zeros(
        n_communities,
        dtype=np.float64,
    )

    touched = np.empty(
        n_communities,
        dtype=np.int32,
    )

    newly = 0
    remaining = 0

    for u in range(n):
        if assigned[u] >= 0:
            continue

        nt = 0
        stamp = u

        for p in range(
            indptr[u],
            indptr[u + 1],
        ):
            v = indices[p]
            c = assigned[v]

            if c < 0:
                continue

            if seen[c] != stamp:
                seen[c] = stamp
                weight_to[c] = 0.0
                touched[nt] = c
                nt += 1

            weight_to[c] += float(
                data[p]
            )

        if nt == 0:
            remaining += 1
            continue

        best_c = touched[0]
        best_w = weight_to[
            best_c
        ]

        for j in range(
            1,
            nt,
        ):
            c = touched[j]
            w = weight_to[c]

            if (
                w > best_w
                or (
                    w == best_w
                    and c < best_c
                )
            ):
                best_c = c
                best_w = w

        next_assigned[u] = (
            best_c
        )

        newly += 1

    return (
        next_assigned,
        newly,
        remaining,
    )


def _compact(
    labels,
):
    labels = np.asarray(
        labels
    )

    if (
        labels.ndim != 1
        or not np.issubdtype(
            labels.dtype,
            np.integer,
        )
        or np.any(
            labels < 0
        )
    ):
        raise ValueError(
            "labels must be 1D nonnegative integers"
        )

    return np.unique(
        labels,
        return_inverse=True,
    )[1].astype(
        np.int32,
        copy=False,
    )


@dataclass(frozen=True)
class ProjectionResult:
    labels: np.ndarray
    direct_orphans: int
    rounds: tuple[dict, ...]
    remaining_final: int


def project_sample_labels_to_full(
    weighted_full_csr,
    sampled_indices,
    sample_labels,
    *,
    max_rounds=15,
):
    graph = (
        weighted_full_csr
        .tocsr()
    )

    graph.sort_indices()

    sampled_indices = np.asarray(
        sampled_indices,
        dtype=np.int64,
    )

    sample_labels = _compact(
        sample_labels
    )

    if (
        sampled_indices.ndim != 1
        or sampled_indices.size
        != sample_labels.size
        or sampled_indices.size == 0
    ):
        raise ValueError(
            "invalid sample arrays"
        )

    if (
        np.any(
            sampled_indices < 0
        )
        or np.any(
            sampled_indices
            >= graph.shape[0]
        )
        or np.unique(
            sampled_indices
        ).size
        != sampled_indices.size
    ):
        raise ValueError(
            "sampled_indices invalid"
        )

    n_communities = (
        int(
            sample_labels.max()
        )
        + 1
    )

    old_to_sample = np.full(
        graph.shape[0],
        -1,
        dtype=np.int32,
    )

    old_to_sample[
        sampled_indices
    ] = np.arange(
        sampled_indices.size,
        dtype=np.int32,
    )

    (
        assigned,
        direct_orphans,
    ) = _direct_projection(
        graph.indptr,
        graph.indices,
        graph.data,
        old_to_sample,
        sample_labels,
        n_communities,
    )

    rounds = []

    for round_id in range(
        1,
        int(
            max_rounds
        )
        + 1,
    ):
        remaining_before = int(
            np.count_nonzero(
                assigned < 0
            )
        )

        if remaining_before == 0:
            break

        (
            assigned_next,
            newly,
            remaining,
        ) = _propagation_round(
            graph.indptr,
            graph.indices,
            graph.data,
            assigned,
            n_communities,
        )

        rounds.append({
            "round":
                round_id,

            "remaining_before":
                remaining_before,

            "assigned":
                int(
                    newly
                ),

            "remaining_after":
                int(
                    remaining
                ),
        })

        assigned = (
            assigned_next
        )

        if newly == 0:
            break

    if np.any(
        assigned[
            sampled_indices
        ]
        != sample_labels
    ):
        raise RuntimeError(
            "sample anchors changed"
        )

    return ProjectionResult(
        labels=np.asarray(
            assigned,
            dtype=np.int32,
        ),

        direct_orphans=int(
            direct_orphans
        ),

        rounds=tuple(
            rounds
        ),

        remaining_final=int(
            np.count_nonzero(
                assigned < 0
            )
        ),
    )
