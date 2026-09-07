
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=False, nogil=True)
def _sort(values, n_values):
    for i in range(1, n_values):
        x = values[i]
        j = i - 1

        while j >= 0 and values[j] > x:
            values[j + 1] = values[j]
            j -= 1

        values[j + 1] = x


@njit(cache=False, nogil=True)
def _core(
    indptr,
    indices,
    data,
    initial_labels,
    resolution,
    max_passes,
):
    n = initial_labels.size
    labels = initial_labels.copy()

    strength = np.zeros(
        n,
        dtype=np.float64,
    )

    self_weight = np.zeros(
        n,
        dtype=np.float64,
    )

    total_weight = 0.0
    max_degree = 0

    for u in range(n):
        degree = indptr[u + 1] - indptr[u]

        if degree > max_degree:
            max_degree = degree

        ku = 0.0
        loop = 0.0

        for p in range(
            indptr[u],
            indptr[u + 1],
        ):
            v = indices[p]
            w = float(data[p])

            if w < 0.0:
                return (
                    labels,
                    np.nan,
                    -1,
                    -1,
                    -1,
                    -1,
                )

            ku += w

            if v == u:
                loop += w

        strength[u] = ku
        self_weight[u] = loop
        total_weight += ku

    if total_weight <= 0.0:
        return (
            labels,
            0.0,
            0,
            0,
            0,
            0,
        )

    counts = np.zeros(
        n,
        dtype=np.int32,
    )

    volumes = np.zeros(
        n,
        dtype=np.float64,
    )

    for u in range(n):
        c = labels[u]

        if c < 0 or c >= n:
            return (
                labels,
                np.nan,
                -2,
                -1,
                -1,
                -1,
            )

        counts[c] += 1
        volumes[c] += strength[u]

    # Pool of currently empty labels.
    next_free = np.full(
        n,
        -1,
        dtype=np.int32,
    )

    free_head = -1

    for c in range(
        n - 1,
        -1,
        -1,
    ):
        if counts[c] == 0:
            next_free[c] = free_head
            free_head = c

    seen = np.full(
        n,
        -1,
        dtype=np.int64,
    )

    weight_to = np.zeros(
        n,
        dtype=np.float64,
    )

    candidates = np.empty(
        max_degree + 1,
        dtype=np.int32,
    )

    stamp = np.int64(0)

    total_gain = 0.0
    total_moves = 0
    empty_moves = 0
    passes_done = 0

    total_sq = total_weight * total_weight

    for pass_index in range(max_passes):
        moved = 0

        for u in range(n):
            source = labels[u]
            ku = strength[u]
            loop_u = self_weight[u]

            stamp += 1
            nc = 0

            for p in range(
                indptr[u],
                indptr[u + 1],
            ):
                c = labels[
                    indices[p]
                ]

                w = float(data[p])

                if seen[c] != stamp:
                    seen[c] = stamp
                    weight_to[c] = w
                    candidates[nc] = c
                    nc += 1
                else:
                    weight_to[c] += w

            _sort(
                candidates,
                nc,
            )

            if seen[source] == stamp:
                w_source_all = weight_to[
                    source
                ]
            else:
                w_source_all = 0.0

            # u's diagonal entry never ceases to be internal.
            w_source_external = (
                w_source_all
                - loop_u
            )

            source_volume = volumes[
                source
            ]

            best_dest = source
            best_delta = 0.0
            best_empty = False

            # --------------------------------------------------
            # Existing neighboring communities.
            # --------------------------------------------------

            for j in range(nc):
                dest = candidates[j]

                if dest == source:
                    continue

                w_dest = weight_to[dest]
                dest_volume = volumes[dest]

                delta_internal = (
                    2.0
                    * (
                        w_dest
                        - w_source_external
                    )
                    / total_weight
                )

                old_null = (
                    source_volume
                    * source_volume
                    + dest_volume
                    * dest_volume
                )

                new_source_volume = (
                    source_volume
                    - ku
                )

                new_dest_volume = (
                    dest_volume
                    + ku
                )

                new_null = (
                    new_source_volume
                    * new_source_volume
                    + new_dest_volume
                    * new_dest_volume
                )

                delta = (
                    delta_internal
                    - resolution
                    * (
                        new_null
                        - old_null
                    )
                    / total_sq
                )

                if delta > best_delta:
                    best_delta = delta
                    best_dest = dest
                    best_empty = False

            # --------------------------------------------------
            # Empty / new community.
            # --------------------------------------------------

            if (
                counts[source] > 1
                and free_head >= 0
            ):
                dest = free_head

                delta_internal = (
                    -2.0
                    * w_source_external
                    / total_weight
                )

                old_null = (
                    source_volume
                    * source_volume
                )

                new_source_volume = (
                    source_volume
                    - ku
                )

                new_null = (
                    new_source_volume
                    * new_source_volume
                    + ku * ku
                )

                delta = (
                    delta_internal
                    - resolution
                    * (
                        new_null
                        - old_null
                    )
                    / total_sq
                )

                if delta > best_delta:
                    best_delta = delta
                    best_dest = dest
                    best_empty = True

            if best_dest == source:
                continue

            if best_empty:
                if best_dest != free_head:
                    return (
                        labels,
                        np.nan,
                        -3,
                        -1,
                        -1,
                        -1,
                    )

                free_head = next_free[
                    best_dest
                ]

                next_free[
                    best_dest
                ] = -1

            counts[source] -= 1
            counts[best_dest] += 1

            volumes[source] -= ku
            volumes[best_dest] += ku

            labels[u] = best_dest

            if counts[source] == 0:
                next_free[source] = free_head
                free_head = source

            total_gain += best_delta
            total_moves += 1
            moved += 1

            if best_empty:
                empty_moves += 1

        passes_done = pass_index + 1

        if moved == 0:
            break

    return (
        labels,
        total_gain,
        0,
        passes_done,
        total_moves,
        empty_moves,
    )


def optimize(
    indptr,
    indices,
    data,
    initial_labels,
    resolution,
    max_passes=1000,
):
    indptr = np.ascontiguousarray(
        indptr,
        dtype=np.int32,
    )

    indices = np.ascontiguousarray(
        indices,
        dtype=np.int32,
    )

    data = np.ascontiguousarray(
        data,
        dtype=np.float32,
    )

    initial_labels = np.ascontiguousarray(
        initial_labels,
        dtype=np.int32,
    )

    (
        labels,
        gain,
        error,
        passes,
        moves,
        empty_moves,
    ) = _core(
        indptr,
        indices,
        data,
        initial_labels,
        float(resolution),
        int(max_passes),
    )

    if error == -1:
        raise ValueError(
            "Negative weights are not supported."
        )

    if error == -2:
        raise ValueError(
            "Labels must lie in [0,n)."
        )

    if error != 0:
        raise RuntimeError(
            f"Internal candidate error {error}."
        )

    return {
        "labels":
            labels,

        "increase":
            float(gain),

        "passes":
            int(passes),

        "moves":
            int(moves),

        "empty_community_moves":
            int(empty_moves),
    }
