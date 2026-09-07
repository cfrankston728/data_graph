
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=False, nogil=True)
def _insertion_sort_int32(
    values,
    n_values,
):
    for i in range(1, n_values):
        x = values[i]
        j = i - 1

        while (
            j >= 0
            and values[j] > x
        ):
            values[j + 1] = values[j]
            j -= 1

        values[j + 1] = x


@njit(cache=False, nogil=True)
def _optimize_empty_aware_newman_core(
    indptr,
    indices,
    data,
    initial_labels,
    resolution,
    max_passes,
):
    """
    Deterministic weighted undirected Newman-modularity local mover.

    Scope:
      - symmetric CSR
      - nonnegative weights
      - no self-loops
      - labels in [0, n)

    Candidate destinations:
      1. every currently occupied neighboring community;
      2. one currently empty community, if the source contains >1 node.

    The empty destination is the missing split operation established by R153.

    This intentionally changes only the move-set semantics. It retains
    sequential Gauss-Seidel updates and repeated full sweeps.
    """

    n = initial_labels.size

    labels = initial_labels.copy()

    node_strength = np.zeros(
        n,
        dtype=np.float64,
    )

    total_weight = 0.0

    max_degree = 0

    for u in range(n):
        degree = (
            indptr[u + 1]
            - indptr[u]
        )

        if degree > max_degree:
            max_degree = degree

        strength = 0.0

        for p in range(
            indptr[u],
            indptr[u + 1],
        ):
            v = indices[p]

            if v == u:
                # Candidate deliberately refuses to silently adopt
                # unqualified self-loop semantics.
                return (
                    labels,
                    np.nan,
                    -1,
                    -1,
                    -1,
                    -1,
                )

            w = float(data[p])

            if w < 0.0:
                return (
                    labels,
                    np.nan,
                    -2,
                    -1,
                    -1,
                    -1,
                )

            strength += w

        node_strength[u] = strength
        total_weight += strength

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
                -3,
                -1,
                -1,
                -1,
            )

        counts[c] += 1
        volumes[c] += node_strength[u]

    # Singly-linked pool of empty labels.
    #
    # Only the head is ever used as the "new community" candidate.
    # All empty labels are objective-equivalent.
    next_free = np.full(
        n,
        -1,
        dtype=np.int32,
    )

    free_head = -1

    # Reverse traversal makes the smallest initially empty label
    # become the head, giving deterministic labeling.
    for c in range(
        n - 1,
        -1,
        -1,
    ):
        if counts[c] == 0:
            next_free[c] = free_head
            free_head = c

    # Community-weight accumulator for neighboring candidates.
    seen = np.full(
        n,
        -1,
        dtype=np.int64,
    )

    weight_to = np.zeros(
        n,
        dtype=np.float64,
    )

    candidate_labels = np.empty(
        max_degree + 1,
        dtype=np.int32,
    )

    stamp = np.int64(0)

    total_increase = 0.0
    total_moves = 0
    total_empty_moves = 0

    passes_done = 0

    total_sq = (
        total_weight
        * total_weight
    )

    for pass_index in range(
        max_passes
    ):
        moves_this_pass = 0

        for u in range(n):
            source = labels[u]
            k_u = node_strength[u]

            stamp += 1

            n_candidates = 0

            for p in range(
                indptr[u],
                indptr[u + 1],
            ):
                v = indices[p]
                c = labels[v]
                w = float(data[p])

                if seen[c] != stamp:
                    seen[c] = stamp
                    weight_to[c] = w

                    candidate_labels[
                        n_candidates
                    ] = c

                    n_candidates += 1

                else:
                    weight_to[c] += w

            # Preserve the old deterministic sorted-community
            # tie surface rather than neighbor encounter order.
            _insertion_sort_int32(
                candidate_labels,
                n_candidates,
            )

            if seen[source] == stamp:
                w_source = weight_to[source]
            else:
                w_source = 0.0

            vol_source = volumes[source]

            best_dest = source
            best_delta = 0.0
            best_is_empty = False

            # Existing neighboring communities.
            for j in range(
                n_candidates
            ):
                dest = candidate_labels[j]

                if dest == source:
                    continue

                w_dest = weight_to[dest]
                vol_dest = volumes[dest]

                # Exact change in:
                #
                # internal/T
                # - gamma * sum(volume_c/T)^2
                #
                # for symmetric no-self-loop CSR.
                delta_internal = (
                    2.0
                    * (
                        w_dest
                        - w_source
                    )
                    / total_weight
                )

                old_null = (
                    vol_source * vol_source
                    + vol_dest * vol_dest
                )

                new_source = (
                    vol_source
                    - k_u
                )

                new_dest = (
                    vol_dest
                    + k_u
                )

                new_null = (
                    new_source * new_source
                    + new_dest * new_dest
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

                # Strict improvement, matching the previous
                # deterministic local-move contract.
                if delta > best_delta:
                    best_delta = delta
                    best_dest = dest
                    best_is_empty = False

            # Leiden-relevant new/empty-community candidate.
            #
            # If source has one node, moving that singleton to another
            # empty label merely renames the community and is skipped.
            if (
                counts[source] > 1
                and free_head >= 0
            ):
                dest = free_head
                vol_dest = 0.0

                delta_internal = (
                    -2.0
                    * w_source
                    / total_weight
                )

                old_null = (
                    vol_source
                    * vol_source
                )

                new_source = (
                    vol_source
                    - k_u
                )

                new_null = (
                    new_source
                    * new_source
                    + k_u * k_u
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

                # Empty candidate is evaluated after sorted occupied
                # candidates and therefore wins only by strict gain.
                if delta > best_delta:
                    best_delta = delta
                    best_dest = dest
                    best_is_empty = True

            if best_dest == source:
                continue

            if best_is_empty:
                # Consume the currently selected empty label.
                if best_dest != free_head:
                    return (
                        labels,
                        np.nan,
                        -4,
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

            volumes[source] -= k_u
            volumes[best_dest] += k_u

            labels[u] = best_dest

            # Source may have become a newly available empty label.
            if counts[source] == 0:
                next_free[source] = free_head
                free_head = source

            total_increase += best_delta
            total_moves += 1
            moves_this_pass += 1

            if best_is_empty:
                total_empty_moves += 1

        passes_done = (
            pass_index + 1
        )

        if moves_this_pass == 0:
            break

    return (
        labels,
        total_increase,
        0,
        passes_done,
        total_moves,
        total_empty_moves,
    )


def optimize_empty_aware_newman(
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

    if (
        indptr.ndim != 1
        or indices.ndim != 1
        or data.ndim != 1
        or initial_labels.ndim != 1
    ):
        raise ValueError(
            "All inputs must be one-dimensional."
        )

    n = initial_labels.size

    if indptr.size != n + 1:
        raise ValueError(
            "indptr length must equal n+1."
        )

    if indices.size != data.size:
        raise ValueError(
            "indices/data size mismatch."
        )

    (
        labels,
        increase,
        error_code,
        passes,
        moves,
        empty_moves,
    ) = _optimize_empty_aware_newman_core(
        indptr,
        indices,
        data,
        initial_labels,
        float(resolution),
        int(max_passes),
    )

    if error_code == -1:
        raise ValueError(
            "Self-loops are not qualified by this candidate."
        )

    if error_code == -2:
        raise ValueError(
            "Negative weights are not supported."
        )

    if error_code == -3:
        raise ValueError(
            "Initial labels must lie in [0,n)."
        )

    if error_code != 0:
        raise RuntimeError(
            f"Internal mover error code {error_code}."
        )

    return {
        "labels":
            labels,

        "increase":
            float(increase),

        "passes":
            int(passes),

        "moves":
            int(moves),

        "empty_community_moves":
            int(empty_moves),
    }
