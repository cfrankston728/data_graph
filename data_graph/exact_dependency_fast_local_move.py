
import numpy as np
from numba import njit


@njit(cache=False, nogil=True)
def _insertion_sort_int32(values, n_values):
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

    node_strength = np.zeros(n, dtype=np.float64)

    total_weight = 0.0
    max_degree = 0

    for u in range(n):
        degree = indptr[u + 1] - indptr[u]

        if degree > max_degree:
            max_degree = degree

        strength = 0.0

        for p in range(indptr[u], indptr[u + 1]):
            v = indices[p]

            if v == u:
                return (
                    labels, np.nan, -1, 0, 0, 0,
                    0, 0, 0,
                    np.zeros(max_passes, dtype=np.int64),
                    np.zeros(max_passes, dtype=np.int64),
                    np.zeros(max_passes, dtype=np.int64),
                )

            w = float(data[p])

            if w < 0.0:
                return (
                    labels, np.nan, -2, 0, 0, 0,
                    0, 0, 0,
                    np.zeros(max_passes, dtype=np.int64),
                    np.zeros(max_passes, dtype=np.int64),
                    np.zeros(max_passes, dtype=np.int64),
                )

            strength += w

        node_strength[u] = strength
        total_weight += strength

    if total_weight <= 0.0:
        return (
            labels, 0.0, 0, 0, 0, 0,
            0, 0, 0,
            np.zeros(max_passes, dtype=np.int64),
            np.zeros(max_passes, dtype=np.int64),
            np.zeros(max_passes, dtype=np.int64),
        )

    # This first exact prototype uses a single uint64 dependency mask.
    # It therefore qualifies the <=64-label warm path only.
    for u in range(n):
        if labels[u] < 0 or labels[u] >= 64:
            return (
                labels, np.nan, -5, 0, 0, 0,
                0, 0, 0,
                np.zeros(max_passes, dtype=np.int64),
                np.zeros(max_passes, dtype=np.int64),
                np.zeros(max_passes, dtype=np.int64),
            )

    counts = np.zeros(n, dtype=np.int32)
    volumes = np.zeros(n, dtype=np.float64)

    for u in range(n):
        c = labels[u]
        counts[c] += 1
        volumes[c] += node_strength[u]

    next_free = np.full(n, -1, dtype=np.int32)
    free_head = -1
    free_count = 0

    for c in range(n - 1, -1, -1):
        if counts[c] == 0:
            next_free[c] = free_head
            free_head = c
            free_count += 1

    seen = np.full(n, -1, dtype=np.int64)
    weight_to = np.zeros(n, dtype=np.float64)

    candidate_labels = np.empty(
        max_degree + 1,
        dtype=np.int32,
    )

    # Exact dependency cache.
    #
    # node_last_event[u] is the global movement event counter at the
    # instant u was last fully evaluated and found stable.
    #
    # dependency_mask[u] contains source + all neighboring communities
    # used in that exact gain calculation.
    #
    # community_last_change[c] records the most recent event that changed
    # community c's volume/count. If no dependency changed afterward, the
    # prior no-move result is still mathematically valid.
    node_last_event = np.full(
        n,
        -1,
        dtype=np.int64,
    )

    dependency_mask = np.zeros(
        n,
        dtype=np.uint64,
    )

    community_last_change = np.zeros(
        64,
        dtype=np.int64,
    )

    free_availability_last_change = np.int64(0)
    event_counter = np.int64(0)

    stamp = np.int64(0)

    total_increase = 0.0
    total_moves = 0
    total_empty_moves = 0
    passes_done = 0

    total_evaluated = 0
    total_skipped = 0
    total_scanned_edges = 0

    evaluated_per_pass = np.zeros(
        max_passes,
        dtype=np.int64,
    )
    skipped_per_pass = np.zeros(
        max_passes,
        dtype=np.int64,
    )
    scanned_edges_per_pass = np.zeros(
        max_passes,
        dtype=np.int64,
    )

    total_sq = total_weight * total_weight

    for pass_index in range(max_passes):
        moves_this_pass = 0

        evaluated_this_pass = 0
        skipped_this_pass = 0
        scanned_this_pass = 0

        for u in range(n):
            can_skip = False

            last_event = node_last_event[u]

            if last_event >= 0:
                changed = False

                mask = dependency_mask[u]

                for c in range(64):
                    bit = np.uint64(1) << np.uint64(c)

                    if (mask & bit) != 0:
                        if community_last_change[c] > last_event:
                            changed = True
                            break

                if (
                    not changed
                    and free_availability_last_change <= last_event
                ):
                    can_skip = True

            if can_skip:
                skipped_this_pass += 1
                total_skipped += 1
                continue

            evaluated_this_pass += 1
            total_evaluated += 1

            source = labels[u]

            if source < 0 or source >= 64:
                return (
                    labels, np.nan, -5,
                    passes_done,
                    total_moves,
                    total_empty_moves,
                    total_evaluated,
                    total_skipped,
                    total_scanned_edges,
                    evaluated_per_pass,
                    skipped_per_pass,
                    scanned_edges_per_pass,
                )

            k_u = node_strength[u]

            stamp += 1
            n_candidates = 0

            dep_mask = (
                np.uint64(1)
                << np.uint64(source)
            )

            for p in range(indptr[u], indptr[u + 1]):
                scanned_this_pass += 1
                total_scanned_edges += 1

                v = indices[p]
                c = labels[v]

                if c < 0 or c >= 64:
                    return (
                        labels, np.nan, -5,
                        passes_done,
                        total_moves,
                        total_empty_moves,
                        total_evaluated,
                        total_skipped,
                        total_scanned_edges,
                        evaluated_per_pass,
                        skipped_per_pass,
                        scanned_edges_per_pass,
                    )

                dep_mask |= (
                    np.uint64(1)
                    << np.uint64(c)
                )

                w = float(data[p])

                if seen[c] != stamp:
                    seen[c] = stamp
                    weight_to[c] = w
                    candidate_labels[n_candidates] = c
                    n_candidates += 1
                else:
                    weight_to[c] += w

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

            for j in range(n_candidates):
                dest = candidate_labels[j]

                if dest == source:
                    continue

                w_dest = weight_to[dest]
                vol_dest = volumes[dest]

                delta_internal = (
                    2.0
                    * (w_dest - w_source)
                    / total_weight
                )

                old_null = (
                    vol_source * vol_source
                    + vol_dest * vol_dest
                )

                new_source = vol_source - k_u
                new_dest = vol_dest + k_u

                new_null = (
                    new_source * new_source
                    + new_dest * new_dest
                )

                delta = (
                    delta_internal
                    - resolution
                    * (new_null - old_null)
                    / total_sq
                )

                if delta > best_delta:
                    best_delta = delta
                    best_dest = dest
                    best_is_empty = False

            if counts[source] > 1 and free_head >= 0:
                dest = free_head

                # If an empty move would create a label outside the
                # 64-label exact fast path, stop rather than silently
                # changing semantics.
                if dest >= 64:
                    # We only need this check if the empty move wins;
                    # compute its gain first.
                    pass

                delta_internal = (
                    -2.0 * w_source / total_weight
                )

                old_null = vol_source * vol_source

                new_source = vol_source - k_u

                new_null = (
                    new_source * new_source
                    + k_u * k_u
                )

                delta = (
                    delta_internal
                    - resolution
                    * (new_null - old_null)
                    / total_sq
                )

                if delta > best_delta:
                    if dest >= 64:
                        return (
                            labels, np.nan, -6,
                            passes_done,
                            total_moves,
                            total_empty_moves,
                            total_evaluated,
                            total_skipped,
                            total_scanned_edges,
                            evaluated_per_pass,
                            skipped_per_pass,
                            scanned_edges_per_pass,
                        )

                    best_delta = delta
                    best_dest = dest
                    best_is_empty = True

            if best_dest == source:
                # Cache only a fully evaluated NO-MOVE result.
                dependency_mask[u] = dep_mask
                node_last_event[u] = event_counter
                continue

            old_free_available = free_count > 0

            if best_is_empty:
                if best_dest != free_head:
                    return (
                        labels, np.nan, -4,
                        passes_done,
                        total_moves,
                        total_empty_moves,
                        total_evaluated,
                        total_skipped,
                        total_scanned_edges,
                        evaluated_per_pass,
                        skipped_per_pass,
                        scanned_edges_per_pass,
                    )

                free_head = next_free[best_dest]
                next_free[best_dest] = -1
                free_count -= 1

            counts[source] -= 1
            counts[best_dest] += 1

            volumes[source] -= k_u
            volumes[best_dest] += k_u

            labels[u] = best_dest

            if counts[source] == 0:
                next_free[source] = free_head
                free_head = source
                free_count += 1

            event_counter += 1

            community_last_change[source] = event_counter
            community_last_change[best_dest] = event_counter

            new_free_available = free_count > 0

            if new_free_available != old_free_available:
                free_availability_last_change = event_counter

            # A moved node has no stable cached no-move evaluation.
            node_last_event[u] = -1
            dependency_mask[u] = np.uint64(0)

            total_increase += best_delta
            total_moves += 1
            moves_this_pass += 1

            if best_is_empty:
                total_empty_moves += 1

        evaluated_per_pass[pass_index] = evaluated_this_pass
        skipped_per_pass[pass_index] = skipped_this_pass
        scanned_edges_per_pass[pass_index] = scanned_this_pass

        passes_done = pass_index + 1

        if moves_this_pass == 0:
            break

    return (
        labels,
        total_increase,
        0,
        passes_done,
        total_moves,
        total_empty_moves,
        total_evaluated,
        total_skipped,
        total_scanned_edges,
        evaluated_per_pass,
        skipped_per_pass,
        scanned_edges_per_pass,
    )


def optimize_exact_dependency_skip(
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
        increase,
        error_code,
        passes,
        moves,
        empty_moves,
        evaluated,
        skipped,
        scanned_edges,
        evaluated_per_pass,
        skipped_per_pass,
        scanned_edges_per_pass,
    ) = _core(
        indptr,
        indices,
        data,
        initial_labels,
        float(resolution),
        int(max_passes),
    )

    if error_code == -1:
        raise ValueError("self-loop encountered")
    if error_code == -2:
        raise ValueError("negative weight encountered")
    if error_code == -4:
        raise RuntimeError("empty-label pool invariant failed")
    if error_code == -5:
        raise RuntimeError(
            "dependency-mask fast path requires all active labels <64"
        )
    if error_code == -6:
        raise RuntimeError(
            "empty move would leave the <=64-label fast path"
        )
    if error_code != 0:
        raise RuntimeError(
            f"internal exact-skip error {error_code}"
        )

    return {
        "labels": labels,
        "increase": float(increase),
        "passes": int(passes),
        "moves": int(moves),
        "empty_community_moves": int(empty_moves),
        "evaluated_nodes": int(evaluated),
        "skipped_nodes": int(skipped),
        "scanned_edges": int(scanned_edges),
        "evaluated_per_pass":
            evaluated_per_pass[:passes].copy(),
        "skipped_per_pass":
            skipped_per_pass[:passes].copy(),
        "scanned_edges_per_pass":
            scanned_edges_per_pass[:passes].copy(),
    }
