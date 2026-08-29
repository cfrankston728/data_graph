from __future__ import annotations

# Neutral first-coarsening core candidate.
# Generated from exact reviewed source artifacts in DG-OPT-086.

from typing import Iterable, Sequence, Tuple
import numpy as np

import numba as nb

def mutual_nn_coarsening_directed(sources, targets, weights, n_nodes):
    """Assign nodes to reciprocal-best-neighbor meta-nodes for coarsening."""
    best_neighbor = np.full(n_nodes, -1, dtype=np.int64)
    best_weight   = np.full(n_nodes, -np.inf)
    for e in range(sources.shape[0]):
        u = sources[e]; v = targets[e]; w = weights[e]
        if w > best_weight[u]:
            best_weight[u] = w; best_neighbor[u] = v
        if w > best_weight[v]:
            best_weight[v] = w; best_neighbor[v] = u

    meta_id = np.full(n_nodes, -1, dtype=np.int64)
    next_id = 0
    for i in range(n_nodes):
        j = best_neighbor[i]
        if j > i and j >= 0 and best_neighbor[j] == i and meta_id[i] == -1:
            meta_id[i] = next_id; meta_id[j] = next_id; next_id += 1
    for i in range(n_nodes):
        if meta_id[i] == -1:
            meta_id[i] = next_id; next_id += 1
    return meta_id, next_id


def aggregate_undirected_edges_with_dist_chunked(s, t, w, d, chunk=8_000_000):
    """Memory-safe aggregation to UNIQUE undirected pairs a<b.
    Weight = SUM of weights; distance = MIN of distances.
    Splits into chunks, sorts/dedups each chunk, concatenates,
    then final dedup via your numba dedup.
    """
    n = s.shape[0]
    if n == 0:
        return (s.astype(np.int64), t.astype(np.int64),
                w.astype(w.dtype), d.astype(d.dtype))

    outs_a, outs_b, outs_w, outs_d = [], [], [], []
    for start in range(0, n, chunk):
        end = min(start + chunk, n)

        a = s[start:end].astype(np.int64, copy=False)
        b = t[start:end].astype(np.int64, copy=False)
        ww = w[start:end].astype(w.dtype, copy=True)
        dd = d[start:end].astype(d.dtype, copy=True)

        # canonicalize to a<b
        mask = a > b
        if mask.any():
            a2 = a.copy(); b2 = b.copy()
            a2[mask] = b[mask]; b2[mask] = a[mask]
            a, b = a2, b2

        # sort by packed key
        keys  = (a << 32) | b
        order = np.argsort(keys)
        a = a[order]; b = b[order]; ww = ww[order]; dd = dd[order]

        # in-chunk combine (sum weights, min dist)
        ua = [a[0]]; ub = [b[0]]; uw = [ww[0]]; ud = [dd[0]]
        for i in range(1, a.size):
            if a[i] == ua[-1] and b[i] == ub[-1]:
                uw[-1] += ww[i]
                if dd[i] < ud[-1]:
                    ud[-1] = dd[i]
            else:
                ua.append(a[i]); ub.append(b[i]); uw.append(ww[i]); ud.append(dd[i])

        outs_a.append(np.asarray(ua, dtype=np.int64))
        outs_b.append(np.asarray(ub, dtype=np.int64))
        outs_w.append(np.asarray(uw, dtype=ww.dtype))
        outs_d.append(np.asarray(ud, dtype=dd.dtype))

    # concatenate chunk results and final dedup (maxw/mind == sum/mind works when each (a,b) appears once per chunk)
    A = np.concatenate(outs_a); B = np.concatenate(outs_b)
    W = np.concatenate(outs_w); D = np.concatenate(outs_d)
    # final dedup: we want SUM weights, MIN distance. Use your existing combiner semantics.
    # dedup_undirected_maxw_mind does MAX for weights; we want SUM → do a tiny local pass:
    keys = (A << 32) | B
    order = np.argsort(keys)
    A = A[order]; B = B[order]; W = W[order]; D = D[order]

    out_a = [A[0]]; out_b = [B[0]]; out_w = [W[0]]; out_d = [D[0]]
    for i in range(1, A.size):
        if A[i] == out_a[-1] and B[i] == out_b[-1]:
            out_w[-1] += W[i]
            if D[i] < out_d[-1]:
                out_d[-1] = D[i]
        else:
            out_a.append(A[i]); out_b.append(B[i]); out_w.append(W[i]); out_d.append(D[i])

    return (np.asarray(out_a, dtype=np.int64),
            np.asarray(out_b, dtype=np.int64),
            np.asarray(out_w, dtype=W.dtype),
            np.asarray(out_d, dtype=D.dtype))


def dedup_undirected_maxw_mind(sel_u, sel_v, sel_w, sel_d):
    """
    Deduplicate a set of *undirected* selections (possibly both u->v and v->u).
    Returns unique a<b with:
      - weight = MAX of weights from either side
      - distance = MIN of distances from either side
    """
    n = sel_u.shape[0]
    a = np.empty(n, dtype=np.int64)
    b = np.empty(n, dtype=np.int64)
    for i in range(n):
        u = sel_u[i]; v = sel_v[i]
        if u < v:
            a[i] = u; b[i] = v
        else:
            a[i] = v; b[i] = u

    keys = (a.astype(np.int64) << 32) | b.astype(np.int64)
    order = np.argsort(keys)
    a = a[order]; b = b[order]
    w = sel_w[order]; d = sel_d[order]

    out_a = np.empty(n, dtype=np.int64)
    out_b = np.empty(n, dtype=np.int64)
    out_w = np.empty(n, dtype=w.dtype)
    out_d = np.empty(n, dtype=d.dtype)

    out = 0
    i = 0
    while i < n:
        ua = a[i]; ub = b[i]
        maxw = w[i]
        mind = d[i]
        i += 1
        while i < n and a[i] == ua and b[i] == ub:
            if w[i] > maxw:
                maxw = w[i]
            if d[i] < mind:
                mind = d[i]
            i += 1
        out_a[out] = ua
        out_b[out] = ub
        out_w[out] = maxw
        out_d[out] = mind
        out += 1

    return out_a[:out], out_b[:out], out_w[:out], out_d[:out]


def sparsify_knn_undirected(a, b, w, d, n_nodes, k):
    """
    Top-k per node on an undirected graph given unique pairs (a<b).
    Returns unique pairs (a'<b') chosen by OR of endpoint selections.
    """
    m = a.shape[0]
    # degrees
    deg = np.zeros(n_nodes, dtype=np.int64)
    for i in range(m):
        deg[a[i]] += 1
        deg[b[i]] += 1

    # row pointers for local 2m storage
    ptr = np.empty(n_nodes + 1, dtype=np.int64)
    ptr[0] = 0
    for i in range(n_nodes):
        ptr[i+1] = ptr[i] + deg[i]
    total = ptr[-1]  # 2m

    nbr  = np.empty(total, dtype=np.int64)
    wts  = np.empty(total, dtype=np.float32)
    dst  = np.empty(total, dtype=np.float32)

    fill = ptr[:-1].copy()
    for i in range(m):
        u = a[i]; v = b[i]; wt = w[i]; di = d[i]
        pu = fill[u]; nbr[pu] = v; wts[pu] = wt; dst[pu] = di; fill[u] = pu + 1
        pv = fill[v]; nbr[pv] = u; wts[pv] = wt; dst[pv] = di; fill[v] = pv + 1

    # pre-count selections
    sel_count = 0
    for u in range(n_nodes):
        du = ptr[u+1] - ptr[u]
        if du > 0:
            sel_count += k if du > k else du

    cand_u = np.empty(sel_count, dtype=np.int64)
    cand_v = np.empty(sel_count, dtype=np.int64)
    cand_w = np.empty(sel_count, dtype=np.float32)
    cand_d = np.empty(sel_count, dtype=np.float32)

    out = 0
    for u in range(n_nodes):
        start = ptr[u]; end = ptr[u+1]; du = end - start
        if du == 0:
            continue

        if du <= k:
            for j in range(du):
                v = nbr[start + j]
                cand_u[out] = u; cand_v[out] = v
                cand_w[out] = wts[start + j]; cand_d[out] = dst[start + j]
                out += 1
        else:
            # simple O(du*k) selector; after coarsening du is modest
            tmp_w = wts[start:end].copy()
            tmp_i = np.empty(k, dtype=np.int64)
            for t in range(k):
                mi = 0; mw = tmp_w[0]
                for r in range(1, du):
                    if tmp_w[r] > mw:
                        mi = r; mw = tmp_w[r]
                tmp_i[t] = mi; tmp_w[mi] = np.float32(-1e38)
            for t in range(k):
                j = tmp_i[t]
                v = nbr[start + j]
                cand_u[out] = u; cand_v[out] = v
                cand_w[out] = wts[start + j]; cand_d[out] = dst[start + j]
                out += 1

    # OR-of-endpoints, dedup to unique (a<b)
    return dedup_undirected_maxw_mind(cand_u[:out], cand_v[:out], cand_w[:out], cand_d[:out])


class BoundedFirstCoarseningAccumulator:
    """
    Semantics-first prototype for amortizing analyzer level-0 selection.

    The accumulator stores at most ``k`` graph-distance neighbors per node.

    It does not alter or replace the canonical full-resolution graph.
    It records only derived state needed to seed the first coarsening prefix.

    Tie semantics
    -------------
    Candidate entries are ordered by:

        (graph_distance, target_node_id)

    For the generator's compact upper-triangular edge stream this reproduces
    the row-major directed ordering used by the analyzer in the verified
    zero-polish path.

    This implementation deliberately uses small Python lists per node so the
    behavioral contract is transparent. It is a semantics prototype, not the
    intended high-performance production implementation.
    """

    def __init__(self, n_nodes: int, k: int):
        self.n_nodes = int(n_nodes)
        self.k = int(k)

        if self.n_nodes < 0:
            raise ValueError("n_nodes must be nonnegative")

        if self.k < 1:
            raise ValueError("k must be >= 1")

        self._entries = [
            []
            for _ in range(
                self.n_nodes
            )
        ]

        self.n_valid_undirected_edges = 0
        self.n_skipped_edges = 0

    def _update_directed(
        self,
        source: int,
        target: int,
        distance: float,
    ) -> None:
        source = int(source)
        target = int(target)
        distance = float(distance)

        if source == target:
            return

        entries = self._entries[
            source
        ]

        for index, (
            old_distance,
            old_target,
        ) in enumerate(entries):
            if old_target != target:
                continue

            if distance >= old_distance:
                return

            del entries[index]
            break

        entries.append(
            (
                distance,
                target,
            )
        )

        entries.sort(
            key=lambda item: (
                item[0],
                item[1],
            )
        )

        if len(entries) > self.k:
            del entries[
                self.k:
            ]

    def add_undirected_edge(
        self,
        source: int,
        target: int,
        distance: float,
    ) -> None:
        source = int(source)
        target = int(target)
        distance = float(distance)

        self._update_directed(
            source,
            target,
            distance,
        )

        self._update_directed(
            target,
            source,
            distance,
        )

        self.n_valid_undirected_edges += 1

    def add_batch(
        self,
        sources: Sequence[int],
        targets: Sequence[int],
        distances: Sequence[float],
        *,
        missing_weight=None,
    ) -> None:
        if not (
            len(sources)
            == len(targets)
            == len(distances)
        ):
            raise ValueError(
                "sources, targets, distances must have equal length"
            )

        for source, target, distance in zip(
            sources,
            targets,
            distances,
        ):
            if (
                missing_weight is not None
                and distance == missing_weight
            ):
                self.n_skipped_edges += 1
                continue

            self.add_undirected_edge(
                source,
                target,
                distance,
            )

    def add_compact_arrays(
        self,
        edge_arr,
        distances,
        *,
        missing_weight=None,
    ) -> None:
        edge_arr = np.asarray(
            edge_arr,
            dtype=np.int64,
        )

        distances = np.asarray(
            distances,
        )

        if (
            edge_arr.ndim != 2
            or edge_arr.shape[1] != 2
        ):
            raise ValueError(
                "edge_arr must have shape (E, 2)"
            )

        self.add_batch(
            edge_arr[:, 0],
            edge_arr[:, 1],
            distances,
            missing_weight=missing_weight,
        )

    def materialize_level0(self):
        sources = []
        targets = []
        distances = []

        for node, entries in enumerate(
            self._entries
        ):
            for distance, target in entries:
                sources.append(node)
                targets.append(target)
                distances.append(distance)

        return (
            np.asarray(
                sources,
                dtype=np.int64,
            ),
            np.asarray(
                targets,
                dtype=np.int64,
            ),
            np.asarray(
                distances,
                dtype=np.float32,
            ),
        )

    @property
    def retained_directed_entries(self) -> int:
        return sum(
            len(entries)
            for entries in self._entries
        )

    @property
    def maximum_directed_entries(self) -> int:
        return self.n_nodes * self.k


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
