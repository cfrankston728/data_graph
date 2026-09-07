
#include <stdint.h>
#include <stdlib.h>
#include <math.h>


typedef struct {
    int32_t id;
    double delta;
} Candidate;


static int candidate_cmp(
    const void *a,
    const void *b
) {
    const Candidate *x =
        (const Candidate *)a;

    const Candidate *y =
        (const Candidate *)b;

    if (x->id < y->id) {
        return -1;
    }

    if (x->id > y->id) {
        return 1;
    }

    return 0;
}


static uint64_t rng_next(
    uint64_t *state
) {
    uint64_t x = *state;

    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;

    *state = x;

    return (
        x
        * UINT64_C(
            2685821657736338717
        )
    );
}


static double rng_uniform(
    uint64_t *state
) {
    return (
        (double)(
            rng_next(state)
            >> 11
        )
        * (
            1.0
            / 9007199254740992.0
        )
    );
}


int canonical_refine_newman_undirected(
    const int32_t *parent,
    const int32_t *indices,
    const int32_t *indptr,
    const float *data,
    const float *node_weights,
    int32_t n,
    float resolution,
    double theta,
    uint64_t seed,
    int32_t *refined,
    int64_t *stats_i64,
    double *stats_f64
) {
    double *parent_vol = NULL;
    double *node_cross = NULL;
    double *cluster_cross = NULL;
    double *cluster_vol = NULL;

    int32_t *cluster_size = NULL;
    int32_t *eligible = NULL;
    int32_t *touched = NULL;

    int64_t *seen = NULL;
    double *weight_to_cluster = NULL;

    Candidate *candidates = NULL;

    int32_t max_parent = 0;

    for (
        int32_t u = 0;
        u < n;
        ++u
    ) {
        if (
            parent[u]
            > max_parent
        ) {
            max_parent =
                parent[u];
        }
    }

    const int32_t n_parent =
        max_parent + 1;

    parent_vol = (double *)calloc(
        (size_t)n_parent,
        sizeof(double)
    );

    node_cross = (double *)calloc(
        (size_t)n,
        sizeof(double)
    );

    cluster_cross = (double *)calloc(
        (size_t)n,
        sizeof(double)
    );

    cluster_vol = (double *)calloc(
        (size_t)n,
        sizeof(double)
    );

    cluster_size = (int32_t *)calloc(
        (size_t)n,
        sizeof(int32_t)
    );

    eligible = (int32_t *)malloc(
        (size_t)n
        * sizeof(int32_t)
    );

    touched = (int32_t *)malloc(
        (size_t)n
        * sizeof(int32_t)
    );

    seen = (int64_t *)calloc(
        (size_t)n,
        sizeof(int64_t)
    );

    weight_to_cluster = (double *)calloc(
        (size_t)n,
        sizeof(double)
    );

    candidates = (Candidate *)malloc(
        (size_t)n
        * sizeof(Candidate)
    );

    if (
        parent_vol == NULL
        || node_cross == NULL
        || cluster_cross == NULL
        || cluster_vol == NULL
        || cluster_size == NULL
        || eligible == NULL
        || touched == NULL
        || seen == NULL
        || weight_to_cluster == NULL
        || candidates == NULL
    ) {
        free(parent_vol);
        free(node_cross);
        free(cluster_cross);
        free(cluster_vol);
        free(cluster_size);
        free(eligible);
        free(touched);
        free(seen);
        free(weight_to_cluster);
        free(candidates);

        return -1;
    }

    if (
        theta <= 0.0
    ) {
        free(parent_vol);
        free(node_cross);
        free(cluster_cross);
        free(cluster_vol);
        free(cluster_size);
        free(eligible);
        free(touched);
        free(seen);
        free(weight_to_cluster);
        free(candidates);

        return -2;
    }

    if (
        seed == 0
    ) {
        seed =
            UINT64_C(
                0x9E3779B97F4A7C15
            );
    }

    /*
     * Initial singleton refined partition.
     */
    for (
        int32_t u = 0;
        u < n;
        ++u
    ) {
        refined[u] = u;

        cluster_size[u] = 1;

        cluster_vol[u] =
            (double)node_weights[u];

        parent_vol[
            parent[u]
        ] += (
            double
        )node_weights[u];
    }

    /*
     * Compute E({u}, Parent(u)-{u}) for every node.
     */
    for (
        int32_t u = 0;
        u < n;
        ++u
    ) {
        const int32_t pu =
            parent[u];

        double cross = 0.0;

        for (
            int32_t p = indptr[u];
            p < indptr[u + 1];
            ++p
        ) {
            const int32_t v =
                indices[p];

            if (
                v != u
                && parent[v] == pu
            ) {
                cross += (
                    double
                )data[p];
            }
        }

        node_cross[u] =
            cross;

        cluster_cross[u] =
            cross;
    }

    /*
     * R from Algorithm A.2:
     *
     * E(v,S-v)
     *   >= gamma * ||v|| * (||S||-||v||)
     */
    int32_t n_eligible = 0;

    for (
        int32_t u = 0;
        u < n;
        ++u
    ) {
        const double ku =
            (double)node_weights[u];

        const double threshold =
            (
                (double)resolution
                * ku
                * (
                    parent_vol[
                        parent[u]
                    ]
                    - ku
                )
            );

        if (
            node_cross[u]
            >= threshold
        ) {
            eligible[
                n_eligible
            ] = u;

            n_eligible += 1;
        }
    }

    /*
     * Visit R in random order.
     */
    for (
        int32_t i =
            n_eligible - 1;
        i > 0;
        --i
    ) {
        const uint64_t r =
            rng_next(
                &seed
            );

        const int32_t j =
            (int32_t)(
                r
                % (
                    (uint64_t)i
                    + 1
                )
            );

        const int32_t tmp =
            eligible[i];

        eligible[i] =
            eligible[j];

        eligible[j] =
            tmp;
    }

    int64_t stamp = 0;

    int64_t visited_singleton = 0;
    int64_t moves = 0;
    int64_t stays = 0;

    int64_t total_candidates = 0;
    int64_t max_candidates = 0;

    int64_t candidate_targets_noncurrent = 0;

    double q_gain_sum = 0.0;
    double min_move_gain = 0.0;
    double max_move_gain = 0.0;

    int have_move_gain = 0;

    for (
        int32_t idx = 0;
        idx < n_eligible;
        ++idx
    ) {
        const int32_t u =
            eligible[idx];

        const int32_t current =
            refined[u];

        /*
         * Canonical singleton-only eligibility.
         */
        if (
            cluster_size[
                current
            ] != 1
        ) {
            continue;
        }

        visited_singleton += 1;

        const int32_t pu =
            parent[u];

        const double ku =
            (double)node_weights[u];

        stamp += 1;

        int32_t n_touched = 0;

        /*
         * Accumulate E(u,C) by neighboring refined community C.
         * Since positive modularity gain requires positive adjacency
         * for gamma > 0, non-neighbor communities need not be
         * materialized.
         */
        for (
            int32_t p = indptr[u];
            p < indptr[u + 1];
            ++p
        ) {
            const int32_t v =
                indices[p];

            if (
                v == u
                || parent[v] != pu
            ) {
                continue;
            }

            const int32_t target =
                refined[v];

            if (
                seen[target]
                != stamp
            ) {
                seen[target] =
                    stamp;

                touched[
                    n_touched
                ] = target;

                n_touched += 1;
            }

            weight_to_cluster[
                target
            ] += (
                double
            )data[p];
        }

        int32_t n_candidates = 0;

        /*
         * Staying singleton is explicitly permitted with ΔQ = 0.
         */
        candidates[
            n_candidates
        ].id = current;

        candidates[
            n_candidates
        ].delta = 0.0;

        n_candidates += 1;

        for (
            int32_t t = 0;
            t < n_touched;
            ++t
        ) {
            const int32_t target =
                touched[t];

            if (
                target == current
                || cluster_size[
                    target
                ] <= 0
            ) {
                continue;
            }

            const double target_vol =
                cluster_vol[
                    target
                ];

            /*
             * T from Algorithm A.2:
             *
             * E(C,S-C)
             *   >= gamma * ||C|| * (||S||-||C||)
             */
            const double target_threshold =
                (
                    (double)resolution
                    * target_vol
                    * (
                        parent_vol[pu]
                        - target_vol
                    )
                );

            if (
                cluster_cross[
                    target
                ]
                < target_threshold
            ) {
                continue;
            }

            /*
             * Exact undirected Newman modularity merge gain
             * on scikit-network's sum-1 normalized adjacency:
             *
             * ΔQ =
             *   2 * [
             *     E(u,target)
             *     - gamma * k_u * k_target
             *   ]
             */
            const double delta =
                2.0
                * (
                    weight_to_cluster[
                        target
                    ]
                    - (
                        (double)resolution
                        * ku
                        * target_vol
                    )
                );

            if (
                delta >= 0.0
            ) {
                candidates[
                    n_candidates
                ].id = target;

                candidates[
                    n_candidates
                ].delta = delta;

                n_candidates += 1;

                candidate_targets_noncurrent += 1;
            }
        }

        /*
         * Explicit deterministic ordering of the finite candidate
         * set. This does not alter canonical probabilities; it only
         * makes a seeded implementation reproducible.
         */
        qsort(
            candidates,
            (size_t)n_candidates,
            sizeof(Candidate),
            candidate_cmp
        );

        total_candidates +=
            n_candidates;

        if (
            n_candidates
            > max_candidates
        ) {
            max_candidates =
                n_candidates;
        }

        double max_delta =
            candidates[0].delta;

        for (
            int32_t c = 1;
            c < n_candidates;
            ++c
        ) {
            if (
                candidates[c].delta
                > max_delta
            ) {
                max_delta =
                    candidates[c].delta;
            }
        }

        double total_soft_weight =
            0.0;

        /*
         * Reuse weight_to_cluster[candidate.id] is NOT safe for
         * softmax weights, so candidate deltas are exponentiated
         * twice: once for total and once for selection.
         */
        for (
            int32_t c = 0;
            c < n_candidates;
            ++c
        ) {
            total_soft_weight +=
                exp(
                    (
                        candidates[c].delta
                        - max_delta
                    )
                    / theta
                );
        }

        const double draw =
            rng_uniform(
                &seed
            )
            * total_soft_weight;

        double cumulative = 0.0;

        int32_t selected =
            n_candidates - 1;

        for (
            int32_t c = 0;
            c < n_candidates;
            ++c
        ) {
            cumulative +=
                exp(
                    (
                        candidates[c].delta
                        - max_delta
                    )
                    / theta
                );

            if (
                draw < cumulative
            ) {
                selected = c;
                break;
            }
        }

        const int32_t target =
            candidates[
                selected
            ].id;

        const double selected_delta =
            candidates[
                selected
            ].delta;

        if (
            target == current
        ) {
            stays += 1;
        }

        else {
            const double weight_u_target =
                weight_to_cluster[
                    target
                ];

            refined[u] =
                target;

            cluster_size[
                current
            ] = 0;

            cluster_size[
                target
            ] += 1;

            cluster_vol[
                current
            ] = 0.0;

            cluster_vol[
                target
            ] += ku;

            cluster_cross[
                target
            ] = (
                cluster_cross[
                    target
                ]
                + node_cross[u]
                - 2.0
                * weight_u_target
            );

            cluster_cross[
                current
            ] = 0.0;

            moves += 1;

            q_gain_sum +=
                selected_delta;

            if (
                !have_move_gain
            ) {
                min_move_gain =
                    selected_delta;

                max_move_gain =
                    selected_delta;

                have_move_gain = 1;
            }

            else {
                if (
                    selected_delta
                    < min_move_gain
                ) {
                    min_move_gain =
                        selected_delta;
                }

                if (
                    selected_delta
                    > max_move_gain
                ) {
                    max_move_gain =
                        selected_delta;
                }
            }
        }

        /*
         * Clear only touched scratch slots.
         */
        for (
            int32_t t = 0;
            t < n_touched;
            ++t
        ) {
            weight_to_cluster[
                touched[t]
            ] = 0.0;
        }
    }

    stats_i64[0] =
        n_eligible;

    stats_i64[1] =
        visited_singleton;

    stats_i64[2] =
        moves;

    stats_i64[3] =
        stays;

    stats_i64[4] =
        total_candidates;

    stats_i64[5] =
        max_candidates;

    stats_i64[6] =
        candidate_targets_noncurrent;

    stats_i64[7] =
        n
        - moves;

    stats_f64[0] =
        q_gain_sum;

    stats_f64[1] =
        have_move_gain
        ? min_move_gain
        : 0.0;

    stats_f64[2] =
        have_move_gain
        ? max_move_gain
        : 0.0;

    free(parent_vol);
    free(node_cross);
    free(cluster_cross);
    free(cluster_vol);
    free(cluster_size);
    free(eligible);
    free(touched);
    free(seen);
    free(weight_to_cluster);
    free(candidates);

    return 0;
}
