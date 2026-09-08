#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cmath>
#include <limits>
#include <vector>

namespace {

constexpr int32_t MODE_FINE = 0;
constexpr int32_t MODE_COARSE = 1;

template <typename WeightT>
int run_stateful_full_convergence(
    int32_t n,
    const int32_t* indptr,
    const int32_t* indices,
    const WeightT* data,
    const int32_t* initial_labels,
    double resolution,
    int32_t mode,
    int32_t max_passes,
    int32_t* out_labels,
    int32_t* out_counts,
    double* out_volumes,
    int32_t* out_next_free,
    int32_t* out_free_head,
    double* out_increase,
    int32_t* out_passes,
    int64_t* out_moves,
    int64_t* out_empty_moves
) {
    if (
        n < 0
        || indptr == nullptr
        || indices == nullptr
        || data == nullptr
        || initial_labels == nullptr
        || out_labels == nullptr
        || out_counts == nullptr
        || out_volumes == nullptr
        || out_next_free == nullptr
        || out_free_head == nullptr
        || out_increase == nullptr
        || out_passes == nullptr
        || out_moves == nullptr
        || out_empty_moves == nullptr
    ) {
        return -100;
    }

    if (mode != MODE_FINE && mode != MODE_COARSE) {
        return -101;
    }

    if (max_passes < 0) {
        return -102;
    }

    if (!std::isfinite(resolution)) {
        return -103;
    }

    if (n == 0) {
        *out_free_head = -1;
        *out_increase = 0.0;
        *out_passes = 0;
        *out_moves = 0;
        *out_empty_moves = 0;
        return 0;
    }

    std::memcpy(
        out_labels,
        initial_labels,
        static_cast<std::size_t>(n) * sizeof(int32_t)
    );

    std::vector<double> strength(
        static_cast<std::size_t>(n),
        0.0
    );

    std::vector<double> self_weight;

    if (mode == MODE_COARSE) {
        self_weight.assign(
            static_cast<std::size_t>(n),
            0.0
        );
    }

    int32_t max_degree = 0;
    double total_weight = 0.0;

    for (int32_t u = 0; u < n; ++u) {
        const int32_t start = indptr[u];
        const int32_t stop = indptr[u + 1];

        if (start < 0 || stop < start) {
            return -104;
        }

        const int32_t degree = stop - start;

        if (degree > max_degree) {
            max_degree = degree;
        }

        double ku = 0.0;
        double loop = 0.0;

        for (int32_t p = start; p < stop; ++p) {
            const int32_t v = indices[p];

            if (v < 0 || v >= n) {
                return -105;
            }

            if (mode == MODE_FINE && v == u) {
                return -1;
            }

            const double w =
                static_cast<double>(data[p]);

            if (!std::isfinite(w) || w < 0.0) {
                return -2;
            }

            ku += w;

            if (mode == MODE_COARSE && v == u) {
                loop += w;
            }
        }

        strength[
            static_cast<std::size_t>(u)
        ] = ku;

        if (mode == MODE_COARSE) {
            self_weight[
                static_cast<std::size_t>(u)
            ] = loop;
        }

        total_weight += ku;
    }

    std::vector<int32_t> counts(
        static_cast<std::size_t>(n),
        0
    );

    std::vector<double> volumes(
        static_cast<std::size_t>(n),
        0.0
    );

    for (int32_t u = 0; u < n; ++u) {
        const int32_t c = out_labels[u];

        if (c < 0 || c >= n) {
            return -3;
        }

        counts[
            static_cast<std::size_t>(c)
        ] += 1;

        volumes[
            static_cast<std::size_t>(c)
        ] += strength[
            static_cast<std::size_t>(u)
        ];
    }

    // Persistent singly-linked pool of empty labels.
    std::vector<int32_t> next_free(
        static_cast<std::size_t>(n),
        -1
    );

    int32_t free_head = -1;

    for (
        int32_t c = n - 1;
        c >= 0;
        --c
    ) {
        if (
            counts[
                static_cast<std::size_t>(c)
            ] == 0
        ) {
            next_free[
                static_cast<std::size_t>(c)
            ] = free_head;

            free_head = c;
        }
    }

    // Preserve state even for a zero-weight graph.
    if (total_weight <= 0.0) {
        std::memcpy(
            out_counts,
            counts.data(),
            static_cast<std::size_t>(n) * sizeof(int32_t)
        );

        std::memcpy(
            out_volumes,
            volumes.data(),
            static_cast<std::size_t>(n) * sizeof(double)
        );

        std::memcpy(
            out_next_free,
            next_free.data(),
            static_cast<std::size_t>(n) * sizeof(int32_t)
        );

        *out_free_head = free_head;
        *out_increase = 0.0;
        *out_passes = 0;
        *out_moves = 0;
        *out_empty_moves = 0;
        return 0;
    }

    std::vector<int64_t> seen(
        static_cast<std::size_t>(n),
        -1
    );

    std::vector<double> weight_to(
        static_cast<std::size_t>(n),
        0.0
    );

    std::vector<int32_t> candidates(
        static_cast<std::size_t>(max_degree + 1),
        0
    );

    int64_t stamp = 0;

    const double total_sq =
        total_weight * total_weight;

    double total_increase = 0.0;
    int64_t total_moves = 0;
    int64_t total_empty_moves = 0;
    int32_t passes_done = 0;

    for (
        int32_t pass_index = 0;
        pass_index < max_passes;
        ++pass_index
    ) {
        int64_t moves_this_pass = 0;

        for (int32_t u = 0; u < n; ++u) {
            const int32_t source =
                out_labels[u];

            const double ku =
                strength[
                    static_cast<std::size_t>(u)
                ];

            const double loop_u =
                (
                    mode == MODE_COARSE
                    ? self_weight[
                        static_cast<std::size_t>(u)
                    ]
                    : 0.0
                );

            // Defensive stamp wrap handling. In practical workloads
            // this branch is unreachable, but preserves correctness.
            if (
                stamp
                == std::numeric_limits<int64_t>::max()
            ) {
                for (
                    std::size_t i = 0;
                    i < seen.size();
                    ++i
                ) {
                    seen[i] = -1;
                }

                stamp = 0;
            }

            ++stamp;

            int32_t n_candidates = 0;

            for (
                int32_t p = indptr[u];
                p < indptr[u + 1];
                ++p
            ) {
                const int32_t v = indices[p];

                const int32_t c =
                    out_labels[v];

                const double w =
                    static_cast<double>(
                        data[p]
                    );

                if (
                    seen[
                        static_cast<std::size_t>(c)
                    ] != stamp
                ) {
                    seen[
                        static_cast<std::size_t>(c)
                    ] = stamp;

                    weight_to[
                        static_cast<std::size_t>(c)
                    ] = w;

                    candidates[
                        static_cast<std::size_t>(
                            n_candidates
                        )
                    ] = c;

                    ++n_candidates;
                } else {
                    weight_to[
                        static_cast<std::size_t>(c)
                    ] += w;
                }
            }

            // Exact insertion-sort tie surface used by the
            // qualified Python and R265 implementations.
            for (
                int32_t i = 1;
                i < n_candidates;
                ++i
            ) {
                const int32_t x =
                    candidates[
                        static_cast<std::size_t>(i)
                    ];

                int32_t j = i - 1;

                while (
                    j >= 0
                    && candidates[
                        static_cast<std::size_t>(j)
                    ] > x
                ) {
                    candidates[
                        static_cast<std::size_t>(j + 1)
                    ] =
                        candidates[
                            static_cast<std::size_t>(j)
                        ];

                    --j;
                }

                candidates[
                    static_cast<std::size_t>(j + 1)
                ] = x;
            }

            double w_source_all = 0.0;

            if (
                seen[
                    static_cast<std::size_t>(source)
                ] == stamp
            ) {
                w_source_all =
                    weight_to[
                        static_cast<std::size_t>(
                            source
                        )
                    ];
            }

            const double w_source_removable =
                (
                    mode == MODE_COARSE
                    ? w_source_all - loop_u
                    : w_source_all
                );

            const double source_volume =
                volumes[
                    static_cast<std::size_t>(
                        source
                    )
                ];

            int32_t best_dest = source;
            double best_delta = 0.0;
            bool best_is_empty = false;

            // Existing neighboring communities.
            for (
                int32_t j = 0;
                j < n_candidates;
                ++j
            ) {
                const int32_t dest =
                    candidates[
                        static_cast<std::size_t>(j)
                    ];

                if (dest == source) {
                    continue;
                }

                const double w_dest =
                    weight_to[
                        static_cast<std::size_t>(
                            dest
                        )
                    ];

                const double dest_volume =
                    volumes[
                        static_cast<std::size_t>(
                            dest
                        )
                    ];

                const double delta_internal =
                    2.0
                    * (
                        w_dest
                        - w_source_removable
                    )
                    / total_weight;

                const double old_null =
                    source_volume * source_volume
                    + dest_volume * dest_volume;

                const double new_source_volume =
                    source_volume - ku;

                const double new_dest_volume =
                    dest_volume + ku;

                const double new_null =
                    new_source_volume
                    * new_source_volume
                    + new_dest_volume
                    * new_dest_volume;

                const double delta =
                    delta_internal
                    - resolution
                    * (
                        new_null
                        - old_null
                    )
                    / total_sq;

                if (delta > best_delta) {
                    best_delta = delta;
                    best_dest = dest;
                    best_is_empty = false;
                }
            }

            // One currently empty label, selected from the
            // persistent free-list head.
            if (
                counts[
                    static_cast<std::size_t>(
                        source
                    )
                ] > 1
                && free_head >= 0
            ) {
                const int32_t dest =
                    free_head;

                const double delta_internal =
                    -2.0
                    * w_source_removable
                    / total_weight;

                const double old_null =
                    source_volume
                    * source_volume;

                const double new_source_volume =
                    source_volume - ku;

                const double new_null =
                    new_source_volume
                    * new_source_volume
                    + ku * ku;

                const double delta =
                    delta_internal
                    - resolution
                    * (
                        new_null
                        - old_null
                    )
                    / total_sq;

                if (delta > best_delta) {
                    best_delta = delta;
                    best_dest = dest;
                    best_is_empty = true;
                }
            }

            if (best_dest == source) {
                continue;
            }

            if (best_is_empty) {
                if (best_dest != free_head) {
                    return -4;
                }

                free_head =
                    next_free[
                        static_cast<std::size_t>(
                            best_dest
                        )
                    ];

                next_free[
                    static_cast<std::size_t>(
                        best_dest
                    )
                ] = -1;
            }

            counts[
                static_cast<std::size_t>(
                    source
                )
            ] -= 1;

            counts[
                static_cast<std::size_t>(
                    best_dest
                )
            ] += 1;

            volumes[
                static_cast<std::size_t>(
                    source
                )
            ] -= ku;

            volumes[
                static_cast<std::size_t>(
                    best_dest
                )
            ] += ku;

            out_labels[u] =
                best_dest;

            if (
                counts[
                    static_cast<std::size_t>(
                        source
                    )
                ] == 0
            ) {
                next_free[
                    static_cast<std::size_t>(
                        source
                    )
                ] = free_head;

                free_head = source;
            }

            total_increase +=
                best_delta;

            ++total_moves;
            ++moves_this_pass;

            if (best_is_empty) {
                ++total_empty_moves;
            }
        }

        passes_done =
            pass_index + 1;

        if (moves_this_pass == 0) {
            break;
        }
    }

    std::memcpy(
        out_counts,
        counts.data(),
        static_cast<std::size_t>(n) * sizeof(int32_t)
    );

    std::memcpy(
        out_volumes,
        volumes.data(),
        static_cast<std::size_t>(n) * sizeof(double)
    );

    std::memcpy(
        out_next_free,
        next_free.data(),
        static_cast<std::size_t>(n) * sizeof(int32_t)
    );

    *out_free_head =
        free_head;

    *out_increase =
        total_increase;

    *out_passes =
        passes_done;

    *out_moves =
        total_moves;

    *out_empty_moves =
        total_empty_moves;

    return 0;
}

}  // namespace


extern "C" int stateful_newman_full_convergence_f32(
    int32_t n,
    const int32_t* indptr,
    const int32_t* indices,
    const float* data,
    const int32_t* initial_labels,
    double resolution,
    int32_t mode,
    int32_t max_passes,
    int32_t* out_labels,
    int32_t* out_counts,
    double* out_volumes,
    int32_t* out_next_free,
    int32_t* out_free_head,
    double* out_increase,
    int32_t* out_passes,
    int64_t* out_moves,
    int64_t* out_empty_moves
) {
    return run_stateful_full_convergence<float>(
        n,
        indptr,
        indices,
        data,
        initial_labels,
        resolution,
        mode,
        max_passes,
        out_labels,
        out_counts,
        out_volumes,
        out_next_free,
        out_free_head,
        out_increase,
        out_passes,
        out_moves,
        out_empty_moves
    );
}


extern "C" int stateful_newman_full_convergence_f64(
    int32_t n,
    const int32_t* indptr,
    const int32_t* indices,
    const double* data,
    const int32_t* initial_labels,
    double resolution,
    int32_t mode,
    int32_t max_passes,
    int32_t* out_labels,
    int32_t* out_counts,
    double* out_volumes,
    int32_t* out_next_free,
    int32_t* out_free_head,
    double* out_increase,
    int32_t* out_passes,
    int64_t* out_moves,
    int64_t* out_empty_moves
) {
    return run_stateful_full_convergence<double>(
        n,
        indptr,
        indices,
        data,
        initial_labels,
        resolution,
        mode,
        max_passes,
        out_labels,
        out_counts,
        out_volumes,
        out_next_free,
        out_free_head,
        out_increase,
        out_passes,
        out_moves,
        out_empty_moves
    );
}
