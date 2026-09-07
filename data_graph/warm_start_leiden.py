
from __future__ import annotations

from typing import Optional, Union

import numpy as np
from scipy import sparse

from sknetwork.clustering import Leiden
from sknetwork.utils.membership import get_membership


class WarmStartLeiden(Leiden):
    """scikit-network Leiden with optional initial membership.

    When initial_membership is None, this follows the installed
    sknetwork.clustering.Leiden.fit algorithm exactly.

    When supplied, initial_membership replaces only Leiden's default
    singleton initialization. All optimization, refinement,
    aggregation, preprocessing, and postprocessing remain the
    scikit-network implementation.
    """

    def fit(
        self,
        input_matrix: Union[sparse.csr_matrix, np.ndarray],
        force_bipartite: bool = False,
        initial_membership: Optional[np.ndarray] = None,
    ) -> "WarmStartLeiden":

        (
            adjacency,
            out_weights,
            in_weights,
            membership,
            index,
        ) = self._pre_processing(
            input_matrix,
            force_bipartite,
        )

        n = adjacency.shape[0]

        if initial_membership is None:
            labels = np.arange(n)

        else:
            labels = np.asarray(
                initial_membership
            )

            if labels.ndim != 1:
                raise ValueError(
                    "initial_membership must be one-dimensional"
                )

            if labels.shape[0] != n:
                raise ValueError(
                    "initial_membership length must match "
                    "the preprocessed adjacency node count: "
                    f"{labels.shape[0]} != {n}"
                )

            if not np.issubdtype(
                labels.dtype,
                np.integer,
            ):
                raise TypeError(
                    "initial_membership must contain integer labels"
                )

            if np.any(labels < 0):
                raise ValueError(
                    "initial_membership labels must be non-negative"
                )

            # Compact arbitrary non-negative labels while preserving
            # partition equivalence.
            _, labels = np.unique(
                labels,
                return_inverse=True,
            )

            labels = labels.astype(
                np.int32,
                copy=False,
            )

        count = 0
        stop = False

        while not stop:
            count += 1

            labels, increase = self._optimize(
                labels,
                adjacency,
                out_weights,
                in_weights,
            )

            _, labels = np.unique(
                labels,
                return_inverse=True,
            )

            labels_original = labels.copy()

            labels_refined = np.arange(
                len(labels)
            )

            labels_refined = self._optimize_refine(
                labels,
                labels_refined,
                adjacency,
                out_weights,
                in_weights,
            )

            _, labels_refined = np.unique(
                labels_refined,
                return_inverse=True,
            )

            (
                labels,
                adjacency,
                out_weights,
                in_weights,
            ) = self._aggregate_refine(
                labels,
                labels_refined,
                adjacency,
                out_weights,
                in_weights,
            )

            n = adjacency.shape[0]

            stop = n == 1
            stop |= increase <= self.tol_aggregation
            stop |= count == self.n_aggregations

            if stop:
                membership = membership.dot(
                    get_membership(
                        labels_original
                    )
                )

            else:
                membership = membership.dot(
                    get_membership(
                        labels_refined
                    )
                )

            self.print_log(
                "Aggregation:",
                count,
                " Clusters:",
                n,
                " Increase:",
                increase,
            )

        self._post_processing(
            input_matrix,
            membership,
            index,
        )

        return self
