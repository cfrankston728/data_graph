from __future__ import annotations

import numpy as np
from numba import njit


@njit
def fisher_rao_latent_gaussian_distance(
    features,
    i,
    j,
):
    """Historical diagonal-Gaussian Fisher-Rao distance used by Step09."""
    n_features = features.shape[1]

    if n_features % 2 != 0:
        raise ValueError(
            "latent Gaussian feature matrix must contain "
            "[mu_1..mu_d, sig_1..sig_d]"
        )

    n_dims = n_features // 2

    eps = 1e-8
    safe_eps = 1e-30

    means_i = (
        features[i, :n_dims]
        + 0.0
    )

    sigma_i = (
        features[
            i,
            n_dims:2 * n_dims,
        ]
        + eps
    )

    means_j = (
        features[j, :n_dims]
        + 0.0
    )

    sigma_j = (
        features[
            j,
            n_dims:2 * n_dims,
        ]
        + eps
    )

    var_term = (
        (
            sigma_i ** 2
            + sigma_j ** 2
        )
        / 2
        + eps
    )

    ratio = (
        var_term
        / (
            sigma_i
            * sigma_j
        )
    )

    log_term = np.log(
        np.maximum(
            ratio,
            1e-16,
        )
    )

    diff = (
        means_i
        - means_j
    )

    d2 = (
        2
        * np.sum(
            diff ** 2
            / var_term
            + log_term
        )
    )

    return np.sqrt(
        max(
            safe_eps,
            d2,
        )
    )


def logspace_latent_gaussian_embedding(
    row,
    noise_scale=1e-6,
):
    """Historical Step09 log-space Gaussian embedding for one dataframe row."""
    n_dims = 0

    while (
        f"latent_{n_dims + 1}"
        in row.index
        and
        f"sig_{n_dims + 1}"
        in row.index
    ):
        n_dims += 1

    if n_dims < 1:
        raise ValueError(
            "row does not contain paired "
            "latent_*/sig_* columns"
        )

    means = np.array(
        [
            row[
                f"latent_{idx + 1}"
            ]
            for idx
            in range(n_dims)
        ],
        dtype=float,
    )

    sigmas = np.array(
        [
            row[
                f"sig_{idx + 1}"
            ]
            for idx
            in range(n_dims)
        ],
        dtype=float,
    )

    sigmas = np.maximum(
        sigmas,
        1e-8,
    )

    emb = np.zeros(
        2 * n_dims,
        dtype=float,
    )

    emb[0::2] = (
        means
        / sigmas
    )

    emb[1::2] = (
        np.sqrt(2.0)
        * np.log(sigmas)
    )

    if noise_scale:
        emb += (
            np.random.randn(
                len(emb)
            )
            * noise_scale
        )

    return emb


def materialize_logspace_latent_gaussian_embedding(
    node_df,
    *,
    n_dims: int,
    noise_scale: float = 1e-6,
    random_seed: int = 0,
):
    """Vectorized row-order equivalent of the historical noisy embedding."""
    n_dims = int(n_dims)

    if n_dims < 1:
        raise ValueError(
            "n_dims must be positive"
        )

    mean_cols = [
        f"latent_{idx + 1}"
        for idx in range(n_dims)
    ]

    sigma_cols = [
        f"sig_{idx + 1}"
        for idx in range(n_dims)
    ]

    missing = [
        column
        for column
        in mean_cols + sigma_cols
        if column
        not in node_df.columns
    ]

    if missing:
        raise ValueError(
            "missing latent Gaussian columns: "
            + repr(missing)
        )

    means = node_df[
        mean_cols
    ].to_numpy(
        dtype=np.float64,
        copy=True,
    )

    sigmas = node_df[
        sigma_cols
    ].to_numpy(
        dtype=np.float64,
        copy=True,
    )

    sigmas = np.maximum(
        sigmas,
        1e-8,
    )

    embedding = np.empty(
        (
            len(node_df),
            2 * n_dims,
        ),
        dtype=np.float64,
    )

    embedding[:, 0::2] = (
        means
        / sigmas
    )

    embedding[:, 1::2] = (
        np.sqrt(2.0)
        * np.log(sigmas)
    )

    if noise_scale:
        rng = np.random.RandomState(
            int(random_seed)
        )

        embedding += (
            rng.randn(
                len(node_df),
                2 * n_dims,
            )
            * float(
                noise_scale
            )
        )

    return embedding


__all__ = [
    "fisher_rao_latent_gaussian_distance",
    "logspace_latent_gaussian_embedding",
    "materialize_logspace_latent_gaussian_embedding",
]
