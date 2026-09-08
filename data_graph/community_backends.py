from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class BackendCapabilities:
    name: str
    device: str
    supports_warm_start: bool
    deterministic: bool
    production_qualified: bool


class CommunityBackend:
    capabilities: BackendCapabilities

    def is_available(self) -> bool:
        return True

    def fit_predict(
        self,
        adjacency,
        *,
        resolution: float,
        initial_membership: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        raise NotImplementedError


class LeanLeidenCPUBackend(CommunityBackend):
    capabilities = BackendCapabilities(
        name="cpu_lean_leiden",
        device="cpu",
        supports_warm_start=True,
        deterministic=True,
        production_qualified=True,
    )

    def is_available(self) -> bool:
        try:
            from .lean_leiden import LeanLeiden  # noqa: F401
            return True
        except Exception:
            return False

    def fit_predict(
        self,
        adjacency,
        *,
        resolution: float,
        initial_membership=None,
        **kwargs,
    ):
        from .lean_leiden import LeanLeiden

        allowed = {
            "modularity",
            "n_aggregations",
            "refinement_theta",
            "refinement_seed",
            "verbose",
            "warm_level0_scheduler",
        }

        unknown = sorted(
            set(kwargs)
            - allowed
        )

        if unknown:
            raise TypeError(
                "unsupported cpu_lean_leiden backend kwargs: "
                + repr(unknown)
            )

        model = LeanLeiden(
            resolution=float(
                resolution
            ),
            modularity=kwargs.get(
                "modularity",
                "newman",
            ),
            n_aggregations=int(
                kwargs.get(
                    "n_aggregations",
                    50,
                )
            ),
            refinement_theta=float(
                kwargs.get(
                    "refinement_theta",
                    0.01,
                )
            ),
            refinement_seed=int(
                kwargs.get(
                    "refinement_seed",
                    155143,
                )
            ),
            verbose=bool(
                kwargs.get(
                    "verbose",
                    False,
                )
            ),
            warm_level0_scheduler=kwargs.get(
                "warm_level0_scheduler",
                None,
            ),
        )

        if initial_membership is None:
            labels = model.fit_predict(
                adjacency
            )
        else:
            labels = model.fit_predict(
                adjacency,
                initial_membership=np.asarray(
                    initial_membership
                ),
            )

        return np.asarray(
            labels,
            dtype=np.int32,
        )


class CuGraphLeidenGPUBackend(CommunityBackend):
    """Lazy peer GPU backend.

    Cold-start use is exposed, but this adapter is deliberately marked
    production_qualified=False until separately qualified.

    Warm-start is not claimed because stock cuGraph Leiden does not expose
    the qualified warm-start semantics used by project-owned LeanLeiden.
    """

    capabilities = BackendCapabilities(
        name="gpu_cugraph_leiden",
        device="gpu",
        supports_warm_start=False,
        deterministic=False,
        production_qualified=False,
    )

    def is_available(self) -> bool:
        try:
            import cudf  # noqa: F401
            import cugraph  # noqa: F401
            return True
        except Exception:
            return False

    def fit_predict(
        self,
        adjacency,
        *,
        resolution: float,
        initial_membership=None,
        **kwargs,
    ):
        if initial_membership is not None:
            raise NotImplementedError(
                "gpu_cugraph_leiden warm-start semantics "
                "are not currently qualified"
            )

        try:
            import cudf
            import cugraph
            from scipy import sparse
        except Exception as exc:
            raise RuntimeError(
                "gpu_cugraph_leiden requires RAPIDS cudf+cugraph"
            ) from exc

        allowed = {
            "max_iter",
            "random_state",
            "theta",
        }

        unknown = sorted(
            set(kwargs)
            - allowed
        )

        if unknown:
            raise TypeError(
                "unsupported gpu_cugraph_leiden backend kwargs: "
                + repr(unknown)
            )

        if not sparse.isspmatrix_csr(
            adjacency
        ):
            adjacency = sparse.csr_matrix(
                adjacency
            )

        coo = adjacency.tocoo(
            copy=False
        )

        mask = (
            coo.row
            < coo.col
        )

        edge_df = cudf.DataFrame({
            "src":
                np.asarray(
                    coo.row[mask],
                    dtype=np.int32,
                ),

            "dst":
                np.asarray(
                    coo.col[mask],
                    dtype=np.int32,
                ),

            "weight":
                np.asarray(
                    coo.data[mask],
                    dtype=np.float32,
                ),
        })

        graph = cugraph.Graph(
            directed=False
        )

        graph.from_cudf_edgelist(
            edge_df,
            source="src",
            destination="dst",
            edge_attr="weight",
            renumber=False,
        )

        leiden_kwargs = {
            "resolution":
                float(
                    resolution
                ),

            "max_iter":
                int(
                    kwargs.get(
                        "max_iter",
                        100,
                    )
                ),
        }

        if (
            kwargs.get(
                "random_state"
            )
            is not None
        ):
            leiden_kwargs[
                "random_state"
            ] = int(
                kwargs[
                    "random_state"
                ]
            )

        if (
            kwargs.get(
                "theta"
            )
            is not None
        ):
            leiden_kwargs[
                "theta"
            ] = float(
                kwargs[
                    "theta"
                ]
            )

        parts, _modularity = (
            cugraph.leiden(
                graph,
                **leiden_kwargs,
            )
        )

        vertices = np.asarray(
            parts[
                "vertex"
            ].to_numpy(),
            dtype=np.int64,
        )

        partitions = np.asarray(
            parts[
                "partition"
            ].to_numpy(),
            dtype=np.int32,
        )

        labels = np.full(
            adjacency.shape[0],
            -1,
            dtype=np.int32,
        )

        labels[
            vertices
        ] = partitions

        if np.any(
            labels < 0
        ):
            raise RuntimeError(
                "GPU Leiden did not return every vertex"
            )

        return labels


_BACKENDS = {}
_ALIASES = {}


def register_community_backend(
    backend,
    *,
    aliases=(),
    replace=False,
):
    name = (
        backend
        .capabilities
        .name
    )

    if (
        not replace
        and name in _BACKENDS
    ):
        raise ValueError(
            "backend already registered: "
            + name
        )

    _BACKENDS[
        name
    ] = backend

    for alias in aliases:
        alias = str(
            alias
        ).lower()

        if (
            not replace
            and alias
            in _ALIASES
        ):
            raise ValueError(
                "backend alias already registered: "
                + alias
            )

        _ALIASES[
            alias
        ] = name


def get_community_backend(
    name,
):
    key = str(
        name
    ).lower()

    canonical = (
        _ALIASES.get(
            key,
            key,
        )
    )

    if canonical not in _BACKENDS:
        raise ValueError(
            "unknown community backend "
            + repr(name)
            + "; registered="
            + repr(
                sorted(
                    _BACKENDS
                )
            )
        )

    return _BACKENDS[
        canonical
    ]


def community_backend_status():
    return {
        name: {
            "device":
                backend.capabilities.device,

            "supports_warm_start":
                backend.capabilities.supports_warm_start,

            "deterministic":
                backend.capabilities.deterministic,

            "production_qualified":
                backend.capabilities.production_qualified,

            "available":
                bool(
                    backend.is_available()
                ),
        }

        for name, backend
        in sorted(
            _BACKENDS.items()
        )
    }


def run_community_backend(
    adjacency,
    *,
    backend="cpu_lean_leiden",
    resolution: float,
    initial_membership=None,
    backend_kwargs=None,
):
    implementation = (
        get_community_backend(
            backend
        )
    )

    if not implementation.is_available():
        raise RuntimeError(
            "community backend unavailable: "
            + implementation.capabilities.name
        )

    if (
        initial_membership
        is not None
        and not implementation
            .capabilities
            .supports_warm_start
    ):
        raise NotImplementedError(
            implementation.capabilities.name
            + " does not support warm start"
        )

    labels = np.asarray(
        implementation.fit_predict(
            adjacency,
            resolution=float(
                resolution
            ),
            initial_membership=
                initial_membership,
            **dict(
                backend_kwargs
                or {}
            ),
        )
    )

    if (
        labels.ndim != 1
        or labels.shape[0]
        != adjacency.shape[0]
    ):
        raise ValueError(
            "invalid backend label shape: "
            + repr(
                labels.shape
            )
        )

    return labels.astype(
        np.int32,
        copy=False,
    )


register_community_backend(
    LeanLeidenCPUBackend(),
    aliases=(
        "cpu",
        "lean_leiden",
        "leiden_csr",
    ),
)

register_community_backend(
    CuGraphLeidenGPUBackend(),
    aliases=(
        "gpu",
        "cugraph",
        "leiden_gpu",
    ),
)
