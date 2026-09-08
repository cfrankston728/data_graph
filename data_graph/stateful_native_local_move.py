from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np


MODE_FINE = 0
MODE_COARSE = 1


def _ptr(dtype):
    return np.ctypeslib.ndpointer(
        dtype=dtype,
        ndim=1,
        flags=("C_CONTIGUOUS", "ALIGNED"),
    )


def _bind(lib, name, data_dtype):
    fn = getattr(lib, name)

    fn.argtypes = [
        ctypes.c_int32,
        _ptr(np.int32),
        _ptr(np.int32),
        _ptr(data_dtype),
        _ptr(np.int32),
        ctypes.c_double,
        ctypes.c_int32,
        ctypes.c_int32,
        _ptr(np.int32),
        _ptr(np.int32),
        _ptr(np.float64),
        _ptr(np.int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
    ]

    fn.restype = ctypes.c_int

    return fn


def run_full_convergence(
    lib_path,
    indptr,
    indices,
    data,
    initial_labels,
    resolution,
    *,
    mode,
    max_passes=1000,
):
    lib_path = Path(lib_path)

    indptr = np.ascontiguousarray(
        indptr,
        dtype=np.int32,
    )

    indices = np.ascontiguousarray(
        indices,
        dtype=np.int32,
    )

    initial_labels = np.ascontiguousarray(
        initial_labels,
        dtype=np.int32,
    )

    if data.dtype == np.float32:
        data = np.ascontiguousarray(
            data,
            dtype=np.float32,
        )
        entry = "stateful_newman_full_convergence_f32"
        data_dtype = np.float32

    elif data.dtype == np.float64:
        data = np.ascontiguousarray(
            data,
            dtype=np.float64,
        )
        entry = "stateful_newman_full_convergence_f64"
        data_dtype = np.float64

    else:
        data = np.ascontiguousarray(
            data,
            dtype=np.float64,
        )
        entry = "stateful_newman_full_convergence_f64"
        data_dtype = np.float64

    n = int(initial_labels.size)

    if indptr.size != n + 1:
        raise ValueError(
            f"indptr size {indptr.size} != n+1 {n+1}"
        )

    if int(indptr[-1]) != int(indices.size):
        raise ValueError(
            "indptr[-1] != indices.size"
        )

    if indices.size != data.size:
        raise ValueError(
            "indices.size != data.size"
        )

    if mode in ("fine", MODE_FINE):
        mode_code = MODE_FINE

    elif mode in ("coarse", MODE_COARSE):
        mode_code = MODE_COARSE

    else:
        raise ValueError(
            f"unknown mode: {mode!r}"
        )

    lib = ctypes.CDLL(str(lib_path))
    fn = _bind(lib, entry, data_dtype)

    out_labels = np.empty(
        n,
        dtype=np.int32,
    )

    out_counts = np.empty(
        n,
        dtype=np.int32,
    )

    out_volumes = np.empty(
        n,
        dtype=np.float64,
    )

    out_next_free = np.empty(
        n,
        dtype=np.int32,
    )

    out_free_head = ctypes.c_int32(-1)
    out_increase = ctypes.c_double(0.0)
    out_passes = ctypes.c_int32(0)
    out_moves = ctypes.c_int64(0)
    out_empty_moves = ctypes.c_int64(0)

    rc = int(
        fn(
            n,
            indptr,
            indices,
            data,
            initial_labels,
            float(resolution),
            mode_code,
            int(max_passes),
            out_labels,
            out_counts,
            out_volumes,
            out_next_free,
            ctypes.byref(out_free_head),
            ctypes.byref(out_increase),
            ctypes.byref(out_passes),
            ctypes.byref(out_moves),
            ctypes.byref(out_empty_moves),
        )
    )

    return {
        "return_code": rc,
        "labels": out_labels,
        "counts": out_counts,
        "volumes": out_volumes,
        "next_free": out_next_free,
        "free_head": int(out_free_head.value),
        "increase": float(out_increase.value),
        "passes": int(out_passes.value),
        "moves": int(out_moves.value),
        "empty_community_moves":
            int(out_empty_moves.value),
        "mode": (
            "fine"
            if mode_code == MODE_FINE
            else "coarse"
        ),
        "data_dtype": str(data.dtype),
        "entry_point": entry,
    }

# ---------------------------------------------------------------------------
# Canonical package-local LeanLeiden adapter.
#
# The low-level R276-qualified ctypes implementation above is unchanged.
# This adapter deliberately preserves the exact six-argument public surface
# of empty_aware_fast_local_move.optimize_empty_aware_newman.
# ---------------------------------------------------------------------------

from pathlib import Path as _Path

_CANONICAL_NATIVE_LIBRARY = (
    _Path(__file__).resolve().with_name(
        "libstateful_native_local_move.so"
    )
)


def optimize_empty_aware_newman(
    indptr,
    indices,
    data,
    initial_labels,
    resolution,
    max_passes=1000,
):
    if not _CANONICAL_NATIVE_LIBRARY.is_file():
        raise RuntimeError(
            "stateful native local-move library is missing: "
            + str(_CANONICAL_NATIVE_LIBRARY)
        )

    result = run_full_convergence(
        _CANONICAL_NATIVE_LIBRARY,
        indptr,
        indices,
        data,
        initial_labels,
        float(resolution),
        mode="fine",
        max_passes=int(max_passes),
    )

    rc = int(result["return_code"])

    if rc != 0:
        raise RuntimeError(
            "stateful native local move failed with return code "
            + str(rc)
        )

    return (
        np.asarray(result["labels"], dtype=np.int32),
        float(result["increase"]),
        int(result["passes"]),
        int(result["moves"]),
        int(result["empty_community_moves"]),
    )

def optimize_empty_aware_newman_result(
    indptr,
    indices,
    data,
    initial_labels,
    resolution,
    max_passes=1000,
):
    """
    LeanLeiden-facing mapping adapter.

    The qualified standalone optimize_empty_aware_newman tuple API remains
    unchanged. This function only converts that frozen five-tuple surface
    into the mapping contract consumed by LeanLeiden.fit_predict.
    """
    (
        labels,
        increase,
        passes,
        moves,
        empty_community_moves,
    ) = optimize_empty_aware_newman(
        indptr,
        indices,
        data,
        initial_labels,
        resolution,
        max_passes,
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
            int(empty_community_moves),
    }
