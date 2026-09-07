from __future__ import annotations

import concurrent.futures
import dataclasses
import hashlib
import json
import math
import os
import random
import threading
from pathlib import Path
from typing import Any, Callable, Hashable, Mapping, Sequence

DEFAULT_SAMPLE_FRACTION = 0.085


@dataclasses.dataclass(frozen=True)
class GraphInputs:
    node_df: Any
    feature_columns: tuple[str, ...]
    premetric_weight_function: Any
    embedding_function: Any
    embedding_values: Any | None
    generator_kwargs: Mapping[str, Any]
    graph_build_kwargs: Mapping[str, Any]
    analyzer_kwargs: Mapping[str, Any]
    leiden_kwargs: Mapping[str, Any]
    source_provenance: Mapping[str, Any]


@dataclasses.dataclass(frozen=True)
class SampleManifest:
    manifest_id: str
    sample_fraction: float
    sample_seed: int
    full_node_count: int
    sampled_indices: tuple[int, ...]
    feature_columns: tuple[str, ...]
    premetric_descriptor: Any
    embedding_function_descriptor: Any
    generator_kwargs: Any
    graph_build_kwargs: Any
    sampled_graph_build_kwargs: Any
    analyzer_kwargs: Any
    leiden_kwargs: Any
    resolutions: tuple[Hashable, ...]
    source_provenance: Any
    node_content_order_sha256: str
    embedding_content_sha256: str


@dataclasses.dataclass
class SampledWorkerResult:
    manifest_id: str
    sampled_indices: tuple[int, ...]
    sampled_generator: Any
    sampled_graph: Any
    sampled_build_info: Any
    labels_by_resolution: dict[Hashable, Any]
    artifact_path: str | None


class ConcurrentSampledBuildHandle:
    def __init__(
        self,
        *,
        full_generator: Any,
        full_build_result: Any,
        sampled_future: concurrent.futures.Future,
        manifest: SampleManifest,
        executor: concurrent.futures.Executor,
        owns_executor: bool,
    ):
        self.full_generator = full_generator
        self.full_build_result = full_build_result
        self.manifest = manifest
        self._sampled_future = sampled_future
        self._executor = executor
        self._owns_executor = owns_executor
        self._joined = False
        self._closed = False

    @property
    def sampled_future_done(self) -> bool:
        return self._sampled_future.done()

    @property
    def joined(self) -> bool:
        return self._joined

    @property
    def full_graph(self) -> Any:
        if not isinstance(self.full_build_result, tuple) or len(self.full_build_result) < 1:
            raise TypeError("full build result must be a tuple whose first item is the built DataGraph")
        return self.full_build_result[0]

    @property
    def full_build_info(self) -> Any:
        if not isinstance(self.full_build_result, tuple) or len(self.full_build_result) < 2:
            raise TypeError("full build result must contain build info as its second item")
        return self.full_build_result[1]

    def join_after_full_csr(
        self,
        full_csr: Any,
        *,
        timeout: float | None = None,
    ) -> SampledWorkerResult:
        if self._closed:
            raise RuntimeError("orchestration handle is closed")
        if self._joined:
            raise RuntimeError("sampled worker has already been joined")
        if full_csr is None:
            raise ValueError("full_csr must already exist before joining sampled worker")

        result = self._sampled_future.result(timeout=timeout)
        if result.manifest_id != self.manifest.manifest_id:
            raise RuntimeError(
                "sampled worker provenance mismatch: "
                + repr(result.manifest_id)
                + " != "
                + repr(self.manifest.manifest_id)
            )

        self._joined = True
        self._shutdown_executor(wait=True)
        return result

    def close_without_assignment(self) -> None:
        if self._closed:
            return
        self._sampled_future.cancel()
        self._shutdown_executor(wait=True)

    def _shutdown_executor(self, *, wait: bool) -> None:
        if self._closed:
            return
        if self._owns_executor:
            self._executor.shutdown(wait=wait, cancel_futures=True)
        self._closed = True


def deterministic_sample_indices(
    full_node_count: int,
    *,
    fraction: float = DEFAULT_SAMPLE_FRACTION,
    seed: int = 0,
) -> tuple[int, ...]:
    if isinstance(full_node_count, bool) or not isinstance(full_node_count, int):
        raise TypeError("full_node_count must be int")
    if full_node_count < 0:
        raise ValueError("full_node_count must be nonnegative")
    if not (0.0 < float(fraction) <= 1.0):
        raise ValueError("fraction must satisfy 0 < fraction <= 1")
    if full_node_count == 0:
        return ()

    count = int(math.floor(full_node_count * float(fraction) + 0.5))
    count = max(1, min(full_node_count, count))
    rng = random.Random(int(seed))
    return tuple(sorted(rng.sample(range(full_node_count), count)))


def _take_rows(obj: Any, indices: Sequence[int]) -> Any:
    if hasattr(obj, "iloc"):
        return obj.iloc[list(indices)]
    try:
        return obj[list(indices)]
    except Exception:
        pass
    if isinstance(obj, tuple):
        return tuple(obj[i] for i in indices)
    return [obj[i] for i in indices]


def _callable_descriptor(obj: Any) -> dict[str, Any]:
    return {
        "type": "callable",
        "module": getattr(obj, "__module__", None),
        "qualname": getattr(obj, "__qualname__", getattr(obj, "__name__", None)),
    }


def _stable_descriptor(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if callable(obj):
        return _callable_descriptor(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {
            "type": type(obj).__module__ + "." + type(obj).__qualname__,
            "fields": _stable_descriptor(dataclasses.asdict(obj)),
        }
    if isinstance(obj, Mapping):
        return {
            str(key): _stable_descriptor(value)
            for key, value in sorted(obj.items(), key=lambda item: str(item[0]))
        }
    if isinstance(obj, (list, tuple)):
        return [_stable_descriptor(value) for value in obj]
    if isinstance(obj, set):
        return sorted(
            (_stable_descriptor(value) for value in obj),
            key=lambda value: json.dumps(value, sort_keys=True, default=str),
        )
    if isinstance(obj, Path):
        return str(obj)

    shape = getattr(obj, "shape", None)
    descriptor = {"type": type(obj).__module__ + "." + type(obj).__qualname__}
    if shape is not None:
        try:
            descriptor["shape"] = list(shape)
        except Exception:
            descriptor["shape"] = str(shape)
    try:
        descriptor["length"] = len(obj)
    except Exception:
        pass
    return descriptor


def _hash_bytes(parts: Sequence[bytes | str]) -> str:
    hasher = hashlib.sha256()
    for part in parts:
        if isinstance(part, str):
            part = part.encode("utf-8")
        hasher.update(len(part).to_bytes(8, byteorder="little", signed=False))
        hasher.update(part)
    return hasher.hexdigest()


def content_fingerprint(value: Any) -> str:
    try:
        import numpy as np
    except Exception:
        np = None
    try:
        import pandas as pd
    except Exception:
        pd = None

    if value is None:
        return _hash_bytes([b"none"])

    if pd is not None and isinstance(value, pd.DataFrame):
        row_hashes = pd.util.hash_pandas_object(value, index=True, categorize=False)
        metadata = {
            "kind": "pandas.DataFrame",
            "shape": list(value.shape),
            "columns": [repr(x) for x in value.columns.tolist()],
            "column_names": [repr(x) for x in value.columns.names],
            "index_names": [repr(x) for x in value.index.names],
            "dtypes": [str(x) for x in value.dtypes.tolist()],
        }
        return _hash_bytes([
            json.dumps(metadata, sort_keys=True, separators=(",", ":")),
            row_hashes.to_numpy(dtype="uint64", copy=False).tobytes(),
        ])

    if np is not None and isinstance(value, np.ndarray):
        metadata = {
            "kind": "numpy.ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        if value.dtype.hasobject:
            payload = json.dumps(value.tolist(), sort_keys=True, default=repr).encode("utf-8")
        else:
            payload = np.ascontiguousarray(value).tobytes()
        return _hash_bytes([
            json.dumps(metadata, sort_keys=True, separators=(",", ":")),
            payload,
        ])

    if hasattr(value, "to_numpy"):
        try:
            return content_fingerprint(value.to_numpy())
        except Exception:
            pass

    return _hash_bytes([
        json.dumps(_stable_descriptor(value), sort_keys=True, default=str, separators=(",", ":"))
    ])


def _validate_materialized_embedding(inputs: GraphInputs) -> None:
    if inputs.embedding_values is None:
        raise ValueError("embedding_values must be materialized before concurrency begins")
    if len(inputs.embedding_values) != len(inputs.node_df):
        raise ValueError("embedding_values row count must equal node_df row count")



def resolve_sampled_graph_build_kwargs(
    full_graph_build_kwargs: Mapping[str, Any],
    *,
    sample_fraction: float,
    sampled_node_count: int,
    sampled_n_neighbors: int | None = None,
) -> dict[str, Any]:
    """
    Resolve effective sampled-graph build kwargs.

    Every full-graph build parameter is preserved except n_neighbors.

    Automatic scaling approximates the same local neighborhood radius after
    uniform node subsampling:

        k_sample ~= sample_fraction * k_full

    The explicit override exists for experimental sweeps.
    """
    kwargs = dict(full_graph_build_kwargs)

    if sampled_node_count < 0:
        raise ValueError("sampled_node_count must be nonnegative")

    if not (0.0 < float(sample_fraction) <= 1.0):
        raise ValueError("sample_fraction must satisfy 0 < f <= 1")

    if "n_neighbors" not in kwargs:
        if sampled_n_neighbors is not None:
            raise ValueError(
                "sampled_n_neighbors override requires full graph "
                "n_neighbors to be explicit"
            )
        return kwargs

    k_full = kwargs["n_neighbors"]

    if isinstance(k_full, bool) or not isinstance(k_full, int):
        raise TypeError("full graph n_neighbors must be an integer")

    if k_full < 1:
        raise ValueError("full graph n_neighbors must be positive")

    if sampled_node_count <= 1:
        raise ValueError(
            "sampled graph requires at least two nodes when n_neighbors "
            "is configured"
        )

    if sampled_n_neighbors is None:
        k_sample = max(
            1,
            int(math.floor(
                float(sample_fraction) * int(k_full) + 0.5
            )),
        )
    else:
        if (
            isinstance(sampled_n_neighbors, bool)
            or not isinstance(sampled_n_neighbors, int)
        ):
            raise TypeError("sampled_n_neighbors must be int or None")

        if sampled_n_neighbors < 1:
            raise ValueError("sampled_n_neighbors must be positive")

        k_sample = int(sampled_n_neighbors)

    k_sample = min(
        int(sampled_node_count) - 1,
        k_sample,
    )

    kwargs["n_neighbors"] = k_sample
    return kwargs

def make_sample_manifest(
    inputs: GraphInputs,
    *,
    sampled_indices: Sequence[int],
    sample_fraction: float,
    sample_seed: int,
    resolutions: Sequence[Hashable],
    sampled_graph_build_kwargs: Mapping[str, Any],
) -> SampleManifest:
    if not inputs.source_provenance:
        raise ValueError("source_provenance must be non-empty")
    _validate_materialized_embedding(inputs)

    node_hash = content_fingerprint(inputs.node_df)
    embedding_hash = content_fingerprint(inputs.embedding_values)
    payload = {
        "sample_fraction": float(sample_fraction),
        "sample_seed": int(sample_seed),
        "full_node_count": len(inputs.node_df),
        "sampled_indices": list(sampled_indices),
        "feature_columns": list(inputs.feature_columns),
        "premetric": _stable_descriptor(inputs.premetric_weight_function),
        "embedding_function": _stable_descriptor(inputs.embedding_function),
        "generator_kwargs": _stable_descriptor(dict(inputs.generator_kwargs)),
        "graph_build_kwargs": _stable_descriptor(dict(inputs.graph_build_kwargs)),
        "sampled_graph_build_kwargs": _stable_descriptor(
            dict(sampled_graph_build_kwargs)
        ),
        "analyzer_kwargs": _stable_descriptor(dict(inputs.analyzer_kwargs)),
        "leiden_kwargs": _stable_descriptor(dict(inputs.leiden_kwargs)),
        "resolutions": [_stable_descriptor(value) for value in resolutions],
        "source_provenance": _stable_descriptor(dict(inputs.source_provenance)),
        "node_content_order_sha256": node_hash,
        "embedding_content_sha256": embedding_hash,
    }
    manifest_id = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()

    return SampleManifest(
        manifest_id=manifest_id,
        sample_fraction=float(sample_fraction),
        sample_seed=int(sample_seed),
        full_node_count=len(inputs.node_df),
        sampled_indices=tuple(int(i) for i in sampled_indices),
        feature_columns=tuple(inputs.feature_columns),
        premetric_descriptor=_stable_descriptor(inputs.premetric_weight_function),
        embedding_function_descriptor=_stable_descriptor(inputs.embedding_function),
        generator_kwargs=_stable_descriptor(dict(inputs.generator_kwargs)),
        graph_build_kwargs=_stable_descriptor(dict(inputs.graph_build_kwargs)),
        sampled_graph_build_kwargs=_stable_descriptor(
            dict(sampled_graph_build_kwargs)
        ),
        analyzer_kwargs=_stable_descriptor(dict(inputs.analyzer_kwargs)),
        leiden_kwargs=_stable_descriptor(dict(inputs.leiden_kwargs)),
        resolutions=tuple(resolutions),
        source_provenance=_stable_descriptor(dict(inputs.source_provenance)),
        node_content_order_sha256=node_hash,
        embedding_content_sha256=embedding_hash,
    )


def _validate_label_vector(labels: Any, *, expected_count: int, resolution: Hashable) -> None:
    if isinstance(labels, (str, bytes)):
        raise TypeError("Leiden labels must be a one-dimensional sequence")
    shape = getattr(labels, "shape", None)
    if shape is not None and len(shape) != 1:
        raise ValueError("Leiden labels must be one-dimensional at resolution " + repr(resolution))
    try:
        observed = len(labels)
    except Exception as exc:
        raise TypeError("Leiden labels must provide len()") from exc
    if observed != expected_count:
        raise ValueError(
            "Leiden label/sample alignment mismatch at resolution "
            + repr(resolution)
            + ": expected "
            + str(expected_count)
            + ", observed "
            + str(observed)
        )


def _labels_descriptor(labels: Any) -> dict[str, Any]:
    try:
        serial = labels.tolist() if hasattr(labels, "tolist") else list(labels)
    except Exception:
        serial = _stable_descriptor(labels)
    payload = json.dumps(serial, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    try:
        length = len(labels)
    except Exception:
        length = None
    return {"length": length, "sha256": hashlib.sha256(payload).hexdigest()}


def _write_provenance_artifact(
    artifact_path: Path,
    *,
    manifest: SampleManifest,
    labels_by_resolution: Mapping[Hashable, Any],
) -> None:
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "sampled-data-graph-provenance-v3",
        "manifest": dataclasses.asdict(manifest),
        "labels_by_resolution": {
            repr(key): _labels_descriptor(value)
            for key, value in labels_by_resolution.items()
        },
    }
    tmp = artifact_path.with_name(
        artifact_path.name + ".tmp." + str(os.getpid()) + "." + str(threading.get_ident())
    )
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    os.replace(tmp, artifact_path)


def _run_sampled_worker(
    *,
    inputs: GraphInputs,
    manifest: SampleManifest,
    sampled_graph_build_kwargs: Mapping[str, Any],
    graph_factory: Callable[[GraphInputs], Any],
    graph_build_call: Callable[[Any, Mapping[str, Any]], Any],
    community_call: Callable[[GraphInputs, Any, Sequence[Hashable]], Mapping[Hashable, Any]],
    provenance_artifact_path: Path | None,
) -> SampledWorkerResult:
    sample_inputs = dataclasses.replace(
        inputs,
        node_df=_take_rows(inputs.node_df, manifest.sampled_indices),
        embedding_values=_take_rows(inputs.embedding_values, manifest.sampled_indices),
        graph_build_kwargs=dict(sampled_graph_build_kwargs),
        source_provenance={
            **dict(inputs.source_provenance),
            "parent_sample_manifest_id": manifest.manifest_id,
        },
    )

    sampled_generator = graph_factory(sample_inputs)
    sampled_build_result = graph_build_call(sampled_generator, sample_inputs.graph_build_kwargs)

    if not isinstance(sampled_build_result, tuple) or len(sampled_build_result) < 2:
        raise TypeError("graph build must return (DataGraph, build_info)")
    sampled_graph = sampled_build_result[0]
    sampled_build_info = sampled_build_result[1]

    labels_by_resolution = dict(
        community_call(sample_inputs, sampled_build_result, manifest.resolutions)
    )

    if set(labels_by_resolution) != set(manifest.resolutions):
        raise ValueError("community result resolution keys do not match requested resolutions")

    for resolution in manifest.resolutions:
        _validate_label_vector(
            labels_by_resolution[resolution],
            expected_count=len(manifest.sampled_indices),
            resolution=resolution,
        )

    if provenance_artifact_path is not None:
        _write_provenance_artifact(
            provenance_artifact_path,
            manifest=manifest,
            labels_by_resolution=labels_by_resolution,
        )

    return SampledWorkerResult(
        manifest_id=manifest.manifest_id,
        sampled_indices=manifest.sampled_indices,
        sampled_generator=sampled_generator,
        sampled_graph=sampled_graph,
        sampled_build_info=sampled_build_info,
        labels_by_resolution=labels_by_resolution,
        artifact_path=str(provenance_artifact_path) if provenance_artifact_path is not None else None,
    )


def launch_sampled_worker_then_build_full(
    *,
    inputs: GraphInputs,
    graph_factory: Callable[[GraphInputs], Any],
    graph_build_call: Callable[[Any, Mapping[str, Any]], Any],
    community_call: Callable[[GraphInputs, Any, Sequence[Hashable]], Mapping[Hashable, Any]],
    resolutions: Sequence[Hashable],
    sample_fraction: float = DEFAULT_SAMPLE_FRACTION,
    sample_seed: int = 0,
    sampled_n_neighbors: int | None = None,
    provenance_artifact_path: str | os.PathLike[str] | None = None,
    executor: concurrent.futures.Executor | None = None,
    worker_start_timeout_s: float = 10.0,
) -> ConcurrentSampledBuildHandle:
    if not resolutions:
        raise ValueError("at least one Leiden resolution is required")
    _validate_materialized_embedding(inputs)

    if bool(dict(inputs.graph_build_kwargs).get("construction_coarsening_seed", False)):
        raise ValueError(
            "construction_coarsening_seed=True is not thread-safe in the current exported generator; "
            "concurrent sampled/full orchestration requires it to remain disabled"
        )

    sampled_indices = deterministic_sample_indices(
        len(inputs.node_df), fraction=sample_fraction, seed=sample_seed
    )

    sampled_graph_build_kwargs = resolve_sampled_graph_build_kwargs(
        inputs.graph_build_kwargs,
        sample_fraction=sample_fraction,
        sampled_node_count=len(sampled_indices),
        sampled_n_neighbors=sampled_n_neighbors,
    )

    manifest = make_sample_manifest(
        inputs,
        sampled_indices=sampled_indices,
        sample_fraction=sample_fraction,
        sample_seed=sample_seed,
        resolutions=resolutions,
        sampled_graph_build_kwargs=sampled_graph_build_kwargs,
    )
    artifact_path = Path(provenance_artifact_path) if provenance_artifact_path is not None else None

    owns_executor = executor is None
    if executor is None:
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="sampled-data-graph"
        )

    worker_entered = threading.Event()

    def worker_wrapper():
        worker_entered.set()
        return _run_sampled_worker(
            inputs=inputs,
            manifest=manifest,
            sampled_graph_build_kwargs=sampled_graph_build_kwargs,
            graph_factory=graph_factory,
            graph_build_call=graph_build_call,
            community_call=community_call,
            provenance_artifact_path=artifact_path,
        )

    sampled_future = executor.submit(worker_wrapper)
    if not worker_entered.wait(timeout=float(worker_start_timeout_s)):
        sampled_future.cancel()
        if owns_executor:
            executor.shutdown(wait=True, cancel_futures=True)
        raise TimeoutError("sampled worker did not enter before full build launch")

    full_generator = graph_factory(inputs)
    try:
        full_build_result = graph_build_call(full_generator, inputs.graph_build_kwargs)
    except BaseException:
        sampled_future.cancel()
        if owns_executor:
            executor.shutdown(wait=True, cancel_futures=True)
        raise

    if not isinstance(full_build_result, tuple) or len(full_build_result) < 2:
        sampled_future.cancel()
        if owns_executor:
            executor.shutdown(wait=True, cancel_futures=True)
        raise TypeError("full graph build must return (DataGraph, build_info)")

    return ConcurrentSampledBuildHandle(
        full_generator=full_generator,
        full_build_result=full_build_result,
        sampled_future=sampled_future,
        manifest=manifest,
        executor=executor,
        owns_executor=owns_executor,
    )
