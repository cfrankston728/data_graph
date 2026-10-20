"""CYC26 full-population topology tests (FAISS-free fake index)."""
import numpy as np
import pytest

from data_graph.full_population_knn import (
    select_ranked_nonself_neighbors,
    iter_full_knn_shards,
    build_full_knn_memmap,
    fit_full_ipca,
)


class ExactIndex:
    def __init__(self, points):
        self.points = points
        self.ntotal = len(points)

    def search(self, queries, n):
        d = ((queries[:, None] - self.points[None]) ** 2).sum(axis=2)
        ids = np.argsort(d, axis=1)[:, :n]
        return np.take_along_axis(d, ids, axis=1), ids


def test_ranked_nonself_neighbors():
    ids = select_ranked_nonself_neighbors(
        np.array([[0, 3, 1, 2], [1, 2, 0, 3]]),
        np.array([0, 1]), k=2, n_nodes=4,
    )
    np.testing.assert_array_equal(ids, [[3, 1], [2, 0]])


def test_reject_duplicate_neighbors():
    with pytest.raises(ValueError, match="duplicate"):
        select_ranked_nonself_neighbors(
            np.array([[0, 1, 1, 2]]), [0], k=2, n_nodes=4,
        )


def test_full_topology_memmap_and_lock(tmp_path):
    rng = np.random.default_rng(42)
    features = rng.normal(size=(17, 3)).astype(np.float32)
    index = ExactIndex(features)
    path = tmp_path / "knn.npy"
    seen = []
    result = build_full_knn_memmap(
        index, features, path, k=5,
        rows_per_shard=3, progress=seen.append,
    )
    assert result.shape == (17, 5)
    assert len(seen) == 6
    expected = np.argsort(
        ((features[:, None] - features[None, :]) ** 2).sum(axis=2),
        axis=1,
    )[:, 1:6]
    np.testing.assert_array_equal(result, expected)
    assert not np.any(result == np.arange(17)[:, None])
    assert not path.with_name("knn.npy.lock").exists()

    lock = path.with_name("knn.npy.lock")
    lock.write_text("other process")
    with pytest.raises(FileExistsError):
        build_full_knn_memmap(index, features, path, k=5)
    assert lock.read_text() == "other process"


def test_ipca_preserves_small_data_shape_with_tail():
    rng = np.random.default_rng(23)
    features = rng.normal(size=(21, 8)).astype(np.float32)
    transformed, info = fit_full_ipca(
        features, n_components=3, io_block=10,
    )
    assert transformed.shape == (21, 3)
    assert np.isfinite(transformed).all()
    assert info["nodes"] == 21
