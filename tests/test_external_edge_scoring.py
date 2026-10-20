import numpy as np
import pytest

from data_graph.external_edge_scoring import (
    ExternalEdgeScoreError,
    score_knn_row_shards,
)


def test_shard_dedup_and_original_order():
    features = np.arange(4, dtype=np.float32)[:, None]
    neighbors = np.array([[1, 2, -1], [0, 1, 2], [0, 3, -1], [2, 3, -1]])
    calls = []

    def scorer(features, src, dst):
        calls.append(np.column_stack([src, dst]))
        return (dst - src).astype(np.float64)

    result = score_knn_row_shards(features, neighbors, scorer, rows_per_shard=2)
    expected = np.array([[1, 2, np.nan], [1, 0, 1],
                         [2, 1, np.nan], [1, 0, np.nan]], dtype=np.float32)
    np.testing.assert_allclose(result, expected, equal_nan=True)
    assert len(calls) == 2
    assert sum(np.all(v == [0, 1]) for v in calls[0]) == 1
    assert all(np.all(x[:, 0] < x[:, 1]) for x in calls)


def test_on_disk_atomic_output_and_single_writer(tmp_path):
    x = np.arange(3, dtype=np.float32)[:, None]
    nbrs = np.array([[1], [0], [-1]])
    output = tmp_path / 'score.npy'

    def score(features, src, dst):
        return np.ones(len(src))

    values = score_knn_row_shards(x, nbrs, score, output_path=output)
    assert isinstance(values, np.memmap)
    np.testing.assert_allclose(values[:2, 0], [1, 1])
    assert np.isnan(values[2, 0])
    assert not (tmp_path / 'score.npy.lock').exists()
    assert not list(tmp_path.glob('*.pending.npy'))
    (tmp_path / 'score.npy.lock').write_text('active writer')
    with pytest.raises(FileExistsError):
        score_knn_row_shards(x, nbrs, score, output_path=output)
    assert (tmp_path / 'score.npy.lock').read_text() == 'active writer'


def test_bad_score_rolls_back(tmp_path):
    x = np.arange(3, dtype=np.float32)[:, None]
    nbrs = np.array([[1], [0], [-1]])
    path = tmp_path / 'bad.npy'
    with pytest.raises(ExternalEdgeScoreError):
        score_knn_row_shards(x, nbrs, lambda *args: np.array([np.nan]), output_path=path)
    assert not path.exists()
    assert not path.with_name(path.name + '.lock').exists()
    assert not list(tmp_path.glob('*.pending.npy'))


def test_bad_neighbors_fail():
    x = np.arange(3, dtype=np.float32)[:, None]
    with pytest.raises(ExternalEdgeScoreError):
        score_knn_row_shards(x, np.array([[4], [0], [-1]]), lambda *a: np.ones(1))
