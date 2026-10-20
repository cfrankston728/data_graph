import numpy as np
import pytest
import torch

from data_graph.native_decoder_fr import DecoderFRSettings, NativeDecoderFRBatcher


class ToyDecoder(torch.nn.Module):
    def reconstruct_from_latents(self, z):
        return torch.softmax(
            torch.stack((z[:, 0], z[:, 1], -z[:, 0]), dim=1),
            dim=1,
        )


def test_bounded_fisher_rao_matches_direct_reference():
    rng = np.random.default_rng(10)
    features = rng.normal(size=(32, 4)).astype(np.float32)
    features[7, :2] = features[3, :2]
    i = np.array([0, 1, 3, 3, 7, 0, 1, 5, 0, 0, 5, 6, 8])
    j = np.array([1, 0, 7, 3, 3, 8, 5, 9, 4, 1, 6, 5, 8])
    opts = DecoderFRSettings(
        latent_dim=2, capacity=3, max_group_edges=3,
        decode_microbatch=2, expected_output_bins=3,
    )
    scorer = NativeDecoderFRBatcher(ToyDecoder(), torch.device("cpu"), settings=opts)
    actual = scorer(features, i, j)

    with torch.inference_mode():
        z = torch.from_numpy(features[:, :2])
        p = ToyDecoder().reconstruct_from_latents(z)
        chord = (p[i].sqrt() - p[j].sqrt()).double().norm(dim=1)
        reference = (4 * torch.asin((chord / 2).clamp(0, 1))).float().numpy()

    np.testing.assert_allclose(actual, reference, atol=1e-6, rtol=3e-5)
    assert actual[2] == 0.0
    assert actual[3] == 0.0
    assert scorer.events[0]["peak_resident_nodes"] <= opts.capacity
    assert scorer.scored_edges == len(i)


def test_invalid_features_are_rejected():
    scorer = NativeDecoderFRBatcher(
        ToyDecoder(), torch.device("cpu"),
        settings=DecoderFRSettings(latent_dim=2, expected_output_bins=3),
    )
    with pytest.raises(ValueError):
        scorer(np.zeros((2, 1)), [0], [1])
