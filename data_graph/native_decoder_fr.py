"""VAE-optional bounded decoder Fisher-Rao scorer, ported from CYC26 V14/V15.

The caller provides a model with reconstruct_from_latents and a Torch device.
No model checkpoints, dataset paths, or training code are imported here.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class DecoderFRSettings:
    latent_dim: int = 5
    capacity: int = 4096
    max_group_edges: int = 4096
    decode_microbatch: int = 64
    expected_output_bins: int | None = 10201
    orientation: str = "max"

    def __post_init__(self):
        if min(self.latent_dim, self.capacity, self.max_group_edges, self.decode_microbatch) < 1:
            raise ValueError("decoder scorer limits must be positive")
        if self.capacity < 2:
            raise ValueError("at least two decoded nodes must fit in cache")
        if self.orientation not in ("min", "max"):
            raise ValueError("invalid orientation")


def plan_bounded_decode_groups(features, source, target, *, settings):
    """Retain V15's latent-PC ordering and strict GPU endpoint bound."""
    z = np.asarray(features, dtype=np.float64)
    i = np.asarray(source, dtype=np.int64)
    j = np.asarray(target, dtype=np.int64)
    if z.ndim != 2 or i.shape != j.shape:
        raise ValueError("invalid node features or edge endpoints")
    if not len(i):
        return [], []
    n = len(z)
    if min(i.min(), j.min()) < 0 or max(i.max(), j.max()) >= n:
        raise ValueError("invalid edge endpoint")
    centered = z - z.mean(0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    rank = np.empty(n, dtype=np.int64)
    rank[np.argsort(centered @ vh[0], kind="stable")] = np.arange(n)
    a, b = rank[i], rank[j]
    if settings.orientation == "min":
        order = np.lexsort((np.maximum(a, b), np.minimum(a, b)))
    else:
        order = np.lexsort((np.minimum(a, b), np.maximum(a, b)))
    groups, required = [], []
    for pos in range(0, len(order), settings.max_group_edges):
        pending = [order[pos:pos + settings.max_group_edges]]
        while pending:
            chunk = pending.pop()
            nodes = np.unique(np.r_[i[chunk], j[chunk]])
            if len(nodes) > settings.capacity:
                if len(chunk) <= 1:
                    raise ValueError("cache capacity cannot accommodate both edge endpoints")
                mid = len(chunk) // 2
                pending.extend((chunk[mid:], chunk[:mid]))
            else:
                groups.append(chunk)
                required.append(nodes.tolist())
    return groups, required


class NativeDecoderFRBatcher:
    """Callable(features, i, j), returning distances in original edge order.

    Uses V14 compact-active-node remapping to avoid a global-node cache,
    V15 PC scheduling and next-use cache eviction, and the stable
    4*asin(||sqrt(p)-sqrt(q)||/2) categorical Fisher-Rao formula.
    """
    def __init__(self, model, device, *, settings=None):
        self.model = model
        self.device = device
        self.settings = settings or DecoderFRSettings()
        self.calls = 0
        self.scored_edges = 0
        self.events = []

    def __call__(self, node_features, i_indices, j_indices):
        import torch

        cfg = self.settings
        i = np.asarray(i_indices, dtype=np.int64).reshape(-1)
        j = np.asarray(j_indices, dtype=np.int64).reshape(-1)
        data = np.asarray(node_features)
        if data.ndim != 2 or data.shape[1] < cfg.latent_dim or i.shape != j.shape:
            raise ValueError("invalid node features or edges")
        if not len(i):
            return np.empty(0, dtype=np.float32)
        if min(i.min(), j.min()) < 0 or max(i.max(), j.max()) >= len(data):
            raise ValueError("endpoint index outside population")
        active, inverse = np.unique(np.r_[i, j], return_inverse=True)
        local = np.ascontiguousarray(data[active], dtype=np.float32)
        if not np.isfinite(local).all():
            raise ValueError("nonfinite node features")
        k = len(i)
        a, b = inverse[:k].astype(np.int64), inverse[k:].astype(np.int64)
        groups, required = plan_bounded_decode_groups(local, a, b, settings=cfg)
        uses = [[] for _ in range(len(active))]
        for gid, nodes in enumerate(required):
            for node in nodes:
                uses[node].append(gid)
        ptr = np.zeros(len(active), dtype=np.int32)
        cache = {}
        result = np.empty(k, dtype=np.float32)
        metrics = dict(edges=k, active_nodes=len(active),
                       groups=len(groups), node_decodes=0,
                       decoder_batches=0, incomplete_evictions=0,
                       peak_resident_nodes=0)
        with torch.inference_mode():
            for gid, (positions, nodes) in enumerate(zip(groups, required)):
                needed = set(nodes)
                missing = [u for u in nodes if u not in cache]
                overflow = len(cache) + len(missing) - cfg.capacity
                if overflow > 0:
                    candidates = [u for u in cache if u not in needed]
                    def next_use(u):
                        return uses[u][ptr[u]] if ptr[u] < len(uses[u]) else float("inf")
                    victims = sorted(candidates,
                                     key=lambda u: (next_use(u), u),
                                     reverse=True)[:overflow]
                    if len(victims) != overflow:
                        raise RuntimeError("cache capacity cannot be respected")
                    for u in victims:
                        metrics["incomplete_evictions"] += int(ptr[u] < len(uses[u]))
                        del cache[u]
                for start in range(0, len(missing), cfg.decode_microbatch):
                    nodes_to_decode = np.asarray(
                        missing[start:start + cfg.decode_microbatch],
                        dtype=np.int64,
                    )
                    inp = torch.from_numpy(
                        np.ascontiguousarray(local[nodes_to_decode, :cfg.latent_dim])
                    ).to(self.device)
                    prediction = self.model.reconstruct_from_latents(inp)
                    if not torch.is_tensor(prediction):
                        raise TypeError("decoder must return a Torch Tensor")
                    flat = prediction.reshape(len(inp), -1).float()
                    if cfg.expected_output_bins is not None and flat.shape[1] != cfg.expected_output_bins:
                        raise ValueError("decoder output bin count mismatch")
                    if not bool(torch.isfinite(flat).all()) or bool((flat < -1e-7).any()):
                        raise ValueError("invalid decoded categorical probabilities")
                    flat = flat.clamp_min(0)
                    mass = flat.sum(1, keepdim=True)
                    if not bool((mass > 0).all()):
                        raise ValueError("zero decoded categorical mass")
                    decoded = (flat / mass).sqrt().contiguous()
                    for node, value in zip(nodes_to_decode, decoded.unbind(0)):
                        cache[int(node)] = value
                    metrics["node_decodes"] += len(nodes_to_decode)
                    metrics["decoder_batches"] += 1
                metrics["peak_resident_nodes"] = max(
                    metrics["peak_resident_nodes"], len(cache)
                )
                if len(cache) > cfg.capacity:
                    raise RuntimeError("decoder cache exceeded bound")
                x = torch.stack([cache[int(u)] for u in a[positions]]).double()
                y = torch.stack([cache[int(u)] for u in b[positions]]).double()
                chord = torch.linalg.vector_norm(x - y, dim=1)
                scores = (4 * torch.asin((chord / 2).clamp(0, 1))).float()
                same = np.all(
                    local[a[positions], :cfg.latent_dim] ==
                    local[b[positions], :cfg.latent_dim],
                    axis=1,
                )
                scores = torch.where(
                    torch.as_tensor(same, device=scores.device), 0., scores
                )
                result[positions] = scores.cpu().numpy()
                for u in nodes:
                    if uses[u][ptr[u]] != gid:
                        raise RuntimeError("cache-use bookkeeping failure")
                    ptr[u] += 1
                    if ptr[u] == len(uses[u]):
                        del cache[u]
        if cache:
            raise RuntimeError("cache not fully released")
        if not np.isfinite(result).all() or np.any(result < 0) or np.any(result > np.pi + 1e-5):
            raise ValueError("invalid Fisher-Rao result")
        self.calls += 1
        self.scored_edges += k
        self.events.append(metrics)
        return result
