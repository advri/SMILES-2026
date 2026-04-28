"""
aggregation.py — Token aggregation strategy and feature extraction
               (student-implemented).

Converts per-token, per-layer hidden states from the extraction loop in
``solution.py`` into flat feature vectors for the probe classifier.

Two stages can be customised independently:

  1. ``aggregate`` — select layers and token positions, pool into a vector.
  2. ``extract_geometric_features`` — optional hand-crafted features
     (enabled by setting ``USE_GEOMETRIC = True`` in ``solution.py``).

Both stages are combined by ``aggregation_and_feature_extraction``, the
single entry point called from the notebook.
"""

import torch
import torch.nn.functional as F


def _resolve_layers(n_layers: int, selected_layers=None):
    """
    hidden_states comes from torch.stack(outputs.hidden_states, dim=1),
    so index 0 is usually the embedding output and the last indices are
    the deepest transformer layers.
    """
    if selected_layers is None:
        # Use the last 4 actual representational layers if possible.
        start = max(1, n_layers - 4)
        selected_layers = list(range(start, n_layers))

    resolved = []
    for idx in selected_layers:
        if idx < 0:
            idx = n_layers + idx
        if idx < 0 or idx >= n_layers:
            raise IndexError(f"Layer index {idx} is out of range for n_layers={n_layers}")
        resolved.append(idx)

    return resolved


def _safe_cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a = a.float()
    b = b.float()
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=1, eps=eps).squeeze(0)


def extract_geometric_features(layer_vectors: torch.Tensor) -> torch.Tensor:
    """
    Hand-crafted geometric features over pooled layer vectors.

    Args:
        layer_vectors: Tensor of shape (L, H), usually mean-pooled vectors from selected layers.

    Returns:
        1D tensor of scalar geometric features.
    """
    layer_vectors = layer_vectors.float()
    L, H = layer_vectors.shape

    feats = []

    # Per-layer norms
    norms = torch.norm(layer_vectors, dim=1)
    feats.extend([
        norms.mean(),
        norms.std(unbiased=False),
        norms.min(),
        norms.max(),
    ])

    # Centered spread across layers
    centered = layer_vectors - layer_vectors.mean(dim=0, keepdim=True)
    centered_norms = torch.norm(centered, dim=1)
    feats.extend([
        centered_norms.mean(),
        centered_norms.std(unbiased=False),
        centered_norms.max(),
    ])

    # Adjacent-layer geometry
    if L >= 2:
        adj_cosines = []
        adj_distances = []

        for i in range(L - 1):
            a = layer_vectors[i]
            b = layer_vectors[i + 1]
            adj_cosines.append(_safe_cosine(a, b))
            adj_distances.append(torch.norm(b - a))

        adj_cosines = torch.stack(adj_cosines)
        adj_distances = torch.stack(adj_distances)

        feats.extend([
            adj_cosines.mean(),
            adj_cosines.std(unbiased=False),
            adj_cosines.min(),
            adj_cosines.max(),
            adj_distances.mean(),
            adj_distances.std(unbiased=False),
            adj_distances.max(),
        ])

        first = layer_vectors[0]
        last = layer_vectors[-1]
        feats.extend([
            _safe_cosine(first, last),
            torch.norm(last - first),
            torch.norm(last) - torch.norm(first),
        ])

    # Coordinate-wise spread summary
    coord_std = layer_vectors.std(dim=0, unbiased=False)
    feats.extend([
        coord_std.mean(),
        coord_std.std(unbiased=False),
        coord_std.max(),
    ])

    return torch.stack([x.float() for x in feats])


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
):
    """
    Aggregate token-level hidden states from multiple layers into one compact feature vector.

    Args:
        hidden_states: Tensor of shape (n_layers, seq_len, hidden_dim)
        attention_mask: Tensor of shape (seq_len,)
        use_geometric: whether to append hand-crafted geometric features

    Returns:
        1D CPU torch tensor
    """
    if hidden_states.ndim != 3:
        raise ValueError(
            f"Expected hidden_states shape (n_layers, seq_len, hidden_dim), got {tuple(hidden_states.shape)}"
        )

    if attention_mask.ndim != 1:
        raise ValueError(
            f"Expected attention_mask shape (seq_len,), got {tuple(attention_mask.shape)}"
        )

    hidden_states = hidden_states.float().cpu()
    attention_mask = attention_mask.float().cpu()

    n_layers, seq_len, hidden_dim = hidden_states.shape
    valid_len = int(attention_mask.sum().item())
    valid_len = max(1, min(valid_len, seq_len))

    selected_layers = _resolve_layers(n_layers)
    feature_blocks = []

    mean_pooled_layers = []
    last_pooled_layers = []

    for layer_idx in selected_layers:
        x = hidden_states[layer_idx, :valid_len, :]  # (T, H)

        mean_pool = x.mean(dim=0)
        max_pool = x.max(dim=0).values
        last_pool = x[-1]
        std_pool = x.std(dim=0, unbiased=False)

        feature_blocks.extend([mean_pool, max_pool, last_pool, std_pool])

        mean_pooled_layers.append(mean_pool)
        last_pooled_layers.append(last_pool)

    mean_stack = torch.stack(mean_pooled_layers, dim=0)  # (L, H)
    last_stack = torch.stack(last_pooled_layers, dim=0)  # (L, H)

    # Cross-layer summaries
    feature_blocks.extend([
        mean_stack.mean(dim=0),
        mean_stack.std(dim=0, unbiased=False),
        mean_stack.max(dim=0).values,
        mean_stack.min(dim=0).values,
    ])

    if mean_stack.size(0) >= 2:
        feature_blocks.extend([
            mean_stack[-1] - mean_stack[0],
            last_stack[-1] - last_stack[0],
        ])

    # Sequence-level summaries from the last selected layer
    last_layer_tokens = hidden_states[selected_layers[-1], :valid_len, :]  # (T, H)
    token_norms = torch.norm(last_layer_tokens, dim=1)
    feature_blocks.append(torch.tensor([
        token_norms.mean(),
        token_norms.std(unbiased=False),
        token_norms.max(),
    ], dtype=torch.float32))

    if use_geometric:
        feature_blocks.append(extract_geometric_features(mean_stack))

    features = torch.cat([block.reshape(-1).float() for block in feature_blocks], dim=0)
    return features.cpu()
