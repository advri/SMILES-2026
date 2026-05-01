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


def extract_geometric_features(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Optional compact geometric features.
    hidden: (n_layers, seq_len, hidden_dim)
    mask  : (seq_len,)
    """
    valid_idx = mask.bool().nonzero(as_tuple=False).squeeze(-1)
    if valid_idx.numel() == 0:
        return torch.zeros(8, dtype=torch.float32)

    x = hidden[:, valid_idx, :]  # (L, T, D)

    last = x[:, -1, :]                 # (L, D)
    mean = x.mean(dim=1)               # (L, D)
    std = x.std(dim=1, unbiased=False) # (L, D)

    feats = [
        torch.norm(last, dim=1).mean(),
        torch.norm(mean, dim=1).mean(),
        torch.norm(std, dim=1).mean(),
        (last - mean).norm(dim=1).mean(),
        last.var(dim=1, unbiased=False).mean(),
        mean.var(dim=1, unbiased=False).mean(),
        std.mean(),
        std.max(),
    ]
    return torch.stack(feats).float()


def _safe_tail(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    x: (L, T, D)
    returns last min(k, T) tokens
    """
    t = x.size(1)
    return x[:, -min(k, t):, :]


def aggregation_and_feature_extraction(
    hidden: torch.Tensor,
    mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """
    hidden: (n_layers, seq_len, hidden_dim)
    mask  : (seq_len,)
    returns 1D feature vector

    Design:
    - keep only last 4 transformer layers
    - focus on final valid tokens (proxy for assistant response)
    - use compact statistics:
        * last token
        * mean over last 8/16/32 tokens
        * global mean
        * tail-vs-global contrast
    """
    valid_idx = mask.bool().nonzero(as_tuple=False).squeeze(-1)
    if valid_idx.numel() == 0:
        out = torch.zeros(4 * hidden.size(-1) * 5, dtype=torch.float32)
        if use_geometric:
            out = torch.cat([out, torch.zeros(8, dtype=torch.float32)], dim=0)
        return out

    x = hidden[:, valid_idx, :]  # (L, T, D)

    # Drop embedding layer if present and keep only final transformer layers
    if x.size(0) > 1:
        x = x[1:]
    x = x[-4:] if x.size(0) >= 4 else x

    # Core pooled representations
    last_token = x[:, -1, :]                     # (L, D)
    full_mean = x.mean(dim=1)                    # (L, D)

    tail8 = _safe_tail(x, 8).mean(dim=1)         # (L, D)
    tail16 = _safe_tail(x, 16).mean(dim=1)       # (L, D)
    tail32 = _safe_tail(x, 32).mean(dim=1)       # (L, D)

    # Contrast features: answer tail vs full context
    diff16 = tail16 - full_mean                  # (L, D)

    features = torch.cat(
        [
            last_token.reshape(-1),
            tail8.reshape(-1),
            tail16.reshape(-1),
            tail32.reshape(-1),
            diff16.reshape(-1),
        ],
        dim=0,
    ).float()

    if use_geometric:
        geom = extract_geometric_features(hidden, mask)
        features = torch.cat([features, geom], dim=0)

    return features
