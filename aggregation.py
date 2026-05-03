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

from __future__ import annotations

import torch

N_SELECTED_LAYERS = 8


def _masked_mean(layer_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token representations using the attention mask.

    Args:
        layer_states: Tensor of shape (n_layers, seq_len, hidden_dim).
        attention_mask: Tensor of shape (seq_len,) with 1 for real tokens.

    Returns:
        Tensor of shape (n_layers, hidden_dim).
    """
    mask = attention_mask.to(dtype=layer_states.dtype, device=layer_states.device)
    mask = mask.view(1, -1, 1)  # (1, seq_len, 1)

    denom = mask.sum(dim=1).clamp(min=1.0)  # (1, 1)
    pooled = (layer_states * mask).sum(dim=1) / denom  # (n_layers, hidden_dim)
    return pooled


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Convert per-token hidden states into a single flat feature vector.

    Args:
        hidden_states:
            Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
            Layer index 0 is the token embedding; index -1 is the final
            transformer layer.

        attention_mask:
            1-D tensor of shape ``(seq_len,)`` with 1 for real tokens and
            0 for padding.

    Returns:
        A 1-D tensor of shape ``(N_SELECTED_LAYERS * hidden_dim,)``.
    """
    if hidden_states.ndim != 3:
        raise ValueError(
            f"hidden_states must have shape (n_layers, seq_len, hidden_dim), "
            f"got {tuple(hidden_states.shape)}"
        )
    if attention_mask.ndim != 1:
        raise ValueError(
            f"attention_mask must have shape (seq_len,), got {tuple(attention_mask.shape)}"
        )

    # Exclude embedding layer.
    transformer_layers = hidden_states[1:]  # (n_transformer_layers, seq_len, hidden_dim)

    if transformer_layers.size(0) < N_SELECTED_LAYERS:
        raise ValueError(
            f"Need at least {N_SELECTED_LAYERS} transformer layers after excluding "
            f"the embedding layer, but got {transformer_layers.size(0)}."
        )

    selected_layers = transformer_layers[-N_SELECTED_LAYERS:]  # (K, seq_len, hidden_dim)
    pooled = _masked_mean(selected_layers, attention_mask)     # (K, hidden_dim)

    # Flatten so the public interface remains unchanged.
    return pooled.reshape(-1)


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Extract optional geometric / statistical features.

    For this experiment we deliberately disable extra geometric features to keep
    the representation clean and make the comparison against previous probes
    easier to interpret.

    Returns:
        Empty 1-D tensor.
    """
    return torch.zeros(
        0,
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Aggregate hidden states and optionally append geometric features.

    Args:
        hidden_states:
            Tensor of shape ``(n_layers, seq_len, hidden_dim)`` for one sample.
        attention_mask:
            1-D tensor of shape ``(seq_len,)`` with 1 for real tokens.
        use_geometric:
            Whether to append geometric features.

    Returns:
        A 1-D float tensor of shape ``(feature_dim,)``.
    """
    agg_features = aggregate(hidden_states, attention_mask)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features
