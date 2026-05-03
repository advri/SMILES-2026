"""
probe.py — Hallucination probe classifier (student-implemented).
Implements ``HallucinationProbe``, a binary classifier that classifies feature
vectors as truthful (0) or hallucinated (1). Called from ``solution.py``
via ``evaluate.run_evaluation``. All four public methods (``fit``,
``fit_hyperparameters``, ``predict``, ``predict_proba``) must be implemented
and their signatures must not change.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Must match aggregation.py
N_SELECTED_LAYERS = 8


class _LayerWiseMLP(nn.Module):
    """Shared MLP applied independently to each layer vector."""

    def __init__(
        self,
        d_in: int,
        n_layers: int,
        hidden_dim: int = 128,
        proj_dim: int = 64,
        dropout: float = 0.35,
    ) -> None:
        super().__init__()

        self.input_norm = nn.LayerNorm(d_in)

        self.shared_mlp = nn.Sequential(
            nn.Linear(d_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.layer_head = nn.Linear(proj_dim, 1)
        self.layer_logits = nn.Parameter(torch.zeros(n_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: Tensor of shape (batch_size, n_layers, d_in)

        Returns:
            Logits of shape (batch_size,)
        """
        x = self.input_norm(x)                       # (B, L, D)
        h = self.shared_mlp(x)                       # (B, L, P)
        per_layer_logits = self.layer_head(h).squeeze(-1)  # (B, L)

        layer_weights = torch.softmax(self.layer_logits, dim=0)  # (L,)
        logits = (per_layer_logits * layer_weights.unsqueeze(0)).sum(dim=1)
        return logits


class HallucinationProbe(nn.Module):
    """Binary classifier that detects hallucinations from hidden-state features."""

    def __init__(self) -> None:
        super().__init__()

        self._net: _LayerWiseMLP | None = None
        self._scaler = StandardScaler()
        self._threshold: float = 0.5

        self._n_layers = N_SELECTED_LAYERS
        self._device = torch.device("cpu")

        # defaults
        self._epochs = 100
        self._batch_size = 32
        self._lr = 1e-3
        self._weight_decay = 1e-2
        self._grad_clip = 1.0

    def _build_network(self, input_dim: int) -> None:
        """Instantiate the layer-wise network.

        Args:
            input_dim: flattened feature dimension = n_layers * hidden_dim
        """
        if input_dim % self._n_layers != 0:
            raise ValueError(
                f"Input dimension {input_dim} is not divisible by "
                f"n_layers={self._n_layers}. "
                "Check that aggregation.py and probe.py use the same number of layers "
                "and that no extra geometric features are appended."
            )

        per_layer_dim = input_dim // self._n_layers

        self._net = _LayerWiseMLP(
            d_in=per_layer_dim,
            n_layers=self._n_layers,
            hidden_dim=128,
            proj_dim=64,
            dropout=0.35,
        ).to(self._device)

    def _reshape_flat_features(self, X: np.ndarray) -> torch.Tensor:
        """Convert flat NumPy features into (N, L, D) torch tensor."""
        X_t = torch.from_numpy(X).float()
        return X_t.view(X_t.shape[0], self._n_layers, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — returns raw logits of shape ``(n_samples,)``.

        Args:
            x: Float tensor of shape ``(n_samples, n_layers, hidden_dim)``.

        Returns:
            1-D tensor of raw logits.
        """
        if self._net is None:
            raise RuntimeError(
                "Network has not been built yet. Call fit() before forward()."
            )

        if x.ndim != 3:
            raise ValueError(
                f"Expected x with shape (n_samples, n_layers, hidden_dim), got {tuple(x.shape)}"
            )

        return self._net(x)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        """Train the probe on labelled feature vectors."""
        torch.manual_seed(42)
        np.random.seed(42)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        X_scaled = self._scaler.fit_transform(X).astype(np.float32)

        self._build_network(X_scaled.shape[1])

        X_t = self._reshape_flat_features(X_scaled)
        y_t = torch.from_numpy(y.astype(np.float32))

        dataset = TensorDataset(X_t, y_t)
        batch_size = min(self._batch_size, len(dataset))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        pos_weight = torch.tensor(
            [n_neg / max(n_pos, 1)],
            dtype=torch.float32,
            device=self._device,
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self._lr,
            weight_decay=self._weight_decay,
        )

        self.train()
        for _ in range(self._epochs):
            for xb, yb in loader:
                xb = xb.to(self._device)
                yb = yb.to(self._device)

                optimizer.zero_grad()
                logits = self(xb)
                loss = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), self._grad_clip)
                optimizer.step()

        self.eval()
        return self

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        """Tune the decision threshold on a validation set to maximise F1."""
        probs = self.predict_proba(X_val)[:, 1]

        candidates = np.unique(np.concatenate([probs, np.linspace(0.0, 1.0, 101)]))
        best_threshold = 0.5
        best_f1 = -1.0

        for t in candidates:
            y_pred_t = (probs >= t).astype(int)
            score = f1_score(y_val, y_pred_t, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_threshold = float(t)

        self._threshold = best_threshold
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels for feature vectors."""
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates."""
        if self._net is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        X = np.asarray(X, dtype=np.float32)
        X_scaled = self._scaler.transform(X).astype(np.float32)
        X_t = self._reshape_flat_features(X_scaled).to(self._device)

        self.eval()
        with torch.no_grad():
            logits = self(X_t)
            prob_pos = torch.sigmoid(logits).cpu().numpy()

        return np.stack([1.0 - prob_pos, prob_pos], axis=1)
