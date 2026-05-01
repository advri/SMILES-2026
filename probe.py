"""
probe.py — Hallucination probe classifier (student-implemented).
Implements ``HallucinationProbe``, a binary classifier that classifies feature
vectors as truthful (0) or hallucinated (1). Called from ``solution.py``
via ``evaluate.run_evaluation``. All four public methods (``fit``,
``fit_hyperparameters``, ``predict``, ``predict_proba``) must be implemented
and their signatures must not change.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class HallucinationProbe:
    """
    Stable linear probe for small-data / high-dimensional regime.
    """

    def __init__(self):
        self.pipeline = None
        self.threshold = 0.5

    def _build_pipeline(self, n_features: int, n_samples: int):
        # Conservative dimensionality reduction for p >> n
        n_components = min(64, max(8, min(n_samples - 1, n_features)))
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=n_components, random_state=42)),
                (
                    "clf",
                    LogisticRegression(
                        C=0.1,
                        class_weight="balanced",
                        max_iter=5000,
                        solver="liblinear",
                        random_state=42,
                    ),
                ),
            ]
        )

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        self.pipeline = self._build_pipeline(X.shape[1], X.shape[0])
        self.pipeline.fit(X, y)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        proba = self.pipeline.predict_proba(X)

        # Safety: always return shape (N, 2)
        if proba.ndim == 1:
            proba = np.stack([1.0 - proba, proba], axis=1)

        if proba.shape[1] == 1:
            p1 = proba[:, 0]
            proba = np.stack([1.0 - p1, p1], axis=1)

        return proba

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(np.int64)
