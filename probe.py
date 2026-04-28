"""
probe.py — Hallucination probe classifier (student-implemented).

Implements ``HallucinationProbe``, a binary MLP that classifies feature
vectors as truthful (0) or hallucinated (1).  Called from ``solution.py``
via ``evaluate.run_evaluation``.  All four public methods (``fit``,
``fit_hyperparameters``, ``predict``, ``predict_proba``) must be implemented
and their signatures must not change.
"""

import numpy as np
import torch
import torch.nn as nn

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class HallucinationProbe(nn.Module):
    """
    Stable and interpretable probe:
        StandardScaler -> PCA -> LogisticRegression
    with optional threshold tuning on validation data.

    Kept as nn.Module because the infrastructure expects that class shape,
    but the actual classifier is sklearn-based for robustness on small data.
    """

    def __init__(
        self,
        pca_components=0.95,
        C=1.0,
        max_iter=2000,
        validation_size=0.15,
        random_state=42,
        tune_threshold=True,
        class_weight="balanced",
    ):
        super().__init__()

        self.pca_components = pca_components
        self.C = C
        self.max_iter = max_iter
        self.validation_size = validation_size
        self.random_state = random_state
        self.tune_threshold = tune_threshold
        self.class_weight = class_weight

        self.scaler = None
        self.pca = None
        self.clf = None

        self.threshold_ = 0.5
        self.is_fitted_ = False
        self.constant_class_ = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train).astype(int)

        unique_classes = np.unique(y_train)
        if len(unique_classes) == 1:
            self.constant_class_ = int(unique_classes[0])
            self.is_fitted_ = True
            return self

        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train,
                y_train,
                test_size=self.validation_size,
                stratify=y_train,
                random_state=self.random_state,
            )
        else:
            X_val = np.asarray(X_val, dtype=np.float32)
            y_val = np.asarray(y_val).astype(int)

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        self.pca = self._build_pca(X_train_scaled)
        if self.pca is not None:
            X_train_proc = self.pca.fit_transform(X_train_scaled)
            X_val_proc = self.pca.transform(X_val_scaled)
        else:
            X_train_proc = X_train_scaled
            X_val_proc = X_val_scaled

        self.clf = LogisticRegression(
            C=self.C,
            penalty="l2",
            solver="liblinear",
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            random_state=self.random_state,
        )
        self.clf.fit(X_train_proc, y_train)

        if self.tune_threshold and len(np.unique(y_val)) > 1:
            val_probs = self.clf.predict_proba(X_val_proc)[:, 1]
            self.threshold_ = self._find_best_threshold(y_val, val_probs)
        else:
            self.threshold_ = 0.5

        self.is_fitted_ = True
        return self

    def predict_proba(self, X):
        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float32)

        if self.constant_class_ is not None:
            return np.full(len(X), float(self.constant_class_), dtype=np.float32)

        X_proc = self._transform(X)
        return self.clf.predict_proba(X_proc)[:, 1].astype(np.float32)

    def predict(self, X, threshold=None):
        probs = self.predict_proba(X)
        thr = self.threshold_ if threshold is None else float(threshold)
        return (probs >= thr).astype(int)

    def evaluate(self, X, y):
        y = np.asarray(y).astype(int)
        probs = self.predict_proba(X)
        preds = self.predict(X)

        result = {
            "accuracy": float(accuracy_score(y, preds)),
            "f1": float(f1_score(y, preds, zero_division=0)),
            "threshold": float(self.threshold_),
        }

        if len(np.unique(y)) > 1:
            result["auroc"] = float(roc_auc_score(y, probs))
        else:
            result["auroc"] = float("nan")

        return result

    def forward(self, X):
        """
        nn.Module-compatible forward. Returns probabilities as a torch tensor.
        """
        probs = self.predict_proba(X)
        return torch.from_numpy(probs)

    def _transform(self, X):
        X_scaled = self.scaler.transform(X)
        if self.pca is not None:
            return self.pca.transform(X_scaled)
        return X_scaled

    def _build_pca(self, X_train_scaled):
        if self.pca_components is None:
            return None

        n_samples, n_features = X_train_scaled.shape

        if isinstance(self.pca_components, float):
            if not (0.0 < self.pca_components < 1.0):
                raise ValueError("Float pca_components must be in (0, 1)")
            return PCA(
                n_components=self.pca_components,
                svd_solver="full",
                random_state=self.random_state,
            )

        n_components = int(self.pca_components)
        n_components = min(n_components, n_features, max(1, n_samples - 1))
        if n_components < 1:
            return None

        return PCA(
            n_components=n_components,
            svd_solver="auto",
            random_state=self.random_state,
        )

    def _find_best_threshold(self, y_true, probs):
        precision, recall, thresholds = precision_recall_curve(y_true, probs)

        if len(thresholds) == 0:
            return 0.5

        f1_scores = 2.0 * precision[:-1] * recall[:-1] / np.clip(
            precision[:-1] + recall[:-1],
            1e-12,
            None,
        )
        best_idx = int(np.nanargmax(f1_scores))
        return float(thresholds[best_idx])

    def _check_is_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError("HallucinationProbe is not fitted yet.")
