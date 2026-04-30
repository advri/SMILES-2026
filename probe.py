"""
probe.py — Hallucination probe classifier (student-implemented).
Implements ``HallucinationProbe``, a binary classifier that classifies feature
vectors as truthful (0) or hallucinated (1). Called from ``solution.py``
via ``evaluate.run_evaluation``. All four public methods (``fit``,
``fit_hyperparameters``, ``predict``, ``predict_proba``) must be implemented
and their signatures must not change.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class HallucinationProbe(nn.Module):
    """
    Experimental v2 probe:
        StandardScaler -> PCA -> LogisticRegression

    Goals of this experiment:
    1. Stronger regularization on small data.
    2. More aggressive dimensionality reduction.
    3. More conservative threshold tuning (balanced accuracy instead of F1).
    4. Stable sklearn-like predict_proba output with shape (N, 2).
    """

    def __init__(
        self,
        pca_components=0.95,
        C=0.1,
        max_iter=4000,
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
        self.best_params_ = None

        # Internal cap to prevent PCA from keeping too many dimensions on tiny data
        self._max_effective_pca_dim = 64

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train).astype(int)

        if X_train.ndim == 1:
            X_train = X_train.reshape(1, -1)

        self.is_fitted_ = False
        self.constant_class_ = None
        self.threshold_ = 0.5
        self.scaler = None
        self.pca = None
        self.clf = None

        unique_classes = np.unique(y_train)
        if len(unique_classes) == 1:
            self.constant_class_ = int(unique_classes[0])
            self.is_fitted_ = True
            return self

        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = self._safe_train_val_split(X_train, y_train)
        else:
            X_val = np.asarray(X_val, dtype=np.float32)
            y_val = np.asarray(y_val).astype(int)
            if X_val.ndim == 1:
                X_val = X_val.reshape(1, -1)

        self.scaler, self.pca, self.clf = self._fit_pipeline(
            X_train=X_train,
            y_train=y_train,
            pca_components=self.pca_components,
            C=self.C,
            class_weight=self.class_weight,
        )

        if self.tune_threshold and len(np.unique(y_val)) > 1:
            val_probs = self._predict_positive_proba_with_pipeline(
                X_val, self.scaler, self.pca, self.clf
            )
            self.threshold_ = self._find_best_threshold_balanced(y_val, val_probs)
        else:
            self.threshold_ = 0.5

        self.is_fitted_ = True
        return self

    def fit_hyperparameters(self, X_train, y_train, X_val=None, y_val=None):
        """
        Small experimental hyperparameter search.
        Chooses params on validation AUROC (primary) and balanced accuracy (tie-breaker),
        then fits the final model via fit() using the selected params.
        """
        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train).astype(int)

        if X_train.ndim == 1:
            X_train = X_train.reshape(1, -1)

        unique_classes = np.unique(y_train)
        if len(unique_classes) == 1:
            self.constant_class_ = int(unique_classes[0])
            self.threshold_ = 0.5
            self.is_fitted_ = True
            self.best_params_ = {
                "pca_components": None,
                "C": self.C,
                "class_weight": self.class_weight,
            }
            return self

        if X_val is None or y_val is None:
            X_subtrain, X_val, y_subtrain, y_val = self._safe_train_val_split(X_train, y_train)
        else:
            X_subtrain = X_train
            y_subtrain = y_train
            X_val = np.asarray(X_val, dtype=np.float32)
            y_val = np.asarray(y_val).astype(int)
            if X_val.ndim == 1:
                X_val = X_val.reshape(1, -1)

        pca_candidates = self._get_pca_candidates(X_subtrain.shape)
        c_candidates = [0.03, 0.1, 0.3, 1.0]
        if self.C not in c_candidates:
            c_candidates.append(float(self.C))
            c_candidates = sorted(set(c_candidates))

        if self.class_weight == "balanced":
            class_weight_candidates = [None, "balanced"]
        else:
            class_weight_candidates = [self.class_weight]

        best = None

        for pca_components in pca_candidates:
            for C in c_candidates:
                for class_weight in class_weight_candidates:
                    try:
                        scaler, pca, clf = self._fit_pipeline(
                            X_train=X_subtrain,
                            y_train=y_subtrain,
                            pca_components=pca_components,
                            C=C,
                            class_weight=class_weight,
                        )

                        val_probs = self._predict_positive_proba_with_pipeline(
                            X_val, scaler, pca, clf
                        )

                        if len(np.unique(y_val)) > 1:
                            val_auroc = float(roc_auc_score(y_val, val_probs))
                        else:
                            val_auroc = float("-inf")

                        threshold = (
                            self._find_best_threshold_balanced(y_val, val_probs)
                            if self.tune_threshold and len(np.unique(y_val)) > 1
                            else 0.5
                        )
                        val_preds = (val_probs >= threshold).astype(int)
                        bal_acc = self._balanced_accuracy(y_val, val_preds)

                        candidate = {
                            "score_auroc": val_auroc,
                            "score_bal_acc": bal_acc,
                            "pca_components": pca_components,
                            "C": C,
                            "class_weight": class_weight,
                        }

                        if best is None:
                            best = candidate
                        else:
                            better = False
                            if candidate["score_auroc"] > best["score_auroc"] + 1e-12:
                                better = True
                            elif abs(candidate["score_auroc"] - best["score_auroc"]) <= 1e-12:
                                if candidate["score_bal_acc"] > best["score_bal_acc"] + 1e-12:
                                    better = True
                            if better:
                                best = candidate

                    except Exception:
                        continue

        if best is not None:
            self.pca_components = best["pca_components"]
            self.C = best["C"]
            self.class_weight = best["class_weight"]
            self.best_params_ = {
                "pca_components": best["pca_components"],
                "C": best["C"],
                "class_weight": best["class_weight"],
                "val_auroc": best["score_auroc"],
                "val_bal_acc": best["score_bal_acc"],
            }
        else:
            self.best_params_ = {
                "pca_components": self.pca_components,
                "C": self.C,
                "class_weight": self.class_weight,
            }

        return self.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    def predict_proba(self, X):
        """
        Returns probabilities in sklearn-compatible shape (N, 2):
            [:, 0] -> P(class=0)
            [:, 1] -> P(class=1)
        """
        self._check_is_fitted()

        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        n = X.shape[0]

        if self.constant_class_ is not None:
            if self.constant_class_ == 1:
                p1 = np.ones(n, dtype=np.float32)
            else:
                p1 = np.zeros(n, dtype=np.float32)
            return np.column_stack([1.0 - p1, p1]).astype(np.float32)

        X_proc = self._transform(X)
        proba = np.asarray(self.clf.predict_proba(X_proc), dtype=np.float32)

        if proba.ndim == 1:
            p1 = proba
            return np.column_stack([1.0 - p1, p1]).astype(np.float32)

        if proba.shape[1] == 2:
            if hasattr(self.clf, "classes_"):
                classes = np.asarray(self.clf.classes_)
                if len(classes) == 2 and not np.array_equal(classes, np.array([0, 1])):
                    idx0 = int(np.where(classes == 0)[0][0])
                    idx1 = int(np.where(classes == 1)[0][0])
                    return np.column_stack([proba[:, idx0], proba[:, idx1]]).astype(np.float32)
            return proba.astype(np.float32)

        if proba.shape[1] == 1:
            p = proba[:, 0].astype(np.float32)
            if hasattr(self.clf, "classes_") and len(self.clf.classes_) == 1:
                only_class = int(self.clf.classes_[0])
                if only_class == 1:
                    p1 = p
                else:
                    p1 = 1.0 - p
            else:
                p1 = p
            return np.column_stack([1.0 - p1, p1]).astype(np.float32)

        raise RuntimeError(f"Unexpected predict_proba output shape: {proba.shape}")

    def predict(self, X, threshold=None):
        probs = self.predict_proba(X)[:, 1]
        thr = self.threshold_ if threshold is None else float(threshold)
        return (probs >= thr).astype(int)

    def evaluate(self, X, y):
        y = np.asarray(y).astype(int)
        probs = self.predict_proba(X)[:, 1]
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
        nn.Module-compatible forward. Returns probabilities as torch tensor
        with shape (N, 2).
        """
        probs = self.predict_proba(X)
        return torch.from_numpy(probs)

    def _safe_train_val_split(self, X, y):
        try:
            return train_test_split(
                X,
                y,
                test_size=self.validation_size,
                stratify=y,
                random_state=self.random_state,
            )
        except ValueError:
            return train_test_split(
                X,
                y,
                test_size=self.validation_size,
                random_state=self.random_state,
            )

    def _fit_pipeline(self, X_train, y_train, pca_components, C, class_weight):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        pca = self._build_pca(X_train_scaled, pca_components)
        if pca is not None:
            X_train_proc = pca.fit_transform(X_train_scaled)
        else:
            X_train_proc = X_train_scaled

        clf = LogisticRegression(
            C=float(C),
            penalty="l2",
            solver="liblinear",
            max_iter=self.max_iter,
            class_weight=class_weight,
            random_state=self.random_state,
        )
        clf.fit(X_train_proc, y_train)

        return scaler, pca, clf

    def _predict_positive_proba_with_pipeline(self, X, scaler, pca, clf):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        X_scaled = scaler.transform(X)
        if pca is not None:
            X_proc = pca.transform(X_scaled)
        else:
            X_proc = X_scaled

        proba = np.asarray(clf.predict_proba(X_proc), dtype=np.float32)

        if proba.ndim == 1:
            return proba.astype(np.float32)

        if proba.shape[1] == 2:
            if hasattr(clf, "classes_"):
                classes = np.asarray(clf.classes_)
                if len(classes) == 2:
                    idx1 = int(np.where(classes == 1)[0][0])
                    return proba[:, idx1].astype(np.float32)
            return proba[:, 1].astype(np.float32)

        if proba.shape[1] == 1:
            if hasattr(clf, "classes_") and len(clf.classes_) == 1:
                only_class = int(clf.classes_[0])
                if only_class == 1:
                    return proba[:, 0].astype(np.float32)
                return np.zeros(X.shape[0], dtype=np.float32)
            return proba[:, 0].astype(np.float32)

        raise RuntimeError(f"Unexpected predict_proba output shape: {proba.shape}")

    def _transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        X_scaled = self.scaler.transform(X)
        if self.pca is not None:
            return self.pca.transform(X_scaled)
        return X_scaled

    def _build_pca(self, X_train_scaled, pca_components):
        if pca_components is None:
            return None

        n_samples, n_features = X_train_scaled.shape
        max_allowed = min(
            self._max_effective_pca_dim,
            n_features,
            max(1, n_samples - 1),
        )

        if max_allowed < 2:
            return None

        if isinstance(pca_components, float):
            if not (0.0 < pca_components < 1.0):
                raise ValueError("Float pca_components must be in (0, 1)")

            probe_pca = PCA(
                n_components=min(n_features, max(1, n_samples - 1)),
                svd_solver="full",
                random_state=self.random_state,
            )
            probe_pca.fit(X_train_scaled)
            cumsum = np.cumsum(probe_pca.explained_variance_ratio_)
            k = int(np.searchsorted(cumsum, pca_components) + 1)
            k = min(max(2, k), max_allowed)

            return PCA(
                n_components=k,
                svd_solver="full",
                random_state=self.random_state,
            )

        n_components = int(pca_components)
        n_components = min(max(2, n_components), max_allowed)

        return PCA(
            n_components=n_components,
            svd_solver="auto",
            random_state=self.random_state,
        )

    def _get_pca_candidates(self, train_shape):
        n_samples, n_features = train_shape
        max_allowed = min(
            self._max_effective_pca_dim,
            n_features,
            max(1, n_samples - 1),
        )

        candidates = [None]

        for k in [8, 16, 32, 64]:
            if 2 <= k <= max_allowed:
                candidates.append(k)

        if isinstance(self.pca_components, int):
            k = min(max(2, int(self.pca_components)), max_allowed)
            candidates.append(k)
        elif isinstance(self.pca_components, float):
            candidates.append(self.pca_components)
        elif self.pca_components is None:
            candidates.append(None)

        dedup = []
        for item in candidates:
            if item not in dedup:
                dedup.append(item)
        return dedup

    def _balanced_accuracy(self, y_true, y_pred):
        y_true = np.asarray(y_true).astype(int)
        y_pred = np.asarray(y_pred).astype(int)

        pos_mask = y_true == 1
        neg_mask = y_true == 0

        if pos_mask.sum() == 0:
            tpr = 1.0
        else:
            tpr = float((y_pred[pos_mask] == 1).mean())

        if neg_mask.sum() == 0:
            tnr = 1.0
        else:
            tnr = float((y_pred[neg_mask] == 0).mean())

        return 0.5 * (tpr + tnr)

    def _find_best_threshold_balanced(self, y_true, probs):
        y_true = np.asarray(y_true).astype(int)
        probs = np.asarray(probs, dtype=np.float32).reshape(-1)

        if len(probs) == 0:
            return 0.5

        candidates = np.unique(np.clip(probs, 0.0, 1.0))
        candidates = np.concatenate(
            [
                np.array([0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float32),
                candidates.astype(np.float32),
            ]
        )
        candidates = np.unique(np.clip(candidates, 0.0, 1.0))

        best_thr = 0.5
        best_score = float("-inf")

        for thr in candidates:
            preds = (probs >= thr).astype(int)
            score = self._balanced_accuracy(y_true, preds)

            better = False
            if score > best_score + 1e-12:
                better = True
            elif abs(score - best_score) <= 1e-12:
                if abs(float(thr) - 0.5) < abs(best_thr - 0.5):
                    better = True

            if better:
                best_score = score
                best_thr = float(thr)

        return float(best_thr)

    def _check_is_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError("HallucinationProbe is not fitted yet.")
