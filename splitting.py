"""
splitting.py — Train / validation / test split utilities (student-implementable).

``split_data`` receives the label array ``y`` and, optionally, the full
DataFrame ``df`` (for group-aware splits).  It must return a list of
``(idx_train, idx_val, idx_test)`` tuples of integer index arrays.

Contract
--------
* ``idx_train``, ``idx_val``, ``idx_test`` are 1-D NumPy arrays of integer
  indices into the full dataset.
* ``idx_val`` may be ``None`` if no separate validation fold is needed.
* All indices must be non-overlapping; together they must cover every sample.
* Return a **list** — one element for a single split, K elements for k-fold.
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split


def split_data(
    y,
    df=None,
    n_splits=5,
    val_size=0.10,
    test_size=0.20,
    random_state=42,
):
    """
    Returns a list of (train_idx, val_idx, test_idx) tuples.

    Strategy:
    - If the dataset is large enough, use stratified outer k-fold test splits.
    - Inside each outer training partition, carve out a stratified validation split.
    - If the dataset is too small for stable k-fold stratification, fall back to
      a single stratified train/val/test split.
    """
    _ = df  # kept for compatibility with the provided infrastructure

    y = np.asarray(y).astype(int)
    idx = np.arange(len(y))

    unique, counts = np.unique(y, return_counts=True)
    if len(unique) < 2:
        raise ValueError("Need at least two classes for stratified splitting.")

    min_class_count = counts.min()

    # Prefer 5-fold CV when possible, otherwise reduce the number of folds.
    effective_splits = min(n_splits, int(min_class_count))

    # Fallback to one stratified split if k-fold is not feasible.
    if effective_splits < 2:
        train_val_idx, test_idx = train_test_split(
            idx,
            test_size=test_size,
            stratify=y,
            random_state=random_state,
        )

        if val_size > 0:
            relative_val_size = val_size / (1.0 - test_size)
            train_idx, val_idx = train_test_split(
                train_val_idx,
                test_size=relative_val_size,
                stratify=y[train_val_idx],
                random_state=random_state,
            )
        else:
            train_idx = train_val_idx
            val_idx = None

        return [(train_idx, val_idx, test_idx)]

    skf = StratifiedKFold(
        n_splits=effective_splits,
        shuffle=True,
        random_state=random_state,
    )

    splits = []

    for fold_id, (train_val_idx, test_idx) in enumerate(skf.split(idx, y)):
        if val_size > 0:
            outer_test_fraction = len(test_idx) / len(y)
            relative_val_size = val_size / max(1e-8, (1.0 - outer_test_fraction))

            # Keep validation size in a safe range.
            relative_val_size = float(np.clip(relative_val_size, 0.05, 0.30))

            train_idx, val_idx = train_test_split(
                train_val_idx,
                test_size=relative_val_size,
                stratify=y[train_val_idx],
                random_state=random_state + fold_id,
            )
        else:
            train_idx = train_val_idx
            val_idx = None

        splits.append((train_idx, val_idx, test_idx))

    return splits
