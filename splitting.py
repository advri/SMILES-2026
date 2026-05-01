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
from sklearn.model_selection import StratifiedKFold


def split_data(y, df=None):
    """
    Returns list of (train_idx, val_idx, test_idx)
    Here val_idx is always None.

    We intentionally remove the tiny validation split because it was too noisy
    and encouraged unstable threshold / hyperparameter tuning.
    """
    y = np.asarray(y).astype(int)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = []

    dummy_X = np.zeros(len(y))
    for train_idx, test_idx in skf.split(dummy_X, y):
        splits.append((train_idx, None, test_idx))

    return splits
