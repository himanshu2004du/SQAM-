"""Feature selection (unsupervised): drop constant and highly correlated features.
Scaling is done inside the model pipeline so it is fit on training folds only."""
import numpy as np
import pandas as pd

META = ["release", "release_idx", "file", "defective"]


def feature_columns(df):
    return [c for c in df.columns if c not in META and pd.api.types.is_numeric_dtype(df[c])]


def prune_features(df, corr_threshold=0.95):
    feats = [c for c in feature_columns(df) if df[c].nunique() > 1]
    if len(feats) < 2:
        return feats
    corr = df[feats].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop = [c for c in upper.columns if (upper[c] > corr_threshold).any()]
    return [c for c in feats if c not in drop]
