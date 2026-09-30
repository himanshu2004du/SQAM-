"""Random Forest with chronological expanding-window validation (train on releases < i, test on i)."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_model(seed=42):
    return make_pipeline(StandardScaler(),
                         RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=seed))


def expanding_window_cv(df, features, min_train_releases=1, seed=42):
    idxs = sorted(df.release_idx.unique())
    rows, importances = [], []
    for i in idxs[min_train_releases:]:
        tr, te = df[df.release_idx < i], df[df.release_idx == i]
        if tr.defective.nunique() < 2 or len(te) == 0:
            rows.append({"test_release_idx": i, "note": "skipped (single-class train or empty test)"})
            continue
        model = make_model(seed).fit(tr[features], tr.defective)
        pred = model.predict(te[features])
        proba = model.predict_proba(te[features])[:, 1]
        rows.append({
            "test_release_idx": i, "n_train": len(tr), "n_test": len(te),
            "precision": precision_score(te.defective, pred, zero_division=0),
            "recall": recall_score(te.defective, pred, zero_division=0),
            "f1": f1_score(te.defective, pred, zero_division=0),
            "mcc": matthews_corrcoef(te.defective, pred),
            "auc": roc_auc_score(te.defective, proba) if te.defective.nunique() > 1 else np.nan,
        })
        importances.append(model[-1].feature_importances_)
    imp = (pd.Series(np.mean(importances, axis=0), index=features).sort_values(ascending=False)
           if importances else pd.Series(dtype=float))
    return pd.DataFrame(rows), imp
