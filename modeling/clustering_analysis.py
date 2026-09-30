"""RQ4: are defective and clean files separable? DBSCAN + a small numpy SOM."""
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def run_dbscan(X, y, eps=1.5, min_samples=5):
    Xs = StandardScaler().fit_transform(X)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(Xs)
    mask = labels != -1
    sil = silhouette_score(Xs[mask], labels[mask]) if len(set(labels[mask])) > 1 else float("nan")
    purity = []
    for c in set(labels[mask]):
        yc = np.asarray(y)[labels == c]
        purity.append(max(yc.mean(), 1 - yc.mean()))
    return {"n_clusters": len(set(labels[mask])), "noise_fraction": float((~mask).mean()),
            "silhouette": float(sil),
            "mean_label_purity": float(np.mean(purity)) if purity else float("nan")}


def run_som(X, y, grid=(5, 5), epochs=200, lr=0.5, seed=0):
    rng = np.random.default_rng(seed)
    Xs = StandardScaler().fit_transform(X)
    gx, gy = grid
    W = rng.normal(size=(gx, gy, Xs.shape[1]))
    coords = np.stack(np.meshgrid(np.arange(gx), np.arange(gy), indexing="ij"), -1)
    sigma0 = max(gx, gy) / 2
    for t in range(epochs):
        frac = t / epochs
        lr_t, sig = lr * (1 - frac), sigma0 * (1 - frac) + 0.5
        x = Xs[rng.integers(len(Xs))]
        bmu = np.unravel_index(np.argmin(((W - x) ** 2).sum(-1)), (gx, gy))
        h = np.exp(-((coords - np.array(bmu)) ** 2).sum(-1) / (2 * sig ** 2))
        W += lr_t * h[..., None] * (x - W)
    bmus = [np.unravel_index(np.argmin(((W - x) ** 2).sum(-1)), (gx, gy)) for x in Xs]
    cells = {}
    for b, lab in zip(bmus, np.asarray(y)):
        cells.setdefault(tuple(int(i) for i in b), []).append(lab)
    return {"occupied_cells": len(cells),
            "cell_defect_rate": {str(k): float(np.mean(v)) for k, v in cells.items()},
            "quantization_error": float(np.mean([np.linalg.norm(x - W[b]) for x, b in zip(Xs, bmus)]))}
