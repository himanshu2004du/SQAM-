"""End-to-end: mine -> label -> preprocess -> cluster -> train/evaluate.
Usage: python run_pipeline.py --repo /path/to/local/clone [--out outputs] [--ext .java]
Optional: --pmd pmd.xml --jacoco jacoco.xml --github owner/repo --token TOKEN
"""
import argparse
import json
import os

from dataset.build_dataset import build_dataset
from mining.mine_coverage_metrics import parse_jacoco_report
from mining.mine_evolution_metrics import get_releases, mine_evolution_metrics
from mining.mine_issue_metrics import mine_issue_metrics
from mining.mine_pmd_metrics import parse_pmd_report
from modeling.clustering_analysis import run_dbscan, run_som
from modeling.train_model import expanding_window_cv
from preprocessing.preprocess import prune_features


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--out", default="outputs")
    p.add_argument("--ext", nargs="+", default=[".java"])
    p.add_argument("--pmd"); p.add_argument("--jacoco")
    p.add_argument("--github"); p.add_argument("--token")
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    print("[1/5] Mining evolution metrics ...")
    evo = mine_evolution_metrics(a.repo, tuple(a.ext))
    pmd = parse_pmd_report(a.pmd) if a.pmd else None
    cov = parse_jacoco_report(a.jacoco) if a.jacoco else None
    issues = None
    if a.github:
        owner, repo = a.github.split("/")
        issues = mine_issue_metrics(owner, repo, get_releases(a.repo), a.token)

    print("[2/5] Building dataset + labels ...")
    df = build_dataset(evo, pmd, cov, issues)
    df.to_csv(f"{a.out}/dataset.csv", index=False)
    print(f"      {len(df)} rows, defect rate {df.defective.mean():.2%}")

    print("[3/5] Preprocessing ...")
    feats = prune_features(df)
    print(f"      {len(feats)} features kept: {feats}")

    print("[4/5] Clustering (RQ4) ...")
    clus = {"dbscan": run_dbscan(df[feats].values, df.defective.values, min_samples=min(5, max(2, len(df) // 10))),
            "som": run_som(df[feats].values, df.defective.values)}
    with open(f"{a.out}/clustering.json", "w") as f:
        json.dump(clus, f, indent=2, default=str)

    print("[5/5] Chronological expanding-window evaluation ...")
    res, imp = expanding_window_cv(df, feats)
    res.to_csv(f"{a.out}/cv_results.csv", index=False)
    imp.to_csv(f"{a.out}/feature_importance.csv", header=["importance"])
    print(res.to_string(index=False))
    print("Done. Results in", a.out)


if __name__ == "__main__":
    main()
