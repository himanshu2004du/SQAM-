"""Merge metrics and add forward-looking defect labels (SZZ-lite).

A file in release R is labelled defective (1) if a bug-fix commit touches it in release R+1.
The last release has no future window, so it is dropped.
"""
import pandas as pd


def add_labels(evo):
    fixes = evo.set_index(["release_idx", "file"])["bugfix_commits"]
    df = evo.copy()
    df["defective"] = [int(fixes.get((r + 1, f), 0) > 0) for r, f in zip(df.release_idx, df.file)]
    return df[df.release_idx < df.release_idx.max()].reset_index(drop=True)


def _merge_by_path_suffix(df, extra):
    extra = extra.assign(file=extra.file.str.replace("\\", "/", regex=False))
    mapping = {f: next((e for e in extra.file if f.endswith(e) or e.endswith(f)), None)
               for f in df.file.unique()}
    return (df.assign(_k=df.file.map(mapping))
              .merge(extra.rename(columns={"file": "_k"}), on="_k", how="left")
              .drop(columns="_k"))


def build_dataset(evo, pmd=None, coverage=None, issues=None):
    df = evo
    for extra in (pmd, coverage):
        if extra is not None and len(extra):
            df = _merge_by_path_suffix(df, extra)
    if issues:
        df["bug_issues_in_release"] = df.release.map(issues).fillna(0)
    return add_labels(df).fillna(0)
