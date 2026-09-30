"""Evolution metrics (MOSER, HASSAN, CHANGE_TYPE) mined per file per release with PyDriller.

Releases are git tags ordered by date. Each commit is assigned to the first release
whose tagged commit date is >= the commit date. Commits after the last tag are ignored.
"""
import math
import re
from collections import defaultdict

import pandas as pd
from git import Repo
from pydriller import Repository

BUGFIX_RE = re.compile(r"\b(fix(e[sd])?|bug(s)?|defect|patch|fault|error)\b", re.I)


def get_releases(repo_path):
    repo = Repo(repo_path)
    rel = [(t.name, t.commit.committed_datetime) for t in repo.tags]
    return sorted(rel, key=lambda x: x[1])


def _entropy(sizes):
    total = sum(sizes)
    if total == 0:
        return 0.0
    return -sum((s / total) * math.log2(s / total) for s in sizes if s > 0)


def mine_evolution_metrics(repo_path, extensions=(".java",)):
    releases = get_releases(repo_path)
    if len(releases) < 2:
        raise ValueError("Need at least 2 tags (releases) in the repository.")
    rel_dates = [d for _, d in releases]

    acc = defaultdict(lambda: {
        "revisions": 0, "authors": set(), "lines_added": 0, "lines_deleted": 0,
        "bugfix_commits": 0, "hassan_entropy": 0.0,
        "ct_add": 0, "ct_modify": 0, "ct_delete": 0, "ct_rename": 0,
    })
    first_seen = {}

    for commit in Repository(repo_path).traverse_commits():
        idx = next((i for i, d in enumerate(rel_dates) if commit.committer_date <= d), None)
        if idx is None:
            continue
        mods = [m for m in commit.modified_files
                if (m.new_path or m.old_path or "").endswith(tuple(extensions))]
        if not mods:
            continue
        h = _entropy([m.added_lines + m.deleted_lines for m in mods])
        is_fix = bool(BUGFIX_RE.search(commit.msg or ""))
        for m in mods:
            path = m.new_path or m.old_path
            first_seen.setdefault(path, commit.committer_date)
            a = acc[(idx, path)]
            a["revisions"] += 1
            a["authors"].add(commit.author.email)
            a["lines_added"] += m.added_lines
            a["lines_deleted"] += m.deleted_lines
            a["hassan_entropy"] += h
            a["bugfix_commits"] += int(is_fix)
            ct = m.change_type.name.lower()
            key = {"add": "ct_add", "delete": "ct_delete", "rename": "ct_rename"}.get(ct, "ct_modify")
            a[key] += 1

    rows = []
    for (idx, path), a in acc.items():
        name, date = releases[idx]
        rows.append({
            "release": name, "release_idx": idx, "file": path,
            "revisions": a["revisions"], "authors": len(a["authors"]),
            "lines_added": a["lines_added"], "lines_deleted": a["lines_deleted"],
            "churn": a["lines_added"] + a["lines_deleted"],
            "age_days": (date - first_seen[path]).days,
            "hassan_entropy": a["hassan_entropy"],
            "ct_add": a["ct_add"], "ct_modify": a["ct_modify"],
            "ct_delete": a["ct_delete"], "ct_rename": a["ct_rename"],
            "bugfix_commits": a["bugfix_commits"],
        })
    return pd.DataFrame(rows).sort_values(["release_idx", "file"]).reset_index(drop=True)
