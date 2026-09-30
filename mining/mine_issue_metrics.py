"""ISSUE metrics: bug-labelled issues opened per release window (repo-level) via GitHub REST API."""
from datetime import datetime

import requests


def mine_issue_metrics(owner, repo, releases, token=None, label="bug"):
    """releases: list of (name, datetime) sorted by date. Returns {release_name: bug_issue_count}."""
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    issues, page = [], 1
    while True:
        r = requests.get(f"https://api.github.com/repos/{owner}/{repo}/issues",
                         headers=headers, timeout=30,
                         params={"state": "all", "labels": label, "per_page": 100, "page": page})
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        issues += [i for i in batch if "pull_request" not in i]
        page += 1
    dates = [datetime.fromisoformat(i["created_at"].replace("Z", "+00:00")) for i in issues]
    out, prev = {}, None
    for name, d in releases:
        out[name] = sum(1 for x in dates if (prev is None or x > prev) and x <= d)
        prev = d
    return out
