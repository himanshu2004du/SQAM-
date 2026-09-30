import os
import subprocess

import pytest

from dataset.build_dataset import build_dataset
from mining.mine_evolution_metrics import mine_evolution_metrics
from run_pipeline import main


def _git(cwd, *args, date=None):
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t.com",
               GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t.com")
    if date:
        env["GIT_AUTHOR_DATE"] = env["GIT_COMMITTER_DATE"] = date
    subprocess.run(["git", *args], cwd=cwd, env=env, check=True, capture_output=True)


@pytest.fixture()
def repo(tmp_path):
    r = str(tmp_path)
    _git(r, "init", "-q")
    day = [0]

    def commit(files, msg):
        day[0] += 1
        for name, body in files.items():
            with open(os.path.join(r, name), "a") as f:
                f.write(body)
        _git(r, "add", "-A")
        _git(r, "commit", "-q", "-m", msg, date=f"2024-01-{day[0]:02d}T12:00:00")

    commit({"Foo.java": "class Foo {}\n", "Bar.java": "class Bar {}\n"}, "init"); _git(r, "tag", "v1.0")
    commit({"Foo.java": "// fix bug in Foo\n"}, "fix bug in Foo")
    commit({"Bar.java": "// feature\n"}, "add feature"); _git(r, "tag", "v1.1")
    commit({"Bar.java": "// fix defect\n"}, "fix defect in Bar")
    commit({"Foo.java": "// more\n"}, "refactor"); _git(r, "tag", "v1.2")
    commit({"Foo.java": "// fix error\n"}, "fix error"); _git(r, "tag", "v1.3")
    return r


def test_mining_columns(repo):
    evo = mine_evolution_metrics(repo)
    assert {"revisions", "authors", "churn", "hassan_entropy", "bugfix_commits"} <= set(evo.columns)
    assert evo.release.nunique() >= 3


def test_labels_are_forward_looking(repo):
    df = build_dataset(mine_evolution_metrics(repo))
    row = df[(df.release == "v1.0") & (df.file == "Foo.java")]
    assert row.defective.iloc[0] == 1
    assert "v1.3" not in set(df.release)


def test_full_pipeline(repo, tmp_path):
    out = str(tmp_path / "out")
    main(["--repo", repo, "--out", out])
    assert os.path.exists(os.path.join(out, "dataset.csv"))
    assert os.path.exists(os.path.join(out, "cv_results.csv"))
