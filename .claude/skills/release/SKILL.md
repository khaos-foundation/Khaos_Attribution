---
name: release
description: Release a new contract version in two halves - a release/<version> branch with the version commit and its pull request, then, once that is merged, the annotated tag on its merge commit. The pull request is merged when its check is green and the tag is pushed; neither waits for the operator. Use when a contract change is ready to become vX.Y.Z.
argument-hint: <version> <one-line subject> | tag <version>
---

# Release the contract

`main` takes a commit only through a pull request whose `pytest` check passed, so a release is two halves: the pull request, merged once its check is green, then the tag. Neither waits for the operator; stop only for a red check that cannot be fixed.

## First half: `<version> <one-line subject>`

`$0` is the new version, `X.Y.Z`; the rest of `$ARGUMENTS` is the release's one-line subject.

1. **Refuse** unless all hold, and say which failed:
   - the working tree is clean (`git status --porcelain` prints nothing);
   - `main` is checked out and, after `git fetch origin`, equal to `origin/main`;
   - `$0` is `X.Y.Z` and newer than `version` in `pyproject.toml`;
   - no tag `v$0` and no branch `release/$0` exists here or on `origin` (`git ls-remote origin refs/tags/v$0 refs/heads/release/$0`).
2. **Branch.** `git switch -c release/$0`.
3. **Test.** `bash scripts/gate.sh pytest` must pass, then `bash scripts/gate.sh torch` must print `PASS torch` with its counts: it runs the suite again, with this checkout's `src` first on the path, under an interpreter that has torch and every package the torch tests need (`KHAOS_TORCH_PYTHON` when set, else the first of this checkout's venv, `python3` and `python` on PATH that has them), so the torch tests the first gate skips run. **Refuse** to go on without both PASS lines from this release's branch. When the torch gate prints FAIL because no interpreter has them, look for one on this machine, such as a sibling room's venv beside this checkout (`../<room>/.venv/bin/python`), and run `KHAOS_TORCH_PYTHON=<that python> bash scripts/gate.sh torch` again; stop and say so only when there is none. Report both PASS lines with their counts.
4. **Set the version** in exactly two places: `version = "$0"` in `pyproject.toml` and `__version__ = "$0"` in `src/khaos_attribution/__init__.py`, then run `uv lock` (never `--upgrade`) so `uv.lock` records it. Run `.venv/bin/python -m pytest -q tests/test_version.py`; it holds the three equal.
5. **Commit** the three files on the branch with the subject `$0: <subject>`, a body saying what the version changes, and the session's attribution trailer. Pass the message on stdin (`git commit -F -`) and treat the commit as made only on exit code 0.
6. **Open the pull request.** `git push -u origin release/$0`, then `gh pr create --base main --head release/$0 --title "$0: <subject>" --body-file -` with the commit's body and both gates' PASS lines with their counts, ending with the session's pull-request attribution line.
7. **Merge on green.** Print the pull request's URL and watch its `pytest` check in the background (`gh pr checks <number> --watch`); do not hold the turn for it. A red check is fixed on the branch, never merged over. When it is green, `gh pr merge <number> --merge --delete-branch`, then run the second half; nothing is tagged before the merge.

## Second half: `tag <version>`

`$1` is the version whose pull request was merged.

1. **Find the merge.** `gh pr list --state merged --head release/$1 --json number,title,mergeCommit` must name exactly one pull request; its `mergeCommit.oid` is the commit to tag and its title, after `$1: `, is the subject. Refuse when there is none, or when a tag `v$1` already exists here or on `origin`.
2. **Bring main up.** `git switch main && git pull --ff-only`, then `git merge-base --is-ancestor <merge commit> main` must succeed.
3. **Check the merge commit is the version.** `git show <merge commit>:pyproject.toml` holds `version = "$1"` and `git show <merge commit>:src/khaos_attribution/__init__.py` holds `__version__ = "$1"`.
4. **Tag** `git tag -a v$1 -m "<subject>" <merge commit>`: annotated, the subject alone as its message. Then, where the merge left the branch here, `git branch -d release/$1`; it is merged, so `-d` takes it.
5. **Push the tag.** `git push origin v$1`, then say in one line that `v$1` is released. Pinning the new tag elsewhere is not this repository's job.
