---
name: release
description: Release a new contract version - set the version, commit, and make the annotated tag, stopping before the push. Use when a contract change is ready to become vX.Y.Z.
argument-hint: <version> <one-line subject>
---

# Release the contract

`$0` is the new version, `X.Y.Z`; the rest of `$ARGUMENTS` is the release's one-line subject.

1. **Refuse** unless all hold, and say which failed:
   - the working tree is clean (`git status --porcelain` prints nothing);
   - `$0` is `X.Y.Z` and newer than `version` in `pyproject.toml`;
   - no tag `v$0` exists here or on `origin` (`git ls-remote --tags origin v$0`).
2. **Test.** `bash scripts/gate.sh pytest` must pass. Its torch-gated tests skip without torch: when an interpreter with torch is on this machine, run `<that python> -m pytest -q` too and report both counts.
3. **Set the version** in exactly two places: `version = "$0"` in `pyproject.toml` and `__version__ = "$0"` in `src/khaos_attribution/__init__.py`. Run `.venv/bin/python -m pytest -q tests/test_version.py`; it holds the two equal.
4. **Commit** both files with the subject `$0: <subject>`, a body saying what the version changes, and the session's attribution trailer. Pass the message on stdin (`git commit -F -`) and treat the commit as made only on exit code 0.
5. **Tag** `git tag -a v$0 -m "<subject>"`: annotated, the subject alone as its message.
6. **Stop before pushing.** Print `git push origin <branch> v$0` and leave it for the operator. Pinning the new tag elsewhere is not this repository's job.
