# Khaos Attribution — working notes

## What this repo is

The contract package the other three repos pin by tag: rights schemas, the estimator, the watermark codec, the seal. Every change is a version; the Space pins the tag in pyproject and the Workshop installs a copy of the sibling checkout, never an editable one, and pins its version in `src/common/contract.py`.

## Comments and docstrings (house rule)

Code here is read by professionals; commentary earns its place or goes.

- A comment says **why**, in one line, only where the reason is not obvious from the code. It never restates the code.
- No dates, no review tags, no attributions, no stories about past bugs. Git history holds history; the code holds the rule.
- Docstrings: one sentence, at most three lines. An Args/Returns block only for a parameter that is not obvious. Module docstrings at most eight lines.
- A constraint that must not be broken gets one or two plain sentences, stated as the rule, not as the incident that taught it.
- Tests: the docstring says what the test guards, in one or two lines.
- Section dividers are one line. No essay blocks, no "(2026-…)", no "(review)", no "(the artist, …)".

A whole-codebase pass on 2026-09-14 removed about 20,000 lines of such commentary across the four repos. Do not put it back.

## Tests

`bash scripts/gate.sh pytest` runs the suite (`.venv/bin/python -m pytest -q`); `--changed` runs what a change touches, once per branch before its PR; the Stop hook in `.claude/settings.json` runs only `--quick` (the test files for what changed since its last run that was not red, within 90 seconds; any it had to skip are counted at every stop until `--changed` or the suite has run them green). The `release` skill makes a version.

Sweeps search with `rg` or `git ls-files`, never a bare `find`; any `find` prunes `data/`.

## Engine descriptors (house rule)

A key added to an engine's `train` defaults replaces that setting on every run and plan that pinned it without naming it in `engine_train_override`, and the name list is part of a plan's hash; a recipe's extra settings go in its `recipes` entry instead, and a resume carries them from the run's record.

## Releases (house rule)

`main` takes a commit only through a pull request whose `pytest` check passed; the branch protection holds everyone to it, admins included. A release is the annotated tag `v<version>` on its pull request's merge commit: the `release` skill's first half opens the `release/<version>` pull request, and its second half tags the merge.

## Licence (house rule)

Khaos is free to use and is NOT open source. Never describe any Khaos repository, package, app or document as open source, open-sourced or MIT-licensed; say "free to use" (and, for the public contract, "source available"). Every repository's `LICENSE` is the Khaos Free Use Licence. Third-party components and the base model keep their own licences and are described as theirs.
