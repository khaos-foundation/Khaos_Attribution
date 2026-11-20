# Khaos Attribution — working notes

## What this repo is

The contract the rooms share: rights schemas and their validators, the estimators, the watermark codec, Content Credentials, the seal, the Guild Register's client, and the engine socket (`src/khaos_attribution/engines/`). Every change is a version; the Space pins the tag in pyproject and the Workshop installs a copy of the sibling checkout, never an editable one, and pins its version in `src/common/contract.py`.

It also holds the Guild Register's service (`src/register_service/`, its deploy scripts in `deploy/`, its reference in `docs/registry-service.md`): one process on its own server, deployed from a tag by that tag's own `deploy/register-server.sh`.

## The Register (house rule)

- The live Register is changed only by the owner's deploy. Code and scripts here are changed freely; nothing here runs against the live server.
- Its ledger and its signed log are permanent and public: a row or an entry is never edited or removed. A withhold hides a body from view, never a leaf; a strike is a new entry.
- Whatever the register writes into the ledger or the public mirror stays in public git history for good: personal words go only where a withhold reaches them.

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

`bash scripts/gate.sh pytest` runs the suite (`.venv/bin/python -m pytest -q`); `--changed` runs what a change touches, once per branch before its PR; the Stop hook in `.claude/settings.json` runs only `--quick` (the test files for what changed since its last run that was not red, within 90 seconds; any it had to skip are counted at every stop until `--changed` or the suite has run them green). `bash scripts/gate.sh torch` runs the suite under an interpreter that has torch and what its tests need (`KHAOS_TORCH_PYTHON`, else this checkout's venv or one on PATH) and fails when there is none; it runs only by name, and the `release` skill, which makes a version, needs its PASS.

Sweeps search with `rg` or `git ls-files`, never a bare `find`; any `find` prunes `data/`, and a `find` that removes never uses `-delete` or `-depth` from a folder that holds `data/`, `artists/` or `outputs/` (both work depth first, where a prune does nothing): remove with `-exec rm` after the prune.

## Engine descriptors (house rule)

A key added to an engine's `train` defaults replaces that setting on every run and plan that pinned it without naming it in `engine_train_override`, and the name list is part of a plan's hash; a recipe's extra settings go in its `recipes` entry instead, and a resume carries them from the run's record.

## Releases (house rule)

`main` takes a commit only through a pull request whose `pytest` check passed; the branch protection holds everyone to it, admins included. A release is the annotated tag `v<version>` on its pull request's merge commit: the `release` skill's first half opens the `release/<version>` pull request, and its second half tags the merge.

## Licence (house rule)

Khaos is free to use and is NOT open source. Never describe any Khaos repository, package, app or document as open source, open-sourced or MIT-licensed; say "free to use" (and, for the public contract, "source available"). Every repository's `LICENSE` is the Khaos Free Use Licence. Third-party components and the base model keep their own licences and are described as theirs.
