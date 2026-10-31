# The Khaos provenance system — how it works today

Current-state reference (2026-08-20). The *decisions and their reasons*
live in `watermarking-v2.md`; this page describes the running system.
Contract version: khaos-attribution **0.17.0** (the live version is pyproject's).

## In one paragraph

Every piece of audio a Khaos model generates carries an inaudible
AudioSeal watermark identifying **the model that made it** — the training
run, and through it the artist, their consent record, and the rights of
the training catalogue. Anyone holding a file can upload it to the
Listening Space's **/verify** page and get that answer. A second,
content-based layer (landmark fingerprints) identifies the **exact
output** — including audio whose watermark was stripped or degraded, and
outputs that have since been deleted. The whole thing is backed by plain
files that sync between machines; the only database is a disposable local
index rebuilt from them.

## The watermark

- **Carrier**: AudioSeal (Meta, MIT), 16-bit message, embedded as a
  resampled delta so the music itself is never resampled.
- **Payload**: an extended-Hamming (16,11) SECDED codeword — 2048 usable
  IDs, single-bit errors corrected, double-bit errors refused rather than
  misread. The codeword **is** the `watermark_id` in every record.
- **What one ID means**: a **training run** (adapter version). All
  outputs of a run share its ID — Platform outputs, Workshop auditions,
  variations, repaints, extends. Exported stems re-embed their source
  output's codeword. A retrain is a new model and a new ID.

### What survives, and the rule that follows

Codecs and resampling keep both the presence and the ID. A short
excerpt keeps the presence but can decode the wrong ID, and it does so
consistently, so no decode-side gate can catch it. /verify therefore
resolves **content first**: a fingerprint match names the exact output
and its run, and the decoded payload is reported as corroboration —
confirmed, "didn't survive this copy", or absent. A raw ID resolves
alone only when no content evidence exists (worded "uncorroborated").
Never embed a second watermark over a marked file; two messages
superimpose into garbage. Stems re-embedding the *same* codeword is
fine. Removal by an adversary is out of scope: this is provenance, not
DRM.

## How IDs are allocated

- **At training completion**, the Workshop allocates the run's codeword
  and stamps it into `model_card.json` (`watermark_id`). Failure to
  allocate fails the run loudly — no silent unmarked adapters. Stub/
  synthetic runs never spend an ID.
- The 2048-payload space is split into **per-machine bands** so three
  machines allocate without coordination: reserved 0–31 (calibration),
  laptop 32–703, studio 704–1375, threadripper 1376–2047. Band comes
  from the machine profile; `KHAOS_WM_BAND` overrides.
- **23 legacy IDs** (the retired per-output scheme) are frozen in the
  contract's `legacy_watermark_ids.json` and permanently excluded — no
  ID is ever owned by both an old output and a model. Old outputs keep
  their marks and resolve via this list forever.
- The card travels with the adapter, so every machine learns a model's
  ID by reading the card. **Sync rule**: all machines take the contract
  pin bump *before* stamped cards are synced to them.

## Life of an output

1. Training completes → run gets `watermark_id` in its model card.
2. Generation (either app) reads the card and embeds that codeword.
   A card without an ID generates anyway — **loudly** warned, sidecar
   records `watermark_id: null` (remedy below).
3. The provenance sidecar records the ID; the Platform's index refreshes
   and queues the audio for fingerprinting (one background worker).
4. Deletion writes a **tombstone** first —
   `outputs/<artist>/.tombstones/<id>.json` with the content hash and
   fingerprints, no audio — then removes the audio and sidecar. Delete
   removes the sound, not the fact it existed.

## What /verify answers

| situation | answer |
|---|---|
| watermark decodes to a served model | **artist-led**: "Made with the *Artist* adapter — consent on file · N catalogue tracks · run + date", plus the exact-output match when content confirms one |
| ID is one of the 23 frozen legacy IDs | the original per-output answer (Platform record, or "Workshop test render") |
| valid ID, no card on this machine | "watermark confirmed; this machine doesn't have the model's card yet — sync adapters" |
| excerpt (payload mismatches the matched output) | the full artist-led answer via content, flagged "the embedded ID didn't survive this copy — identity confirmed by content match" |
| no watermark, content matches | "No watermark detected, but this audio matches output X (N aligned landmarks)" — always a match with numbers, never a definitive claim |
| deleted output | any of the above, flagged "since deleted — identified from its tombstone" |
| fingerprinting can't read the file | says so ("content matching could not read this file format") instead of pretending no match |

/verify works even on a machine without audioseal installed — content
matching stands alone.

**Rights ride the answer** (2026-08-20): an identified output brings its
computed attribution estimate's splits — writers, publishers, masters ℗,
unattributed % — when the estimate exists; its absence is stated (with
the Compute remedy), a Workshop audition says estimates are computed for
Listening Space outputs, and a deleted output says its estimate went
with it. A model-level answer states the catalogue's rights coverage counted
from the artist bundle, never just asserted.

## The fingerprint layer

In-house landmark constellation (`khaos_attribution/fingerprint.py`,
format v1, numpy-only): spectral peaks → paired landmarks → 25-bit
hashes; matching is aligned-offset voting. A confident match needs ≥12
aligned votes, ≥5% of the query's landmarks, ≥8 distinct hashes,
**prominence** — the winning offset at 1.5× the best offset outside its
2 s neighbourhood — **and dominance** — the winner must double the
runner-up candidate. Proven against excerpts and 128k MP3. Known
limitations, documented in the contract: same-tempo percussive loops are
near-identical to this whole algorithm class, and dominance is what
separates the true source; a sustained chord aligns at any offset, and
prominence is what separates a copied passage from the same chord played
again.

## The provenance index

`Khaos_Platform…/.provenance_cache/provenance_index.sqlite3` — a
**derived, rebuildable** SQLite (WAL). It is *never* the source of truth
and *never* syncs between machines (it sits outside the synced
`outputs/` tree; each machine rebuilds its own from the files that do
sync). Tables: models, outputs, legacy_ids, fingerprints, tombstones.
Refresh is a file scan swapped in one short transaction — a row whose
canonical file is gone simply isn't re-inserted. One background worker
fingerprints queued audio; /verify never blocks on it and says
"catching up" when it lags. A verify that decodes a watermark but finds
no content match triggers one on-demand rescan, so a fresh Workshop
audition becomes matchable without a server restart (its first ask may
still say "catching up"; the next one matches). Delete the file any time; the next server
start rebuilds it.

The **Engine Room** reads the index (strictly read-only) and reports:
unstamped served adapters (with the backfill remedy), index
missing/unreadable, fingerprint lag, or the healthy census.

## Operator how-tos

- **Adapter warns "no watermark_id"** →
  `PYTHONPATH=src .venv/bin/python scripts/backfill_model_watermarks.py`
  in the Workshop (dry-run; add `--apply`), then sync the card.
- **Band exhausted at training completion** → the run fails with the
  message; bands hold 672 runs per machine, so investigate before
  expanding anything.
- **Survival testing** →
  `Khaos_Attribution/scripts/watermark_survival.py <wav>…` (uses the
  reserved band; never run it against production IDs).
- **Env knobs**: `KHAOS_WM_BAND` (allocation band override),
  `KHAOS_WORKSHOP_ARTISTS` (path the Platform index scans for auditions;
  empty string disables — tests use this to stay hermetic).

## Current gaps (tracked)

- The Wwise-conversion leg of the survival test waits for the Wwise MCP
  (Vorbis, Wwise's music codec, already passes).
- Stem exports aren't fed to the fingerprint queue yet (the index schema
  is ready; mixes are covered).
- The Studio's Workshop run card for `chris_green_theleap` is unstamped
  until the backfill runs there (its Platform card is stamped).

## Pointer map

| thing | where |
|---|---|
| contract (codec, allocation, fingerprints, schemas) | `Khaos_Attribution/src/khaos_attribution/{watermark,fingerprint,validation}.py` |
| design decisions | `Khaos_Attribution/docs/watermarking-v2.md` |
| index DDL + rebuild rules | `Khaos_Attribution/docs/provenance-index.md` |
| allocation + backfill | Workshop `src/training/training.py`, `scripts/backfill_model_watermarks.py` |
| embedding at generation | Platform `server.py` runner; Workshop `scripts/infer.py` |
| /verify | Platform `routers/system.py` + `static/shell.js` |
| index + worker + tombstones | Platform `provenance_index.py`, `routers/outputs.py` |
| Engine Room coverage check | `khaos_diag/collectors/shared.py`, `checks.py` |

## Amendment — the 2026-08-22 reset

The "23 legacy IDs … permanently excluded" above was true until
2026-08-22, when the operator deleted every output ever made and reset the
ID space (`watermarking-v2.md` §3a). The frozen list is empty now; the
history of what was freed lives in `watermark_resets.json`, and `/verify`
answers "freed by the reset" for those IDs instead of the per-output
answer described here.
