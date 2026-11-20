# The Khaos provenance system — how it works today

Current-state reference. The *decisions and their reasons* live in
`watermarking-v2.md`; this page describes the running system at the
contract version in `pyproject.toml`.

## In one paragraph

Every piece of audio a Khaos model generates carries an inaudible
AudioSeal watermark naming **the bucket of its model**: a number derived
from the machine and the training run, which many models share, so the
mark narrows a check to a few models and never names one alone. The Guild
Register's serial names the model, and through it the artist, their
consent record and the rights of the training catalogue. Each file also
carries signed Content Credentials, its second mark. A content-based layer
(landmark fingerprints) identifies the **exact output**, including audio
whose watermark was stripped or degraded and outputs since deleted. A
piece that cannot be marked is refused: nothing is stored. Anyone holding a
file can ask a Listening Space's **/verify** page, the Register, or the
`khaos-verify` command.

## The watermark

- **Carrier**: AudioSeal (Meta, MIT), 16-bit message, embedded as a
  resampled delta so the music itself is never resampled.
- **Payload**: an extended-Hamming (16,11) SECDED codeword — 2048 usable
  IDs, single-bit errors corrected, double-bit errors refused rather than
  misread. The codeword **is** the `watermark_id` in every record.
- **What one ID means**: a **bucket** of training runs. All outputs of a
  run carry its bucket's codeword — Space pieces, Workshop takes,
  variations, repaints, extends; exported stems re-embed their source
  output's codeword. A retrain is a new model, usually in another bucket.
  A codeword is never a register number.

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

## How a run's codeword is chosen

- **Derived, never allocated.** At training completion the Workshop
  derives the run's bucket from its machine key and run id
  (`watermark.derive_payload`: a hash into the payloads 32–2047; 0–31 are
  reserved for calibration) and stamps the codeword into
  `model_card.json` (`watermark_id`; the manifest says
  `watermark_source: derived`). Nothing is coordinated between machines
  and nothing can run out.
- **The Register's serial names the model.** Publishing a model gives it
  a serial (from 65536), countersigned by the Register; the seal's ring
  reads that serial, never the codeword.
- **Earlier schemes.** Per-output IDs and per-machine bands came before;
  the 2026-08-22 reset freed every per-output ID (`watermark_resets.json`),
  and the frozen list `legacy_watermark_ids.json` is empty. A card
  stamped from a band still reads, as its own number.
- The card travels with the adapter, so every machine learns a model's
  codeword by reading the card.

## Life of an output

1. Training completes → run gets `watermark_id` in its model card.
2. Generation (either room) reads the card and embeds that codeword,
   then signs the file's Content Credentials. A card without an ID is
   **refused**: nothing is rendered or stored (the remedy is below).
3. The provenance sidecar records the ID; the Platform's index refreshes
   and queues the audio for fingerprinting (one background worker).
4. Deletion writes a **tombstone** first —
   `outputs/<artist>/.tombstones/<id>.json` with the content hash and
   fingerprints, no audio — then removes the audio and sidecar. Delete
   removes the sound, not the fact it existed.

## What /verify answers

| situation | answer |
|---|---|
| watermark decodes to a bucket this machine serves | the models of that bucket here, **artist-led** once a content match names the exact output: "Made with the *Artist* adapter — consent on file · N catalogue tracks · run + date" |
| ID freed by the 2026-08-22 reset | "freed by the reset", never whatever model the ID is issued to later |
| valid ID, no card on this machine | "watermark confirmed; this machine doesn't have the model's card yet — sync adapters" |
| excerpt (payload mismatches the matched output) | the full artist-led answer via content, flagged "the embedded ID didn't survive this copy — identity confirmed by content match" |
| no watermark, content matches | "No watermark detected, but this audio matches output X (N aligned landmarks)" — always a match with numbers, never a definitive claim |
| deleted output | any of the above, flagged "since deleted — identified from its tombstone" |
| fingerprinting can't read the file | says so ("content matching could not read this file format") instead of pretending no match |

/verify works even on a machine without audioseal installed — content
matching stands alone.

Audio you submit for checking is deleted as soon as the check finishes. We keep standard server logs for 30 days. This holds for a Listening Space's /verify and for the register's
`POST /c2pa/v1/matches/byContent`, the two places that take audio.

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
runner-up candidate, unless the runners-up are near-copies of it (a
same-seed re-render) and exactly one of them was stored with the query
file's sha256: that one is named (the Space's index, Space decision 0007;
the Register since 0.53.1). Proven against excerpts and 128k MP3. Known
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

- **A model is refused for "no watermark ID"** →
  `PYTHONPATH=src .venv/bin/python scripts/backfill_model_watermarks.py`
  in the Workshop (dry-run; add `--apply`), then sync the card.
- **Survival testing** →
  `Khaos_Attribution/scripts/watermark_survival.py <wav>…` (uses the
  reserved payloads; never run it against production IDs).
- **Env knob**: `KHAOS_WORKSHOP_ARTISTS` (path the Space's index scans
  for auditions; empty string disables — tests use this to stay hermetic).

## Pointer map

| thing | where |
|---|---|
| contract (codec, derivation, fingerprints, schemas, verifier) | `Khaos_Attribution/src/khaos_attribution/{watermark,fingerprint,validation,verify}.py` |
| design decisions | `Khaos_Attribution/docs/watermarking-v2.md` |
| index DDL + rebuild rules | `Khaos_Attribution/docs/provenance-index.md` |
| derivation + backfill | Workshop `src/common/register.py`, `src/training/training.py`, `scripts/backfill_model_watermarks.py` |
| embedding at generation | Platform `server.py` runner; Workshop `scripts/infer.py` |
| /verify | Platform `routers/system.py` + `static/shell.js` |
| index + worker + tombstones | Platform `provenance_index.py`, `routers/outputs.py` |
| Engine Room coverage check | `khaos_diag/collectors/shared.py`, `checks.py` |
| the Guild Register (serials, the index of released pieces, the signed log) | `Khaos_Attribution/src/register_service/`, `docs/registry-service.md` |
