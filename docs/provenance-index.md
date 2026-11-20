# The provenance index — schema and rebuild rules (watermarking v2 §4)

A **derived, rebuildable SQLite index**. Canonical truth stays in plain
files (sidecars, model cards, rights.json, tombstone records); the index
is destroyed and rebuilt without loss, and no writer ever treats it as a
place to record something that exists nowhere else.

Location: beside the Listening Space's outputs folder, never inside it:
`<outputs>/../.provenance_cache/provenance_index.sqlite3` (WAL), so the
synced `outputs/` tree never carries it. The Engine Room may open it
read-only. Never synced between machines — each machine rebuilds its own
from the files that do sync. The DDL below is `_DDL` in the Space's
`src/listening_space/provenance_index.py`, which is the truth; this page
follows it.

## DDL (index_schema_version 3)

```sql
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
-- keys: index_schema_version, and one 'tombstone:<generation_id>' per
-- tombstone whose landmarks are in (its file's mtime and size)

CREATE TABLE models (
  watermark_id INTEGER NOT NULL,        -- the run's codeword: a bucket many models share
  run_id TEXT NOT NULL,
  artist_id TEXT NOT NULL,
  adapter_hash TEXT,
  artist_name TEXT,
  consent_statement TEXT,               -- snapshot for display; card is truth
  card_path TEXT NOT NULL,
  PRIMARY KEY (artist_id, run_id)       -- a codeword names a bucket, so never the key
);
CREATE INDEX models_watermark ON models (watermark_id);

CREATE TABLE outputs (
  generation_id TEXT PRIMARY KEY,
  watermark_id INTEGER,                 -- its model's codeword
  artist_id TEXT NOT NULL,
  run_id TEXT,
  audio_path TEXT NOT NULL,
  sidecar_path TEXT NOT NULL,
  created_at_utc TEXT,
  home TEXT NOT NULL                    -- 'platform' | 'workshop'
);

CREATE TABLE legacy_ids (
  watermark_id INTEGER PRIMARY KEY,     -- a pre-v2 per-output codeword; the
  generation_id TEXT                    -- frozen list is empty since the reset
);

CREATE TABLE fingerprints (
  hash INTEGER NOT NULL,                -- landmark hash (fingerprint format v1)
  generation_id TEXT NOT NULL,
  kind TEXT NOT NULL,                   -- 'mix' | stem name | 'tombstone'
  offset_ms INTEGER NOT NULL,
  UNIQUE (hash, generation_id, kind, offset_ms)
);
CREATE INDEX fingerprints_hash ON fingerprints (hash);
CREATE INDEX fingerprints_gen ON fingerprints (generation_id);

CREATE TABLE tombstones (
  generation_id TEXT PRIMARY KEY,
  artist_id TEXT NOT NULL,
  run_id TEXT,
  watermark_id INTEGER,
  content_sha256 TEXT NOT NULL,
  deleted_at_utc TEXT NOT NULL
);
```

## Rebuild and staleness

- On open: an `index_schema_version` other than the code's → the file is
  deleted and rebuilt from scratch.
- A refresh (at start, when the served models change, and when a verify
  meets a marked file nobody indexed) scans the files with no transaction
  open and swaps the rows in one short transaction; each file's row is
  memoised by its mtime and size, so an unchanged file costs one stat.
  Torn or unreadable files are skipped and surface on a later pass.
- Fingerprinting runs as a queued post-generation step on one background
  worker, never a bulk scan on the serving hot path.
- /verify never blocks on a rebuild: it reads the current index and says
  the exact-output layer is catching up when the queue is not empty,
  instead of silently answering UNKNOWN.

## Fingerprint format

Format v1 is the contract's `khaos_attribution/fingerprint.py`: spectral
peaks paired into landmarks, each a 25-bit hash with its offset in
milliseconds; matching is aligned-offset voting.
