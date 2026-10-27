# The registry service — leased IDs, central fingerprints, one consent root

> **Superseded in part, 0.26.0 (October 2026):** leases were never built. What shipped is `khaos_attribution/registry.py` and `register_service/`: a serial per model from 65,536, countersigned records, an outbox, enrolment by invite, withdrawal; the watermark payload is a derived bucket. §3's leases and §7's phase 4 do not apply; the rest of the shape stands. The plan of record is the Workshop's `docs/plans/register-and-publish.md`.

> **0.45.0 (October 2026): the signed append-only log.** Every admission and strike is an entry in a Merkle log in the Certificate Transparency construction (`khaos_attribution/translog`, `register_service/translog.py`): RFC 8785 entries with a 128-bit salt, RFC 6962 leaf and node hashes, inclusion and consistency proofs verified as RFC 9162 has them, and a signed tree head (Ed25519 over the RFC 8785 JSON of `{log_id, root_sha256, timestamp, tree_size, version}`) after every append, every head kept. The log has its own key (`REGISTER_LOG_KEY_FILE`, `REGISTER_LOG_DEV_KEY`, its own ceremony `register-log-key make|switch`); its public half is committed at `src/khaos_attribution/translog/log-key.pub` and served at `GET /log/key`. `Store.register` appends an `admit` and `Store.withdraw` a `revoke` in the same transaction as the ledger row, which stays as history; on first start a `genesis` entry carries the ledger's tip hash and length, then every existing record is admitted in serial order with its own time (and struck if withdrawn). A machine enrolled with the `warden` role appends strikes, manual admits and withholds through `POST /log/append` (signed, naming the sitting); only wardens. Public reads, cached with ETags and bounded per address: `GET /log/sth`, `/log/sth/{size}`, `/log/proof/{leaf_hash}[?size=]`, `/log/consistency?from=&to=`, `/log/entries?start=&end=` (at most 1000), `/log/key`, `/log/anchors`, and the static export at `/mirror/log/...`, which is also pushed to the public mirror and is all `khaos-log-verify` and the `/verify-entry` page need. A withheld body leaves public view (and the mirror's working tree, not its git history) while every proof stays valid; the withholding is itself an entry. The newest root is stamped daily with OpenTimestamps and pending receipts upgraded four times a day (`register-log-stamp.timer`, `register-log-upgrade.timer`, `register_service/anchor.py` behind `Timestamper`); reading the register never depends on it. Verifiers trust only proved data (the leaf hashes against the signed root, every body against its leaf, records and strikes only through proved entries, a missing body only with a proved withhold or inside the public delay); a database behind its published log is blocked until `log-recover`; one standing admit per record, and a withdrawal strikes every one; exports are serialised and never move `latest.json` backwards; a daily `heartbeat` entry keeps freshness in the tree, every size has exactly one head, and verifiers warn after 48 hours; warden entries name their machine; new bodies reach the public git mirror seven days after the head that first held them, and a verifier exits 4 when held-back entries follow its target; the register fails closed when no published copy (the public repository's raw address by default) vouches for its database; `log-recover` replays a lost entry whose body was never published as a leaf alone and marks it lost; a record republished after a recovery keeps its serial. A lost entry leaves every model admitted before it unresolved (not served, verifiers exit 4) until a warden's `revoke` or `confirm`; a copy whose newest head is over 48 hours old exits 5; a witness head counts only when signed by the key that signed that size here; the unwitnessed open is a one-shot marker file refused for a log with entries; `data/log-started` (kept in backups) stops an empty log opening on "nothing published"; the switch destroys the retired development log key. Unresolved models are under review, never listed as withdrawn; a `confirm` names the lost leaf it answers (for every earlier admit, or one it `confirms`); a recovery reserves one serial per lost entry, returned only to a record presenting this register's countersignature for it. Any admit appended while a loss is open names it in `pending_lost` and stays under review until a confirm after it answers that loss; reservations are carried by the lost-entry withholds, so a later restore rebuilds them; earlier countersigning keys are kept for checking a presented countersignature.

> **0.28.0 (October 2026): training-track claims.** A Workshop claims each training track of its registered model (`POST /api/claims`: the record's hash, a track id the record itself lists, the track's fingerprint and four-phase query, the artist's declaration), queued in the outbox behind the record; withdrawing the model frees its claims. A recording may train any number of models, any account's: a claim is a record, never a lock. Another account's claim on a recording already claimed is taken and linked to it, and both models' pages say how many of their training tracks another account's model also trains on. `POST /api/claims/check` (Workshop machines only) answers none, this account's, or also another account's, never naming the other account's model, and gives the serial of the released output the recording is, if any. Matching reads confidence against the shorter of the two fingerprints, so a claimed track inside a long piece is found either way round. `GET /api/models/{serial}/claims` gives the counts (claimed, shared). Only fingerprints are sent, never audio. The Workshop checks at import and before Publish (`ingestion/ownership.py`, `common/register.claimed_elsewhere`).

> **0.33.0 (October 2026): after the deep review.** Every store write is one transaction (the command-line tool and the service share the ledger, so a half-written entry would break its chain); a query's phases are bounded (at most 8, each at most the landmark cap) through one parser used by `/api/verify`, the claims doors and the lookup; the signed routes run off the event loop and the claim matcher matches before it locks; the vote queries read only the landmarks that share a hash with the query, under a covering index; the watermark bucket is an indexed column on `records`, filled on open for an older file; the public pages read the ledger check at most once a minute (`ledger_status`), the real check staying whole; `ledger.jsonl` is written atomically; content lookups take at most two minutes of audio. One withdrawn rule lives on the index's rows (`withdrawn`, set at `/api/withdraw`): no door offers a withdrawn model's pieces, and `/api/served` refuses `serving` for one (410). **A released piece its maker takes down** is told to the register by the machine that released it (`POST /api/outputs/remove`; client `remove_output`; outbox kind `remove_output`): its fingerprint stays, so `/api/verify` still names it and now says `removed_at`, while the C2PA lookup no longer offers it (410). `licence`, `rights_stamp` and `musical` (bars arithmetic) join the contract from their byte-identical copies in the two rooms; `REGISTER_PUBLIC_URL` is the address any verifier may ask; the `rooms` extra installs what a room needs in one word.

> **0.32.0 (October 2026): the C2PA soft-binding lookup.** The register answers the C2PA Soft Binding Resolution API under `/c2pa/v1` (`register_service/softbinding.py`): `GET services/supportedAlgorithms`; `GET`/`POST matches/byBinding` with `dev.khaos.landmarks.1` (base64 of the JSON fingerprint phases); `POST matches/byContent` (audio, at most 10 MB, the front door's limit; decoded on the register within a fixed memory budget, at most 96 kHz and two channels, two at a time, and fingerprinted); `GET manifests/{manifestId}` (`application/c2pa`, the manifest store a piece was released with). A match is always by fingerprint, and only a released piece whose manifest store is held, from a model that is not withdrawn, is offered (a withdrawn model's manifest answers 410). A Khaos watermark (`dev.khaos.audioseal-bucket.1`, the ID in decimal) names a group of models, so it is refused as a binding and taken only as the hint that narrows a content search. `POST /api/outputs` takes an optional `manifest` (base64 of the file's `C2PA` RIFF chunk, `credentials.manifest_store`); only the machine that released a piece can set or replace it, and a manifest that is not a store is not kept while the release still stands. Rate limits per address: content 10/min (a verify key lifts it), bindings 30/min, manifests 120/min. The two algorithm names are Khaos's own until they are entered on the C2PA soft-binding algorithm list.

Plan, not commitment (2026-08-22). Code is the truth; this is the shape
to build towards once the Workshop release work starts (it is the same
"move trust out of the client" move as that plan).

## 1. Why

Today's model identity is an 11-bit watermark payload — **2048 codewords
in total** — allocated by each training machine *scanning the model cards
it can see* and taking a number from its own **band** (§3 of
`watermarking-v2.md`: laptop 32–703, studio 704–1375, threadripper
1376–2047). The bands exist only because the machines never talk. Three
consequences followed this week:

- the scan *is* the ledger, so removing an artist had to write a retired
  record beside the artists directory or its IDs would have been re-issued;
- a Listening Space can only `/verify` what it made itself — its
  provenance index is local;
- revocation is a file deletion on one machine; other Spaces learn nothing.

A central **registry service** owns the ledger, the fingerprint index and
the consent record. It does **not** replace filesystem handoff between
the apps, and it must never make an offline machine unable to finish a
training run — hence leases (§3).

## 2. What it holds

| table | rows | written by |
|---|---|---|
| `leases` | machine key, payload block (start, size), issued/expires | service |
| `ids` | codeword → state (leased / stamped / retired), run id, artist id, card hash, machine | Workshop (stamp), either app (retire) |
| `cards` | run id, artist id, adapter hash, card JSON, signature | Workshop at training completion |
| `consent` | artist id, statement hash, granted/withdrawn timestamps | Workshop (grant), Platform or Workshop (withdraw) |
| `fingerprints` | output id, artist id, run id, fingerprint rows (the contract's `fingerprint_array`), stems flag | Platform at store time; Workshop for auditions |
| `audit` | append-only log of every call | service |

Fingerprints are spectral-peak hashes, not audio. Nothing in the service
can reproduce a catalogue.

## 3. Leases, not live calls

Training completes offline. The allocator therefore never calls the
service at completion time; it takes from a **leased block**:

1. When online, a machine asks `POST /leases {machine, size}` and receives
   a contiguous block of payloads (default 32) it alone may stamp from.
   Blocks are accounted, expire (90 days) unless renewed, and are never
   re-leased while any ID in them is stamped.
2. `allocate_model_watermark_id` takes the next unused payload **from the
   lease file** (`<data root>/.watermark_lease.json`), falling back to the
   static band only when the machine has never held a lease — and says so
   in the run manifest (`watermark_source: lease | band`).
3. Everything the machine did while offline — cards, stamps, retirements,
   audition fingerprints — sits in an **outbox**
   (`<data root>/.registry_outbox/`) and syncs in order when online.
   Sync is idempotent: every record carries a content hash.

The static bands thus become the fallback of last resort; once every
machine has leased, the band table is retired from the contract.

## 4. The API (five calls the apps make)

| call | who | when |
|---|---|---|
| `POST /leases` | Workshop | online, when the local lease has < 8 free payloads or is near expiry |
| `POST /cards` | Workshop | training completion (via outbox); the service signs the card (Ed25519) and returns the signature the Platform's gate can check |
| `POST /retire` | both apps | artist removal; the service marks the IDs retired and records withdrawal in `consent` |
| `POST /fingerprints` | Platform (outputs), Workshop (auditions) | at store time (via outbox) |
| `POST /verify` | anyone, no key | a fingerprint array (and optional decoded codeword) → the same content-first answer `/verify` gives today, for any output any machine made |

Machines authenticate with a per-machine key issued at install; the
verify call is public and read-only. The service is small: FastAPI,
SQLite to start, Postgres when a second site exists.

## 5. What changes in the apps

- **Contract** (`khaos_attribution`): `registry.py` — lease file
  read/write, outbox append/sync, the request/response shapes, card
  signature verification. The static `MACHINE_BANDS` stay until §3 step 3.
- **Workshop**: the allocator prefers the lease; the outbox syncs on the
  System tab ("registry: 3 records waiting") and at startup when online;
  removal posts `retire`.
- **Platform**: the model-card gate also checks the service signature
  when one is present (unsigned cards stay servable until the cut-over
  date in the release plan); `/verify` consults the service when local
  resolution fails; withdrawal posts `retire` and a Space that learns of
  a withdrawal at startup refuses the bundle even if the files are still
  on disk.
- **Engine Room**: a registry-sync audit (outbox depth, lease headroom,
  unsigned cards) — UNKNOWN is not OK.

## 6. What it does not fix

The 2048 ceiling is the payload, not the partition. Central allocation
reclaims the slack of the bands; it does not make the space bigger. The
escape is the direction v2 already took — **content-first** resolution:
with a central fingerprint index, the fingerprint identifies the exact
output and the watermark only corroborates. When the space runs short,
the watermark can carry something coarse ("Khaos-made, issued in epoch
N") and identity lives in the index. That is a later decision; the
service makes it possible without re-embedding anything.

## 7. Phases

1. **Service + leases** (offline-safe): leases, `ids`, `retire`, outbox
   sync; the allocator prefers the lease. Bands remain the fallback.
2. **Cards and consent**: `POST /cards` with signatures; the Platform
   gate verifies when present; withdrawal propagates at startup.
3. **Fingerprints + public verify**: outputs and auditions register;
   `/verify` answers for any machine's output; the Public Listening
   Space's verify page points here.
4. **Retire the bands** once every machine has leased.

Each phase ships with the sign-off ritual; phase 1 is a few days of work
and is the one that removes this week's retired-IDs file from the critical
path.
