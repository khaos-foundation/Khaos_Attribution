# The Guild Register — how it works today

Current-state reference for `src/register_service` (the service) and `src/khaos_attribution/registry.py` (the client,
the record and the outbox), at the contract version in `pyproject.toml`. Running it on a server is
`deploy/README.md`; checking an entry in its log is the README's "Verify an entry in the Guild Register". The original
lease-based plan and the notes each release added to it are kept in `docs/history/registry-service-plan.md`.

## What it is

One process (FastAPI under uvicorn, one worker by design: the serial and the ledger's chain are kept under one
process's lock) on loopback port 8912, behind Caddy. It numbers the fine-tuned models artists publish, countersigns
each record, keeps a hash-chained public ledger and a signed append-only log of every admission and strike, holds the
fingerprints of released pieces so anyone can ask whether audio is one, and the training tracks each model claims.
The register never sees audio except a clip sent to the soft-binding lookup, deleted when the answer is made.

## Machines and roles

A machine enrols with a one-time invite the operator makes (`register-cli invite <account> --role ...`); its Ed25519
key id is the first 16 hex characters of the sha256 of its raw public key. The invite fixes its role:

| role | what it may do |
|---|---|
| `workshop` | publish its account's records and withdraw them, release its account's models' pieces, claim and check training tracks |
| `space` | report where a model is served (`serving`, `removed`) and release pieces of the models it reports serving |
| `warden` | append the Guild's entries to the log: strikes, admits, confirms and withholds, each naming its sitting |

Each door takes only the roles in its row; any other enrolled machine gets 403. Every signed call carries the
machine's signature over the canonical JSON body (sorted keys, no spaces) in `X-Machine-Key-Id` and
`X-Machine-Signature`; the register knows the machine and checks the signature over the raw bytes before it parses
the body. A withdrawal, a served report, a release or its removal, a claim and a log entry also carry a nonce: the
same body again gets its first answer back, another body under a used nonce is refused as a replay. An enrolment, a
record (the same record again gets the same serial) and a claim check take none.

## Routes

Signed:

| route | what it does |
|---|---|
| `POST /api/enrol` | an invite and a public key make an enrolled machine |
| `POST /api/records` | a signed record gets the next serial (from 65536) and the register's countersignature; the same record again gets the same answer |
| `POST /api/withdraw` | the publishing account withdraws its model; final |
| `POST /api/served` | a Space says it serves a model, or no longer does; its address is kept as an https origin (http only on loopback) |
| `POST /api/outputs` | a released piece's fingerprint (and Content Credentials manifest store, and file sha256) into the index under its model's bucket |
| `POST /api/outputs/remove` | the machine that released a piece takes it down; its fingerprint stays, the lookup no longer offers it |
| `POST /api/claims` | a Workshop claims one of its model's training tracks, on the artist's declaration |
| `POST /api/claims/check` | at import: whether a recording already trains a registered model, and whether it is a released piece |
| `POST /log/append` | a warden's entry |

Public:

| route | what it answers |
|---|---|
| `POST /api/verify` | a four-phase fingerprint (and the codeword, when one was heard, and the file's sha256): the released piece it is, the bucket's models, or nothing; `X-Khaos-Verify-Key` gives a key holder its own lane |
| `GET /api/models/{serial}` | the model's public record as the artist chose to show it, its state and where it is served; an account's reason for a withdrawal is never in it |
| `GET /api/models/{serial}/claims` | how many of its training tracks are claimed, and how many another account's model also trained on |
| `GET /api/withdrawn` | `withdrawn` (what a Space must not serve), `struck` (the wardens', which a warden may admit again) and `unresolved` (under review after a recovery) |
| `GET /api/ledger?since=` | the ledger's rows after `since`, 1,000 a page, the next named in a `Link` header |
| `GET /api/public-key` | the countersigning key and whether it is a development one |
| `GET /log/...`, `/mirror/log/...` | the signed log's heads, proofs, entries, key and anchors, and its static export; 404 while the log is off |
| `/c2pa/v1/...` | the C2PA Soft Binding Resolution API: by fingerprint, by content, and a released piece's manifest store |
| `/`, `/models/{serial}`, `/verify-entry`, `/sitemap.xml`, `/robots.txt` | the public pages |
| `GET /health` | the register's version, whether the ledger's chain is unbroken, the counts, whether the log is on, and why it is blocked when it is |

## Model states

`registered`, then either `withdrawn` (by its account; final: no admit undoes it) or `struck` (a warden's revoke left
no admit standing; a warden's admit makes it `registered` again). Neither withdrawn nor struck is served: listed in
`/api/withdrawn`, its pieces not offered by any door, `serving` and new releases refused with 410. A model admitted
before an entry a recovery lost is **under review** (`unresolved`) until a warden answers the loss; a Space keeps
serving it and says so.

## Stores

All under `REGISTER_DATA_DIR`:

| file | holds |
|---|---|
| `register.sqlite3` | accounts, invites, machines, nonces, records, where each is served, the countersigning keys ever used, the hash-chained ledger, and the signed log (leaves, bodies, nodes, heads, reservations, log keys, anchors, export state) |
| `index.sqlite3` | released pieces, their landmark fingerprints and manifest stores, training-track claims and their fingerprints |
| `mirror/` | `models/<serial>.json`, `ledger.jsonl` and `log/` (the full export: `sth/`, `hashes.json`, `bodies.json`, `proofs/`, `consistency/`, `key.json`, `anchors/`), pushed to the public repository |
| `mirror-public/log/` | the export with each new body held back up to seven days, which is what reaches public git history |
| `verify-keys.json` | the hashes of the verify keys issued (a key itself is shown once) |
| `log-started` | the marker that this register started its log |

Both databases carry the shape of file they are in SQLite's `user_version`. A file from before that stamp opens and
is stamped; a file a newer register wrote is refused untouched, and the service will not start on it.
`python -m register_service check-version <folder>` says so of a backup, and the restore runs it before it stops the
service.

The nightly backup and its restore are `deploy/register-backup.sh` and `register-restore.sh`; the signing keys are
never in a backup and have their own ceremony (`deploy/key-ceremony.md`).

## Limits

Bodies: 16 kB for an enrolment, 8 MB for other signed calls and `/api/verify`, 24 MB for a claim and a claim check (a
twelve-minute track's four phases), 1 MB for a manifest store; Caddy takes 10 MB, and 25 MB on the two claim routes.

Rate per address and minute: 240 reads (the log's files in a bucket of their own), 120 signed calls, 600 claims and
claim checks, 20 verifies and 6 enrolments; a verify key 600. An IPv6 address counts by its /64. The soft-binding
lookup: 10 content matches, 30 by binding, 120 manifests, and two decodes at a time; a third is answered 429 with
`Retry-After`, never queued.

## The log

Every admission and strike is a leaf of an RFC 6962 Merkle log: RFC 8785 entries with a 128-bit salt, a signed tree
head per size under the log's own key, inclusion and consistency proofs. `Store.register` appends an `admit` and
`Store.withdraw` a `revoke` (or a `withdraw`, for a model the wardens already struck) in the same transaction as the
ledger row. An account's reason for a withdrawal goes only in the log's revoke and the private records table, where a
withhold reaches it; the ledger row carries none.

A register whose log never started runs with it off: every other door answers, the log's routes give 404, and
nothing is appended, until `REGISTER_LOG_ENABLE=1` lets the service's next start begin the log with its genesis.

The register fails closed: at start it compares its database with every published copy and appends nothing while one
is ahead of it or apart (`register-cli log-recover` puts lost entries back), and it appends nothing while a leaf below
the newest is missing. The full rules are in `deploy/README.md` ("The log") and the README.
