# khaos_attribution

JSON schemas and validators for artist attribution. The package bundles five
schemas — a **provenance record** attached to every generated output, a
**model card** describing an artist adapter, **track rights**, an
**attribution estimate**, and an output **tombstone** — plus Python
functions that validate records against them using
[jsonschema](https://python-jsonschema.readthedocs.io/), and the shared
implementation modules the two apps pin: `watermark` (AudioSeal v2 —
run-level SECDED codewords, retirement, the recorded reset history),
`fingerprint`, `estimator` and `blend` (the attribution estimate and
its blend of signals), `diagnostics` (whether that estimate's similarity
signal varied with the output at all, or returns the same tracks whatever
was generated) and `aspects` (per-aspect shares — harmony, rhythm, timbre —
which are descriptor agreement, never causal influence), `embedding`,
`lyric_align`, `prompt_expansion` (Simple mode's rules and exemplars),
`caption_fit` (whether a prompt reads like the artist's captions and keeps the request),
`caption_vocabulary` (the claim words captions and lints read), `translation` (Expand's translator:
the prompt, the lint, the request check and the fallback to the listener's own words),
and `catalogue` (the artist's usual tempo/key/time signature).

Part of [Khaos](https://www.khaos.dev/), a free-to-use toolkit for music creators to train AI models on their own work. The method and its published results are at [khaos.dev/research](https://www.khaos.dev/research).

This repo is the *data* contract of the Khaos ecosystem; the *visual*
contract (shared design language for the toolkit and listening-space UIs)
lives in the `Khaos_Platform` repo's `DESIGN.md`.

## Installation

Directly from GitHub:

```bash
pip install "git+https://github.com/khaos-foundation/Khaos_Attribution.git"
```

## Usage

```python
from khaos_attribution import validate_provenance_record, validate_model_card

validate_provenance_record(record)   # returns the record if valid
validate_model_card(card)
```

Both functions raise `khaos_attribution.AttributionValidationError` with a
message listing every problem found (missing fields, wrong types, out-of-range
values) when a record does not match its schema.

The raw schema files are bundled with the package and can be loaded with
`khaos_attribution.load_schema("provenance_record.schema.json")`.

## Verify a file

Every Khaos output carries two marks: an inaudible watermark and a signed, time-stamped record (C2PA Content
Credentials) saying it is AI-generated. To check a file:

```bash
pip install "khaos-attribution[verify,detector] @ git+https://github.com/khaos-foundation/Khaos_Attribution.git"
khaos-verify song.wav                      # WAV, FLAC, MP3, OGG; M4A and AAC need FFmpeg
khaos-verify song.mp3 --register https://register.example --json
```

The answer is one of five verdicts. **confirmed**: the audio matches a released Khaos output in the Register's
index. **signed**: an intact signed record claims Khaos made it; the signing machine is not yet certified, so this
alone is not proof. **tampered**: the file was changed after it was signed. **marked**: a watermark alone, which
is not proof either. **none**: no Khaos mark was found, which never proves a person made the audio. Files longer
than 20 minutes are not decoded. Authorities, researchers and the press can ask for a verify key at
foundation@khaos.dev; it lifts the per-address limit (`--key` or `KHAOS_VERIFY_KEY`).

## Verify an entry in the Guild Register

The Guild Register is a tamper-evident public register: every model the Guild admits, and every strike, is an entry in
a signed append-only log built the way Certificate Transparency logs are (RFC 6962 / RFC 9162). Nobody, the Foundation
included, can change a past entry without every later signed head showing it; anyone can check an entry without asking
the register; and the log's root is anchored daily with OpenTimestamps, so an entry can be shown to have existed by a date.

```bash
pip install "khaos-attribution[registry] @ git+https://github.com/khaos-foundation/Khaos_Attribution.git"
khaos-log-verify record.json                      # a signed record, a log entry, a leaf hash or a record hash
khaos-log-verify <leaf hash> --mirror ./ledger    # a clone of the public mirror; the register need not be up
```

The command uses only the log's public key committed in this package (`src/khaos_attribution/translog/log-key.pub`,
placed after the log key's ceremony; until then pass the published key with `--key`) and the mirror's static files. It
trusts nothing the mirror says that the signed head does not prove: the leaf hashes must make the signed root, every
published body must hash to its leaf, a missing body is accepted only when a withhold entry in the log names it (or,
in the public git mirror, while it is inside the publication delay), a record is found only through its own proved
admit entry, and a strike only through a proved revoke (or the withhold of one, which still names what it struck). It
proves the entry's inclusion, says whether it was struck and why, checks that the newest head extends the last one this
machine saw (kept in `~/.khaos/log-verify-state.json`), and warns when the newest head is more than 48 hours old: the
register appends a `heartbeat` entry (its time and nothing else) every day, so an older head means the mirror may be
stale. Exit codes: 0 proved, 1 the proof failed, 2 not found, 3 no key or mirror could be read, 4 proved but a strike
after it could be hidden (newer entries not yet public in this mirror, or an entry after it lost in a recovery and not
since confirmed or struck by the wardens), 5 proved but the newest head is over 48 hours old, so this copy may be
frozen. The same check runs in a browser at
`https://register.khaos.dev/verify-entry`.

What is signed, exactly: an entry is the RFC 8785 canonical JSON of its fields (`version`, `kind` admit/revoke/withhold/
genesis, `time` in RFC 3339 UTC, a random 128-bit `salt`, and the kind's own fields); its leaf hash is
SHA-256(0x00 || entry); nodes are SHA-256(0x01 || left || right); a signed tree head is Ed25519 over the RFC 8785 JSON of
`{log_id, root_sha256, timestamp, tree_size, version}`. RFC 8785 does not normalise unicode: anyone writing an entry
sends its strings in NFC, or the same name in two forms is two different entries. A strike is an appended `revoke` that
names the admitted leaf; nothing is ever changed or deleted. A withheld entry keeps its leaf, so every proof and head
stays valid while its body is no longer served. The public git mirror receives a new entry's body only seven days after
it is appended, counted from the signed head that first held it (heads, leaf hashes, proofs, withholds and heartbeats
at once), so a body withheld within that week never enters the public history; a body withheld later stays in that history, and withholding then hides it from the live register only.

## Provenance record fields

A provenance record travels with a generated output and says exactly how it
was made.

| Field | Meaning |
| --- | --- |
| `schema_version` | Which version of the provenance record schema this record follows, written as three numbers like `1.0.0`. Lets readers know what fields to expect. |
| `artist_name` | The human-readable name of the artist whose adapter was used to generate the output. |
| `artist_id` | A stable, unique identifier for that artist. Names can change or clash; this identifier never does. |
| `adapter_version` | The version of the artist's adapter (the fine-tuned add-on to the base model) that produced this output. |
| `adapter_hash` | A cryptographic fingerprint (hex string) of the adapter's weights. Proves exactly which adapter file was used — if the weights change, the hash changes. |
| `prompt` | The text prompt that was given to the model to produce this output. |
| `timestamp` | When the output was generated, in ISO 8601 date-time form, e.g. `2026-08-10T12:00:00Z`. |
| `base_model` | The name and version of the underlying base model the adapter was applied to. |
| `base_model_licence` | The licence the base model is distributed under, so downstream users know what terms apply. |
| `watermark_id` | *(optional)* A small whole number between 0 and 65535 embedded in the audio watermark (AudioSeal, v2: the adapter RUN's ID from its model card), tying the file back to this record. `null` when the serving machine could not mark the output — the `/verify` surfaces say so loudly. |
| `generation_mode` | *(optional, since v0.2.0)* How the output was produced: `text2music` for a fresh generation (the default when omitted), or `retake`, `repaint`, `extend`, `audio2audio` for outputs derived from an earlier one. |
| `source_generation_id` | *(optional, since v0.2.0)* When the output was derived from an earlier generation, the generation ID of that source output, so lineage can be traced through the registry. `null` (or omitted) for fresh generations. |

## Model card fields

A model card describes one artist adapter: who consented, what it was trained
on, and what it extends.

| Field | Meaning |
| --- | --- |
| `schema_version` | Which version of the model card schema this card follows, written as three numbers like `1.0.0`. |
| `artist_name` | The human-readable name of the artist the adapter represents. |
| `artist_id` | The stable, unique identifier for that artist — the same one used in provenance records. |
| `consent_statement` | A plain-language statement recording that the artist consented to an adapter being trained on their work. |
| `training_catalogue` | The list of tracks the adapter was trained on. Each entry has a `title` (the track's name) and a `duration` (its length in seconds). At least one track is required. |
| `training_date` | The date the adapter was trained, as `YYYY-MM-DD`. |
| `adapter_version` | The version of the adapter this card describes — match it against the `adapter_version` in provenance records. |
| `base_model` | The name and version of the base model the adapter was trained against. |
| `base_model_licence` | The licence the base model is distributed under. |

The model card and provenance record reject unknown fields; the attribution
estimate's `method` block deliberately does NOT (`additionalProperties: true`),
which is where `reliability` and `aspects` ride — the document's top level is
closed and its `schema_version` is a const, so a key there would invalidate
every estimate ever written. Both schemas otherwise reject unknown fields, so typos in field names fail validation
rather than passing silently.

## What the attribution estimate carries about its own signal

Since estimator 0.5.0 `method.similarity_scores` stores the raw per-track
cosines, so a producer can hand the earlier documents for an adapter to
`estimator.recent_similarity_from_documents` and the next estimate can judge
whether the similarity signal varies across outputs at all
(`method.reliability`); when that verdict is `collapsed` the influence
shares fall back to the exposure prior — the number changes, not just the
footnote.

## The seal and marks of the Guild of Fine Tuners

`khaos_attribution.seal` draws the Guild's hallmarks as dependency-free SVG.
`render_seal(number, artist_mark, standard_version, date_letter)` is the
ceremonial seal — `REGISTER No:` and the register number on an arc, the
interlaced Khaos star, three hand-struck punches — for certificates and the
Register (the number is opaque: issued at admission, set as given, never
derived here);
`render_marks_row(artist_mark, standard_version, date_letter)` is the small
functional row of four punches (artist, standard, Guild, year) for model
cards, model pages and cover art. `seal_for_provenance(document)` and
`marks_row_for_provenance(document)` read the values from a verified
provenance document and **refuse** unless `embedded_agrees_with_sidecar` is
`True`. Type is vector paths from a committed glyph table (Red Hat Display
Bold, OFL — licence beside it); there is no `<text>`, no font and nothing
fetched at runtime, and the module's `SPEC.md` says why that must stay so —
and why the interlaced star is never drawn small. `seal.artwork.place_seal_on_artwork(...)` composites the seal onto hero
artwork top right (a new image, the master untouched; Pillow via the
`artwork` extra). `python -m
khaos_attribution.seal seal|marks …` writes either asset; rebuild the glyph
table with `tools/build_glyphs.py` only when the charset or the font changes.

## Where rights data came from

A track's rights record names its `source`: `operator-entered`, `society-confirmed` (a collecting society's own record), `distributor`, `musicbrainz` or `other`. From 0.22.0 no other value validates, so a record written with an earlier name for the society value has to be re-saved.

## Running the tests

```bash
pip install -e ".[test]"
pytest
```

## Licence

Khaos is free to use and is not open source. The source may be readable, but reading it grants no right to copy, modify or redistribute it; see `LICENSE`. Components written by others keep their own licences, and the base model's licence is its own.
