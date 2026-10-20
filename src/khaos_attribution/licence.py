"""The licence chain, where it is used: the plain-language reading of a base model's licence, what this platform
does and does not warrant, and the installers' acceptance record. The licence NAME shown anywhere is the model card's
own `base_model_licence`, never a literal typed here; an unknown name gets an honest sentence, not a summary nobody
checked. Stdlib only and runnable as a script: the installers call it before a virtual environment exists.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The licence the pinned ACE-Step 1.5 engine is published under — the
# LICENSE file at the root of the checkout, and `license: mit` on the
# weights' own model card. Pinned by test to the trainer's
# `_BASE_MODEL_LICENCE` (Workshop) and the engine's `base_model_licence`
# (Space), so the installer's statement cannot drift from what the cards
# say. The installer is the one surface with no card to read it from.
ENGINE_LICENCE = "MIT"
ENGINE_NAME = "ACE-Step 1.5"

# What a licence permits downstream, in one sentence, for a musician.
# Keyed by the exact `base_model_licence` value the cards carry.
BASE_MODEL_LICENCE_SUMMARIES = {
    "MIT": ("MIT is a permissive licence: the base model, and anything you make "
            "with it, can be used, changed, shared and sold commercially without asking, "
            "provided the ACE-Step copyright notice travels with any copy of "
            "the model; it says nothing about the artist's music — those rights "
            "come from the artist's consent and rights records."),
    "Stability AI Community License": (
        "The Stability AI Community License lets the base model, and what you make with it, be used and sold "
        "commercially while the maker's yearly revenue stays under Stability's threshold (one million US dollars at "
        "the time of writing); above it a separate licence from Stability is needed, and 'Powered by Stability AI' "
        "travels with the model. Its text encoder is Google's and carries the Gemma terms of use. It says nothing "
        "about the artist's music — those rights come from the artist's consent and rights records."),
    "MIT License": ("The MIT License is permissive: the base model, and anything you make with it, can be used, changed, "
                    "shared and sold commercially without asking, provided the model's copyright notices travel with any "
                    "copy of it. The model's own notice still applies: mimetic-a0's says some of the synthetic audio it "
                    "learned from was rendered with a sound font whose samples' origin is uncertain. It says nothing "
                    "about the artist's music — those rights come from the artist's consent and rights records."),
}

# A licence this module has no reading of: say so, name it, and send the
# reader to the text. Never a summary of something nobody looked at.
UNKNOWN_LICENCE_SUMMARY = ("No plain-language summary is on file for the {name} "
                           "licence — read the licence itself before using an "
                           "output commercially.")

# One paragraph, the same on every surface, behind a collapsed "What this
# means". Written for a musician deciding whether they can use a piece —
# the aim is real licensing, so it says what the chain of evidence IS, not
# "experimental only".
WHAT_THIS_MEANS_LABEL = "What this means"
# The warranty as a tree: a heading with its points, or a sentence alone.
# Pages draw it nested; the paragraph is the tree flattened and joined, so
# the installer's statement and every page say the same words.
WARRANTY_TREE = (
    ("What Khaos warrants:", (
        "every adapter was trained only on music its artist consented to;",
        "every output carries a watermark and a provenance record naming the artist, the adapter and the base model;",
        "every attribution estimate states its method.",
    )),
    ("What Khaos does not warrant:", (
        "that an output resembles no other music;",
        "that it can be registered or cleared anywhere;",
        "that the base model's licence covers the artist's music — it does not.",
    )),
    ("Your rights in an output come from the artist, through the consent and rights records on file.", ()),
    ("Khaos gives you the evidence to license it for real, on the artist's terms, not this software's.", ()),
)
WARRANTY_POINTS = tuple(" ".join((head, *points)) for head, points in WARRANTY_TREE)
WARRANTY_STATEMENT = " ".join(WARRANTY_POINTS)

# The installer's acceptance marker: written beside the repo once the
# statement has been read and acknowledged; --check reports its state.
ACCEPTANCE_MARKER = ".khaos-licence-accepted.json"
ACCEPTANCE_STATES = ("accepted", "stale", "missing")


def licence_summary(name) -> str:
    """The one-sentence downstream reading of a licence, by its NAME as the
    card or record carries it. Whitespace around the name is forgiven; a
    name this module has no sentence for gets the honest unknown sentence
    with the name in it, and an empty name is reported as unrecorded."""
    key = str(name or "").strip()
    if not key:
        return UNKNOWN_LICENCE_SUMMARY.format(name="unrecorded")
    return BASE_MODEL_LICENCE_SUMMARIES.get(key) or UNKNOWN_LICENCE_SUMMARY.format(name=key)


def licence_line(card) -> dict:
    """The licence block a surface renders, read from a model card or a
    provenance record and never retyped: the base model and licence name
    are the document's own fields (None when absent — absence is shown,
    never filled), the sentence is keyed by that name, the warranty
    paragraph is the same everywhere."""
    doc = card if isinstance(card, dict) else {}
    name = doc.get("base_model_licence")
    name = str(name).strip() if isinstance(name, str) and name.strip() else None
    model = doc.get("base_model")
    model = str(model).strip() if isinstance(model, str) and model.strip() else None
    return {"base_model": model,
            "base_model_licence": name,
            "permits": licence_summary(name),
            "warranty": WARRANTY_STATEMENT,
            "warranty_points": list(WARRANTY_POINTS),
            "warranty_tree": [[head, list(points)] for head, points in WARRANTY_TREE],
            "what_this_means": WHAT_THIS_MEANS_LABEL}


def _wrap(text: str, width: int = 76, indent: str = "  ") -> str:
    import textwrap
    return textwrap.fill(text, width=width, initial_indent=indent, subsequent_indent=indent)


def install_statement() -> str:
    """What the installer prints once and asks the operator to accept: the
    engine, its licence, the downstream sentence, the warranty paragraph.
    The acceptance marker records this text's hash, so a change to any of
    these words asks for a fresh acknowledgement on the next run."""
    return "\n".join([
        f"Khaos runs on the {ENGINE_NAME} base model, which its authors publish under the "
        f"{ENGINE_LICENCE} licence (the LICENSE file in the engine checkout this installer pins).",
        "",
        _wrap(licence_summary(ENGINE_LICENCE)),
        "",
        _wrap(WARRANTY_STATEMENT),
        "",
        "Every model card, provenance record and export names this licence. Accepting it",
        "here is acknowledged once per machine and recorded beside this repository.",
    ])


def statement_hash() -> str:
    """SHA-256 of the statement as printed — the marker's proof of WHICH
    words were accepted."""
    return hashlib.sha256(install_statement().encode("utf-8")).hexdigest()


def read_acceptance(marker) -> dict | None:
    """The marker's record, or None when absent or unreadable (an
    unreadable marker is not an acceptance)."""
    try:
        loaded = json.loads(Path(marker).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


def acceptance_state(marker) -> str:
    """'accepted' when the marker records THIS statement's hash; 'stale'
    when it records another (the words changed since — read again);
    'missing' when there is no usable marker."""
    record = read_acceptance(marker)
    if record is None:
        return "missing"
    return "accepted" if record.get("statement_sha256") == statement_hash() else "stale"


def record_acceptance(marker, accepted_by: str | None = None) -> dict:
    """Write the marker: licence name, statement hash, date, who. The
    date is UTC to the second; `accepted_by` defaults to the login name
    so a shared machine's marker says whose acknowledgement it holds."""
    record = {"licence": ENGINE_LICENCE,
              "base_model": ENGINE_NAME,
              "statement_sha256": statement_hash(),
              "accepted_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "accepted_by": accepted_by or _login_name()}
    path = Path(marker)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    return record


def _login_name() -> str:
    # $USER first: os.getlogin() answers for the controlling terminal, which
    # under sudo, a launcher or a sandbox is not the person accepting.
    name = os.environ.get("USER")
    if name:
        return name
    try:
        return os.getlogin()
    except OSError:
        return "unknown"


def acceptance_report(marker) -> str:
    """One line for the installer: what the marker says, in the operator's
    words — accepted (when, by whom), stale, or missing."""
    state = acceptance_state(marker)
    record = read_acceptance(marker) or {}
    if state == "accepted":
        when = str(record.get("accepted_at_utc") or "")[:10] or "an unrecorded date"
        who = record.get("accepted_by") or "unknown"
        return (f"base model licence ({record.get('licence') or ENGINE_LICENCE}) accepted "
                f"on {when} by {who}")
    if state == "stale":
        return ("base model licence statement has changed since it was accepted — "
                "it needs reading and accepting again")
    return "base model licence not yet accepted on this machine"


def main(argv: list[str] | None = None) -> int:
    """`statement` prints the text; `hash` its hash; `check <marker>`
    prints the report and exits 0 accepted / 1 missing / 2 stale;
    `accept <marker>` records the acceptance and prints the report."""
    args = list(sys.argv[1:] if argv is None else argv)
    command = args[0] if args else "statement"
    if command == "statement":
        print(install_statement())
        return 0
    if command == "hash":
        print(statement_hash())
        return 0
    if command in ("check", "accept"):
        if len(args) < 2:
            print(f"usage: licence.py {command} <marker-path>", file=sys.stderr)
            return 64
        marker = args[1]
        if command == "accept":
            record_acceptance(marker)
        print(acceptance_report(marker))
        state = acceptance_state(marker)
        return {"accepted": 0, "missing": 1, "stale": 2}[state]
    print(f"unknown command {command!r}: statement | hash | check <marker> | accept <marker>",
          file=sys.stderr)
    return 64


if __name__ == "__main__":
    sys.exit(main())
