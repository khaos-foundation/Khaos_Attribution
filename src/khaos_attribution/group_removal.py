"""Group removal (W): a piece's split read from its loss under the served adapter and under one adapter per writer
split pattern trained without that pattern's tracks.

A group's share is the positive part of its rise (group adapter's loss minus the served adapter's), normalised; the
group's share reaches writers by the split its tracks share. Nothing is fitted per catalogue.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence

from khaos_attribution import blend

SCHEMA = "khaos.group_removal/0.1.0"
METHOD_KIND = "group_removal"
# The estimate schema version a group-removal estimate is stamped with; the other kinds keep theirs.
ESTIMATE_SCHEMA_VERSION = "1.2.0"


class NoRise(ValueError):
    """No group's removal raised the piece's loss: the adapters say nothing about where it came from."""


class LossNotFinite(NoRise):
    """A loss under one of the adapters is not a number: nothing is said about where the piece came from."""


class RecordRefused(ValueError):
    """A group-removal record that is malformed, or not the one this adapter and these rights were measured for."""


# ── groups ───────────────────────────────────────────────────────────────────

def pattern_key(rights: Mapping | None) -> str:
    """A track's writer split pattern: writer names with their shares, order-free."""
    writers = (rights or {}).get("writers") or []
    return json.dumps(sorted((str(w["name"]), float(w["share_pct"])) for w in writers))


def split_groups(tracks: Sequence[str], rights_by_track: Mapping[str, Mapping]) -> list[list[str]]:
    """The tracks partitioned by writer split pattern; tracks sorted within a group, groups ordered by pattern."""
    by: dict[str, list[str]] = {}
    for t in sorted(tracks):
        by.setdefault(pattern_key(rights_by_track.get(t)), []).append(t)
    return [by[k] for k in sorted(by)]


def partition_sha256(groups: Sequence[Sequence[str]]) -> str:
    """Which tracks share a group, whatever the writers are called: a rights edit that keeps it keeps the adapters."""
    canon = sorted(sorted(str(t) for t in g) for g in groups)
    return hashlib.sha256(json.dumps(canon, separators=(",", ":")).encode("utf-8")).hexdigest()


def group_ids(groups: Sequence[Sequence[str]]) -> list[str]:
    return [f"g{i + 1}" for i in range(len(groups))]


# ── one piece ────────────────────────────────────────────────────────────────

def group_shares(full_loss: float, group_losses: Sequence[float]) -> list[float]:
    """Each group's share of the piece: its positive rise over the served adapter's loss, normalised."""
    if not all(math.isfinite(float(x)) for x in (full_loss, *group_losses)):
        raise LossNotFinite("a loss under one of the adapters is not finite, so the piece was not measured")
    rises = [max(float(g) - float(full_loss), 0.0) for g in group_losses]
    total = sum(rises)
    if not total > 0:
        raise NoRise("no group's removal raised this piece's loss")
    return [r / total for r in rises]


def track_shares(shares: Sequence[float], groups: Sequence[Sequence[str]],
                 training_share: Mapping[str, float] | None = None) -> dict[str, float]:
    """A group's share spread over its tracks by the training-share rule (evenly where it gives none).

    Writers are the same across a group's tracks, so this only moves publisher and master money inside a group."""
    out: dict[str, float] = {}
    rule = training_share or {}
    for share, tracks in zip(shares, groups, strict=True):
        w = [max(float(rule.get(t, 0.0)), 0.0) for t in tracks]
        tot = sum(w)
        for t, wt in zip(tracks, w, strict=True):
            out[t] = share * (wt / tot if tot > 0 else 1.0 / len(tracks))
    return out


def money(shares: Sequence[float], groups: Sequence[Sequence[str]], rights_by_track: Mapping[str, Mapping],
          training_share: Mapping[str, float] | None = None) -> dict:
    """Writer, publisher and master shares of the piece, as ``blend.money_splits`` gives them (point values)."""
    by_track = track_shares(shares, groups, training_share)
    return blend.money_splits(by_track, {}, dict(rights_by_track))


# ── the bundle's record ──────────────────────────────────────────────────────

def build_record(*, groups: Sequence[Sequence[str]], served_sha256: str, served_epoch: int | None,
                 adapters_sha256: Mapping[str, str], noise_seed: int, device: str, compute_dtype: str,
                 measured_error: Mapping | None = None, abstention: Mapping | None = None) -> dict:
    """The record a bundle carries beside its group adapters (keys: ``g1``.. in the groups' order)."""
    ids = group_ids(groups)
    if sorted(adapters_sha256) != sorted(ids):
        raise RecordRefused(f"adapters for {sorted(adapters_sha256)}, groups {ids}")
    return {"schema": SCHEMA, "groups": {i: sorted(g) for i, g in zip(ids, groups, strict=True)},
            "partition_sha256": partition_sha256(groups), "served": {"sha256": served_sha256, "epoch": served_epoch},
            "adapters_sha256": dict(adapters_sha256), "noise_seed": int(noise_seed), "device": device,
            "compute_dtype": compute_dtype, "measured_error": dict(measured_error) if measured_error else None,
            "abstention": dict(abstention) if abstention else None}


def read_record(record: Mapping) -> list[list[str]]:
    """The record's groups in order, after its own checks."""
    if record.get("schema") != SCHEMA:
        raise RecordRefused(f"schema {record.get('schema')!r}, not {SCHEMA}")
    groups_by_id = record.get("groups") or {}
    ids = [f"g{i + 1}" for i in range(len(groups_by_id))]
    if not ids or sorted(groups_by_id) != sorted(ids):
        raise RecordRefused("the record's groups are not g1..gN")
    groups = [list(groups_by_id[i]) for i in ids]
    tracks = [t for g in groups for t in g]
    if not all(groups) or len(tracks) != len(set(tracks)):
        raise RecordRefused("the record's groups are empty or share a track")
    if partition_sha256(groups) != record.get("partition_sha256"):
        raise RecordRefused("the record's partition hash is not its groups'")
    if sorted(record.get("adapters_sha256") or {}) != sorted(ids):
        raise RecordRefused("the record does not pin one adapter per group")
    return groups


def adapter_refusal(record: Mapping, weights_sha256: Mapping[str, str]) -> str | None:
    """Why the adapter files at hand are not the ones the record pins (``{group id: sha256 of its weights}``)."""
    pinned = record.get("adapters_sha256") or {}
    for gid in group_ids(read_record(record)):
        if weights_sha256.get(gid) != pinned.get(gid):
            return f"group adapter {gid} is not the one its record pins"
    return None


def split(record: Mapping, losses: Mapping[str, float], rights_by_track: Mapping[str, Mapping],
          training_share: Mapping[str, float] | None = None) -> dict:
    """A piece's split from its losses keyed ``served`` and by the record's group ids; groups come from the record,
    never re-derived from the rights, so a renamed writer cannot move money between groups."""
    groups = read_record(record)
    ids = group_ids(groups)
    missing = [k for k in ("served", *ids) if k not in losses]
    if missing:
        raise RecordRefused(f"no loss for {', '.join(missing)}")
    shares = group_shares(losses["served"], [losses[i] for i in ids])
    return {"groups": [{"group_id": i, "track_ids": g, "share": s} for i, g, s in zip(ids, groups, shares, strict=True)],
            "money": money(shares, groups, rights_by_track, training_share)}


def stale_because(record: Mapping, *, served_sha256: str, rights_by_track: Mapping[str, Mapping]) -> str | None:
    """Why the record no longer measures this served adapter under these rights, or None when it does."""
    groups = read_record(record)
    if (record.get("served") or {}).get("sha256") != served_sha256:
        return "the served adapter has changed since its group adapters were trained"
    tracks = [t for g in groups for t in g]
    if partition_sha256(split_groups(tracks, rights_by_track)) != record["partition_sha256"]:
        return "the rights now group the tracks differently from the group adapters"
    return None
