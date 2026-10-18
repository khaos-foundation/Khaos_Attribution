"""Measured influence: the D-TRAK kernel over an adapter's gradient index,
per-track shares, and the estimate document both rooms write from them.

The index is built once per adapter where the training pairs are; scoring
needs only the index and the output's projected gradient. Influence here is
what a retraining ground truth measured it to be, on the catalogue it was
measured on; the measurement itself is the caller's to supply; resemblance (CLAP) is a different quantity
and travels beside it under its own name.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from khaos_attribution import ATTRIBUTION_ESTIMATE_SCHEMA_VERSION, blend
from khaos_attribution.validation import validate_attribution_estimate

INFLUENCE_METHOD = "dtrak"
INFLUENCE_VERSION = "0.1.0"
FEATURE_TARGET = "vnorm"
LAMBDA_MULTIPLE = 10.0
PROJECTION_DIM = 8192
TIMESTEPS_PER_PAIR = 10
INDEX_MATRIX = "gradients.npy"
INDEX_RECORD = "index.json"
INDEX_SCHEMA = "khaos.gradient_index/0.1.0"
NORM_EPS = 1e-12
UNASSIGNED = "unassigned"


BASE_CAVEAT = (
    "Measured influence: how much of this output's adapter gradient each training track "
    "explains, read by the D-TRAK kernel over the adapter's gradient index.")

CALIBRATION_SCHEMA = "khaos.influence_calibration/0.1.0"
# One fixed factor per track, fitted once on outputs scored against the retraining ground truth: a track the
# kernel habitually under- or over-weights is corrected by that much. Under this many fit outputs no factor is read.
CALIBRATION_MIN_OUTPUTS = 60
CALIBRATED_CAVEAT = (
    "Calibrated: each track's share is the kernel's times one fixed factor for that track, fitted on this "
    "catalogue's own retraining ground truth; nothing is fitted per output.")


ABSTENTION_SCHEMA = "khaos.abstention/0.1.0"
# The rule's two numbers are these percentiles of the adapter's own outputs: about one in twenty is refused.
ABSTENTION_GAIN_PERCENTILE = 2.5
ABSTENTION_LOSS_PERCENTILE = 97.5
ABSTENTION_MIN_OUTPUTS = 40
# The share of base-model outputs the test's rule refused at the least: each adapter's own control is read against it.
ABSTENTION_CONTROL_BAR = 0.9
# The numbers are read only for outputs within this band of the length they were taken at. Measured on one
# adapter at one, two and three times that length, where they held; shorter and longer are unmeasured.
ABSTENTION_LENGTH_BAND = (0.5, 3.0)
NOT_ATTRIBUTABLE = ("Not attributable: this adapter does not explain this output clearly better than the base "
                    "model does, so no share is given to any track.")


class InfluenceRefused(ValueError):
    """The index or the query cannot be scored, with the reason."""


class NotAttributable(InfluenceRefused):
    """The adapter's own loss says it did not shape this output: no percentage is given."""

    def __init__(self, reading: Mapping):
        super().__init__(NOT_ATTRIBUTABLE)
        self.reading = dict(reading)


@dataclass
class InfluenceIndex:
    """The index's raw projected gradients, one row per training pair."""
    phi: np.ndarray
    pair_ids: list
    track_ids: list
    invalid: np.ndarray
    meta: dict

    @property
    def dim(self) -> int:
        return int(self.phi.shape[1]) if self.phi.ndim == 2 else 0


def load_index(index_dir) -> InfluenceIndex:
    """Read ``gradients.npy`` + ``index.json``; rows are rebuilt to their raw
    scale from the stored norm, and a zero or non-finite row is marked invalid."""
    d = Path(index_dir)
    record_path, matrix_path = d / INDEX_RECORD, d / INDEX_MATRIX
    if not record_path.is_file() or not matrix_path.is_file():
        raise InfluenceRefused(f"no gradient index under {d}")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("schema") != INDEX_SCHEMA:
        raise InfluenceRefused(f"index at {d} is schema {record.get('schema')!r}, not {INDEX_SCHEMA!r}")
    rows = list(record.pop("rows", []))
    matrix = np.load(str(matrix_path))
    if matrix.ndim != 2 or matrix.shape[0] != len(rows):
        raise InfluenceRefused(f"index at {d} has a matrix that does not match its rows")
    norms = np.asarray([float(r.get("gradient_norm", float("nan"))) for r in rows], dtype=np.float64)
    invalid = ~np.isfinite(norms) | (norms <= NORM_EPS)
    phi = np.asarray(matrix, dtype=np.float64) * np.where(invalid, 0.0, norms)[:, None]
    return InfluenceIndex(phi=phi, pair_ids=[r["pair_id"] for r in rows],
                          track_ids=[r["track_id"] for r in rows], invalid=invalid, meta=record)


def index_sha256(index_dir) -> str:
    """One hash over the matrix bytes and the record, so a bundle can name the exact index."""
    d = Path(index_dir)
    h = hashlib.sha256()
    for name in (INDEX_MATRIX, INDEX_RECORD):
        h.update((d / name).read_bytes())
    return h.hexdigest()


def kernel_scores(phi: np.ndarray, q: np.ndarray, lam: float) -> np.ndarray:
    """ϕ(x)ᵀ (ΦᵀΦ + λI)⁻¹ Φᵀ in the dual (n × n) form Φᵀ(ΦΦᵀ + λI)⁻¹, one
    score per training pair."""
    phi = np.asarray(phi, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    if phi.ndim != 2 or phi.shape[1] != q.size:
        raise InfluenceRefused(f"features are {phi.shape}, the query has {q.size} entries")
    if lam <= 0:
        raise InfluenceRefused("lambda must be positive")
    gram = phi @ phi.T + float(lam) * np.eye(phi.shape[0])
    return np.linalg.solve(gram, phi @ q)


def lambda_for(index: InfluenceIndex, multiple: float = LAMBDA_MULTIPLE) -> float:
    """λ as a multiple of the mean squared row norm of the valid rows."""
    valid = index.phi[~index.invalid]
    scale = float(np.mean(np.sum(valid ** 2, axis=1))) if len(valid) else 1.0
    return float(multiple) * scale


def per_track(scores: Sequence[float], track_ids: Sequence[str]) -> dict:
    """Each track's total over its pairs."""
    out: dict = {}
    for s, t in zip(scores, track_ids):
        out[t] = out.get(t, 0.0) + float(s)
    return out


def shares(totals: Mapping[str, float]) -> dict:
    """Positive totals normalised to one; a track that did not help gets zero."""
    pos = {t: max(0.0, float(v)) for t, v in totals.items()}
    s = sum(pos.values())
    return {t: (v / s if s > 0 else 0.0) for t, v in pos.items()}


def calibrated_shares(totals: Mapping[str, float], factors: Mapping[str, float]) -> dict:
    """The kernel's shares, each scaled by its track's factor and renormalised; a track without
    a factor keeps the kernel's own weight; empty when the kernel names no track."""
    kernel = shares(totals)
    scaled = {t: v * float(factors.get(t, 1.0)) for t, v in kernel.items()}
    total = sum(scaled.values())
    if total <= 0:
        return {}
    return {t: v / total for t, v in sorted(scaled.items())}


def index_engine(meta: Mapping | None) -> str | None:
    """The engine an index's record says its adapter runs on: its own field, else its base model id, else ACE-Step's
    (a record from before engines were named); None when the record names something that is not a base model id."""
    from khaos_attribution.base_model import ACESTEP_ENGINE, engine_of  # noqa: PLC0415
    if (meta or {}).get("engine"):
        return str(meta["engine"])
    base = (meta or {}).get("base_model_id")
    try:
        return engine_of(str(base)) if base else ACESTEP_ENGINE
    except ValueError:
        return None


def calibration_engine(block: Mapping) -> str:
    """The engine a calibration's factors were measured on; one from before engines were named is ACE-Step's."""
    from khaos_attribution.base_model import ACESTEP_ENGINE  # noqa: PLC0415
    return str((block.get("fitted_on") or {}).get("engine") or ACESTEP_ENGINE)


def instrument(engine: str | None, runtime: str | None) -> str | None:
    """An engine on one runtime: a measure is its own instrument there, so the engine alone names its torch runtime."""
    if not engine:
        return None
    return f"{engine}-{runtime}" if runtime and runtime != "torch" else str(engine)


def index_instrument(meta: Mapping | None, base_model_id: str | None = None) -> str | None:
    """The instrument an index was built on: its engine (its own field, else the base model id given or recorded,
    else ACE-Step's) and the runtime it records."""
    engine = index_engine({**(meta or {}), **({"base_model_id": base_model_id} if base_model_id and not (meta or {}).get("engine") else {})})
    return instrument(engine, (meta or {}).get("runtime"))


def calibration_instrument(block: Mapping) -> str:
    """The instrument a calibration's factors were measured on."""
    return instrument(calibration_engine(block), (block.get("fitted_on") or {}).get("runtime"))


def calibration_of(meta: Mapping | None) -> dict | None:
    """The calibration an index's record carries, when it was measured on the index's own engine; else None, since
    one instrument's subset adapters say nothing about another's. For what a page summarises; the card reads the
    block whole through read_calibration, which refuses it in words."""
    block = (meta or {}).get("calibration")
    if not isinstance(block, Mapping) or not block.get("factors"):
        return None
    return dict(block) if calibration_instrument(block) == index_instrument(meta) else None


def calibration_sha256(block: Mapping | None) -> str | None:
    """The identity of a calibration block as written, so a document can say which one it was read with."""
    if not block:
        return None
    return hashlib.sha256(json.dumps(block, sort_keys=True, default=str).encode()).hexdigest()


def read_calibration(block: Mapping | None, *, index_tracks: Sequence[str], artist_id: str | None = None,
                     on: str | None = None) -> dict:
    """Whether an index's calibration block is applied, and why not: it must be this artist's, measured on the
    index's instrument (``on``), carry the schema, finite non-negative factors fitted on enough outputs, its own
    validation and a sane error, and at least one factor for a track the index holds."""
    if not block:
        return {"applied": False, "reason": "no calibration: the kernel alone"}
    if not isinstance(block, Mapping):
        return {"applied": False, "reason": "calibration is not a record"}
    if on is not None and calibration_instrument(block) != on:
        return {"applied": False, "reason": f"calibration was measured on {calibration_instrument(block)}, and this "
                                            f"index was built on {on}"}
    if block.get("schema") != CALIBRATION_SCHEMA:
        return {"applied": False, "reason": f"calibration schema {block.get('schema')!r} is not {CALIBRATION_SCHEMA}"}
    if artist_id is not None and block.get("artist_id") != artist_id:
        return {"applied": False, "reason": f"calibration is {block.get('artist_id')}'s, not {artist_id}'s"}
    validation = block.get("validation")
    if not isinstance(validation, Mapping) or not isinstance(validation.get("money_on_the_right_tracks"), (int, float)):
        return {"applied": False, "reason": "calibration carries no validation of its own"}
    err = block.get("measured_error_pp")
    if err is not None and not (isinstance(err, (int, float)) and not isinstance(err, bool)
                                and np.isfinite(float(err)) and 0 < float(err) <= 50):
        return {"applied": False, "reason": f"calibration's measured error {err!r} is not a number of points between 0 and 50"}
    factors = block.get("factors")
    if not isinstance(factors, Mapping) or not factors:
        return {"applied": False, "reason": "calibration carries no factors"}
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in factors.values()):
        return {"applied": False, "reason": "a calibration factor is not a number"}
    bad = [t for t, v in factors.items() if not (float(v) >= 0.0 and np.isfinite(float(v)))]
    if bad:
        return {"applied": False, "reason": f"calibration factors for {len(bad)} track(s) are negative or not finite"}
    n = block.get("fit_outputs")
    if not isinstance(n, int) or n < CALIBRATION_MIN_OUTPUTS:
        return {"applied": False, "reason": f"calibration fitted on {n} outputs, under {CALIBRATION_MIN_OUTPUTS}"}
    held = set(index_tracks)
    if not any(t in held for t in factors):
        return {"applied": False, "reason": "no calibration factor names a track this index holds"}
    unfitted = sorted(held - set(factors))
    return {"applied": True, "reason": None, "factors": {t: float(v) for t, v in factors.items()},
            "coverage": {"fitted": len(held) - len(unfitted), "of": len(held), "unfitted": unfitted,
                         "extra": sorted(set(factors) - held)}}


def _finite_pair(row: Mapping) -> tuple[float, float] | None:
    """``(loss with, loss without)`` when both are finite numbers, else None: a string or a NaN is not a loss."""
    on, off = row.get("with"), row.get("without")
    for v in (on, off):
        if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)):
            return None
    on, off = float(on), float(off)
    return (on, off) if np.isfinite(on) and np.isfinite(off) else None


def abstention_thresholds(losses: Sequence[Mapping], *, device: str, compute_dtype: str,
                          control: Sequence[Mapping] | None = None, validation: Mapping | None = None,
                          **measured_on) -> dict:
    """The rule's two numbers from an adapter's own outputs, each ``{"with", "without"}``: the low
    percentile of the gain and the high percentile of the loss with the adapter, to four places.
    ``control`` is the base model's outputs measured the same way: how many the numbers refuse."""
    pairs = [_finite_pair(r) for r in losses]
    rows = [(off - on, on) for pr in pairs if pr is not None for on, off in (pr,)]
    if len(rows) < ABSTENTION_MIN_OUTPUTS:
        raise InfluenceRefused(f"the abstention rule needs {ABSTENTION_MIN_OUTPUTS} measured outputs of the adapter's "
                               f"own, not {len(rows)}")
    gains, with_adapter = [g for g, _ in rows], [w for _, w in rows]
    gain_min = round(float(np.percentile(gains, ABSTENTION_GAIN_PERCENTILE)), 4)
    loss_max = round(float(np.percentile(with_adapter, ABSTENTION_LOSS_PERCENTILE)), 4)
    reserved = {"schema", "gain_min", "loss_max", "n_outputs", "own_accepted", "gain_percentile", "loss_percentile",
                "device", "compute_dtype", "control", "validation"}
    clash = sorted(reserved & set(measured_on))
    if clash:
        raise TypeError(f"measured_on may not carry the rule's own fields: {', '.join(clash)}")
    return {**measured_on, "schema": ABSTENTION_SCHEMA, "gain_min": gain_min, "loss_max": loss_max,
            "n_outputs": len(rows), "own_accepted": sum(1 for g, w in rows if accepts(g, w, gain_min, loss_max)),
            "gain_percentile": ABSTENTION_GAIN_PERCENTILE, "loss_percentile": ABSTENTION_LOSS_PERCENTILE,
            "device": str(device), "compute_dtype": str(compute_dtype),
            "control": control_reading(control, gain_min, loss_max),
            "validation": dict(validation) if validation else None}


def control_reading(control: Sequence[Mapping] | None, gain_min: float, loss_max: float) -> dict | None:
    """How the two numbers treat base-model outputs of the same prompts: the adapter's own proof
    that its check tells its outputs from the base model's. None when no control was measured."""
    pairs = [_finite_pair(r) for r in (control or [])]
    rows = [(off - on, on) for pr in pairs if pr is not None for on, off in (pr,)]
    if not rows:
        return None
    refused = sum(1 for g, w in rows if not accepts(g, w, gain_min, loss_max))
    rate = refused / len(rows)
    return {"n": len(rows), "refused": refused, "rate": round(rate, 4), "bar": ABSTENTION_CONTROL_BAR,
            "met": rate >= ABSTENTION_CONTROL_BAR}


def accepts(gain: float | None, loss_with: float | None, gain_min: float, loss_max: float) -> bool:
    """The rule on one output: the adapter explains it better than the base model by more than the gain, and well enough."""
    return gain is not None and loss_with is not None and gain > gain_min and loss_with < loss_max


def abstention_unreadable(thresholds: Mapping | None, *, device_type: str, compute_dtype: str,
                          duration_sec: float | None = None, index_sha256: str | None = None) -> str | None:
    """Why the rule cannot be read for this output on this machine, or None when it can: an adapter
    published without its two numbers, numbers taken beside another index, another kind of device or
    precision than they were taken on, or an output much shorter or longer than the outputs they came from."""
    if not thresholds or thresholds.get("gain_min") is None or thresholds.get("loss_max") is None:
        return ("this adapter was published without the two numbers the check needs; build its influence "
                "index again at Publish")
    if index_sha256 and thresholds.get("index_sha256") and thresholds["index_sha256"] != index_sha256:
        return "its two numbers were taken beside another influence index; build the index again at Publish"
    taken_on = str(thresholds.get("device") or "").split(":")[0]
    if taken_on != str(device_type).split(":")[0]:
        return (f"its two numbers were taken on {taken_on or 'an unrecorded device'}, and this machine "
                f"scores on {device_type}")
    if str(thresholds.get("compute_dtype")) != str(compute_dtype):
        return (f"its two numbers were taken at {thresholds.get('compute_dtype')} precision, and this machine "
                f"scores at {compute_dtype}")
    taken_at = thresholds.get("render_seconds")
    if taken_at and duration_sec is not None:
        low, high = (float(taken_at) * b for b in ABSTENTION_LENGTH_BAND)
        if not low <= float(duration_sec) <= high:
            return (f"its two numbers were taken on {float(taken_at):g}-second outputs, and this one is "
                    f"{float(duration_sec):.0f} seconds")
    return None


def abstention_reading(thresholds: Mapping | None, losses: Mapping | None, *, device_type: str,
                       compute_dtype: str, duration_sec: float | None = None,
                       index_sha256: str | None = None) -> dict:
    """The rule read for one output from its two losses, or why it was not read."""
    def unchecked(reason: str) -> dict:
        return {"checked": False, "accepted": None, "reason": reason}
    why = abstention_unreadable(thresholds, device_type=device_type, compute_dtype=compute_dtype,
                                duration_sec=duration_sec, index_sha256=index_sha256)
    if why:
        return unchecked(why)
    pair = _finite_pair(losses or {})
    if pair is None:
        return unchecked("the output's loss could not be measured")
    on, off = pair
    gain = off - on
    return {"checked": True, "accepted": accepts(gain, float(on), float(thresholds["gain_min"]),
                                                 float(thresholds["loss_max"])),
            "gain": round(gain, 6), "loss_with": round(float(on), 6), "loss_without": round(float(off), 6),
            "gain_min": float(thresholds["gain_min"]), "loss_max": float(thresholds["loss_max"]), "reason": None}


def read_abstention(thresholds: Mapping | None, module, index_meta: Mapping, files: Sequence, *,
                    duration_sec: float | None = None, index_sha256: str | None = None, paired_loss=None) -> dict:
    """The rule read for one output on a live module, the one call both rooms' scorers make. Its two
    losses are measured only when the numbers can be used here; a refused output raises ``NotAttributable``.
    ``paired_loss`` is the engine's measure of the output with and without the adapter; the default is ACE-Step's."""
    from khaos_attribution import gradient as G  # noqa: PLC0415
    here = {"device_type": str(module.device_type), "compute_dtype": G.dtype_tag(module.dtype),
            "duration_sec": duration_sec, "index_sha256": index_sha256}
    if abstention_unreadable(thresholds, **here):
        losses = None
    else:
        try:
            losses = (paired_loss or G.paired_loss)(module, index_meta, list(files))
        except G.GradientUnavailable as exc:
            # An adapter that cannot be switched off is not read; it is never refused for it.
            return {"checked": False, "accepted": None, "reason": str(exc)}
    reading = abstention_reading(thresholds, losses, **here)
    if reading["checked"] and not reading["accepted"]:
        raise NotAttributable(reading)
    return reading


REFUSAL_KIND = "not_attributable"


def refusal_record(generation_id: str, reading: Mapping, index_meta: Mapping) -> dict:
    """What is kept beside an output the rule refused: the reading, and the index it was read under."""
    return {"generation_id": str(generation_id), REFUSAL_KIND: True, "reason": NOT_ATTRIBUTABLE,
            "reading": dict(reading), "index_sha256": index_meta.get("index_sha256"),
            "adapter_sha256": index_meta.get("adapter_sha256"),
            "refused_at_utc": datetime.now(timezone.utc).isoformat()}


def refusal_stands(record: Mapping | None, index_meta: Mapping | None) -> bool:
    """Whether a kept refusal still answers for its output. It stops only when the adapter it was read
    for is served with an index or numbers it was not read under: only then can the output be read afresh."""
    if not record or not record.get(REFUSAL_KIND):
        return False
    meta = index_meta or {}
    if not meta or meta.get("adapter_sha256") != record.get("adapter_sha256"):
        return True
    numbers = meta.get("abstention") or {}
    if numbers.get("gain_min") is None or numbers.get("loss_max") is None:
        return True
    reading = record.get("reading") or {}
    return (meta.get("index_sha256") == record.get("index_sha256")
            and numbers.get("gain_min") == reading.get("gain_min")
            and numbers.get("loss_max") == reading.get("loss_max"))


def writer_shares(track_shares: Mapping[str, float], rights_by_track: Mapping[str, dict] | None) -> dict:
    """Track shares through each track's writer split to writer shares; a
    track without writers counts as unassigned."""
    out: dict = {}
    for track, share in track_shares.items():
        writers = ((rights_by_track or {}).get(track) or {}).get("writers") or []
        total = sum(float(w.get("share_pct") or 0.0) for w in writers)
        if not writers or total <= 0:
            out[UNASSIGNED] = out.get(UNASSIGNED, 0.0) + float(share)
            continue
        for w in writers:
            out[w["name"]] = out.get(w["name"], 0.0) + float(share) * float(w.get("share_pct") or 0.0) / total
    return out


def attribute(index: InfluenceIndex, feature: Sequence[float], *, multiple: float = LAMBDA_MULTIPLE) -> dict:
    """One output's per-pair scores, per-track totals and shares under the kernel."""
    f = np.asarray(feature, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(f)):
        raise InfluenceRefused("the output's feature is non-finite")
    lam = lambda_for(index, multiple)
    scores = np.where(index.invalid, 0.0, kernel_scores(index.phi, f, lam))
    totals = per_track(scores, index.track_ids)
    return {"scores": scores, "track_totals": totals, "shares": shares(totals), "lam": lam,
            "lambda_multiple": float(multiple)}


def exposure_from_index(index: InfluenceIndex) -> dict:
    """Each track's share of the index's valid rows: the exposure prior the document shows beside influence."""
    counts: dict = {}
    for t, bad in zip(index.track_ids, index.invalid):
        if not bad:
            counts[t] = counts.get(t, 0) + 1
    total = sum(counts.values())
    return {t: n / total for t, n in counts.items()} if total else {}


def resemblance_block(clap_document: Mapping, influence_shares: Mapping[str, float]) -> dict:
    """The CLAP estimate read beside influence: its similarity shares (none
    when the signal was uninformative: the blend is then only the exposure
    prior), and how far the two splits agree."""
    informative = bool(clap_document.get("method", {}).get("similarity_informative"))
    clap = ({t["track_id"]: float(t["similarity_share_pct"]) / 100.0
             for t in clap_document.get("influence", []) if t.get("similarity_share_pct") is not None}
            if informative else {})
    tracks = sorted(set(influence_shares) | set(clap))
    a = [float(influence_shares.get(t, 0.0)) for t in tracks]
    b = [clap.get(t, 0.0) for t in tracks]
    money = (1.0 - 0.5 * sum(abs(x - y) for x, y in zip(a, b))) if clap else None
    return {"method": "clap_blend", "estimator_version": str(clap_document.get("method", {}).get("estimator_version")),
            "similarity_informative": informative,
            "shares_pct": {t: round(v * 100, 4) for t, v in clap.items()},
            "money_agreement_with_influence": (round(money, 4) if money is not None else None),
            "rank_agreement_with_influence": (_spearman(a, b) if clap else None),
            "note": "resemblance to the training audio, not influence on this output"}


def _agreement(block: Mapping, influence_shares: Mapping[str, float]) -> dict:
    """The resemblance block's two agreement numbers read against the shares a document shows."""
    clap = {t: float(v) / 100.0 for t, v in (block.get("shares_pct") or {}).items()}
    tracks = sorted(set(influence_shares) | set(clap))
    a = [float(influence_shares.get(t, 0.0)) for t in tracks]
    b = [clap.get(t, 0.0) for t in tracks]
    money = (1.0 - 0.5 * sum(abs(x - y) for x, y in zip(a, b))) if clap else None
    return {"money_agreement_with_influence": (round(money, 4) if money is not None else None),
            "rank_agreement_with_influence": (_spearman(a, b) if clap else None)}


def _spearman(a: Sequence[float], b: Sequence[float]) -> float | None:
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if x.size < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    rx, ry = _ranks(x), _ranks(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def _ranks(v: np.ndarray) -> np.ndarray:
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(v.size, dtype=np.float64)
    i = 0
    while i < v.size:
        j = i
        while j + 1 < v.size and v[order[j + 1]] == v[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def build_influence_estimate(*, generation_id: str, artist_id: str, adapter_version: str,
                             track_totals: Mapping[str, float], rights: Mapping[str, dict],
                             fallback_titles: Mapping[str, str | None],
                             index: InfluenceIndex, index_sha256: str, adapter_sha256: str | None,
                             lam: float, lambda_multiple: float = LAMBDA_MULTIPLE,
                             dataset_hash: str | None = None, dataset_hash_note: str | None = None,
                             resemblance: Mapping | None = None,
                             extra_caveats: Sequence[str] = (), abstention: Mapping | None = None,
                             calibration: Mapping | None = None, validation: Mapping | None = None,
                             measured_error_pp: float | None = None, base_model_id: str | None = None) -> dict:
    """Shares, ranges and money as a validated ``dtrak`` document; a refused output gets none.
    A readable calibration block supplies the shares, validation and error; otherwise the caller's
    ``validation`` and ``measured_error_pp`` are the kernel's, and without an error no estimate is made."""
    if abstention is not None and abstention.get("checked") and not abstention.get("accepted"):
        raise NotAttributable(abstention)
    cal = read_calibration(calibration, index_tracks=index.track_ids, artist_id=artist_id,
                           on=index_instrument(index.meta, base_model_id) or "an engine the record does not name")
    share = calibrated_shares(track_totals, cal["factors"]) if cal["applied"] else shares(track_totals)
    if cal["applied"] and not share and any(v > 0 for v in shares(track_totals).values()):
        raise InfluenceRefused("the calibration gives every track the kernel named a factor of zero; nothing to split")
    if not share:
        raise InfluenceRefused("no track totals to build an estimate from")
    if sum(share.values()) <= 0:
        raise InfluenceRefused("no training track has positive influence on this output; nothing to split")
    caveats = [BASE_CAVEAT, *extra_caveats]
    if cal["applied"]:
        caveats.append(CALIBRATED_CAVEAT)
        cov = cal["coverage"]
        if cov["unfitted"]:
            caveats.append(f"{len(cov['unfitted'])} of this index's {cov['of']} tracks had no factor fitted and keep "
                           "the kernel's own weight.")
    elif calibration:
        caveats.append(f"Calibration not applied: {cal['reason']}.")
    if abstention is not None and not abstention.get("checked"):
        caveats.append(f"Not checked for outputs this adapter did not shape: {abstention.get('reason')}.")
    if dataset_hash_note:
        caveats.append(dataset_hash_note)
    without_rights = [t for t in share if t not in rights]
    if without_rights:
        caveats.append(f"{len(without_rights)} of {len(share)} influencing tracks have no rights record; "
                       "their influence is reported as unattributed, not redistributed.")
    error_pp = (calibration.get("measured_error_pp") if cal["applied"] else None) or measured_error_pp
    if error_pp is None:
        raise InfluenceRefused("no measured error travels with this index; a share without its error is not given")
    error_pp = float(error_pp)
    err = error_pp / 100.0
    ranges = {t: (max(0.0, v - err), min(1.0, v + err)) for t, v in share.items()}
    splits = blend.money_splits(share, ranges, rights)
    pct = blend.largest_remainder_pcts(share)
    exposure = exposure_from_index(index)
    if resemblance is not None and cal["applied"]:
        resemblance = {**resemblance, **_agreement(resemblance, share)}
    resemblance_shares = {t: v / 100.0 for t, v in (resemblance or {}).get("shares_pct", {}).items()}
    influence = []
    for track_id, v in sorted(share.items(), key=lambda kv: -kv[1]):
        lo, hi = ranges[track_id]
        influence.append({
            "track_id": track_id,
            "title": rights[track_id]["title"] if track_id in rights else fallback_titles.get(track_id),
            "exposure_share_pct": round(exposure.get(track_id, 0.0) * 100, 4),
            "similarity_share_pct": (round(resemblance_shares[track_id] * 100, 4)
                                     if track_id in resemblance_shares else None),
            "influence_share_pct": pct[track_id],
            "blended_share_pct": pct[track_id],
            "share_range_pct": [round(lo * 100, 4), round(hi * 100, 4)],
        })
    meta = index.meta
    document = {
        "schema_version": ATTRIBUTION_ESTIMATE_SCHEMA_VERSION,
        "generation_id": generation_id,
        "artist_id": artist_id,
        "adapter_version": adapter_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": {
            "kind": INFLUENCE_METHOD,
            "estimator_version": INFLUENCE_VERSION,
            "exposure_basis": "training pairs per track, from the index rows",
            "embedding_model": None,
            "embedding_version": None,
            "similarity_informative": bool((resemblance or {}).get("similarity_informative", False)),
            "temperature": None,
            "feature_target": str(meta.get("target", FEATURE_TARGET)),
            "lambda_multiple": float(lambda_multiple),
            "lambda": float(lam),
            "projection_dim": int(index.dim),
            "timesteps": list(meta.get("timesteps") or []),
            "noise_seed": meta.get("noise_seed"),
            "n_pairs": int(len(index.pair_ids)),
            "n_tracks": int(len(set(index.track_ids))),
            "index_sha256": index_sha256,
            "adapter_sha256": adapter_sha256,
            "dataset_hash": dataset_hash,
            "measured_error_pp": error_pp,
            "validation": dict(calibration["validation"]) if cal["applied"] and calibration.get("validation")
            else (dict(validation) if validation else None),
            "calibrated": bool(cal["applied"]),
        },
        "influence": influence,
        "splits": splits,
        "caveats": caveats,
    }
    if abstention is not None:
        document["method"]["abstention"] = dict(abstention)
    if cal["applied"]:
        document["method"]["calibration"] = {
            "schema": calibration.get("schema"), "factors": cal["factors"], "coverage": cal["coverage"],
            "fitted_on": dict(calibration.get("fitted_on") or {}), "fit_outputs": calibration.get("fit_outputs"),
            "kernel_shares": shares(track_totals), "block_sha256": calibration_sha256(calibration)}
    elif calibration:
        document["method"]["calibration"] = {"applied": False, "reason": cal["reason"],
                                             "block_sha256": calibration_sha256(calibration)}
    if resemblance is not None:
        document["resemblance"] = dict(resemblance)
    return validate_attribution_estimate(document)
