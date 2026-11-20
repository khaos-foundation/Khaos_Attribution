"""Assemble one attribution estimate document.

The math lives in blend.py; this module owns the caveat wording, rounding,
method stamp and validation gate so both apps produce documents that differ
only in their data source. The caller supplies embeddings, rights and run
metadata; the result has already passed validate_attribution_estimate.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

import numpy as np

from khaos_attribution import aspects as _aspects
from khaos_attribution import ATTRIBUTION_ESTIMATE_SCHEMA_VERSION, blend
from khaos_attribution import diagnostics as _diagnostics
from khaos_attribution.validation import validate_attribution_estimate

BASE_CAVEAT = ("This is an estimate with the stated method, "
               "not a legal statement of ownership.")

# How many earlier outputs' similarity columns a producer should hand back.
RECENT_LIMIT = 64


def recent_similarity_from_documents(documents: Iterable[dict], *,
                                     adapter_version: str,
                                     exclude_generation_id: str | None = None,
                                     limit: int = RECENT_LIMIT
                                     ) -> dict[str, dict[str, float]]:
    """The earlier outputs' similarity columns (``method.similarity_scores``),
    read back from this adapter's estimate documents for ``recent_similarity``.

    Documents for other adapters, without the block, or for the output being
    estimated are skipped; the newest ``limit`` by ``created_at`` are kept.
    """
    dated: list[tuple[str, str, dict[str, float]]] = []
    for doc in documents:
        if not isinstance(doc, dict) or doc.get("adapter_version") != adapter_version:
            continue
        gid = doc.get("generation_id")
        if not gid or gid == exclude_generation_id:
            continue
        column = (doc.get("method") or {}).get("similarity_scores")
        if not isinstance(column, dict) or not column:
            continue
        try:
            dated.append((str(doc.get("created_at") or ""), gid,
                          {t: float(v) for t, v in column.items()}))
        except (TypeError, ValueError):
            continue
    dated.sort(key=lambda item: item[0])
    return {gid: column for _, gid, column in dated[-limit:]}


def build_estimate(*, generation_id: str, artist_id: str,
                   adapter_version: str,
                   run_tracks: set[str],
                   segment_counts: dict[str, int],
                   n_training_pairs: int | None,
                   embeddings: np.ndarray,
                   row_track_ids: list[str],
                   output_embedding: np.ndarray,
                   rights: dict[str, dict],
                   fallback_titles: dict[str, str | None],
                   embedding_model: str | None,
                   embedding_version: str | None,
                   exposure_basis: str,
                   extra_caveats: tuple[str, ...] = (),
                   recent_similarity: dict | None = None,
                   output_metadata: dict | None = None,
                   track_metadata: dict[str, dict] | None = None) -> dict:
    """The full pipeline after embedding: exposure → similarity → blend →
    ranges → money → validated document.

    Every value in `rights` must be a schema-valid track_rights record;
    `exposure_basis` names the data source, the one field that legitimately
    differs between apps. Raises ValueError when no training track has
    embeddings, the index and array disagree, or an embedding is not finite.
    """
    if len(row_track_ids) != len(embeddings):
        raise ValueError(
            f"Embedding index lists {len(row_track_ids)} rows but the array "
            f"holds {len(embeddings)} — the store/bundle is torn (files "
            f"copied at different times?). Refusing to estimate from it.")
    tracks_with_rows = set(row_track_ids)
    missing_rows = sorted(t for t, n in segment_counts.items() if n > 0 and t not in tracks_with_rows)
    if missing_rows:
        raise ValueError(
            f"{len(missing_rows)} training track(s) have segments but no embedding rows — the store "
            "is torn the other way (counts ahead of the array). Refusing to estimate from it.")
    if not np.isfinite(output_embedding).all():
        raise ValueError(
            f"The output's embedding for {generation_id} contains NaN or "
            f"infinity — the audio could not be embedded; nothing to compare.")
    if len(embeddings) and not np.isfinite(embeddings).all():
        raise ValueError(
            f"The stored embeddings for {adapter_version} contain NaN or "
            f"infinity — re-run the embedding stage before estimating.")
    caveats = [BASE_CAVEAT, *extra_caveats]

    # Zero segments is as absent as unlisted; both are named, not dropped.
    missing = sorted(t for t in run_tracks if segment_counts.get(t, 0) <= 0)
    if missing:
        caveats.append(
            f"{len(missing)} of {len(run_tracks)} training tracks have no "
            f"stored embeddings and are excluded from the estimate: "
            f"{', '.join(missing[:5])}" + ("…" if len(missing) > 5 else ""))
    if n_training_pairs and sum(segment_counts.values()) != n_training_pairs:
        caveats.append(
            f"The embedding store holds {sum(segment_counts.values())} "
            f"segments for these tracks but the run trained on "
            f"{n_training_pairs} pairs — the store has changed since "
            f"training, and exposure shares reflect the store as it is now.")
    exposure = blend.exposure_shares(segment_counts)
    if not exposure:
        raise ValueError(f"No embeddings for any training track of "
                         f"{adapter_version} — nothing to estimate from")

    scores = blend.similarity_scores(output_embedding, embeddings,
                                     row_track_ids, set(exposure))
    sim_weights, temperature = blend.similarity_weights(scores)
    if sim_weights is None:
        caveats.append(
            "Single-track adapter: influence is the whole output by "
            "construction; similarity adds nothing."
            if len(run_tracks) == 1 else
            "Acoustic similarity was uninformative (score spread below the "
            "noise floor); blended shares equal the exposure prior.")

    # Read before the blend so a collapsed signal changes the number, not
    # just the footnote.
    reliability = _reliability_block(recent_similarity, scores)
    if reliability.get("verdict") == "collapsed":
        sim_weights, temperature = None, None
        caveats.append(
            "Across recent outputs this catalogue's similarity signal did not vary with "
            "the output (" + str(reliability.get("why", "")) + "). Read these shares as "
            "the exposure prior.")

    blended = blend.blend_shares(exposure, sim_weights)
    if sim_weights is None:
        # No signal: the sweep has nothing to vary, so the interval is degenerate.
        ranges = {t: (v, v) for t, v in blended.items()}
    else:
        ranges = blend.temperature_sweep_ranges(exposure, scores, temperature)

    without_rights = [t for t in blended if t not in rights]
    if without_rights:
        caveats.append(
            f"{len(without_rights)} of {len(blended)} influencing tracks "
            f"have no rights record; their influence is reported as "
            f"unattributed, not redistributed.")
    splits = blend.money_splits(blended, ranges, rights)

    titles = {t: (rights[t]["title"] if t in rights
                  else fallback_titles.get(t))
              for t in blended}

    blended_pct = blend.largest_remainder_pcts(blended)
    shares = []
    for track_id, share in sorted(blended.items(), key=lambda kv: -kv[1]):
        lo, hi = ranges[track_id]
        shares.append({
            "track_id": track_id,
            "title": titles[track_id],
            "exposure_share_pct": round(exposure[track_id] * 100, 4),
            "similarity_share_pct": (round(sim_weights[track_id] * 100, 4)
                                     if sim_weights else None),
            "blended_share_pct": blended_pct[track_id],
            "share_range_pct": [round(lo * 100, 4), round(hi * 100, 4)],
        })

    document = {
        "schema_version": ATTRIBUTION_ESTIMATE_SCHEMA_VERSION,
        "generation_id": generation_id,
        "artist_id": artist_id,
        "adapter_version": adapter_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": {
            "estimator_version": blend.ESTIMATOR_VERSION,
            "exposure_basis": exposure_basis,
            "embedding_model": embedding_model,
            "embedding_version": embedding_version,
            "similarity_informative": sim_weights is not None,
            "temperature": temperature,
            # Raw per-track cosines, stored so the next estimate can judge
            # the signal across outputs.
            "similarity_scores": {t: round(float(v), 6) for t, v in scores.items()},
        },
        "influence": shares,
        "splits": splits,
        "caveats": caveats,
    }
    # These live under `method`, which is open, because the top level is
    # closed and a new key there would invalidate every estimate written.
    # `reliability` reads variation across outputs; `similarity_informative`
    # reads the spread within one.
    document["method"]["reliability"] = reliability
    if output_metadata is not None and track_metadata:
        document["method"]["aspects"] = _aspects.aspect_shares(
            output_metadata, {t: m for t, m in track_metadata.items() if t in blended},
            timbre_scores=scores, timbre_weights=sim_weights)
    return validate_attribution_estimate(document)


def _reliability_block(recent: dict | None, this_output: dict[str, float]) -> dict:
    """The collapse verdict over the earlier columns plus this output's.

    Until an adapter has ``diagnostics.MIN_QUERIES`` outputs on file the
    verdict is ``unknown``.
    """
    if not recent:
        return {"verdict": "unknown",
                "why": ("no earlier outputs were supplied, so the similarity signal "
                        "could not be judged across generations"),
                "method": "aria_collapse_diagnostics"}
    # Intersection, not union: filling missing tracks with 0.0 would invent
    # exactly the variation this looks for.
    covered = set(this_output)
    for col in recent.values():
        covered &= set(col)
    tracks = sorted(covered)
    if not tracks:
        return {"verdict": "unknown",
                "why": ("the earlier outputs do not all score a track this estimate scores, "
                        "so there is no matrix to read without inventing values"),
                "method": "aria_collapse_diagnostics"}
    columns = [[col[t] for t in tracks] for col in recent.values()]
    columns.append([this_output[t] for t in tracks])
    return _diagnostics.reliability(np.asarray(columns, dtype=float).T)
