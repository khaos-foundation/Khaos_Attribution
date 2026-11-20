"""The index keys each landmark by its hash: an index from before moves over in one transaction and answers every
verify, import check and claim count as it did, a landmark a fingerprint repeats still votes each time, and no
lookup scans the landmarks."""
import random
import sqlite3

import pytest

pytest.importorskip("cryptography")
pytest.importorskip("fastapi")

from register_service.index import INDEX_VERSION, Index  # noqa: E402

V1 = """
CREATE TABLE outputs (generation_id TEXT PRIMARY KEY, serial INTEGER NOT NULL, payload INTEGER NOT NULL,
  artist_id TEXT NOT NULL, released_at TEXT NOT NULL, released_by TEXT NOT NULL, kind TEXT NOT NULL, landmarks INTEGER NOT NULL,
  withdrawn INTEGER NOT NULL DEFAULT 0, removed_at TEXT, sha256 TEXT);
CREATE TABLE fingerprints (hash INTEGER NOT NULL, generation_id TEXT NOT NULL, offset_ms INTEGER NOT NULL);
CREATE INDEX fingerprints_hash ON fingerprints (hash);
CREATE INDEX fingerprints_gen ON fingerprints (generation_id);
CREATE INDEX outputs_payload ON outputs (payload);
CREATE INDEX fingerprints_vote ON fingerprints (hash, generation_id, offset_ms);
CREATE TABLE manifests (generation_id TEXT PRIMARY KEY, store BLOB NOT NULL, sha256 TEXT NOT NULL, kept_at TEXT NOT NULL);
CREATE TABLE claims (claim_id INTEGER PRIMARY KEY, account_id INTEGER NOT NULL, serial INTEGER NOT NULL,
  track_id TEXT NOT NULL, declaration TEXT NOT NULL, claimed_at TEXT NOT NULL, claimed_by TEXT NOT NULL,
  landmarks INTEGER NOT NULL, UNIQUE (serial, track_id));
CREATE TABLE claim_fingerprints (hash INTEGER NOT NULL, claim_id INTEGER NOT NULL, offset_ms INTEGER NOT NULL);
CREATE INDEX claim_fingerprints_hash ON claim_fingerprints (hash);
CREATE INDEX claim_fingerprints_vote ON claim_fingerprints (hash, claim_id, offset_ms);
CREATE INDEX claim_fingerprints_claim ON claim_fingerprints (claim_id);
CREATE TABLE claim_shares (claim_id INTEGER NOT NULL, other_claim INTEGER NOT NULL, votes INTEGER NOT NULL,
  at TEXT NOT NULL, PRIMARY KEY (claim_id, other_claim));
"""


def _prints(seed: int, n: int):
    rng = random.Random(seed)
    pool = [rng.randrange(1 << 25) for _ in range(300)]
    # Offsets on a coarse grid, so a print repeats a landmark now and then, as a real one can.
    return rng, pool, [[[rng.choice(pool), rng.randrange(0, 20000, 250)] for _ in range(rng.randint(5, 80))] for _ in range(n)]


def _filled(path, seed=5, pieces=200, claims=60):
    """The same releases and claims through the index's own doors."""
    rng, pool, prints = _prints(seed, pieces + claims)
    index = Index(path)
    for n, fp in enumerate(prints[:pieces]):
        index.release(f"gen-{n:03d}", 65536 + n % 9, 32 + n % 4, "a", fp, "k", "space", sha256=f"{n:064x}")
    for n, fp in enumerate(prints[pieces:]):
        # Some claims are a released piece's own print, so they match one another across accounts.
        print_ = prints[n] if n % 3 == 0 else fp
        index.claim(1 + n % 3, 65536 + n % 9, f"track-{n}", print_, "declared", "k")
    index.withdraw_model(65536 + 4)
    index.remove_output("gen-007", 65536 + 7)
    return index, rng, pool, prints


def _as_version_one(fresh: Index, path) -> None:
    """The fresh index's contents, written in the layout before version 2: a row per landmark, repeats and all."""
    old = sqlite3.connect(path)
    old.executescript(V1)
    cols = "generation_id, serial, payload, artist_id, released_at, released_by, kind, landmarks, withdrawn, removed_at, sha256"
    for row in fresh.db.execute(f"SELECT {cols} FROM outputs ORDER BY oid"):
        old.execute(f"INSERT INTO outputs ({cols}) VALUES ({','.join('?' * 11)})", tuple(row))
    for row in fresh.db.execute("SELECT o.generation_id AS g, l.hash AS h, l.offset_ms AS o, l.n AS n FROM landmarks l"
                                " JOIN outputs o ON o.oid = l.oid"):
        old.executemany("INSERT INTO fingerprints (hash, generation_id, offset_ms) VALUES (?, ?, ?)", [(row["h"], row["g"], row["o"])] * row["n"])
    for row in fresh.db.execute("SELECT * FROM claims"):
        old.execute("INSERT INTO claims VALUES (?, ?, ?, ?, ?, ?, ?, ?)", tuple(row))
    for row in fresh.db.execute("SELECT * FROM claim_landmarks"):
        old.executemany("INSERT INTO claim_fingerprints (hash, claim_id, offset_ms) VALUES (?, ?, ?)",
                        [(row["hash"], row["claim_id"], row["offset_ms"])] * row["n"])
    for row in fresh.db.execute("SELECT * FROM claim_shares"):
        old.execute("INSERT INTO claim_shares VALUES (?, ?, ?, ?)", tuple(row))
    old.execute("PRAGMA user_version = 1")
    old.commit()
    old.close()


def _answers(index: Index, rng, pool, prints) -> list:
    out = []
    for n in range(40):
        query = [[(rng.choice(pool), rng.randrange(0, 20000, 250)) for _ in range(rng.randint(1, 60))]]
        if n % 2:
            query[0] += [tuple(x) for x in prints[rng.randrange(len(prints))]]
        for payload in (None, 33):
            answer = index.verify(query, payload, sha256=f"{n:064x}")
            out.append(answer)
        out.append(index.check_claim(query, 1 + n % 3))
    out += [index.claims_of(65536 + s) for s in range(9)]
    return out


def test_an_index_from_before_moves_over_and_answers_as_it_did(tmp_path):
    fresh, rng, pool, prints = _filled(tmp_path / "fresh.sqlite3")
    _as_version_one(fresh, tmp_path / "old.sqlite3")
    before = (tmp_path / "old.sqlite3").stat().st_size
    moved = Index(tmp_path / "old.sqlite3")
    assert moved.db.execute("PRAGMA user_version").fetchone()[0] == INDEX_VERSION
    tables = {r[0] for r in moved.db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert not tables & {"fingerprints", "claim_fingerprints", "outputs_v1"}
    for table in ("outputs", "landmarks", "claims", "claim_landmarks", "claim_shares"):
        assert sorted(map(tuple, moved.db.execute(f"SELECT * FROM {table}"))) == sorted(map(tuple, fresh.db.execute(f"SELECT * FROM {table}"))), table
    assert _answers(moved, random.Random(1), pool, prints) == _answers(fresh, random.Random(1), pool, prints)
    assert (tmp_path / "old.sqlite3").stat().st_size < before
    Index(tmp_path / "old.sqlite3")


def test_a_repeated_landmark_still_votes_each_time(tmp_path):
    index = Index(tmp_path / "i.sqlite3")
    index.release("twice", 65536, 32, "a", [[7, 100], [7, 100], [8, 200]], "k", "space")
    assert index.db.execute("SELECT n FROM landmarks WHERE hash = 7").fetchone()[0] == 2
    assert index.output("twice")["landmarks"] == 3 and "oid" not in index.output("twice")
    assert sorted(index._candidates([[(7, 100), (8, 200)]], None)["twice"]) == [(7, 100), (7, 100), (8, 200)]


def test_no_lookup_scans_the_landmarks(tmp_path):
    """Every statement a verify and an import check run is planned as seeks into the landmarks' key."""
    index, rng, pool, prints = _filled(tmp_path / "i.sqlite3", pieces=60, claims=20)
    seen = []
    index.db.set_trace_callback(seen.append)
    query = [[tuple(x) for x in prints[3]] + [(rng.choice(pool), 0) for _ in range(20)]]
    index.verify(query, None)
    index.verify(query, 33)
    index.check_claim(query, 1)
    index.db.set_trace_callback(None)
    plans = [" | ".join(r[3] for r in index.db.execute("EXPLAIN QUERY PLAN " + s)) for s in seen
             if s.lstrip().upper().startswith("SELECT") and "landmarks" in s]
    assert plans
    for plan in plans:
        assert "SCAN f" not in plan and "SCAN landmarks" not in plan and "SCAN claim_landmarks" not in plan, plan
