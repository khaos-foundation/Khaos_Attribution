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


def _version_one_file(path, null_id=False):
    """A small index in the layout before version 2, one landmark row orphaned, optionally a piece with no id."""
    db = sqlite3.connect(path)
    db.executescript(V1)
    db.execute("INSERT INTO outputs (generation_id, serial, payload, artist_id, released_at, released_by, kind, landmarks)"
               " VALUES ('gen-a', 65536, 32, 'a', 'then', 'k', 'space', 3)")
    db.executemany("INSERT INTO fingerprints VALUES (?, ?, ?)", [(7, "gen-a", 100), (7, "gen-a", 100), (8, "gen-a", 200), (9, "gone", 0)])
    if null_id:
        db.execute("INSERT INTO outputs (generation_id, serial, payload, artist_id, released_at, released_by, kind, landmarks)"
                   " VALUES (NULL, 65537, 32, 'a', 'then', 'k', 'space', 0)")
    db.execute("PRAGMA user_version = 1")
    db.commit()
    db.close()


class _Interrupting:
    """The index's connection, but the statement named raises: a crash at that point of the start."""

    def __init__(self, real, statement, error):
        self.real, self.statement, self.error = real, statement, error

    def execute(self, sql, *args):
        if sql.strip().upper() == self.statement:
            raise self.error
        return self.real.execute(sql, *args)

    def __getattr__(self, name):
        return getattr(self.real, name)


@pytest.mark.parametrize("error", [KeyboardInterrupt(), sqlite3.OperationalError("disk full")])
def test_a_start_stopped_after_the_move_commits_leaves_a_version_two_file(tmp_path, monkeypatch, error):
    """The stamp is inside the move: a crash before the compaction leaves the new shape stamped 2, never 1."""
    from register_service.store import file_version
    path = tmp_path / "i.sqlite3"
    _version_one_file(path)
    real = Index.db
    monkeypatch.setattr(Index, "db", property(lambda self: _Interrupting(real.fget(self), "VACUUM", error)))
    if isinstance(error, KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            Index(path)
    else:
        Index(path)
    monkeypatch.undo()
    assert file_version(path) == INDEX_VERSION
    again = Index(path)
    assert sorted(again._candidates([[(7, 100), (8, 200)]], None)["gen-a"]) == [(7, 100), (7, 100), (8, 200)]


def test_a_move_says_what_it_left_behind_and_refuses_a_piece_with_no_id(tmp_path, caplog):
    from register_service.store import file_version
    _version_one_file(tmp_path / "ok.sqlite3")
    with caplog.at_level("WARNING", logger="register_service"):
        Index(tmp_path / "ok.sqlite3")
    assert "1 landmark rows name no released piece" in caplog.text and "moved to version 2" in caplog.text
    _version_one_file(tmp_path / "bad.sqlite3", null_id=True)
    with pytest.raises(sqlite3.IntegrityError):
        Index(tmp_path / "bad.sqlite3")
    assert "refused, nothing changed" in caplog.text
    assert file_version(tmp_path / "bad.sqlite3") == 1
    tables = {r[0] for r in sqlite3.connect(tmp_path / "bad.sqlite3").execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert {"fingerprints", "outputs"} <= tables and "landmarks" not in tables


def test_a_file_from_before_any_version_moves_over_too(tmp_path):
    """Version 0: no withdrawn, removed or hash columns, and no claim tables."""
    path = tmp_path / "i.sqlite3"
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE outputs (generation_id TEXT PRIMARY KEY, serial INTEGER NOT NULL, payload INTEGER NOT NULL,"
               " artist_id TEXT NOT NULL, released_at TEXT NOT NULL, released_by TEXT NOT NULL, kind TEXT NOT NULL, landmarks INTEGER NOT NULL)")
    db.execute("CREATE TABLE fingerprints (hash INTEGER NOT NULL, generation_id TEXT NOT NULL, offset_ms INTEGER NOT NULL)")
    db.execute("INSERT INTO outputs VALUES ('old', 65536, 32, 'a', 'then', 'k', 'space', 1)")
    db.execute("INSERT INTO fingerprints VALUES (5, 'old', 0)")
    db.commit()
    db.close()
    index = Index(path)
    assert index.output("old")["withdrawn"] == 0 and index.verify([[(5, 0)]], None)["grade"] in ("output", "none")
    assert index._candidates([[(5, 0)]], None) == {"old": [(5, 0)]}
    assert index.claims_of(65536) == {"claimed": 0, "shared": 0}
    assert index.db.execute("PRAGMA user_version").fetchone()[0] == INDEX_VERSION
