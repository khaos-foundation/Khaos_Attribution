"""The index picks its candidates in the database: the pieces sharing most hashes with a query, and only their
shared landmarks, reach Python, and they are the pieces and landmarks the row-by-row reading picked."""
import random

import pytest

pytest.importorskip("cryptography")
pytest.importorskip("fastapi")

import register_service.index as I  # noqa: E402
from register_service.index import Index  # noqa: E402


def _reference(index, phases, payload):
    """Every matching row read into Python, then the cut: the reading the database's own cut must equal."""
    hashes = sorted({h for q in phases for h, _ in q})
    rows: dict = {}
    for i in range(0, len(hashes), 900):
        chunk = hashes[i:i + 900]
        q = ("SELECT f.generation_id AS g, f.hash AS h, f.offset_ms AS o FROM fingerprints f"
             " JOIN outputs ou ON ou.generation_id = f.generation_id WHERE ou.withdrawn = 0 AND f.hash IN (%s)" % ",".join("?" * len(chunk)))
        params = list(chunk)
        if payload is not None:
            q += " AND ou.payload = ?"
            params.append(payload)
        for row in index.db.execute(q, params):
            rows.setdefault(row["g"], []).append((row["h"], row["o"]))
    top = sorted(rows, key=lambda g: len(rows[g]), reverse=True)[:I.CANDIDATES]
    return {g: rows[g] for g in top}


def _filled(tmp_path, seed: int, pieces: int = 40) -> tuple:
    rng = random.Random(seed)
    index = Index(tmp_path / f"i{seed}.sqlite3")
    pool = [rng.randrange(1 << 25) for _ in range(400)]
    prints = []
    for n in range(pieces):
        fp = [[rng.choice(pool), rng.randrange(0, 60000, 50)] for _ in range(rng.randint(5, 120))]
        index.release(f"g{rng.randrange(10**6):06d}-{n}", 65536 + n % 7, 32 + n % 3, "a", fp, "k", "space")
        prints.append(fp)
    index.withdraw_model(65536 + 3)
    return index, pool, prints, rng


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("cut", [3, 10, 50])
def test_the_databases_cut_is_the_cut_of_the_row_by_row_reading(tmp_path, monkeypatch, seed, cut):
    """Ties at the cut included: the same pieces in the same order, each with the same landmarks in the same order."""
    monkeypatch.setattr(I, "CANDIDATES", cut)
    index, pool, prints, rng = _filled(tmp_path, seed)
    for _ in range(6):
        query = [[(rng.choice(pool), rng.randrange(0, 60000, 50)) for _ in range(rng.randint(1, 300))] for _ in range(rng.randint(1, 4))]
        if rng.random() < 0.4:
            query[0] += [tuple(x) for x in rng.choice(prints)]
        for payload in (None, 32, 33):
            mine, theirs = index._candidates(query, payload), _reference(index, query, payload)
            assert list(mine) == list(theirs)
            assert mine == theirs
            assert index.verify(query, payload)["grade"] in ("output", "bucket", "none")
    assert index._candidates([[]], None) == {}


def test_only_the_candidates_landmarks_leave_the_database(tmp_path, monkeypatch):
    """A query that shares hashes with every piece still reads the rows of the pieces it keeps, and no others."""
    monkeypatch.setattr(I, "CANDIDATES", 2)
    index, pool, prints, rng = _filled(tmp_path, 99, pieces=60)
    query = [[(h, 0) for h in pool]]
    kept = index._candidates(query, None)
    everything = _reference(index, query, None)
    monkeypatch.setattr(I, "CANDIDATES", 10 ** 6)
    matching = sum(len(v) for v in _reference(index, query, None).values())
    fetched = []
    real = index.db
    class Counting:
        def execute(self, sql, params=()):
            cur = real.execute(sql, params)
            if sql.lstrip().upper().startswith("SELECT"):
                rows = cur.fetchall()
                fetched.append(len(rows))
                return iter(rows)
            return cur
        def executemany(self, sql, seq):
            return real.executemany(sql, seq)
    monkeypatch.setattr(I, "CANDIDATES", 2)
    monkeypatch.setattr(Index, "db", property(lambda self: Counting()))
    again = index._candidates(query, None)
    assert again == kept == everything and len(kept) == 2
    assert sum(fetched) <= 2 + sum(len(v) for v in kept.values()) < matching / 5
