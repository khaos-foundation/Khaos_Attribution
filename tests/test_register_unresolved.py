"""The unresolved models are read in a bounded number of queries, whatever the log's size, and are exactly the
models the per-record reading names: every public page and every Space's poll asks for them."""
import hashlib
import json
import random

import pytest

pytest.importorskip("cryptography")
pytest.importorskip("fastapi")
pytest.importorskip("rfc8785")

from khaos_attribution import translog as T  # noqa: E402
from register_service.store import Store  # noqa: E402


def _reference_records(log) -> set:
    """The reading record by record, as admit_states gives it: the rule the one-pass reading must equal."""
    out = set()
    for r in log.db.execute("SELECT DISTINCT record_sha256 FROM log_leaves WHERE kind = 'admit'").fetchall():
        states = set(log.admit_states(r["record_sha256"]).values())
        if "unresolved" in states and "standing" not in states:
            out.add(r["record_sha256"])
    return out


def _reference_serials(store) -> list:
    shas = _reference_records(store.log)
    return sorted(r["serial"] for r in store.db.execute("SELECT serial, record_sha256 FROM records").fetchall()
                  if r["record_sha256"] in shas)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _generated(tmp_path, seed: int, steps: int, lost_share: float) -> Store:
    """A log of admits, re-admits, strikes, lost leaves and the wardens' answers in a random order. An admit's
    pending_lost names only leaves the log already holds as lost, as the register writes it."""
    rng = random.Random(seed)
    store = Store(tmp_path / f"r{seed}.sqlite3")
    log = store.log
    admits, lost, serial = [], [], 65536
    with store._tx():
        for step in range(steps):
            roll = rng.random()
            if roll < lost_share:
                leaf = _sha(f"lost-{seed}-{step}")
                log.replay_leaf_locked(leaf)
                lost.append(leaf)
            elif roll < lost_share + 0.12 and admits:
                leaf, n, sha = rng.choice(admits)
                if not log.db.execute("SELECT 1 FROM log_leaves WHERE kind = 'revoke' AND refers = ?", (leaf,)).fetchone():
                    entry = T.make_entry("revoke", revokes=leaf, serial=n, record_sha256=sha, reason="struck")
                    log._insert_leaf(T.jcs(entry), entry)
            elif roll < lost_share + 0.30 and lost:
                fields = {"answers": rng.choice(lost), "sitting": "S1"}
                if admits and rng.random() < 0.6:
                    leaf, n, sha = rng.choice(admits)
                    fields.update(confirms=leaf, serial=n, record_sha256=sha)
                entry = T.make_entry("confirm", **fields)
                log._insert_leaf(T.jcs(entry), entry)
            else:
                if admits and rng.random() < 0.25:
                    _, n, sha = rng.choice(admits)
                else:
                    serial += 1
                    n, sha = serial, _sha(f"record-{seed}-{serial}")
                    if rng.random() < 0.9:
                        store.db.execute("INSERT INTO records (serial, record_sha256, machine_key_id, account_id, signed_json,"
                                         " counter_json, state, published_at) VALUES (?, ?, 'k', 1, '{}', '{}', 'registered', 't')",
                                         (n, sha))
                pending = rng.sample(lost, rng.randint(0, min(2, len(lost)))) if lost and rng.random() < 0.5 else None
                entry = T.make_entry("admit", serial=n, record_sha256=sha, pending_lost=pending)
                _, leaf = log._insert_leaf(T.jcs(entry), entry)
                admits.append((leaf, n, sha))
    return store


@pytest.mark.parametrize("seed", range(40))
def test_one_pass_names_the_models_the_per_record_reading_names(tmp_path, seed):
    store = _generated(tmp_path, seed, steps=120, lost_share=(0.0, 0.03, 0.1, 0.25)[seed % 4])
    assert set(store.log.unresolved_records()) == _reference_records(store.log)
    assert store.unresolved_serials() == _reference_serials(store)


def test_the_generated_logs_hold_every_state_the_rule_tells_apart(tmp_path):
    """The comparison above means something only if the logs it reads have unresolved, standing and struck admits."""
    seen, some_unresolved = set(), 0
    for seed in range(40):
        store = _generated(tmp_path, seed, steps=120, lost_share=(0.0, 0.03, 0.1, 0.25)[seed % 4])
        some_unresolved += bool(store.unresolved_serials())
        for r in store.db.execute("SELECT DISTINCT record_sha256 FROM log_leaves WHERE kind = 'admit'").fetchall():
            seen.update(store.log.admit_states(r["record_sha256"]).values())
    assert seen == {"unresolved", "standing", "struck"} and some_unresolved >= 10


def _statements(store, call) -> int:
    seen = []
    store.db.set_trace_callback(seen.append)
    try:
        call()
    finally:
        store.db.set_trace_callback(None)
    return len(seen)


def test_a_log_that_lost_nothing_is_answered_without_reading_its_admits(tmp_path):
    counts = []
    for seed, steps in ((1000, 60), (1001, 600)):
        store = _generated(tmp_path, seed, steps=steps, lost_share=0.0)
        assert store.unresolved_serials() == []
        counts.append(_statements(store, store.unresolved_serials))
    assert counts[0] == counts[1] <= 2, counts


def test_a_log_with_losses_is_read_in_passes_not_once_per_model(tmp_path):
    """The number of queries follows the wardens' answers, never the number of admitted models."""
    small = _generated(tmp_path, 2000, steps=80, lost_share=0.1)
    large = _generated(tmp_path, 2001, steps=1200, lost_share=0.1)
    for store in (small, large):
        assert store.unresolved_serials() == _reference_serials(store) != []
    admits = lambda s: s.db.execute("SELECT COUNT(*) FROM log_leaves WHERE kind = 'admit'").fetchone()[0]  # noqa: E731
    confirms = lambda s: s.db.execute("SELECT COUNT(*) FROM log_leaves WHERE kind = 'confirm'").fetchone()[0]  # noqa: E731
    assert admits(large) > 5 * admits(small)
    assert _statements(large, large.unresolved_serials) <= 8 + confirms(large) + len(large.unresolved_serials()) // 400


def test_the_kinds_are_indexed(tmp_path):
    store = Store(tmp_path / "r.sqlite3")
    plan = " ".join(r["detail"] for r in store.db.execute(
        "EXPLAIN QUERY PLAN SELECT idx, leaf_hash FROM log_leaves WHERE kind = 'unavailable'"))
    assert "USING" in plan and "INDEX" in plan and "SCAN" not in plan, plan
