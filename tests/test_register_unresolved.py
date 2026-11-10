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


def _work(store, call) -> int:
    """The database's own count of the work a call made it do: virtual-machine steps, in hundreds."""
    ticks = [0]

    def tick():
        ticks[0] += 1
        return 0
    store.db.set_progress_handler(tick, 100)
    try:
        call()
    finally:
        store.db.set_progress_handler(None, 0)
    return ticks[0]


def _grown(tmp_path, name: str, models: int, lost: str | None) -> Store:
    """``models`` admitted records; with ``lost``, one leaf lost half-way and what the wardens did about it:
    'answered' for every admit, or 'open'. Two later admits name the loss either way."""
    store = Store(tmp_path / f"{name}.sqlite3")
    log = store.log
    with store._tx():
        for n in range(models):
            sha = _sha(f"{name}-{n}")
            store.db.execute("INSERT INTO records (serial, record_sha256, machine_key_id, account_id, signed_json, counter_json,"
                             " state, published_at) VALUES (?, ?, 'k', 1, '{}', '{}', 'registered', 't')", (65536 + n, sha))
            entry = T.make_entry("admit", serial=65536 + n, record_sha256=sha)
            log._insert_leaf(T.jcs(entry), entry)
            if lost and n == models // 2:
                leaf = _sha(f"{name}-lost")
                log.replay_leaf_locked(leaf)
        if lost:
            for n in (models, models + 1):
                sha = _sha(f"{name}-{n}")
                store.db.execute("INSERT INTO records (serial, record_sha256, machine_key_id, account_id, signed_json, counter_json,"
                                 " state, published_at) VALUES (?, ?, 'k', 1, '{}', '{}', 'registered', 't')", (65536 + n, sha))
                entry = T.make_entry("admit", serial=65536 + n, record_sha256=sha, pending_lost=[leaf])
                log._insert_leaf(T.jcs(entry), entry)
            if lost == "answered":
                entry = T.make_entry("confirm", answers=leaf, sitting="S1")
                log._insert_leaf(T.jcs(entry), entry)
    return store


def test_a_log_that_lost_nothing_costs_the_same_whatever_its_size(tmp_path):
    """Measured in the database's own work, so reading every record to answer 'none' is seen."""
    small, large = _grown(tmp_path, "small", 50, None), _grown(tmp_path, "large", 2000, None)
    assert small.unresolved_serials() == large.unresolved_serials() == []
    assert _work(large, large.unresolved_serials) <= _work(small, small.unresolved_serials) + 1


def test_an_answered_loss_costs_by_the_loss_not_by_the_log(tmp_path):
    """One lost leaf the wardens have answered for every model: nothing is under review, and finding that out reads
    the loss, its answer and the admits that named it, however many models the log holds."""
    small, large = _grown(tmp_path, "small", 50, "answered"), _grown(tmp_path, "large", 2000, "answered")
    for store in (small, large):
        assert store.unresolved_serials() == _reference_serials(store) == []
    assert _work(large, large.unresolved_serials) <= 2 * _work(small, small.unresolved_serials) + 2


def test_an_open_loss_puts_the_models_before_it_and_the_ones_naming_it_under_review(tmp_path):
    store = _grown(tmp_path, "open", 60, "open")
    assert store.unresolved_serials() == _reference_serials(store) == [65536 + n for n in range(31)] + [65536 + 60, 65536 + 61]


def test_a_log_from_before_the_waiting_mark_is_marked_when_it_is_opened(tmp_path):
    """An older database has no mark of which admits name a loss; opening it fills the mark from the bodies."""
    store = _grown(tmp_path, "old", 40, "open")
    expected = store.unresolved_serials()
    store.db.execute("UPDATE log_leaves SET waits = NULL")
    reopened = Store(store.path)
    assert reopened.unresolved_serials() == expected == _reference_serials(reopened)
    assert reopened.db.execute("SELECT COUNT(*) FROM log_leaves WHERE kind = 'admit' AND waits IS NULL").fetchone()[0] == 0


def test_the_kinds_are_indexed(tmp_path):
    store = Store(tmp_path / "r.sqlite3")
    plan = " ".join(r["detail"] for r in store.db.execute(
        "EXPLAIN QUERY PLAN SELECT idx, leaf_hash FROM log_leaves WHERE kind = 'unavailable'"))
    assert "USING" in plan and "INDEX" in plan and "SCAN" not in plan, plan
