"""Publishes arriving at once: every record gets a serial of its own, the serials have no gap, each has one admit in
a log whose tree and heads still re-derive, and the same record sent twice at once is one record."""
import hashlib
import threading

import pytest

pytest.importorskip("cryptography")
pytest.importorskip("fastapi")
pytest.importorskip("rfc8785")

from khaos_attribution import registry as R  # noqa: E402
from register_service.store import Store, now_utc  # noqa: E402


def _signed(name: str) -> dict:
    sha = hashlib.sha256(name.encode()).hexdigest()
    return {"record_sha256": sha, "record": {"artist_id": name, "public_view": "model", "model": {"watermark_payload": 40}}}


def test_publishes_at_once_get_serials_of_their_own_and_leave_a_whole_log(tmp_path):
    store = Store(tmp_path / "register.sqlite3")
    store.log.signer = R.SigningKey.generate()
    with store._tx():
        store.log.remember_key_locked()
        store.log.genesis_locked()
    account = store.create_account("artist")
    key = R.SigningKey.generate()
    threads, per_thread, start = 8, 12, threading.Barrier(8)
    answers, errors = [], []

    def publish(t: int) -> None:
        start.wait(10)
        for i in range(per_thread):
            # Every third record is also sent by the next thread at the same time.
            for name in (f"record-{t}-{i}", f"shared-{(t + (i % 3 == 0)) % threads}-{i}"):
                signed = _signed(name)
                try:
                    serial, counter, already = store.register(signed, "k", account,
                                                              lambda n, at, s=signed: R.countersign(s, n, key, at or now_utc()))
                    answers.append((name, serial, counter["serial"], already))
                except Exception as exc:  # noqa: BLE001 — collected, the test fails on any
                    errors.append(repr(exc))
    workers = [threading.Thread(target=publish, args=(t,)) for t in range(threads)]
    for w in workers:
        w.start()
    for w in workers:
        w.join(60)
    assert errors == []
    by_name: dict = {}
    for name, serial, countersigned, _ in answers:
        assert serial == countersigned
        by_name.setdefault(name, set()).add(serial)
    assert all(len(serials) == 1 for serials in by_name.values()), "one record, one serial, however many times it was sent"
    serials = sorted(next(iter(s)) for s in by_name.values())
    assert serials == list(range(R.FIRST_SERIAL, R.FIRST_SERIAL + len(by_name))), "no serial twice, none skipped"
    assert sum(1 for *_, already in answers if not already) == len(by_name)
    assert store.serials() == serials and store.log.size() == 1 + len(by_name)
    for name in by_name:
        assert list(store.log.admit_states(_signed(name)["record_sha256"]).values()) == ["standing"]
    assert store.log.problems(store.log.signer.public_pem()) == [] and store.ledger_ok()
    assert [h["tree_size"] for h in store.log.heads()] == list(range(1, store.log.size() + 1)), "a signed head for every size"
