"""The outbox keeps an entry the register refused for now (403) and sends it later, never letting a removal overtake
the release it removes; an entry can go first; a refusal never reuses a number; an unknown kind blocks nothing; the
machine key is private from its first byte; a record's writer shares are fractions that add up."""
import os
import stat
import sys

import pytest

sys.path.insert(0, "src")
pytest.importorskip("cryptography")

from khaos_attribution import registry as R  # noqa: E402
from khaos_attribution.engines.worker_process import POSIX_ONLY  # noqa: E402
from tests.test_registry import _record  # noqa: E402


def _client(monkeypatch, answer):
    client = R.RegisterClient("https://register.example", R.SigningKey.generate())
    calls = []

    def post(path, body):
        calls.append((path, body))
        return answer(path, body)
    monkeypatch.setattr(client, "_post", post)
    return client, calls


def _forbidden_until(served: set):
    """A register that answers 403 to a release of a serial no 'served' has reached yet."""
    def answer(path, body):
        if path == "/api/served":
            served.add(body["serial"])
        if path == "/api/outputs" and body["serial"] not in served:
            raise R.RegisterError(403, "a Space releases outputs of the models it reports serving")
        if path == "/api/outputs/remove" and body["generation_id"] not in released:
            raise R.RegisterError(404, "no such released piece")
        if path == "/api/outputs":
            released.add(body["generation_id"])
        return {"ok": True}
    released: set = set()
    return answer


def test_a_403_is_held_not_set_aside_and_a_serving_report_behind_it_still_goes(monkeypatch, tmp_path):
    served: set = set()
    client, calls = _client(monkeypatch, _forbidden_until(served))
    box = R.Outbox(tmp_path)
    for i in range(3):
        box.append("release", {"generation_id": f"g{i}", "serial": 65536, "fingerprint": [[1, 2]], "nonce": f"n{i:015d}"})
    box.append("served", {"serial": 65536, "space_url": "https://space.example", "state": "serving", "nonce": "s" * 16})
    client.sync(box)
    assert [c[0] for c in calls] == ["/api/outputs", "/api/served"], "one call for the held serial, then the report"
    assert box.refused() == [] and len(box.pending()) == 3
    held = box.held()
    assert [e["body"]["generation_id"] for e in held] == ["g0"] and held[0]["held"]["status"] == 403
    client.sync(box)
    assert box.pending() == [] and box.held() == [] and box.refused() == []


def test_a_removal_never_overtakes_the_release_it_removes(monkeypatch, tmp_path):
    served: set = set()
    client, calls = _client(monkeypatch, _forbidden_until(served))
    box = R.Outbox(tmp_path)
    box.append("release", {"generation_id": "g1", "serial": 65536, "fingerprint": [[1, 2]], "nonce": "a" * 16})
    box.append("remove_output", {"generation_id": "g1", "nonce": "b" * 16})
    box.append("served", {"serial": 65536, "space_url": "https://space.example", "state": "serving", "nonce": "c" * 16})
    client.sync(box)
    assert box.refused() == [] and [e["kind"] for _, e in box.pending()] == ["release", "remove_output"]
    client.sync(box)
    assert box.refused() == [] and box.pending() == []
    assert [c[0] for c in calls][-2:] == ["/api/outputs", "/api/outputs/remove"]


def test_an_entry_can_go_first_and_a_pending_duplicate_moves_ahead(tmp_path):
    box = R.Outbox(tmp_path)
    box.append("release", {"generation_id": "g1"})
    served = box.append("served", {"serial": 1})
    box.append("release", {"generation_id": "g2"})
    assert box.append("served", {"serial": 1}, first=True) == served
    assert [e["kind"] for _, e in box.pending()] == ["served", "release", "release"]
    box.append("served", {"serial": 2}, first=True)
    assert [e["body"].get("serial") for _, e in box.pending()] == [1, 2, None, None], "firsts keep their own order"
    (path, _), *_ = box.pending()
    box.mark_sent(path, {})
    assert box.append("served", {"serial": 1}, first=True) == served and len(box.pending()) == 3, "a sent entry is never sent again"


def test_a_number_given_to_an_entry_later_refused_is_never_given_again(tmp_path):
    box = R.Outbox(tmp_path)
    box.append("claim", {"n": 1})
    (path, _), = box.pending()
    box.refuse(path, {"status": 422})
    box.append("claim", {"n": 2})
    (path2, _), = box.pending()
    assert path2.name.split("-")[0] != path.name.split("-")[0]


def test_an_entry_of_a_kind_this_client_does_not_know_is_set_aside(monkeypatch, tmp_path):
    client, calls = _client(monkeypatch, lambda path, body: {"ok": True})
    box = R.Outbox(tmp_path)
    box.append("a_newer_kind", {"x": 1})
    box.append("served", {"serial": 1, "space_url": "https://space.example", "state": "serving"})
    client.sync(box)
    assert box.pending() == [] and [e["kind"] for e in box.refused()] == ["a_newer_kind"]


@pytest.mark.skipif(os.name != "posix", reason=POSIX_ONLY)
def test_the_machine_key_is_private_even_when_chmod_fails(monkeypatch, tmp_path):
    def no_chmod(*a, **k):
        raise OSError("not supported here")
    monkeypatch.setattr(os, "chmod", no_chmod)
    old = os.umask(0o022)
    try:
        R.SigningKey.load_or_create(tmp_path / "key.pem")
    finally:
        os.umask(old)
    assert stat.S_IMODE((tmp_path / "key.pem").stat().st_mode) == 0o600


@pytest.mark.parametrize("writers", [[{"name": "W", "share": 1.5}], [{"name": "W", "share": -0.1}],
                                     [{"name": "W", "share": 0.7}, {"name": "V", "share": 0.5}]])
def test_writer_shares_are_fractions_that_add_up_to_no_more_than_the_whole(writers):
    rec = _record()
    rec["training_set"]["tracks"][0]["writers"] = writers
    assert any("share" in p for p in R.record_problems(rec))


def test_writer_shares_from_rights_that_sum_to_100_within_their_tolerance_are_taken():
    rec = _record()
    rec["training_set"]["tracks"][0]["writers"] = [{"name": "W", "share": 50.005 / 100}, {"name": "V", "share": 50.005 / 100}]
    assert R.record_problems(rec) == []


def test_the_unused_public_key_file_name_is_gone():
    assert not hasattr(R, "MACHINE_KEY_PUBLIC_FILE")
