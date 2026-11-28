"""The hub is asked first at the pinned revision, the mirror second, and a file is installed only when it matches its pin;
nothing here reaches the real hub or the real mirror, and no request ever carries a credential."""
import errno
import hashlib
import http.server
import importlib.util
import json
import os
import re
import threading
import urllib.error
from pathlib import Path

import pytest

from khaos_attribution import weights as W

REPO, REV = "org/pinned", "a" * 40
SMALL = b'{"x": 1}'
BIG = bytes((i * 7 + i // 251) % 256 for i in range(300_000))
FILES = {"config.json": SMALL, "w/big.bin": BIG}


def _pins(files=FILES, repo=REPO):
    return {repo: {"revision": REV, "files": tuple(
        (p, len(d), hashlib.sha256(d).hexdigest()) for p, d in files.items())}}


class _Fake(http.server.BaseHTTPRequestHandler):
    """A hub (kind "hub": /<repo>/resolve/<rev>/<path>, optionally through a redirect) or a mirror (/<repo>/<rev>/<path>)."""
    protocol_version = "HTTP/1.0"

    def log_message(self, *a):
        pass

    def do_GET(self):
        s = self.server.state
        s["seen"].append({"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()}})
        if self.path.startswith("/cdn/"):
            name = self.path[len("/cdn/"):]
        elif s["kind"] == "hub":
            rest = self.path.split("/resolve/", 1)[1] if "/resolve/" in self.path else ""
            name = rest.partition("/")[2]
            if s["redirect"]:
                self.send_response(302)
                self.send_header("Location", f"/cdn/{name}")
                self.end_headers()
                return
        elif self.path == f"/{REPO}/manifest.json" and s.get("manifest"):
            body = json.dumps(s["manifest"]).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        else:
            name = self.path.removeprefix(f"/{REPO}/{REV}/")
        mode = s["mode"]
        if isinstance(mode, int):
            self.send_error(mode)
            return
        if isinstance(mode, list) and mode and isinstance(mode[0], int):
            self.send_error(mode.pop(0))
            return
        data = s["files"].get(name)
        if data is None:
            self.send_error(404)
            return
        if mode == "corrupt":
            data = bytes(b ^ 1 for b in data)
        start = 0
        asked = self.headers.get("Range")
        if asked:
            start = int(asked.removeprefix("bytes=").rstrip("-"))
        body = data[start:]
        self.send_response(206 if asked else 200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        cut = s["drops"].pop(0) if s["drops"] and len(body) > 50_000 else None
        self.wfile.write(body if cut is None else body[:cut])


def _serve(kind, files=FILES, mode="ok", redirect=False, drops=()):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Fake)
    server.state = {"kind": kind, "files": dict(files), "mode": mode, "redirect": redirect, "drops": list(drops), "seen": []}
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@pytest.fixture
def rig(tmp_path, monkeypatch):
    """A pin for org/pinned, a fake hub and a fake mirror holding its files, and a login that must never be sent."""
    hub, mirror = _serve("hub"), _serve("mirror")
    monkeypatch.setattr(W, "PINS", _pins())
    monkeypatch.setenv("HF_ENDPOINT", f"http://127.0.0.1:{hub.server_address[1]}")
    monkeypatch.setenv("HF_TOKEN", "hf_secret_token_value")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hfhome"))
    (tmp_path / "hfhome").mkdir()
    (tmp_path / "hfhome" / "token").write_text("hf_secret_file_token")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    sleeps = []
    monkeypatch.setattr(W, "_sleep", sleeps.append)
    W._skip.clear()

    class Rig:
        pass

    r = Rig()
    r.hub, r.mirror, r.sleeps = hub, mirror, sleeps
    r.mirror_url = f"http://127.0.0.1:{mirror.server_address[1]}"
    yield r
    hub.shutdown()
    mirror.shutdown()


def _fetch(rig, dest, **kw):
    log = []
    return W.fetch_into_dir(REPO, dest, base=rig.mirror_url, progress=log.append, **kw), log


def _no_credential(*servers):
    for server in servers:
        for seen in server.state["seen"]:
            assert not {"authorization", "cookie", "proxy-authorization"} & set(seen["headers"]), seen
            assert "hf_secret" not in json.dumps(seen)


def test_a_file_comes_from_the_hub_at_the_pinned_revision_and_the_mirror_is_not_asked(rig, tmp_path):
    served, log = _fetch(rig, tmp_path / "d")
    assert served == REV
    assert (tmp_path / "d" / "w" / "big.bin").read_bytes() == BIG and (tmp_path / "d" / "config.json").read_bytes() == SMALL
    assert rig.mirror.state["seen"] == []
    assert all(s["path"].startswith(f"/{REPO}/resolve/{REV}/") for s in rig.hub.state["seen"])
    assert "big.bin: from Hugging Face" in log


def test_no_request_to_either_host_ever_carries_a_credential_or_names_this_project(rig, tmp_path):
    """HF_TOKEN and a saved login are set; neither is read, and the hub is sent no name for this project."""
    rig.hub.state["mode"] = rig.mirror.state["mode"] = "corrupt"
    with pytest.raises(W.PinMismatch):
        _fetch(rig, tmp_path / "d")
    assert rig.hub.state["seen"] and rig.mirror.state["seen"], "both hosts were asked"
    _no_credential(rig.hub, rig.mirror)
    for seen in rig.hub.state["seen"]:
        assert "khaos" not in seen["headers"].get("user-agent", "").lower()


def test_a_hub_that_redirects_to_its_cdn_is_followed_and_a_partial_copy_resumes_across_the_redirect(rig, tmp_path):
    rig.hub.state["redirect"] = True
    dest = tmp_path / "d"
    (dest / "w").mkdir(parents=True)
    (dest / "w" / "big.bin.part").write_bytes(BIG[:100_000])
    assert _fetch(rig, dest)[0] == REV
    assert (dest / "w" / "big.bin").read_bytes() == BIG and not (dest / "w" / "big.bin.part").exists()
    cdn = [s for s in rig.hub.state["seen"] if s["path"] == "/cdn/w/big.bin"]
    assert [s["headers"].get("range") for s in cdn] == ["bytes=100000-"], "the CDN was asked for the rest only"
    _no_credential(rig.hub)


def test_a_hub_file_that_fails_its_pin_is_discarded_and_the_mirror_serves_the_right_one(rig, tmp_path):
    rig.hub.state["mode"] = "corrupt"
    served, log = _fetch(rig, tmp_path / "d")
    assert served == REV and (tmp_path / "d" / "w" / "big.bin").read_bytes() == BIG
    assert not list((tmp_path / "d").rglob("*.part"))
    assert any("did not match the pinned hash" in line for line in log) and "big.bin: from the Khaos mirror" in log


def test_a_file_no_source_serves_correctly_is_never_installed_and_no_fallback_is_allowed(rig, tmp_path):
    rig.hub.state["mode"] = rig.mirror.state["mode"] = "corrupt"
    with pytest.raises(W.PinMismatch, match="did not match"):
        _fetch(rig, tmp_path / "d")
    assert not (tmp_path / "d" / "config.json").exists() and not list((tmp_path / "d").rglob("*.part"))
    code = W.main(["fetch", REPO, "--into", str(tmp_path / "cli"), "--base", rig.mirror_url])
    assert code == 1, "the command line says a mismatch, never the exit code that sends a script to the hub library"


def test_a_hub_that_has_become_gated_is_asked_once_for_the_set_then_the_mirror_serves_the_rest(rig, tmp_path):
    rig.hub.state["mode"] = 401
    served, log = _fetch(rig, tmp_path / "d")
    assert served == REV and (tmp_path / "d" / "w" / "big.bin").read_bytes() == BIG
    assert len(rig.hub.state["seen"]) == 1, "one 401, then the rest of the set went straight to the mirror"
    assert any("needs an account" in line for line in log)
    _no_credential(rig.hub)


def test_the_hub_is_asked_again_once_a_skip_has_expired(rig, tmp_path, monkeypatch):
    rig.hub.state["mode"] = 403
    clock = [1000.0]
    monkeypatch.setattr(W, "_now", lambda: clock[0])
    _fetch(rig, tmp_path / "a")
    asked = len(rig.hub.state["seen"])
    _fetch(rig, tmp_path / "b")
    assert len(rig.hub.state["seen"]) == asked, "still skipped"
    clock[0] += W.SKIP_SECONDS + 1
    rig.hub.state["mode"] = "ok"
    _fetch(rig, tmp_path / "c")
    assert len(rig.hub.state["seen"]) > asked and (tmp_path / "c" / "w" / "big.bin").read_bytes() == BIG


def test_a_blocked_hub_costs_one_failure_for_the_whole_install_then_the_mirror_serves(rig, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", "http://127.0.0.1:9")
    served, log = _fetch(rig, tmp_path / "d")
    assert served == REV and (tmp_path / "d" / "w" / "big.bin").read_bytes() == BIG
    assert [line for line in log if "did not answer" in line] == ["Hugging Face did not answer; using the Khaos mirror"]
    assert rig.sleeps == [], "a host that never answers is not waited on"


def test_a_hub_server_error_goes_to_the_mirror_without_a_retry_and_a_mirror_one_is_retried(rig, tmp_path):
    rig.hub.state["mode"] = 503
    rig.mirror.state["mode"] = [503, 503]
    assert _fetch(rig, tmp_path / "d")[0] == REV
    assert len(rig.hub.state["seen"]) == 1
    assert rig.sleeps == [1.0, 2.0], "the mirror's two 503s were waited out, the hub's was not"


def test_a_download_that_breaks_after_bytes_flowed_resumes_with_a_range(rig, tmp_path):
    rig.hub.state["drops"] = [60_000, 90_000]
    assert _fetch(rig, tmp_path / "d")[0] == REV
    assert (tmp_path / "d" / "w" / "big.bin").read_bytes() == BIG
    ranges = [s["headers"].get("range") for s in rig.hub.state["seen"] if s["path"].endswith("w/big.bin")]
    assert ranges == [None, "bytes=60000-", "bytes=150000-"]
    assert rig.sleeps == [1.0, 2.0] and rig.mirror.state["seen"] == []


def test_a_stream_that_keeps_breaking_gives_up_on_that_source_and_the_mirror_serves(rig, tmp_path):
    rig.hub.state["drops"] = [1_000] * 10
    assert _fetch(rig, tmp_path / "d")[0] == REV
    assert (tmp_path / "d" / "w" / "big.bin").read_bytes() == BIG
    assert len([s for s in rig.hub.state["seen"] if s["path"].endswith("big.bin")]) == W.ATTEMPTS


def test_offline_means_the_hub_is_not_asked(rig, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert _fetch(rig, tmp_path / "d")[0] == REV
    assert rig.hub.state["seen"] == [] and rig.mirror.state["seen"]


def test_a_set_with_no_pin_is_served_by_the_mirror_alone_as_before(rig, tmp_path, monkeypatch):
    monkeypatch.setattr(W, "PINS", {})
    rig.mirror.state["manifest"] = {"repo": REPO, "revision": REV, "files": [
        {"path": p, "size": len(d), "sha256": hashlib.sha256(d).hexdigest()} for p, d in FILES.items()]}
    assert W.fetch_into_dir(REPO, tmp_path / "d", base=rig.mirror_url) == REV
    assert (tmp_path / "d" / "w" / "big.bin").read_bytes() == BIG and rig.hub.state["seen"] == []
    rig.mirror.state["mode"] = "corrupt"
    with pytest.raises(W.MirrorError) as raised:
        W.fetch_into_dir(REPO, tmp_path / "e", base=rig.mirror_url)
    assert not isinstance(raised.value, W.PinMismatch)


def test_a_revision_other_than_the_pin_is_not_served_from_the_pin(rig, tmp_path):
    assert W.fetch_into_dir(REPO, tmp_path / "d", base=rig.mirror_url, revision="b" * 40) is None
    assert rig.hub.state["seen"] == []


def test_a_hub_address_is_fetched_through_the_pin_by_the_same_rule(rig, tmp_path):
    assert W.fetch_url(f"https://huggingface.co/{REPO}/resolve/{REV}/w/big.bin", tmp_path / "big.bin", base=rig.mirror_url)
    assert (tmp_path / "big.bin").read_bytes() == BIG and rig.mirror.state["seen"] == []
    assert not W.fetch_url(f"https://huggingface.co/{REPO}/resolve/{'b' * 40}/w/big.bin", tmp_path / "x", base=rig.mirror_url)


def test_the_hub_cache_layout_is_built_from_hub_bytes_too(rig, tmp_path):
    assert W.fetch_into_hub_cache(REPO, tmp_path / "hub", base=rig.mirror_url) == REV
    snap = tmp_path / "hub" / "models--org--pinned" / "snapshots" / REV
    assert (snap / "w" / "big.bin").read_bytes() == BIG and (tmp_path / "hub" / "models--org--pinned" / "refs" / "main").read_text() == REV


def test_the_check_command_reads_the_pin_without_any_network(rig, capsys):
    assert W.main(["check", REPO, "--base", "http://127.0.0.1:9"]) == 0
    assert capsys.readouterr().out.strip() == REV


# ── anonymity ────────────────────────────────────────────────────────────────


def test_the_client_never_reads_or_sends_a_credential():
    """It holds no code that reads a token or names an Authorization header; its only token words are the ones it removes."""
    source = Path(W.__file__).read_text()
    assert "authorization" not in source.lower()
    assert not re.search(r"environ(\.get)?\W+[\"']HF_TOKEN", source)


def test_the_environment_given_to_the_hub_library_holds_no_token_and_turns_the_saved_login_off():
    env = W.anonymous_hub_env({"HF_TOKEN": "x", "HUGGING_FACE_HUB_TOKEN": "y", "HUGGINGFACE_HUB_TOKEN": "z", "PATH": "/bin"})
    assert env == {"PATH": "/bin", "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1", "NETRC": os.devnull}


def test_inside_anonymous_hub_the_hub_library_builds_no_authorization_header(tmp_path, monkeypatch):
    hub = pytest.importorskip("huggingface_hub")
    constants = pytest.importorskip("huggingface_hub.constants")
    monkeypatch.setenv("HF_TOKEN", "hf_secret_token_value")
    (tmp_path / "token").write_text("hf_secret_file_token")
    monkeypatch.setattr(constants, "HF_TOKEN_PATH", str(tmp_path / "token"))
    monkeypatch.setattr(constants, "HF_HUB_DISABLE_IMPLICIT_TOKEN", False)
    from huggingface_hub.utils import build_hf_headers
    assert "authorization" in {k.lower() for k in build_hf_headers()}, "the stand-in login is seen when not anonymous"
    with W.anonymous_hub():
        assert "authorization" not in {k.lower() for k in build_hf_headers()}
        assert "HF_TOKEN" not in os.environ
    assert os.environ["HF_TOKEN"] == "hf_secret_token_value" and constants.HF_HUB_DISABLE_IMPLICIT_TOKEN is False
    assert hub


# ── the pins ─────────────────────────────────────────────────────────────────


def _pin_script():
    spec = importlib.util.spec_from_file_location("pin_weights", Path(__file__).resolve().parents[1] / "scripts" / "pin_weights.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_pin_is_a_plain_set_of_files_with_a_commit_and_a_hash_each():
    from khaos_attribution.weights_pins import PINS
    assert len(PINS) >= 16
    for repo, pin in PINS.items():
        assert re.fullmatch(r"[0-9a-f]{40}", pin["revision"]), repo
        doc = W._pinned(repo)
        W._checked(repo, doc)
        paths = [f["path"] for f in doc["files"]]
        assert paths and len(paths) == len(set(paths)), repo


def test_the_pins_module_is_what_pin_weights_py_writes_from_the_pins():
    """scripts/pin_weights.py renders the module byte for byte, so a regeneration shows only what changed."""
    from khaos_attribution import weights_pins
    script = _pin_script()
    assert script.render(weights_pins.PINS) == Path(weights_pins.__file__).read_text()
    assert set(weights_pins.PINS) == set(script.REPOS)


# ── a wrong partial copy, a wrong length, no source at all ───────────────────


def test_a_stale_partial_copy_that_is_wrong_is_dropped_and_the_file_fetched_whole_before_anything_is_refused(rig, tmp_path):
    dest = tmp_path / "d"
    (dest / "w").mkdir(parents=True)
    (dest / "w" / "big.bin.part").write_bytes(b"\0" * 100_000)
    assert _fetch(rig, dest)[0] == REV and (dest / "w" / "big.bin").read_bytes() == BIG
    assert [s["headers"].get("range") for s in rig.hub.state["seen"] if s["path"].endswith("big.bin")] == ["bytes=100000-", None]


def test_a_hub_prefix_that_was_wrong_does_not_make_the_mirror_look_wrong(rig, tmp_path):
    """The hub sends wrong bytes and stalls; the mirror then continues from them, fails the pin and is refetched whole."""
    rig.hub.state["mode"] = "corrupt"
    rig.hub.state["drops"] = [1_000] * 10
    assert _fetch(rig, tmp_path / "d")[0] == REV
    assert (tmp_path / "d" / "w" / "big.bin").read_bytes() == BIG


def test_an_answer_whose_length_is_not_the_pinned_files_is_refused_without_reading_it(rig, tmp_path):
    rig.hub.state["files"] = {**FILES, "w/big.bin": BIG + b"x"}
    assert _fetch(rig, tmp_path / "d")[0] == REV and (tmp_path / "d" / "w" / "big.bin").read_bytes() == BIG
    assert len([s for s in rig.hub.state["seen"] if s["path"].endswith("big.bin")]) == 1
    assert not list((tmp_path / "d").rglob("*.part"))


def test_a_pinned_set_no_source_answers_for_is_not_left_to_the_hub_library(rig, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", "http://127.0.0.1:9")
    with pytest.raises(W.Unserved) as raised:
        W.fetch_into_dir(REPO, tmp_path / "d", base="http://127.0.0.1:9")
    assert raised.value.pinned and "Hugging Face" in str(raised.value) and "the Khaos mirror" in str(raised.value)
    assert W.main(["fetch", REPO, "--into", str(tmp_path / "cli"), "--base", "http://127.0.0.1:9"]) == W.EXIT_UNREACHABLE == 4
    monkeypatch.setattr(W, "PINS", {})
    assert W.main(["fetch", REPO, "--into", str(tmp_path / "cli2"), "--base", "http://127.0.0.1:9"]) == W.EXIT_NO_MIRROR == 3


def test_writing_pins_needs_the_hubs_word_for_them(capsys):
    with pytest.raises(SystemExit):
        _pin_script().main(["--write"])
    assert "--verify-hub" in capsys.readouterr().err


# ── the callers hold the refusal ─────────────────────────────────────────────


def test_a_caller_falls_back_to_the_hub_library_only_where_nothing_was_pinned():
    from khaos_attribution.engines.environment import from_mirror
    def raising(exc):
        def fetch():
            raise exc
        return fetch
    assert from_mirror(lambda: "rev") is True and from_mirror(lambda: None) is False
    assert from_mirror(raising(W.Unserved("none", pinned=False))) is False
    assert from_mirror(raising(W.MirrorError("odd manifest"))) is False
    for refusal in (W.PinMismatch("bad"), W.Unserved("none", pinned=True), W.NotPlaced("held"), W.NoSpace("full"),
                    OSError("x")):
        with pytest.raises(type(refusal)):
            from_mirror(raising(refusal))


@pytest.mark.parametrize("refusal", [W.PinMismatch("bad"), W.Unserved("no source answered (Connection refused)", pinned=True),
                                     W.NotPlaced("x could not be written"), W.NoSpace("No space left on the drive")])
def test_a_refused_set_is_reported_and_the_hub_library_is_never_run(monkeypatch, tmp_path, refusal):
    from khaos_attribution.engines.mimetic import provision as M
    from khaos_attribution.engines.sa3 import provision as S

    def refuse(fetch):
        raise refusal

    def never(*a, **k):
        raise AssertionError("the hub library ran after a refusal")
    monkeypatch.setattr(S, "venv_python", lambda *a: Path("py"))
    monkeypatch.setattr(S, "cached", lambda *a, **k: False)
    monkeypatch.setattr(S, "_from_mirror", refuse)
    monkeypatch.setattr(S, "_run", never)
    lines = S.weights(tmp_path, "torch", tmp_path)
    assert lines and all(not ok for ok, _ in lines) and all(("pinned hash" in note or "could not be downloaded" in note
                                                             or str(refusal) in note) for _, note in lines)
    monkeypatch.setattr(M, "venv_python", lambda *a: Path("py"))
    monkeypatch.setattr(M, "cached", lambda *a, **k: False)
    monkeypatch.setattr(M, "_from_mirror", refuse)
    monkeypatch.setattr(M, "_run", never)
    lines = M.weights(tmp_path)
    assert lines and all(not ok for ok, _ in lines)


def test_the_clap_text_prefetch_stops_on_a_refusal_and_does_not_call_the_hub_library(monkeypatch, tmp_path):
    import sys
    import types
    from khaos_attribution import embedding
    calls = []
    fake = types.SimpleNamespace(snapshot_download=lambda *a, **k: calls.append(a))
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    for refusal in (W.PinMismatch("bad"), W.Unserved("none", pinned=True), W.NotPlaced("held")):
        def refuse(*a, **k):
            raise refusal
        monkeypatch.setattr(W, "fetch_into_hub_cache", refuse)
        with pytest.raises(type(refusal)):
            embedding.clap_text_prefetch(tmp_path / "cache")
    assert calls == []


def test_every_revision_the_contract_names_for_a_pinned_set_is_the_pin():
    """A revision that differs from the pin silently leaves the set unpinned, so the descriptors and the CLAP lists agree with it."""
    from khaos_attribution import embedding
    from khaos_attribution.weights_pins import PINS
    named = [(repo, pin) for repo, pin, _ in embedding.CLAP_TEXT_SNAPSHOTS]
    named += [(w["repo"], w["revision"]) for path in Path(W.__file__).parent.glob("engines/*/engine.json")
              for w in json.loads(path.read_text())["weights"] if w.get("revision")]
    assert named
    assert [(repo, rev) for repo, rev in named if repo in PINS and PINS[repo]["revision"] != rev] == []


# ── a file this computer cannot write is never fetched again unchecked ───────


def _held(monkeypatch, times):
    """Path.replace of a .part refused `times` times (None: always), as Windows refuses a file another program holds."""
    real, calls = Path.replace, []

    def replace(self, target):
        if self.name.endswith(".part"):
            calls.append(self.name)
            if times is None or len(calls) <= times:
                raise PermissionError(13, "The process cannot access the file because it is being used by another process",
                                      str(self))
        return real(self, target)
    monkeypatch.setattr(Path, "replace", replace)
    return calls


def test_a_checked_file_another_program_holds_is_put_in_place_once_it_is_let_go(rig, tmp_path, monkeypatch):
    calls = _held(monkeypatch, 2)
    served, _ = _fetch(rig, tmp_path / "d")
    assert served == REV and (tmp_path / "d" / "w" / "big.bin").read_bytes() == BIG
    assert calls == ["config.json.part"] * 3 + ["big.bin.part"] and rig.sleeps == list(W.PLACE_WAITS[:2])
    assert [s["path"] for s in rig.hub.state["seen"]].count(f"/{REPO}/resolve/{REV}/w/big.bin") == 1


def test_a_checked_file_held_throughout_is_refused_plainly_kept_and_used_by_the_next_fetch(rig, tmp_path, monkeypatch):
    _held(monkeypatch, None)
    with pytest.raises(W.NotPlaced) as raised:
        _fetch(rig, tmp_path / "d")
    assert "matched its pin" in str(raised.value) and "antivirus" in str(raised.value)
    assert (tmp_path / "d" / "config.json.part").read_bytes() == SMALL and rig.mirror.state["seen"] == []
    assert not W.may_fall_back(raised.value)
    asked = len(rig.hub.state["seen"])
    monkeypatch.undo()
    monkeypatch.setattr(W, "PINS", _pins())
    monkeypatch.setenv("HF_ENDPOINT", f"http://127.0.0.1:{rig.hub.server_address[1]}")
    monkeypatch.setattr(W, "_sleep", lambda s: None)
    served, _ = _fetch(rig, tmp_path / "d")
    assert served == REV and (tmp_path / "d" / "config.json").read_bytes() == SMALL
    assert not (tmp_path / "d" / "config.json.part").exists()
    assert [s["path"] for s in rig.hub.state["seen"][asked:]].count(f"/{REPO}/resolve/{REV}/config.json") == 0


class _FullDrive:
    """A file whose writes fail as a full drive's do."""

    def __init__(self, fh):
        self.fh = fh

    def write(self, data):
        raise OSError(errno.ENOSPC, "No space left on device")

    def __getattr__(self, name):
        return getattr(self.fh, name)


def test_a_full_drive_says_no_space_left_and_no_source_is_blamed_or_skipped(rig, tmp_path, monkeypatch):
    real = Path.open
    monkeypatch.setattr(Path, "open", lambda self, mode="r", *a, **k: (
        _FullDrive(real(self, mode, *a, **k)) if self.name.endswith(".part") and "b" in mode and mode[0] in "aw"
        else real(self, mode, *a, **k)))
    log = []
    with pytest.raises(W.NoSpace) as raised:
        W.fetch_into_dir(REPO, tmp_path / "d", base=rig.mirror_url, progress=log.append)
    assert str(raised.value).startswith("No space left on the drive for config.json.part") and "GB free" in str(raised.value)
    assert len(rig.hub.state["seen"]) == 1 and rig.mirror.state["seen"] == [] and rig.sleeps == []
    assert W._skip == {} and not any("did not answer" in line for line in log)
    assert not W.may_fall_back(raised.value)


def test_a_connection_reset_mid_file_is_still_a_network_break_and_resumes(rig, tmp_path, monkeypatch):
    import errno as E
    real_get, resets = W._get, []

    def get(url, *a, **k):
        answer = real_get(url, *a, **k)
        if url.endswith("big.bin") and not resets:
            read = answer.read

            def broken(n=-1):
                if resets:
                    raise ConnectionResetError(E.ECONNRESET, "Connection reset by peer")
                resets.append(1)
                return read(100_000)
            answer.read = broken
        return answer
    monkeypatch.setattr(W, "_get", get)
    served, _ = _fetch(rig, tmp_path / "d")
    assert served == REV and (tmp_path / "d" / "w" / "big.bin").read_bytes() == BIG and resets
    ranges = [s["headers"].get("range") for s in rig.hub.state["seen"] if s["path"].endswith("big.bin")]
    assert ranges[0] is None and ranges[1] == "bytes=100000-"


def test_a_folder_that_cannot_be_made_or_a_link_that_cannot_be_placed_is_refused_as_this_computers(rig, tmp_path, monkeypatch):
    (tmp_path / "file").write_text("in the way")
    with pytest.raises(W.NotPlaced) as raised:
        _fetch(rig, tmp_path / "file" / "d")
    assert "on this computer" in str(raised.value)

    def refuse(blob, link):
        raise PermissionError(13, "Access is denied", str(link))
    monkeypatch.setattr(W, "_place_link", refuse)
    with pytest.raises(W.NotPlaced):
        W.fetch_into_hub_cache(REPO, tmp_path / "cache", base=rig.mirror_url)


def test_the_command_line_stops_on_a_file_it_cannot_put_in_place_and_never_says_fall_back(rig, tmp_path, monkeypatch, capsys):
    _held(monkeypatch, None)
    assert W.main(["fetch", REPO, "--into", str(tmp_path / "cli"), "--base", rig.mirror_url]) == W.EXIT_LOCAL == 5
    assert "antivirus" in capsys.readouterr().err
    monkeypatch.setattr(W, "fetch_into_dir", lambda *a, **k: (_ for _ in ()).throw(OSError("odd")))
    assert W.main(["fetch", REPO, "--into", str(tmp_path / "cli2"), "--base", rig.mirror_url]) == W.EXIT_LOCAL


def test_only_a_set_with_no_pin_that_no_source_served_may_fall_back():
    assert W.may_fall_back(W.Unserved("none", pinned=False)) and W.may_fall_back(W.MirrorError("odd manifest"))
    for exc in (W.Unserved("none", pinned=True), W.PinMismatch("bad"), W.NotPlaced("held"), W.NoSpace("full"),
                OSError("x"), ValueError("x")):
        assert not W.may_fall_back(exc), exc


def test_a_refusal_on_this_computer_is_reported_in_its_own_words():
    from khaos_attribution.engines.environment import refused
    ok, line = refused("org/set", W.NoSpace("No space left on the drive for x in y (1.0 GB free): make room, then fetch again"))
    assert ok is False and line == "org/set: No space left on the drive for x in y (1.0 GB free): make room, then fetch again"
