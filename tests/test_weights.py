"""The weights mirror client: files land where the installers and the hub's readers look, every byte checked
against the manifest, a partial copy resumed, and a mirror that cannot serve a set answers None so the caller
falls back to the hub."""
import hashlib
import http.server
import json
import threading
from pathlib import Path

import pytest

from khaos_attribution import weights as W


@pytest.fixture
def mirror(tmp_path):
    """A mirror on loopback holding one repository at one revision, with one file whose manifest lies."""
    root = tmp_path / "mirror"
    repo, rev = "org/model", "a" * 40
    files = {"config.json": b'{"x": 1}', "weights/big.bin": bytes(range(256)) * 4000, "LICENSE": b"terms"}
    for path, data in files.items():
        (root / repo / rev / path).parent.mkdir(parents=True, exist_ok=True)
        (root / repo / rev / path).write_bytes(data)
    manifest = {"repo": repo, "revision": rev,
                "files": [{"path": p, "size": len(d), "sha256": hashlib.sha256(d).hexdigest()} for p, d in files.items()]}
    (root / repo / "manifest.json").write_text(json.dumps(manifest))
    liar = {"repo": "org/liar", "revision": "b" * 40,
            "files": [{"path": "f.bin", "size": 5, "sha256": "0" * 64}]}
    (root / "org/liar" / ("b" * 40)).mkdir(parents=True)
    (root / "org/liar" / ("b" * 40) / "f.bin").write_bytes(b"12345")
    (root / "org/liar" / "manifest.json").write_text(json.dumps(liar))
    handler = type("H", (http.server.SimpleHTTPRequestHandler,), {"log_message": lambda *a: None})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), lambda *a, **k: handler(*a, directory=str(root), **k))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}", repo, rev, files
    server.shutdown()


def test_files_land_in_a_plain_folder_filtered_by_pattern_and_resume_a_partial_copy(mirror, tmp_path):
    base, repo, rev, files = mirror
    dest = tmp_path / "plain"
    assert W.fetch_into_dir(repo, dest, ["config.json", "weights/*"], base=base) == rev
    assert (dest / "config.json").read_bytes() == files["config.json"]
    assert (dest / "weights" / "big.bin").read_bytes() == files["weights/big.bin"] and not (dest / "LICENSE").exists()
    # A partial copy is finished from where it stopped, never restarted from nothing.
    (dest / "weights" / "big.bin").unlink()
    (dest / "weights" / "big.bin.part").write_bytes(files["weights/big.bin"][:100_000])
    assert W.fetch_into_dir(repo, dest, ["weights/*"], base=base) == rev
    assert (dest / "weights" / "big.bin").read_bytes() == files["weights/big.bin"] and not (dest / "weights" / "big.bin.part").exists()


def test_the_hub_cache_layout_is_what_the_hub_itself_reads(mirror, tmp_path):
    base, repo, rev, files = mirror
    cache = tmp_path / "hub"
    assert W.fetch_into_hub_cache(repo, cache, base=base) == rev
    repo_dir = cache / "models--org--model"
    assert (repo_dir / "refs" / "main").read_text() == rev
    snapshot = repo_dir / "snapshots" / rev
    for path, data in files.items():
        assert (snapshot / path).is_symlink() and (snapshot / path).read_bytes() == data
        assert (repo_dir / "blobs" / hashlib.sha256(data).hexdigest()).is_file()
    hub = pytest.importorskip("huggingface_hub")
    found = hub.try_to_load_from_cache(repo, "config.json", cache_dir=str(cache))
    assert isinstance(found, str) and Path(found).read_bytes() == files["config.json"], "the hub resolves it offline"


def test_a_file_that_fails_its_check_is_discarded_and_refused(mirror, tmp_path):
    base = mirror[0]
    with pytest.raises(W.MirrorError, match="did not match"):
        W.fetch_into_dir("org/liar", tmp_path / "liar", base=base)
    assert not (tmp_path / "liar" / "f.bin").exists() and not (tmp_path / "liar" / "f.bin.part").exists()


def test_a_mirror_that_cannot_serve_the_set_answers_none_so_the_caller_falls_back(mirror, tmp_path):
    base, repo, rev, _ = mirror
    assert W.manifest("org/absent", base=base) is None
    assert W.fetch_into_dir("org/absent", tmp_path / "x", base=base) is None
    assert W.fetch_into_dir(repo, tmp_path / "x", base=base, revision="c" * 40) is None, "another revision is not this set"
    assert W.manifest(repo, base="http://127.0.0.1:9", timeout=1.0) is None, "unreachable is None, never an exception"


def test_the_command_line_says_fallback_in_its_exit_code(mirror, tmp_path, capsys):
    base, repo, rev, _ = mirror
    assert W.main(["fetch", repo, "--into", str(tmp_path / "cli"), "--pattern", "config.json", "--base", base]) == 0
    assert capsys.readouterr().out.strip() == rev and (tmp_path / "cli" / "config.json").is_file()
    assert W.main(["fetch", "org/absent", "--into", str(tmp_path / "cli"), "--base", base]) == W.EXIT_NO_MIRROR
    assert W.main(["fetch", "org/liar", "--into", str(tmp_path / "cli2"), "--base", base]) == 1
    assert W.main(["check", repo, "--base", base]) == 0 and capsys.readouterr().out.strip() == rev
    assert W.main(["fetch", repo, "--hub-cache", str(tmp_path / "hub2"), "--base", base]) == 0
    assert (tmp_path / "hub2" / "models--org--model" / "refs" / "main").read_text() == rev


def test_the_mirror_address_comes_from_the_environment_or_the_default(monkeypatch):
    monkeypatch.delenv(W.MIRROR_ENV, raising=False)
    assert W.base_url() == "https://weights.khaos.dev"
    monkeypatch.setenv(W.MIRROR_ENV, "https://example.test/weights/")
    assert W.base_url() == "https://example.test/weights"
    assert W.base_url("http://x/") == "http://x"


def test_the_mlx_ports_files_land_where_the_port_reads_them(mirror, tmp_path, monkeypatch):
    """Stability keeps the port's files under MLX/; the engine's weights step puts them under models/mlx/."""
    from khaos_attribution.engines.sa3 import provision as P
    base, _, rev, _ = mirror
    root = tmp_path / "mirror" / "stabilityai" / "stable-audio-3-optimized"
    (root / rev / "MLX").mkdir(parents=True)
    (root / rev / "MLX" / "t5gemma_f16.npz").write_bytes(b"npz")
    (root / rev / "LICENSE_GEMMA.md").write_bytes(b"terms")
    files = [{"path": "MLX/t5gemma_f16.npz", "size": 3, "sha256": hashlib.sha256(b"npz").hexdigest()},
             {"path": "LICENSE_GEMMA.md", "size": 5, "sha256": hashlib.sha256(b"terms").hexdigest()}]
    (root / "manifest.json").write_text(json.dumps({"repo": "stabilityai/stable-audio-3-optimized", "revision": rev, "files": files}))
    monkeypatch.setenv(W.MIRROR_ENV, base)
    mlx = tmp_path / "port"
    assert P._mlx_from_mirror("stabilityai/stable-audio-3-optimized", mlx, ["models/mlx/t5gemma_f16.npz"]) == rev
    assert (mlx / "models" / "mlx" / "t5gemma_f16.npz").read_bytes() == b"npz" and not (mlx / "MLX").exists()
    assert (mlx / "models" / "mlx" / "LICENSE_GEMMA.md").read_bytes() == b"terms", "the terms travel with the files, beside them"
    # A second run finds the file where it was placed and fetches nothing again.
    (mlx / "models" / "mlx" / "t5gemma_f16.npz").write_bytes(b"npz")
    assert P._mlx_from_mirror("stabilityai/stable-audio-3-optimized", mlx, ["models/mlx/t5gemma_f16.npz"]) == rev


def test_an_empty_file_in_a_set_is_written_whole_and_never_resumed(mirror, tmp_path):
    base, _, rev, _ = mirror
    root = tmp_path / "mirror" / "org" / "withempty"
    (root / rev).mkdir(parents=True)
    (root / rev / "empty.txt").write_bytes(b"")
    (root / "manifest.json").write_text(json.dumps({"repo": "org/withempty", "revision": rev,
        "files": [{"path": "empty.txt", "size": 0, "sha256": hashlib.sha256(b"").hexdigest()}]}))
    assert W.fetch_into_dir("org/withempty", tmp_path / "e", base=base) == rev
    assert (tmp_path / "e" / "empty.txt").is_file() and (tmp_path / "e" / "empty.txt").stat().st_size == 0


def test_a_file_fetched_by_address_comes_from_the_mirror_or_says_it_cannot(mirror, tmp_path):
    """A hub address maps to its repository's set (a pinned revision must match); any other address to the url
    manifest; anything the mirror does not hold answers False so the caller goes to the address itself."""
    base, repo, rev, files = mirror
    root = tmp_path / "mirror"
    assert W.fetch_url(f"https://huggingface.co/{repo}/resolve/{rev}/config.json", tmp_path / "a" / "c.json", base=base)
    assert (tmp_path / "a" / "c.json").read_bytes() == files["config.json"]
    assert W.fetch_url(f"https://huggingface.co/{repo}/resolve/main/LICENSE", tmp_path / "a" / "L", base=base), "main takes the served revision"
    assert not W.fetch_url(f"https://huggingface.co/{repo}/resolve/{'d' * 40}/config.json", tmp_path / "a" / "x", base=base)
    assert not W.fetch_url(f"https://huggingface.co/{repo}/resolve/{rev}/absent.bin", tmp_path / "a" / "y", base=base)
    data = b"model bytes" * 100
    (root / "urls" / "github.com" / "o" / "r").mkdir(parents=True)
    (root / "urls" / "github.com" / "o" / "r" / "m.pt").write_bytes(data)
    (root / "urls" / "manifest.json").write_text(json.dumps({"repo": "urls", "revision": "2026-10-18", "files": [
        {"url": "https://github.com/o/r/m.pt", "path": "github.com/o/r/m.pt", "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}]}))
    assert W.fetch_url("https://github.com/o/r/m.pt", tmp_path / "b" / "m.pt", base=base)
    assert (tmp_path / "b" / "m.pt").read_bytes() == data
    assert not W.fetch_url("https://example.org/other.pt", tmp_path / "b" / "o.pt", base=base)
    assert not W.fetch_url("https://github.com/o/r/m.pt", tmp_path / "c" / "m.pt", base="http://127.0.0.1:9"), "unreachable is False"
