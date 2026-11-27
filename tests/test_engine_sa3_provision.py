"""Stable Audio 3's provision slot behind fakes: --check reports what is missing without changing anything, the
pin is compared not assumed, every listed weight is accounted for, and the report says CHECK OK only when all is."""
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from khaos_attribution.engines import contract_guard as CG
from khaos_attribution.engines import environment as EN
from khaos_attribution.engines.sa3 import provision as P


# Provisioning moves and removes checkouts: nothing here may reach outside the test's folder.
pytestmark = pytest.mark.usefixtures("confined")


@pytest.fixture(autouse=True)
def _temp_inside(tmp_path, monkeypatch):
    """Temporary folders the code under test makes (the hub's among them) are made inside the test's folder too."""
    import tempfile
    inside = tmp_path / "tmp"
    inside.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(inside))


def _patch(monkeypatch, name, fake):
    """A shared environment helper, patched both where sa3's provision binds it and in the shared module."""
    monkeypatch.setattr(P, {"run": "_run"}.get(name, name), fake)
    monkeypatch.setattr(EN, name, fake)


def _done(code=0, out="", err=""):
    return subprocess.CompletedProcess(args=[], returncode=code, stdout=out, stderr=err)


def test_check_reports_a_missing_checkout_and_changes_nothing(tmp_path, monkeypatch):
    calls = []
    _patch(monkeypatch, "run", lambda cmd, cwd=None, env=None: calls.append(cmd) or _done())
    report = P.provision(tmp_path / "sa3", tmp_path / "w", device="cuda", check_only=True)
    assert not report.passed and "not cloned" in report.text() and "CHECK FAILED" in report.text()
    assert calls == [] and not (tmp_path / "sa3").exists()


def test_check_compares_the_pin_and_the_environment_and_each_weight(tmp_path, monkeypatch):
    """Torch weights count only in the Hugging Face cache, the one place Stability's package reads them from."""
    root = tmp_path / "sa3"
    (root / ".git").mkdir(parents=True)
    pin = P.DESCRIPTOR.commit

    def fake_run(cmd, cwd=None, env=None):
        cmd = [str(c) for c in cmd]
        if "rev-parse" in cmd:
            return _done(out=pin + "\n")
        if "-c" in cmd and "import torch" in cmd[-1]:
            return _done(0, "ok")
        return _done()
    _patch(monkeypatch, "run", fake_run)
    monkeypatch.setattr(P, "requirements_missing", lambda python: None)
    (root / ".venv" / "bin").mkdir(parents=True)
    (root / ".venv" / "bin" / "python").write_text("")
    have = set()
    monkeypatch.setattr(P, "cached", lambda repo, files: repo in have)
    report = P.provision(root, tmp_path / "w", device="cuda", check_only=True)
    text = report.text()
    assert "at the pin" in text and "Flash Attention" in text
    assert "stabilityai/stable-audio-3-medium-base missing" in text and "google" not in text and "medium missing" not in text
    assert "stable-audio-3-optimized" not in text, "the MLX port's set is not the torch runtime's"
    assert not report.passed
    have.add("stabilityai/stable-audio-3-medium-base")
    assert P.provision(root, tmp_path / "w", device="cuda", check_only=True).passed, "one open repository is the whole torch set"


def test_an_install_fills_the_cache_through_the_engines_own_downloader(tmp_path, monkeypatch):
    """A set with no pin is the one the hub library is left to; a pinned one is never installed through it."""
    from khaos_attribution import weights
    monkeypatch.setattr(weights, "PINS", {})
    monkeypatch.delenv("PYTHONPATH", raising=False)
    root = tmp_path / "sa3"
    (root / ".git").mkdir(parents=True)
    (root / ".venv" / "bin").mkdir(parents=True)
    (root / ".venv" / "bin" / "python").write_text("")
    calls = []

    def fake_run(cmd, cwd=None, env=None):
        cmd = [str(c) for c in cmd]
        calls.append((cmd, env))
        if "rev-parse" in cmd:
            return _done(out=P.DESCRIPTOR.commit + "\n")
        if "-c" in cmd and "import torch" in cmd[-1]:
            return _done(0, "ok")
        if "snapshot_download" in " ".join(cmd):
            return _done(1, err="Connection timed out") if calls and sum("snapshot_download" in " ".join(c) for c, _ in calls) > 1 else _done()
        return _done()
    _patch(monkeypatch, "run", fake_run)
    monkeypatch.setattr(P, "requirements_missing", lambda python: None)
    monkeypatch.setattr(P, "cached", lambda repo, files: any("snapshot_download" in " ".join(c) and c[-2] == repo for c, _ in calls))
    report = P.provision(root, tmp_path / "w", device="cuda")
    text = report.text()
    downloads = [c for c, _ in calls if "snapshot_download" in " ".join(c)]
    assert [c[-2] for c in downloads] == ["stabilityai/stable-audio-3-medium-base"], "one open repository, nothing gated"
    assert "t5gemma-b-b-ul2/*" in downloads[0][-1] and "svd_bases" not in downloads[0][-1], "the weights, the config and the encoder, not the SVD bases"
    assert all(env.get("HF_HUB_OFFLINE") == "0" and "PYTHONPATH" not in env for c, env in calls if "snapshot_download" in " ".join(c))
    assert report.passed and "hf auth login" not in text


def test_without_git_the_code_comes_from_the_archive_at_the_pin(tmp_path, monkeypatch):
    import io
    import tarfile
    from khaos_attribution.engines.pins import checkout_commit
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        data = b"print('sa3')\n"
        info = tarfile.TarInfo("stable-audio-3-abc/setup.py"); info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
        info = tarfile.TarInfo("stable-audio-3-abc/stable_audio_3/__init__.py"); info.size = 0
        tar.addfile(info, io.BytesIO(b""))
    seen = {}

    class _Resp(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): return False
    monkeypatch.setattr(P, "git_usable", lambda: False)
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout=0: seen.setdefault("url", url) and _Resp(buf.getvalue()))
    root = tmp_path / "sa3"
    ok, note = P.checkout(root)
    assert ok and "unpacked at the pin" in note and (root / "setup.py").read_text() == "print('sa3')\n"
    assert seen["url"] == f"https://github.com/Stability-AI/stable-audio-3/archive/{P.DESCRIPTOR.commit}.tar.gz"
    assert checkout_commit(root) == P.DESCRIPTOR.commit, "the pin is readable without a .git"
    assert P.checkout(root, check_only=True)[0]


def test_a_moved_pin_is_named_not_moved_under_check(tmp_path, monkeypatch):
    root = tmp_path / "sa3"
    (root / ".git").mkdir(parents=True)
    _patch(monkeypatch, "run", lambda cmd, cwd=None, env=None: _done(out="deadbeef" * 5 + "\n") if "rev-parse" in [str(c) for c in cmd] else _done())
    ok, note = P.checkout(root, check_only=True)
    assert not ok and "pin is" in note


def test_the_mlx_runtime_wants_the_exported_base_weights(tmp_path, monkeypatch):
    root = tmp_path / "sa3"
    (root / ".git").mkdir(parents=True)
    mlx_python = root / "optimized" / "mlx" / ".venv" / "bin" / "python"
    mlx_python.parent.mkdir(parents=True)
    mlx_python.write_text("")
    pin = P.DESCRIPTOR.commit
    _patch(monkeypatch, "run", lambda cmd, cwd=None, env=None: _done(out=pin + "\n") if "rev-parse" in [str(c) for c in cmd] else _done(0, "ok"))
    weights = tmp_path / "w"
    monkeypatch.setattr(P, "cached", lambda repo, files: True)
    mlx = root / "optimized" / "mlx"
    monkeypatch.setattr(P, "requirements_missing", lambda python: "cannot load the room's contract (no jsonschema)")
    assert "runs the engine but cannot load the room's contract" in P.provision(root, weights, device="mps",
                                                                                check_only=True).text(), \
        "the MLX measure worker loads the room's contract, so an MLX environment that cannot is not ready"
    monkeypatch.setattr(P, "requirements_missing", lambda python: None)
    (root / P.MLX_TRAINER).parent.mkdir(parents=True, exist_ok=True)
    (root / P.MLX_TRAINER).write_text("lac = mx.array(\n    " + P.TRAINER_MASK_ONES + "\n    np.zeros((1, 256, crop_len)))\n")
    assert "all-ones inpaint mask" in P.provision(root, weights, device="mps", check_only=True).text()
    P.mlx_trainer_patch(root)
    report = P.provision(root, weights, device="mps", check_only=True)
    text = report.text()
    assert "stable-audio-3-optimized missing" in text and "MLX base weights missing" in text and not report.passed
    assert "t5gemma" not in text, "the MLX port carries its own text encoder"
    for rel in ("models/mlx/dit_medium-base_f16.npz", "models/mlx/dit_medium_f16.npz", "models/mlx/same_l_decoder_f32.npz",
                "models/mlx/same_l_encoder_f32.npz", "models/mlx/t5gemma_f16.npz"):
        (mlx / rel).parent.mkdir(parents=True, exist_ok=True)
        (mlx / rel).write_bytes(b"npz")
    assert not P.installed(root, "mlx", weights)
    # An install links the port's ready-made base file where the trainer looks; no model repository is pulled on MLX.
    (root / P.MLX_TRAINER).write_text("lac = mx.array(\n    " + P.TRAINER_MASK_ONES + "\n    np.zeros((1, 256, crop_len)))\n")
    report = P.provision(root, weights, device="mps")
    assert report.passed and (weights / P.BASE_NPZ).resolve() == (mlx / "models/mlx/dit_medium-base_f16.npz").resolve()
    assert P.trainer_mask(root) == "zeros", "an install leaves the MLX trainer on the generation mask"
    assert P.installed(root, "mlx", weights) and P.provision(root, weights, device="mps", check_only=True).passed
    (mlx / "models/mlx/t5gemma_f16.npz").unlink()
    (mlx / "models/mlx/t5gemma_f16.npz").symlink_to(tmp_path / "gone")
    assert "optimized missing" in P.provision(root, weights, device="mps", check_only=True).text(), "a dangling link is missing"
    with pytest.raises(SystemExit):
        P.main(["--help"])


def test_an_unpacked_checkout_moves_to_a_new_pin_and_keeps_its_environments(tmp_path, monkeypatch):
    root = tmp_path / "sa3"
    (root / ".venv" / "bin").mkdir(parents=True)
    (root / ".venv" / "bin" / "python").write_text("")
    (root / "optimized" / "mlx" / "models" / "mlx").mkdir(parents=True)
    (root / "optimized" / "mlx" / "models" / "mlx" / "t5gemma_f16.npz").write_bytes(b"npz")
    (root / ".khaos-commit").write_text("0" * 40 + "\n")
    monkeypatch.setattr(P, "git_usable", lambda: False)
    unpacked = {}
    def fake_unpack(url, where, commit):
        unpacked["url"] = url
        P.unpack_archive.__wrapped__(where, commit) if False else None
        # the real mover, with the archive replaced by one file
        part = where.with_name(where.name + ".part"); part.mkdir(); (part / "setup.py").write_text("")
        (part / ".khaos-commit").write_text(commit + "\n")
        for rel in P.KEPT_ACROSS_PINS:
            if (where / rel).exists():
                (part / rel).parent.mkdir(parents=True, exist_ok=True); (where / rel).rename(part / rel)
        import shutil; shutil.rmtree(where); part.replace(where)
    monkeypatch.setattr(P, "unpack_archive", fake_unpack)
    assert not P.checkout(root, check_only=True)[0]
    ok, note = P.checkout(root)
    assert ok and "moved to the pin" in note and P.DESCRIPTOR.commit in unpacked["url"]
    assert (root / ".venv" / "bin" / "python").is_file() and (root / "optimized/mlx/models/mlx/t5gemma_f16.npz").is_file()


def _no_git_here(monkeypatch):
    """git is not on PATH: the slot is told so, and any git run fails as Windows does."""
    from khaos_attribution.engines import pins
    monkeypatch.setattr(P, "git_usable", lambda: False)
    real = pins.subprocess.run

    def run(cmd, *a, **k):
        if cmd and cmd[0] == "git":
            raise FileNotFoundError(2, "The system cannot find the file specified")
        return real(cmd, *a, **k)
    monkeypatch.setattr(pins.subprocess, "run", run)

    def fake_run(cmd, cwd=None, env=None):
        if cmd and str(cmd[0]) == "git":
            raise FileNotFoundError(2, "The system cannot find the file specified")
        return _done()
    _patch(monkeypatch, "run", fake_run)


def test_a_clone_at_the_pin_needs_no_git_to_be_read(tmp_path, monkeypatch):
    """A git clone whose git has gone is read from its own files, and at the pin nothing runs git."""
    root = tmp_path / "sa3"
    (root / ".git").mkdir(parents=True)
    (root / ".git" / "HEAD").write_text(P.DESCRIPTOR.commit + "\n")
    _no_git_here(monkeypatch)
    assert P.checkout(root, check_only=True) == (True, f"Stable Audio 3 code at the pin {P.DESCRIPTOR.commit[:12]}")
    assert P.checkout(root)[0]


def test_a_clone_off_the_pin_with_no_git_moves_by_the_archive(tmp_path, monkeypatch):
    """A clone at an older commit, with git gone, takes the archive at the pin and keeps its environment."""
    root = tmp_path / "sa3"
    (root / ".git" / "refs" / "heads").mkdir(parents=True)
    (root / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (root / ".git" / "refs" / "heads" / "main").write_text("0" * 40 + "\n")
    (root / ".venv" / "Scripts").mkdir(parents=True)
    (root / ".venv" / "Scripts" / "python.exe").write_text("")
    _no_git_here(monkeypatch)
    moved = {}

    def fake_unpack(url, where, commit):
        moved["url"] = url
        part = where.with_name(where.name + ".part"); part.mkdir(); (part / ".khaos-commit").write_text(commit + "\n")
        (where / ".venv").rename(part / ".venv")
        import shutil; shutil.rmtree(where); part.replace(where)
    monkeypatch.setattr(P, "unpack_archive", fake_unpack)
    ok, note = P.checkout(root, check_only=True)
    assert not ok and "at 000000000000, pin is" in note
    ok, note = P.checkout(root)
    assert ok and "moved to the pin" in note and P.DESCRIPTOR.commit in moved["url"]
    assert (root / ".venv" / "Scripts" / "python.exe").is_file()
    assert P.checkout(root, check_only=True)[0]


def _read_only_objects(git_dir):
    """A clone's objects as Git for Windows leaves them: the file and its folder read-only."""
    import os
    objects = git_dir / "objects" / "ab"
    objects.mkdir(parents=True)
    (objects / "cdef").write_bytes(b"x")
    os.chmod(objects / "cdef", 0o444)
    os.chmod(objects, 0o555)


def test_the_archive_replaces_a_clone_with_read_only_objects_and_keeps_its_environment(tmp_path, monkeypatch):
    """The real move, over a clone whose objects are read-only and a leftover of an earlier move that is too."""
    import io
    import tarfile
    import urllib.request
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo("stable-audio-3-abc/setup.py"); info.size = 0
        tar.addfile(info, io.BytesIO(b""))

    class _Resp(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): return False
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout=0: _Resp(buf.getvalue()))
    root = tmp_path / "sa3"
    (root / ".git").mkdir(parents=True)
    (root / ".git" / "HEAD").write_text("0" * 40 + "\n")
    _read_only_objects(root / ".git")
    (root / ".venv" / "Scripts").mkdir(parents=True)
    (root / ".venv" / "Scripts" / "python.exe").write_text("venv")
    _read_only_objects(tmp_path / "sa3.old" / ".git")
    _no_git_here(monkeypatch)
    ok, note = P.checkout(root)
    assert ok and "moved to the pin" in note, note
    assert (root / ".venv" / "Scripts" / "python.exe").read_text() == "venv"
    assert not (root / ".git").exists() and (root / "setup.py").is_file()
    assert not (tmp_path / "sa3.old").exists() and not (tmp_path / "sa3.part").exists()
    assert P.checkout(root, check_only=True)[0]


def test_a_leftover_that_cannot_be_cleared_stops_the_move_before_anything_moves(tmp_path, monkeypatch):
    """When the earlier move's leftover stays, the checkout and its environment are left exactly as they were."""
    root = tmp_path / "sa3"
    (root / ".venv").mkdir(parents=True)
    (root / ".venv" / "keep").write_text("")
    (tmp_path / "sa3.old").mkdir()
    real = P.remove_tree

    def stuck(path):
        if path.name == "sa3.old":
            raise OSError(f"{path} could not be removed")
        real(path)
    _patch(monkeypatch, "remove_tree", stuck)
    import io
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout=0: io.BytesIO(b""))
    with pytest.raises(OSError):
        P.unpack_archive("https://example.invalid/x.tar.gz", root, "a" * 40)
    assert (root / ".venv" / "keep").is_file() and not (tmp_path / "sa3.part").exists()


def test_a_checkout_that_cannot_be_set_aside_gets_its_environment_back(tmp_path, monkeypatch):
    """A swap refused (a file open in the checkout) leaves the environment where it was, never in a folder cleared next."""
    import io
    import pathlib
    import tarfile
    import urllib.request
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo("stable-audio-3-abc/setup.py"); info.size = 0
        tar.addfile(info, io.BytesIO(b""))

    class _Resp(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): return False
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout=0: _Resp(buf.getvalue()))
    root = tmp_path / "sa3"
    (root / ".venv").mkdir(parents=True)
    (root / ".venv" / "keep").write_text("venv")
    (root / ".khaos-commit").write_text("0" * 40 + "\n")
    real = pathlib.Path.replace

    def refused(self, target):
        if pathlib.Path(target).name == "sa3.old":
            raise PermissionError(13, "The process cannot access the file because it is being used by another process")
        return real(self, target)
    monkeypatch.setattr(pathlib.Path, "replace", refused)
    with pytest.raises(OSError):
        P.unpack_archive("https://example.invalid/x.tar.gz", root, P.DESCRIPTOR.commit)
    assert (root / ".venv" / "keep").read_text() == "venv"
    assert not (tmp_path / "sa3.part" / ".venv").exists()


def test_a_fresh_clone_off_the_pin_is_moved_by_git_not_replaced(tmp_path, monkeypatch):
    """With git here, a clone just made is checked out at the pin; the archive is never fetched over it."""
    root = tmp_path / "sa3"
    monkeypatch.setattr(P, "git_usable", lambda: True)
    calls = []

    def fake_run(cmd, cwd=None, env=None):
        cmd = [str(c) for c in cmd]
        calls.append(cmd)
        if "clone" in cmd:
            (root / ".git").mkdir(parents=True)
        if "rev-parse" in cmd:
            return _done(out="1" * 40 + "\n")
        return _done()
    _patch(monkeypatch, "run", fake_run)
    monkeypatch.setattr(P, "unpack_archive", lambda *a, **k: pytest.fail("the archive replaced a git clone"))
    ok, note = P.checkout(root)
    assert ok and note == f"checked out {P.DESCRIPTOR.commit[:12]}"
    assert ["git", "-C", str(root), "checkout", "-q", "--detach", P.DESCRIPTOR.commit] in calls


def test_on_windows_the_environment_has_its_own_layout_and_no_flash_attention(tmp_path, monkeypatch):
    from khaos_attribution.engines.sa3 import render as R
    monkeypatch.setattr(R, "WINDOWS", True)
    monkeypatch.setattr(P, "WINDOWS", True)
    root = tmp_path / "sa3"
    (root / ".git").mkdir(parents=True)
    assert R.venv_python(root, "torch") == root / ".venv" / "Scripts" / "python.exe"
    calls = []

    def fake_run(cmd, cwd=None, env=None):
        cmd = [str(c) for c in cmd]
        calls.append(cmd)
        if "rev-parse" in cmd:
            return _done(out=P.DESCRIPTOR.commit + "\n")
        if "export" in cmd:
            return _done(out="einops==0.8.0\ntorch==2.7.1\n")
        if "venv" in cmd:
            (root / ".venv" / "Scripts").mkdir(parents=True, exist_ok=True)
            (root / ".venv" / "Scripts" / "python.exe").write_text("")
        if "-c" in cmd and "import torch" in cmd[-1]:
            return _done(0, "ok")
        return _done()
    _patch(monkeypatch, "run", fake_run)
    _patch(monkeypatch, "uv_path", lambda: "uv")
    monkeypatch.setattr(P, "requirements_missing", lambda python: None)
    monkeypatch.setattr(P, "requirements_into", lambda python: (True, "contract"))
    monkeypatch.setattr(P, "cached", lambda repo, files: True)
    app_python = tmp_path / "app-python"
    app_python.mkdir()
    for name in ("msvcp140.dll", "vcruntime140.dll"):
        (app_python / name).write_bytes(b"runtime")
    from khaos_attribution.engines import environment as E
    monkeypatch.setattr(P, "carry_cpp_runtime", lambda venv, site: E.carry_cpp_runtime(venv, site, app_python, platform="win32"))
    report = P.provision(root, tmp_path / "w", device="cuda")
    assert report.passed, report.text()
    runtime = root / ".venv" / E.CPP_RUNTIME_DIR
    assert sorted(p.name for p in runtime.iterdir()) == ["msvcp140.dll", "vcruntime140.dll"]
    assert (root / ".venv" / "Lib" / "site-packages" / E.CPP_RUNTIME_PTH).is_file()
    assert "C++ runtime carried from" in report.text() and "(msvcp140.dll, vcruntime140.dll)" in report.text()
    joined = [" ".join(c) for c in calls]
    assert not any("flash-attn" in c for c in joined), "no Flash Attention build on Windows"
    assert not any("flash_attn" in c for c in joined), "the probe does not ask for it either"
    assert any("import torch, stable_audio_3, soundfile" in c for c in joined), "soundfile is the Windows reader"
    assert any("Scripts/python.exe" in c or "Scripts\\python.exe" in c for c in joined)
    assert P.installed(root, "torch", tmp_path / "w")


def test_the_contract_comes_from_the_nearest_source(tmp_path, monkeypatch):
    """A checkout named in the environment, then a shipped wheel, then the room's sibling checkout when it holds this
    room's code, then the tag."""
    import shutil
    import khaos_attribution
    from khaos_attribution import __version__
    root = tmp_path / "engines" / "stable-audio-3"
    root.mkdir(parents=True)
    monkeypatch.delenv(P.CONTRACT_SRC_ENV, raising=False)
    monkeypatch.delenv(P.CONTRACT_WHEEL_ENV, raising=False)
    tag = f"khaos-attribution @ git+{P.CONTRACT_REPO}@v{__version__}"
    assert P.contract_spec(root) == tag
    sibling = tmp_path / "engines" / "Khaos_Attribution"
    shutil.copytree(Path(khaos_attribution.__file__).parent, sibling / "src" / "khaos_attribution",
                    ignore=shutil.ignore_patterns("__pycache__"))
    (sibling / "pyproject.toml").write_text("")
    assert P.contract_spec(root) == str(sibling), "the sibling holds this room's code"
    (sibling / "src" / "khaos_attribution" / "newer.py").write_text("x = 1\n")
    assert P.contract_spec(root) == tag, "a sibling that has moved on from this room's contract is not installed"
    monkeypatch.setenv(P.CONTRACT_WHEEL_ENV, str(tmp_path / "k.whl"))
    assert P.contract_spec(root) == str(tmp_path / "k.whl")
    monkeypatch.setenv(P.CONTRACT_SRC_ENV, str(tmp_path / "src"))
    assert P.contract_spec(root) == str(tmp_path / "src")


def test_the_contract_counts_as_present_only_when_its_code_is_the_same(tmp_path, monkeypatch):
    """A version string alone would pass an engine environment holding older code of the same version."""
    here = CG.contract_fingerprint()
    assert here and here.split()[0] == __import__("khaos_attribution").__version__
    monkeypatch.setattr(CG, "contract_fingerprint", lambda python=None: here if python is None else here.split()[0] + " other")
    assert not CG.contract_present(tmp_path / "python")
    monkeypatch.setattr(CG, "contract_fingerprint", lambda python=None: here)
    assert CG.contract_present(tmp_path / "python")


def test_the_contract_hash_ignores_line_endings_and_names_a_same_version_mismatch(tmp_path, monkeypatch):
    """A Windows checkout and a wheel of the same code hash alike; the same version with other code is said so."""
    from khaos_attribution.engines.pins import contract_code_hash
    for name, text in (("lf", b"x = 1\ny = 2\n"), ("crlf", b"x = 1\r\ny = 2\r\n")):
        (tmp_path / name).mkdir()
        (tmp_path / name / "m.py").write_bytes(text)
    assert contract_code_hash(tmp_path / "lf").split()[1] == contract_code_hash(tmp_path / "crlf").split()[1]
    ours = CG.contract_fingerprint()
    monkeypatch.setattr(CG, "contract_fingerprint", lambda python=None: ours if python is None else ours.split()[0] + " other")
    assert "built from other code" in CG.contract_mismatch(tmp_path / "python")
    monkeypatch.setattr(CG, "contract_fingerprint", lambda python=None: ours if python is None else None)
    assert "older than" in CG.contract_mismatch(tmp_path / "python")


def test_flash_attention_comes_ready_made_on_linux_and_matches_the_pins(monkeypatch):
    """A server with no CUDA compiler still gets Flash Attention; the wheel names the torch, Python and version pinned."""
    from khaos_attribution.engines.sa3 import provision as P
    wheel = P.FLASH_ATTN_WHEEL
    torch_minor = ".".join(P.TORCH.split("==")[1].split(".")[:2])
    assert f"torch{torch_minor}" in wheel and "cu12" in wheel and P.TORCH_INDEX.endswith("cu128")
    assert f"cp{P.PYTHON_VERSION.replace('.', '')}" in wheel
    assert f"flash_attn-{P.FLASH_ATTN.split('==')[1]}+" in wheel and f"/v{P.FLASH_ATTN.split('==')[1]}/" in wheel
    monkeypatch.setattr(P.sys, "platform", "linux")
    assert P.flash_attn_requirement("x86_64") == wheel
    assert P.flash_attn_requirement("aarch64") == P.FLASH_ATTN


def _torch_env_fakes(monkeypatch, root, venv_result, seen):
    def fake_run(cmd, cwd=None, env=None):
        cmd = [str(c) for c in cmd]
        seen.append((cmd, env))
        if "venv" in cmd:
            return venv_result
        if "export" in cmd:
            return _done(out="einops==0.8.0\n")
        return _done()
    _patch(monkeypatch, "run", fake_run)
    _patch(monkeypatch, "uv_path", lambda: "uv")


def test_a_venv_uv_cannot_make_is_reported_in_uvs_words_and_nothing_installs(tmp_path, monkeypatch):
    """A failed `uv venv` stops the install with uv's own reason, not a later step's missing-venv error."""
    root = tmp_path / "sa3"
    root.mkdir()
    seen = []
    _torch_env_fakes(monkeypatch, root, _done(2, err="error: No interpreter found for Python 3.11"), seen)
    ok, note = P.environment(root, "torch")
    assert not ok
    assert "Python 3.11" in note and "No interpreter found" in note, note
    assert not any("pip" in cmd for cmd, _ in seen), "no install step runs without a venv"


def test_a_folder_left_by_a_failed_try_is_replaced_on_the_next(tmp_path, monkeypatch):
    """A retry over a half-made .venv asks uv to replace it, so a fixed cause lets the install through."""
    root = tmp_path / "sa3"
    (root / ".venv" / "Lib").mkdir(parents=True)
    seen = []
    _torch_env_fakes(monkeypatch, root, _done(), seen)
    ok, note = P.environment(root, "torch")
    venv_calls = [env for cmd, env in seen if "venv" in cmd]
    assert venv_calls and venv_calls[0].get("UV_VENV_CLEAR") == "1"
    assert not ok and "left no interpreter" in note, note


def test_mimetic_stops_at_a_venv_uv_cannot_make_and_leaves_no_folder(tmp_path, monkeypatch):
    """mimetic-a0 stops on uv's reason before it writes its lock under .venv, so a retry finds no half-made folder."""
    from khaos_attribution.engines.mimetic import provision as M
    root = tmp_path / "mimetic"
    root.mkdir()
    seen = []
    _torch_env_fakes(monkeypatch, root, _done(2, err="error: No interpreter found for Python 3.11"), seen)
    monkeypatch.setattr(M, "_run", P._run)
    monkeypatch.setattr(M, "uv_path", lambda: "uv")
    monkeypatch.setattr(EN, "uv_path", lambda: "uv")
    ok, note = M.environment(root, "cuda")
    assert not ok and "No interpreter found" in note, note
    assert not (root / ".venv").exists()


def test_where_uv_cannot_get_the_pinned_python_the_running_one_stands_in(tmp_path, monkeypatch):
    """A failed pinned venv is made again on the interpreter running the install, and the log says so."""
    root = tmp_path / "sa3"
    root.mkdir()
    python = P.venv_python(root, "torch")
    calls = []

    def fake_run(cmd, cwd=None, env=None):
        cmd = [str(c) for c in cmd]
        calls.append(cmd)
        if "venv" in cmd:
            if cmd[cmd.index("--python") + 1] != P.sys.executable:
                return _done(2, err="error: No interpreter found for Python 3.11")
            python.parent.mkdir(parents=True, exist_ok=True)
            python.write_text("")
        if "export" in cmd:
            return _done(out="einops==0.8.0\n")
        return _done(0, "ok")
    _patch(monkeypatch, "run", fake_run)
    _patch(monkeypatch, "uv_path", lambda: "uv")
    monkeypatch.setattr(P, "requirements_into", lambda python: (True, "contract"))
    monkeypatch.setattr(P, "WINDOWS", True)
    ok, note = P.environment(root, "torch")
    assert ok, note
    assert [c[c.index("--python") + 1] for c in calls if "venv" in c] == ["3.11", P.sys.executable]
    assert "No interpreter found" in note and "made on this Python" in note, note


def test_a_real_uv_falls_back_to_the_running_python(tmp_path, monkeypatch):
    """With the real uv, a version it cannot get still ends in a venv whose interpreter runs."""
    import shutil
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is not on PATH")
    if not (3, 11) <= P.sys.version_info[:2] <= (3, 12):
        pytest.skip("the fallback takes only a Python from 3.11 to 3.12")
    monkeypatch.setenv("UV_PYTHON_DOWNLOADS", "never")
    venv = tmp_path / ".venv"
    python = venv / ("Scripts/python.exe" if P.WINDOWS else "bin/python")
    ok, note = P.make_venv(uv, venv, python, "3.6", fallback=True)
    assert ok, note
    assert "made on this Python" in note
    assert subprocess.run([str(python), "-c", "print('ok')"], capture_output=True, text=True).stdout.strip() == "ok"


def test_off_windows_there_is_no_fallback(tmp_path, monkeypatch):
    """Off Windows the ready-made Flash Attention wheel is for 3.11 only, so a failed 3.11 venv stops the install."""
    root = tmp_path / "sa3"
    root.mkdir()
    seen = []
    _torch_env_fakes(monkeypatch, root, _done(2, err="error: No interpreter found for Python 3.11"), seen)
    monkeypatch.setattr(P, "WINDOWS", False)
    ok, note = P.environment(root, "torch")
    assert not ok and "No interpreter found" in note and "this Python" not in note, note
    assert len([cmd for cmd, _ in seen if "venv" in cmd]) == 1


def test_the_fallback_takes_only_a_python_from_3_11_to_3_12(monkeypatch, tmp_path):
    """A running Python outside the range the engine and the contract take is refused, with uv's first reason."""
    calls = []
    _patch(monkeypatch, "run", lambda cmd, cwd=None, env=None: calls.append(cmd) or _done(2, err="error: no 3.11"))
    for version in ((3, 10, 0), (3, 13, 0)):
        monkeypatch.setattr(P.sys, "version_info", version)
        calls.clear()
        ok, note = P.make_venv("uv", tmp_path / ".venv", tmp_path / "py", "3.11", fallback=True)
        assert not ok and "no 3.11" in note and len(calls) == 1, (version, note)


def test_an_environment_whose_interpreter_cannot_start_is_made_again(tmp_path, monkeypatch):
    """A venv whose base Python moved is rebuilt, not patched in place."""
    root = tmp_path / "sa3"
    python = P.venv_python(root, "torch")
    python.parent.mkdir(parents=True)
    python.write_text("")
    seen = []

    def fake_run(cmd, cwd=None, env=None):
        cmd = [str(c) for c in cmd]
        seen.append(cmd)
        if cmd[:3] == [str(python), "-c", "pass"]:
            return _done(1, err="No Python at 'C:\\old\\python.exe'")
        if "-c" in cmd:
            return _done(1)
        if "export" in cmd:
            return _done(out="einops==0.8.0\n")
        return _done()
    _patch(monkeypatch, "run", fake_run)
    _patch(monkeypatch, "uv_path", lambda: "uv")
    monkeypatch.setattr(P, "requirements_into", lambda python: (True, "contract"))
    P.environment(root, "torch")
    venv_at = next(i for i, c in enumerate(seen) if "venv" in c)
    assert any("pip" in c and "install" in c for c in seen[venv_at:]), "the install runs into the venv made again"


def test_an_install_that_leaves_another_contract_is_refused_in_words(tmp_path, monkeypatch):
    """The contract is read back from the engine's environment after it is installed, never assumed."""
    _patch(monkeypatch, "uv_path", lambda: "uv")
    _patch(monkeypatch, "run", lambda cmd, cwd=None, env=None: _done())
    monkeypatch.setattr(CG, "contract_present", lambda python: False)
    monkeypatch.setattr(CG, "contract_mismatch", lambda python: "holds contract 0.41.0, this room runs 0.57.0")
    ok, note = CG.contract_into(tmp_path / "python", tmp_path)
    assert not ok and "holds contract 0.41.0, this room runs 0.57.0" in note
    monkeypatch.setattr(CG, "contract_present", lambda python: True)
    assert CG.contract_into(tmp_path / "python", tmp_path)[0]


def _built_env(tmp_path, monkeypatch, record=None, lock=b"lock v1\n"):
    """An existing torch environment that runs the engine and holds this contract, with a build record or none."""
    root = tmp_path / "sa3"
    python = P.venv_python(root, "torch")
    python.parent.mkdir(parents=True)
    python.write_text("")
    (root / "uv.lock").write_bytes(lock)
    if record is not None:
        (root / ".venv" / P.BUILD_RECORD).write_text(json.dumps(record))
    seen = []

    def fake_run(cmd, cwd=None, env=None):
        cmd = [str(c) for c in cmd]
        seen.append(cmd)
        if "export" in cmd:
            return _done(out="einops==0.8.0\n")
        return _done(0, "ok")
    _patch(monkeypatch, "run", fake_run)
    _patch(monkeypatch, "uv_path", lambda: "uv")
    monkeypatch.setattr(P, "requirements_missing", lambda python: None)
    monkeypatch.setattr(P, "requirements_into", lambda python: (True, "contract"))
    return root, seen


def _installs(seen) -> list:
    return [c for c in seen if "pip" in c and "install" in c]


def test_an_environment_with_no_build_record_is_a_note_and_nothing_is_reinstalled(tmp_path, monkeypatch):
    """A live engine environment built before the record was kept is left as it is, by --check and by an install."""
    root, seen = _built_env(tmp_path, monkeypatch)
    ok, note = P.environment(root, "torch", check_only=True)
    assert ok and "no build record" in note
    ok, note = P.environment(root, "torch")
    assert ok and _installs(seen) == [] and not any("venv" in c for c in seen)
    assert not (root / ".venv" / P.BUILD_RECORD).exists(), "a record is written only by a build"


def test_an_environment_built_at_this_pin_and_lock_is_left_alone(tmp_path, monkeypatch):
    lock = b"lock v1\n"
    root, seen = _built_env(tmp_path, monkeypatch, {"commit": P.DESCRIPTOR.commit,
                                                    "lock_sha256": hashlib.sha256(lock).hexdigest()}, lock)
    assert P.environment(root, "torch", check_only=True)[0]
    assert P.environment(root, "torch")[0] and _installs(seen) == []


def test_an_environment_built_at_another_pin_fails_the_check_and_is_rebuilt_in_place(tmp_path, monkeypatch):
    """After the pin moves, the old dependencies are not left under the new code: the lock is installed again into the
    same environment, and the record then names the pin and the lock it was built from."""
    root, seen = _built_env(tmp_path, monkeypatch, {"commit": "0" * 40, "lock_sha256": "old"})
    ok, note = P.environment(root, "torch", check_only=True)
    assert not ok and "built at 000000000000" in note and P.DESCRIPTOR.commit[:12] in note
    assert _installs(seen) == []
    ok, note = P.environment(root, "torch")
    assert ok, note
    assert not any("venv" in c and "uv" == c[0] for c in seen), "the environment is not made again"
    assert any("-r" in c for c in _installs(seen)), "Stability's lock is installed again"
    record = json.loads((root / ".venv" / P.BUILD_RECORD).read_text())
    assert record == {"commit": P.DESCRIPTOR.commit, "lock_sha256": hashlib.sha256(b"lock v1\n").hexdigest()}


def test_a_fresh_build_pins_every_extra_and_records_itself(tmp_path, monkeypatch):
    root = tmp_path / "sa3"
    root.mkdir()
    (root / "uv.lock").write_bytes(b"lock v2\n")
    python = P.venv_python(root, "torch")
    seen = []

    def fake_run(cmd, cwd=None, env=None):
        cmd = [str(c) for c in cmd]
        seen.append(cmd)
        if "venv" in cmd:
            python.parent.mkdir(parents=True, exist_ok=True)
            python.write_text("")
        if "export" in cmd:
            return _done(out="einops==0.8.0\n")
        return _done(0, "ok")
    _patch(monkeypatch, "run", fake_run)
    _patch(monkeypatch, "uv_path", lambda: "uv")
    monkeypatch.setattr(P, "requirements_into", lambda python: (True, "contract"))
    ok, note = P.environment(root, "torch")
    assert ok, note
    extras = next(c for c in _installs(seen) if any(x.startswith("psutil") for x in c))
    named = [x for x in extras if x.split("==")[0] in ("setuptools", "wheel", "ninja", "psutil", "matplotlib", "wandb")]
    assert len(named) == 6 and all("==" in x for x in named), named
    assert json.loads((root / ".venv" / P.BUILD_RECORD).read_text())["lock_sha256"] == hashlib.sha256(b"lock v2\n").hexdigest()


def test_the_ready_made_flash_attention_is_installed_only_once_its_sha256_matches(tmp_path, monkeypatch):
    import io
    body = b"a wheel"
    served = []

    def urlopen(url, timeout=None):
        served.append(url)
        return io.BytesIO(body)
    monkeypatch.setattr(P.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(P, "FLASH_ATTN_WHEEL_SHA256", hashlib.sha256(b"another wheel").hexdigest())
    path, why = P.fetch_flash_attn(tmp_path)
    assert path is None and "sha256" in why and not list(tmp_path.glob("*.whl")), "a wheel that does not match is not kept"
    monkeypatch.setattr(P, "FLASH_ATTN_WHEEL_SHA256", hashlib.sha256(body).hexdigest())
    path, why = P.fetch_flash_attn(tmp_path)
    assert why is None and path.read_bytes() == body and path.name == P.FLASH_ATTN_WHEEL.rsplit("/", 1)[1]
    assert P.fetch_flash_attn(tmp_path)[0] == path and len(served) == 2, "a wheel already here and matching is not fetched again"


def test_a_failed_flash_attention_fetch_stops_the_install_with_its_reason(tmp_path, monkeypatch):
    root = tmp_path / "sa3"
    root.mkdir()
    (root / "uv.lock").write_bytes(b"lock\n")
    python = P.venv_python(root, "torch")
    seen = []

    def fake_run(cmd, cwd=None, env=None):
        cmd = [str(c) for c in cmd]
        seen.append(cmd)
        if "venv" in cmd:
            python.parent.mkdir(parents=True, exist_ok=True)
            python.write_text("")
        if "export" in cmd:
            return _done(out="einops==0.8.0\n")
        return _done(0, "ok")
    _patch(monkeypatch, "run", fake_run)
    _patch(monkeypatch, "uv_path", lambda: "uv")
    monkeypatch.setattr(P, "flash_attn_requirement", lambda machine=None: P.FLASH_ATTN_WHEEL)
    monkeypatch.setattr(P, "fetch_flash_attn", lambda work: (None, "the wheel did not match its sha256"))
    ok, note = P.environment(root, "torch")
    assert not ok and "did not match its sha256" in note
    assert not any("flash" in " ".join(c) for c in _installs(seen))
    assert not (root / ".venv" / P.BUILD_RECORD).exists()


@pytest.mark.parametrize("engine", ["stable-audio-3", "mimetic-a0"])
def test_a_weight_set_whose_licence_has_obligations_fetches_its_licence_files(engine):
    from khaos_attribution.engines import descriptor
    d = descriptor(engine)
    assert d.licence.get("obligations")
    for w in d.weights:
        if w.get("runtime") != "mlx":
            assert {"LICENSE*", "NOTICE*"} <= set(w.get("patterns") or []), w["repo"]


def _cached_without_licence(tmp_path, monkeypatch):
    snapshot = tmp_path / "hub" / "snapshots" / "rev"
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").write_bytes(b"w")
    monkeypatch.setattr(P, "cached", lambda repo, files: True)
    monkeypatch.setattr(P, "cached_path", lambda repo, name: str(snapshot / name))
    return snapshot


def test_check_notes_a_cached_set_with_no_licence_beside_it_without_failing(tmp_path, monkeypatch):
    """Weights already cached are not fetched again, so their licence files are reported, never assumed."""
    snapshot = _cached_without_licence(tmp_path, monkeypatch)
    root = tmp_path / "sa3"
    monkeypatch.setattr(P, "checkout", lambda root, check_only=False: (True, "at the pin"))
    monkeypatch.setattr(P, "environment", lambda root, runtime, check_only=False: (True, "imports"))
    report = P.provision(root, tmp_path / "w", device="cuda", check_only=True)
    assert report.passed and "[note] no licence file beside stabilityai/stable-audio-3-medium-base" in report.text()
    (snapshot / "LICENSE.md").write_text("terms")
    (snapshot / "NOTICE").write_text("notice")
    assert P.licence_notes("torch") == [] and "[note]" not in P.provision(root, tmp_path / "w", device="cuda", check_only=True).text()


def test_the_weights_check_the_workshop_reads_stays_a_list_of_passes_and_failures(tmp_path, monkeypatch):
    """The Workshop counts every weights() line that is not ok as missing weights; a cached set with no licence file
    beside it is a pass there, its note said only in the report."""
    _cached_without_licence(tmp_path, monkeypatch)
    lines = P.weights(tmp_path / "sa3", "torch", tmp_path / "w", check_only=True)
    assert lines and all(ok is True for ok, _line in lines), lines
    assert [line for ok, line in lines if not ok] == [], "what the Workshop's engine_weights_missing would read"
    assert P.licence_notes("torch"), "the gap is still said, apart from the check"


def test_an_mlx_environment_built_at_another_pin_fails_the_check_and_runs_the_ports_installer_again(tmp_path, monkeypatch):
    root = tmp_path / "sa3"
    python = P.venv_python(root, "mlx")
    python.parent.mkdir(parents=True)
    python.write_text("")
    (root / P.MLX_SUBDIR / "requirements.txt").write_text("mlx==0.1\n")
    (python.parent.parent / P.BUILD_RECORD).write_text(json.dumps({"commit": "1" * 40, "lock_sha256": "x"}))
    seen = []
    _patch(monkeypatch, "run", lambda cmd, cwd=None, env=None: seen.append([str(c) for c in cmd]) or _done(0, "ok"))
    _patch(monkeypatch, "uv_path", lambda: "uv")
    monkeypatch.setattr(P, "requirements_missing", lambda python: None)
    monkeypatch.setattr(P, "requirements_into", lambda python: (True, "contract"))
    ok, note = P.environment(root, "mlx", check_only=True)
    assert not ok and "built at 111111111111" in note and not any("install.sh" in c for c in seen)
    assert P.environment(root, "mlx")[0] and any("install.sh" in c for c in seen)
    assert json.loads((python.parent.parent / P.BUILD_RECORD).read_text())["commit"] == P.DESCRIPTOR.commit


def test_the_cpp_runtime_is_carried_from_the_app_on_windows_only_and_never_over_what_is_there(tmp_path):
    """An engine's environment takes Microsoft's C++ runtime from the app's own Python folder, never the machine's,
    and a .pth adds its folder to every library search from the interpreter's start."""
    from khaos_attribution.engines import environment as E
    source, venv = tmp_path / "app", tmp_path / ".venv"
    site = venv / "Lib" / "site-packages"
    source.mkdir()
    for name in ("msvcp140.dll", "msvcp140_atomic_wait.dll", "vcruntime140.dll", "python312.dll"):
        (source / name).write_bytes(b"app")
    (venv / E.CPP_RUNTIME_DIR).mkdir(parents=True)
    (venv / E.CPP_RUNTIME_DIR / "vcruntime140.dll").write_bytes(b"own")
    assert E.carry_cpp_runtime(venv, site, source, platform="darwin") == []
    assert not (site / E.CPP_RUNTIME_PTH).exists(), "nothing on another system"
    assert E.carry_cpp_runtime(venv, site, source, platform="win32") == ["msvcp140.dll", "msvcp140_atomic_wait.dll"]
    assert (venv / E.CPP_RUNTIME_DIR / "vcruntime140.dll").read_bytes() == b"own"
    assert not (venv / E.CPP_RUNTIME_DIR / "python312.dll").exists(), "only the C++ runtime is carried"
    line = (site / E.CPP_RUNTIME_PTH).read_text(encoding="utf-8")
    assert line.startswith("import ") and line.count("\n") == 1, "a .pth runs a line only when it starts with import"
    assert "os.add_dll_directory(os.path.join(sys.prefix, 'khaos_cpp_runtime'))" in line
    assert line.isascii(), "Windows reads .pth files in its own code page: no path, nothing but ASCII"


def _ready_windows_environment(tmp_path, monkeypatch, carried: list):
    """A Windows torch environment that already runs the engine, with the carry recorded."""
    from khaos_attribution.engines.sa3 import render as R
    monkeypatch.setattr(R, "WINDOWS", True)
    monkeypatch.setattr(P, "WINDOWS", True)
    root = tmp_path / "sa3"
    python = R.venv_python(root, "torch")
    python.parent.mkdir(parents=True)
    python.write_text("")
    _patch(monkeypatch, "run", lambda cmd, cwd=None, env=None: _done(0, "ok"))
    monkeypatch.setattr(P, "build_stale", lambda root, runtime: None)
    monkeypatch.setattr(P, "build_record", lambda root, runtime: {"recorded": True})
    monkeypatch.setattr(P, "requirements_missing", lambda python: None)
    monkeypatch.setattr(P, "carry_cpp_runtime", lambda venv, site: carried.append(venv) or [])
    return root


def test_an_environment_made_before_the_runtime_gets_it_at_the_next_provision_and_a_check_changes_nothing(
        tmp_path, monkeypatch):
    carried = []
    root = _ready_windows_environment(tmp_path, monkeypatch, carried)
    ok, _note = P.environment(root, "torch", check_only=True)
    assert ok and carried == [], "a check carries nothing"
    ok, _note = P.environment(root, "torch")
    assert ok and carried == [root / ".venv"]


def test_an_existing_environment_gets_the_runtime_and_a_copy_that_fails_is_said(tmp_path, monkeypatch):
    """A failed copy is a refusal in words, never a traceback."""
    root = tmp_path / "sa3"
    calls = []
    monkeypatch.setattr(P, "carry_cpp_runtime", lambda venv, site: calls.append(venv) or ["msvcp140.dll"])
    assert P._carry_runtime(root) == (["msvcp140.dll"], "") and calls == [root / ".venv"]

    def refuse(venv, site):
        raise PermissionError("denied")
    monkeypatch.setattr(P, "carry_cpp_runtime", refuse)
    carried, why = P._carry_runtime(root)
    assert carried == [] and "could not be made to run on its own: denied" in why


def test_torchaudio_s_ffmpeg_bindings_leave_the_windows_environment_only(tmp_path):
    """Through torio's bindings the engine would load an FFmpeg found on the machine; on Windows they leave."""
    from khaos_attribution.engines import environment as E
    lib = tmp_path / "site" / "torio" / "lib"
    lib.mkdir(parents=True)
    for name in ("_torio_ffmpeg4.pyd", "libtorio_ffmpeg6.pyd", "__init__.py"):
        (lib / name).write_bytes(b"")
    assert E.leave_out_ffmpeg_bindings(tmp_path / "site", platform="darwin") == []
    assert len(list(lib.iterdir())) == 3
    assert E.leave_out_ffmpeg_bindings(tmp_path / "site", platform="win32") == ["_torio_ffmpeg4.pyd", "libtorio_ffmpeg6.pyd"]
    assert [p.name for p in lib.iterdir()] == ["__init__.py"]
    assert E.leave_out_ffmpeg_bindings(tmp_path / "nowhere", platform="win32") == []


def test_a_binding_held_open_is_passed_over_not_a_failure(tmp_path, monkeypatch):
    from khaos_attribution.engines import environment as E
    lib = tmp_path / "torio" / "lib"
    lib.mkdir(parents=True)
    (lib / "_torio_ffmpeg4.pyd").write_bytes(b"")
    (lib / "_torio_ffmpeg5.pyd").write_bytes(b"")
    real = Path.unlink

    def held(self, *a, **k):
        if self.name == "_torio_ffmpeg4.pyd":
            raise PermissionError("in use")
        return real(self, *a, **k)
    monkeypatch.setattr(Path, "unlink", held)
    assert E.leave_out_ffmpeg_bindings(tmp_path, platform="win32") == ["_torio_ffmpeg5.pyd"]


def test_making_the_environment_stand_alone_also_leaves_the_ffmpeg_bindings_out(tmp_path, monkeypatch):
    root = tmp_path / "sa3"
    left = []
    monkeypatch.setattr(P, "carry_cpp_runtime", lambda venv, site: [])
    monkeypatch.setattr(P, "leave_out_ffmpeg_bindings", lambda site: left.append(site) or [])
    assert P._carry_runtime(root) == ([], "") and left == [root / ".venv" / "Lib" / "site-packages"]
