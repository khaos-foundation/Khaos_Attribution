"""Stable Audio 3's provision slot behind fakes: --check reports what is missing without changing anything, the
pin is compared not assumed, every listed weight is accounted for, and the report says CHECK OK only when all is."""
import subprocess

import pytest

from khaos_attribution.engines.sa3 import provision as P


def _done(code=0, out="", err=""):
    return subprocess.CompletedProcess(args=[], returncode=code, stdout=out, stderr=err)


def test_check_reports_a_missing_checkout_and_changes_nothing(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(P, "_run", lambda cmd, cwd=None, env=None: calls.append(cmd) or _done())
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
    monkeypatch.setattr(P, "_run", fake_run)
    monkeypatch.setattr(P, "contract_present", lambda python: True)
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
    monkeypatch.setattr(P, "_run", fake_run)
    monkeypatch.setattr(P, "contract_present", lambda python: True)
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
    monkeypatch.setattr(P, "_run", lambda cmd, cwd=None, env=None: _done(out="deadbeef" * 5 + "\n") if "rev-parse" in [str(c) for c in cmd] else _done())
    ok, note = P.checkout(root, check_only=True)
    assert not ok and "pin is" in note


def test_the_mlx_runtime_wants_the_exported_base_weights(tmp_path, monkeypatch):
    root = tmp_path / "sa3"
    (root / ".git").mkdir(parents=True)
    mlx_python = root / "optimized" / "mlx" / ".venv" / "bin" / "python"
    mlx_python.parent.mkdir(parents=True)
    mlx_python.write_text("")
    pin = P.DESCRIPTOR.commit
    monkeypatch.setattr(P, "_run", lambda cmd, cwd=None, env=None: _done(out=pin + "\n") if "rev-parse" in [str(c) for c in cmd] else _done(0, "ok"))
    weights = tmp_path / "w"
    monkeypatch.setattr(P, "cached", lambda repo, files: True)
    mlx = root / "optimized" / "mlx"
    monkeypatch.setattr(P, "contract_present", lambda python: False)
    assert "runs the engine but holds" in P.provision(root, weights, device="mps", check_only=True).text(), \
        "the MLX measure worker projects with the contract, so an MLX environment without it is not ready"
    monkeypatch.setattr(P, "contract_present", lambda python: True)
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
        if "-c" in cmd and "import torch" in cmd[-1]:
            (root / ".venv" / "Scripts").mkdir(parents=True, exist_ok=True)
            (root / ".venv" / "Scripts" / "python.exe").write_text("")
            return _done(0, "ok")
        return _done()
    monkeypatch.setattr(P, "_run", fake_run)
    monkeypatch.setattr(P, "uv_path", lambda: "uv")
    monkeypatch.setattr(P, "contract_present", lambda python: True)
    monkeypatch.setattr(P, "contract_into", lambda python, root: (True, "contract"))
    monkeypatch.setattr(P, "cached", lambda repo, files: True)
    report = P.provision(root, tmp_path / "w", device="cuda")
    assert report.passed, report.text()
    joined = [" ".join(c) for c in calls]
    assert not any("flash-attn" in c for c in joined), "no Flash Attention build on Windows"
    assert not any("flash_attn" in c for c in joined), "the probe does not ask for it either"
    assert any("Scripts/python.exe" in c or "Scripts\\python.exe" in c for c in joined)
    assert P.installed(root, "torch", tmp_path / "w")


def test_the_contract_comes_from_the_nearest_source(tmp_path, monkeypatch):
    """A checkout named in the environment, then a shipped wheel, then the room's sibling checkout, then the tag."""
    from khaos_attribution import __version__
    root = tmp_path / "engines" / "stable-audio-3"
    root.mkdir(parents=True)
    monkeypatch.delenv(P.CONTRACT_SRC_ENV, raising=False)
    monkeypatch.delenv(P.CONTRACT_WHEEL_ENV, raising=False)
    assert P.contract_spec(root) == f"khaos-attribution @ git+{P.CONTRACT_REPO}@v{__version__}"
    (tmp_path / "engines" / "Khaos_Attribution").mkdir()
    (tmp_path / "engines" / "Khaos_Attribution" / "pyproject.toml").write_text("")
    assert P.contract_spec(root) == str(tmp_path / "engines" / "Khaos_Attribution")
    monkeypatch.setenv(P.CONTRACT_WHEEL_ENV, str(tmp_path / "k.whl"))
    assert P.contract_spec(root) == str(tmp_path / "k.whl")
    monkeypatch.setenv(P.CONTRACT_SRC_ENV, str(tmp_path / "src"))
    assert P.contract_spec(root) == str(tmp_path / "src")


def test_the_contract_counts_as_present_only_when_its_code_is_the_same(tmp_path, monkeypatch):
    """A version string alone would pass an engine environment holding older code of the same version."""
    here = P.contract_fingerprint()
    assert here and here.split()[0] == __import__("khaos_attribution").__version__
    monkeypatch.setattr(P, "contract_fingerprint", lambda python=None: here if python is None else here.split()[0] + " other")
    assert not P.contract_present(tmp_path / "python")
    monkeypatch.setattr(P, "contract_fingerprint", lambda python=None: here)
    assert P.contract_present(tmp_path / "python")


def test_the_contract_hash_ignores_line_endings_and_names_a_same_version_mismatch(tmp_path, monkeypatch):
    """A Windows checkout and a wheel of the same code hash alike; the same version with other code is said so."""
    from khaos_attribution.engines.pins import contract_code_hash
    for name, text in (("lf", b"x = 1\ny = 2\n"), ("crlf", b"x = 1\r\ny = 2\r\n")):
        (tmp_path / name).mkdir()
        (tmp_path / name / "m.py").write_bytes(text)
    assert contract_code_hash(tmp_path / "lf").split()[1] == contract_code_hash(tmp_path / "crlf").split()[1]
    ours = P.contract_fingerprint()
    monkeypatch.setattr(P, "contract_fingerprint", lambda python=None: ours if python is None else ours.split()[0] + " other")
    assert "built from other code" in P.contract_mismatch(tmp_path / "python")
    monkeypatch.setattr(P, "contract_fingerprint", lambda python=None: ours if python is None else None)
    assert "older than" in P.contract_mismatch(tmp_path / "python")


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
