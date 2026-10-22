"""Stable Audio 3's provision slot: Stability's code at the descriptor's pin, its own environment for this
machine's runtime (torch with Flash Attention on CUDA; the MLX port on Apple silicon), the weights the descriptor
lists, and the MLX base weights exported once. --check reports without changing anything. Nothing here handles a
Hugging Face account: every weight set comes from Khaos's own mirror first, the hub only when the mirror cannot serve it."""

from __future__ import annotations

from khaos_attribution import weights as mirror

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from khaos_attribution.engines.sa3 import DESCRIPTOR
from khaos_attribution.engines.sa3.render import MLX_SUBDIR, WINDOWS, find_root, runtime_for, venv_python
from khaos_attribution.engines.sa3.train import BASE_NPZ

# Stability's CUDA 12.6 build of torch 2.7.1 cannot drive a Blackwell card; its CUDA 12.8 build can.
TORCH = "torch==2.7.1"
TORCHAUDIO = "torchaudio==2.7.1"
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
FLASH_ATTN = "flash-attn==2.8.3"
PYTHON_VERSION = "3.11"


@dataclass
class Report:
    checks: list = field(default_factory=list)

    def ok(self, what: str) -> None:
        self.checks.append((True, what))

    def bad(self, what: str) -> None:
        self.checks.append((False, what))

    @property
    def passed(self) -> bool:
        return all(ok for ok, _ in self.checks)

    def text(self) -> str:
        return "\n".join(f"[{'ok' if ok else 'FAIL'}] {what}" for ok, what in self.checks) + \
            ("\nCHECK OK" if self.passed else "\nCHECK FAILED")


def uv_path() -> str:
    found = shutil.which("uv") or os.path.expanduser("~/.local/bin/uv")
    if not Path(found).exists():
        raise RuntimeError("uv is not installed; the engine's environment is made with it")
    return found


def _run(cmd: list, cwd: Path | None = None, env: dict | None = None) -> subprocess.CompletedProcess:
    # The engine's interpreters never see the room's PYTHONPATH: its packages are not theirs.
    env = {k: v for k, v in (env if env is not None else os.environ).items() if k != "PYTHONPATH"}
    return subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None, env=env, capture_output=True, text=True)


def git_usable() -> bool:
    """Whether git here runs without a dialog: a Mac without the developer tools has a git stub that opens one."""
    if not shutil.which("git"):
        return False
    if sys.platform == "darwin":
        return _run(["xcode-select", "-p"]).returncode == 0
    return True


def archive_url(repo: str, commit: str) -> str:
    base = repo[:-4] if repo.endswith(".git") else repo
    return f"{base}/archive/{commit}.tar.gz"


# What an unpacked checkout keeps when it moves to a new pin: its environments and the weights beside them.
KEPT_ACROSS_PINS = (".venv", "checkpoints", f"{MLX_SUBDIR}/.venv", f"{MLX_SUBDIR}/models/mlx")


def unpack_archive(url: str, root: Path, commit: str, kept: tuple = KEPT_ACROSS_PINS) -> None:
    """Stability's code at the commit from its archive, when git is not usable here: the top folder stripped,
    the commit recorded in .khaos-commit so the pin can still be read; an earlier unpack's environments and
    weights carried across."""
    import io  # noqa: PLC0415
    import tarfile  # noqa: PLC0415
    import urllib.request  # noqa: PLC0415
    from khaos_attribution.engines.pins import COMMIT_FILE  # noqa: PLC0415
    with urllib.request.urlopen(url, timeout=120) as resp:  # noqa: S310 — the descriptor's own code repository
        data = resp.read()
    part = root.with_name(root.name + ".part")
    shutil.rmtree(part, ignore_errors=True)
    part.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        members = [m for m in tar.getmembers() if "/" in m.name]
        for m in members:
            m.name = m.name.split("/", 1)[1]
        tar.extractall(part, members=members, filter="data")
    (part / COMMIT_FILE).write_text(commit + "\n", encoding="utf-8")
    for rel in kept:
        if (root / rel).exists():
            (part / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.rmtree(part / rel, ignore_errors=True)
            shutil.move(str(root / rel), str(part / rel))
    old = root.with_name(root.name + ".old")
    shutil.rmtree(old, ignore_errors=True)
    if root.exists():
        root.replace(old)
    part.replace(root)
    shutil.rmtree(old, ignore_errors=True)


MLX_TRAINER = Path("optimized") / "mlx" / "scripts" / "lora_train_mlx.py"
# Khaos trains under the generation mask (all zeros) the renderer and the torch trainer use, never the MLX trainer's ones.
TRAINER_MASK_ONES = "np.concatenate([np.ones((1, 1, crop_len), np.float32),"
TRAINER_MASK_ZEROS = "np.concatenate([np.zeros((1, 1, crop_len), np.float32),"


def trainer_mask(root: Path) -> str | None:
    """"zeros" or "ones": the inpaint mask the MLX trainer at ``root`` conditions on, or None when it is not found."""
    try:
        text = (Path(root) / MLX_TRAINER).read_text(encoding="utf-8")
    except OSError:
        return None
    if text.count(TRAINER_MASK_ZEROS) == 1 and TRAINER_MASK_ONES not in text:
        return "zeros"
    if text.count(TRAINER_MASK_ONES) == 1:
        return "ones"
    return None


def mlx_trainer_patch(root: Path, *, check_only: bool = False) -> tuple:
    """The MLX trainer conditioned on the generation mask; (ok, note)."""
    mask = trainer_mask(root)
    if mask == "zeros":
        return True, "the MLX trainer conditions on the generation mask, as the renderer does"
    if mask is None:
        return False, f"the MLX trainer's inpaint conditioning was not found in {MLX_TRAINER}; the patch needs review at this pin"
    if check_only:
        return False, "the MLX trainer still conditions on the all-ones inpaint mask; provision to patch it"
    path = Path(root) / MLX_TRAINER
    path.write_text(path.read_text(encoding="utf-8").replace(TRAINER_MASK_ONES, TRAINER_MASK_ZEROS), encoding="utf-8")
    return trainer_mask(root) == "zeros", "the MLX trainer patched to condition on the generation mask"


def checkout(root: Path, *, check_only: bool = False) -> tuple:
    """Stability's code at the descriptor's commit; (ok, note)."""
    from khaos_attribution.engines.pins import checkout_commit  # noqa: PLC0415
    repo, commit = DESCRIPTOR.doc["code"]["repo"], DESCRIPTOR.commit
    if not (root / ".git").is_dir() and checkout_commit(root) is None:
        if check_only:
            return False, f"Stable Audio 3 code not cloned at {root}"
        root.parent.mkdir(parents=True, exist_ok=True)
        if git_usable():
            done = _run(["git", "clone", "-q", repo, root])
            if done.returncode != 0:
                return False, f"clone failed: {done.stderr.strip()[-300:]}"
        else:
            try:
                unpack_archive(archive_url(repo, commit), root, commit)
            except (OSError, ValueError) as exc:
                return False, f"the code archive could not be fetched: {exc}"
            return True, f"Stable Audio 3 code unpacked at the pin {commit[:12]}"
    head = _run(["git", "-C", root, "rev-parse", "HEAD"]).stdout.strip() if (root / ".git").is_dir() else (checkout_commit(root) or "")
    if head == commit:
        return True, f"Stable Audio 3 code at the pin {commit[:12]}"
    if check_only:
        return False, f"Stable Audio 3 code at {head[:12] or 'an unknown commit'}, pin is {commit[:12]}"
    if not (root / ".git").is_dir():
        try:
            unpack_archive(archive_url(repo, commit), root, commit)
        except (OSError, ValueError) as exc:
            return False, f"the code archive could not be fetched: {exc}"
        return True, f"Stable Audio 3 code moved to the pin {commit[:12]}"
    _run(["git", "-C", root, "fetch", "-q", "origin", commit])
    # Khaos's own patch comes off before the move and goes back on at the new pin.
    _run(["git", "-C", root, "checkout", "-q", "--", str(MLX_TRAINER)])
    done = _run(["git", "-C", root, "checkout", "-q", "--detach", commit])
    return (done.returncode == 0), (f"checked out {commit[:12]}" if done.returncode == 0 else f"checkout failed: {done.stderr.strip()[-300:]}")


def environment(root: Path, runtime: str, *, check_only: bool = False) -> tuple:
    """The engine's own interpreter: (ok, note)."""
    if runtime == "mlx":
        mlx = root / MLX_SUBDIR
        python = venv_python(root, "mlx")
        if python.is_file():
            probe = _run([python, "-c", "import mlx.core, safetensors; print('ok')"], cwd=mlx)
            if probe.returncode == 0:
                # The MLX measure worker projects its features with the contract, as the torch one does.
                if contract_present(python):
                    return True, "the MLX runtime imports"
                if check_only:
                    return False, f"the MLX environment at {python} runs the engine but {contract_mismatch(python)}"
                return contract_into(python, root)
        if check_only:
            return False, f"no MLX environment at {python}"
        done = _run(["bash", "install.sh", "-y"], cwd=mlx, env={**os.environ, "UV": uv_path()})
        if done.returncode != 0:
            return False, f"the MLX installer failed: {done.stderr.strip()[-300:]}"
        _run([uv_path(), "pip", "install", "-q", "--python", python, "safetensors"])
        if not python.is_file():
            return False, "the MLX runtime did not install"
        ok, note = contract_into(python, root)
        return ok, ("the MLX runtime installed; " + note) if ok else note
    python = venv_python(root, "torch")
    # Flash Attention has no Windows build; Stability's model falls back to PyTorch's own attention without it.
    probe_code = ("import torch, stable_audio_3" + ("" if WINDOWS else ", flash_attn")
                  + "; assert torch.cuda.is_available(); print('ok')")
    probe_note = "torch sees the GPU and stable_audio_3 imports" + ("" if WINDOWS else "; Flash Attention imports")
    if python.is_file():
        probe = _run([python, "-c", probe_code])
        if probe.returncode == 0:
            if contract_present(python):
                return True, probe_note
            if check_only:
                return False, f"the environment at {python} runs the engine but {contract_mismatch(python)}"
            ok, note = contract_into(python, root)
            return ok, (probe_note + "; " + note) if ok else note
        if check_only:
            return False, f"the environment at {python} cannot run the engine: {probe.stderr.strip()[-200:]}"
    elif check_only:
        return False, f"no environment at {python}"
    uv = uv_path()
    work = root / "work"
    work.mkdir(exist_ok=True)
    if not python.is_file():
        _run([uv, "venv", "-q", "--python", PYTHON_VERSION, root / ".venv"])
    export = _run([uv, "export", "--frozen", "--no-hashes", "--extra", "lora", "--no-emit-project", "--no-dev"], cwd=root)
    if export.returncode != 0 or not export.stdout.strip():
        return False, f"Stability's lock could not be exported: {export.stderr.strip()[-300:]}"
    locked = "\n".join(l for l in export.stdout.splitlines()
                       if not (l.startswith(("torch==", "torchaudio==", "triton==", "nvidia-", "#", "-e")) or not l.strip()))
    (work / "stability-locked.txt").write_text(locked + "\n")
    steps = [
        [uv, "pip", "install", "-q", "--python", python, "-r", work / "stability-locked.txt", f"{TORCH}+cu128",
         f"{TORCHAUDIO}+cu128", "--extra-index-url", TORCH_INDEX, "--index-strategy", "unsafe-best-match"],
        [uv, "pip", "install", "-q", "--python", python, "--no-sources", "--no-deps", "-e", root],
        [uv, "pip", "install", "-q", "--python", python, "setuptools", "wheel", "ninja", "psutil", "matplotlib", "wandb"],
        *([] if WINDOWS else [[uv, "pip", "install", "--python", python, "--no-build-isolation", FLASH_ATTN]]),
    ]
    for cmd in steps:
        done = _run(cmd, cwd=root)
        if done.returncode != 0:
            return False, f"environment step failed ({' '.join(str(c) for c in cmd[:4])}…): {done.stderr.strip()[-300:]}"
    ok, note = contract_into(python, root)
    if not ok:
        return False, f"the engine's environment installed; {note}"
    probe = _run([python, "-c", probe_code])
    if probe.returncode != 0:
        return False, f"the environment installed but cannot run the engine: {probe.stderr.strip()[-200:]}"
    return True, f"the engine's environment installed; {probe_note}"


CONTRACT_SRC_ENV = "KHAOS_CONTRACT_SRC"
CONTRACT_WHEEL_ENV = "KHAOS_CONTRACT_WHEEL"
CONTRACT_REPO = "https://github.com/khaos-foundation/Khaos_Attribution.git"


def contract_spec(root: Path) -> str:
    """Where the contract comes from for the engine's environment: a checkout named in KHAOS_CONTRACT_SRC, a wheel
    named in KHAOS_CONTRACT_WHEEL (the app ships one), the room's sibling checkout, else the tagged release."""
    from khaos_attribution import __version__  # noqa: PLC0415
    src = os.environ.get(CONTRACT_SRC_ENV, "").strip()
    if src:
        return src
    wheel = os.environ.get(CONTRACT_WHEEL_ENV, "").strip()
    if wheel:
        return wheel
    sibling = root.parent / "Khaos_Attribution"
    if (sibling / "pyproject.toml").is_file():
        return str(sibling)
    return f"khaos-attribution @ git+{CONTRACT_REPO}@v{__version__}"


CONTRACT_FINGERPRINT = "from khaos_attribution.engines.pins import contract_code_hash; print(contract_code_hash())"


def contract_fingerprint(python: Path | None = None) -> str | None:
    """"<version> <hash>" of the contract an interpreter imports (this process's own when None), or None when it
    imports none or one too old to say."""
    if python is None:
        from khaos_attribution.engines.pins import contract_code_hash  # noqa: PLC0415
        return contract_code_hash()
    done = _run([python, "-c", CONTRACT_FINGERPRINT])
    return done.stdout.strip() if done.returncode == 0 else None


def contract_mismatch(python: Path) -> str:
    """Why the engine's environment does not hold this contract, in words."""
    theirs, ours = contract_fingerprint(python), contract_fingerprint()
    if not theirs:
        return "holds no contract, or one older than 0.41.1"
    if theirs.split()[0] == ours.split()[0]:
        return f"holds contract {theirs.split()[0]} built from other code than this room's"
    return f"holds contract {theirs.split()[0]}, this room runs {ours.split()[0]}"


def contract_present(python: Path) -> bool:
    theirs = contract_fingerprint(python)
    return theirs is not None and theirs == contract_fingerprint()


def contract_into(python: Path, root: Path) -> tuple:
    """The contract's gradient maths into the engine's environment, the version this room runs: the measure worker
    projects its features with it."""
    from khaos_attribution import __version__  # noqa: PLC0415
    # With its base dependencies (jsonschema, numpy): the engine's own pins are Stability's, the contract's are light.
    done = _run([uv_path(), "pip", "install", "-q", "--python", python, "--reinstall-package", "khaos-attribution", contract_spec(root)])
    if done.returncode != 0:
        return False, f"the contract could not be installed into the engine's environment: {done.stderr.strip()[-200:]}"
    return True, f"khaos-attribution {__version__} in the engine's environment"


def cached(repo: str, filenames: list) -> bool:
    """Whether the Hugging Face cache holds every named file of the repo: the only place Stability's torch
    package reads weights from."""
    try:
        from huggingface_hub import try_to_load_from_cache  # noqa: PLC0415
    except ImportError:
        return False
    for name in filenames:
        found = try_to_load_from_cache(repo, name)
        if not isinstance(found, str):
            return False
    return True


# Stability's repository keeps the port's files under MLX/; the port reads them under models/mlx/.
MLX_REPO_DIR = "MLX"


def _mlx_from_mirror(repo: str, mlx: Path, components: list) -> str | None:
    """The port's files from the mirror, each placed straight where the port reads it (with the licence and the
    notice beside them), so a retry finds what already landed and the checkout gains no file of its own."""
    by_name = {Path(c).name: c for c in components}
    folder = str(Path(components[0]).parent) if components else "models/mlx"

    def place(path: str) -> str:
        name = Path(path).name
        return by_name.get(name) or f"{folder}/{name}"

    return mirror.fetch_into_dir(repo, mlx, [f"{MLX_REPO_DIR}/{n}" for n in by_name] + ["LICENSE*", "NOTICE"], place=place)


def _from_mirror(fetch) -> bool:
    """Whether the mirror served the set; a mirror that cannot is a fallback to the hub, never an error here."""
    try:
        return fetch() is not None
    except (mirror.MirrorError, OSError):
        return False


def mlx_base_link(root: Path, weights_dir: Path) -> Path:
    """Where the MLX trainer's base weights are expected, a link to the port's ready-made file."""
    return weights_dir / BASE_NPZ


def weights(root: Path, runtime: str, weights_dir: Path, *, check_only: bool = False) -> list:
    """Every weight set the descriptor lists for this runtime: torch sets into the Hugging Face cache by the
    engine's own downloader (a gated set only once the user's own login has been approved); the MLX port's set
    through its own weights module, its ready-made base file linked where the trainer looks."""
    python = venv_python(root, runtime)
    out = []
    for w in DESCRIPTOR.weights:
        if w.get("runtime") and w["runtime"] != runtime:
            continue
        required = w.get("required", True)
        if w.get("runtime") == "mlx":
            mlx = root / MLX_SUBDIR
            parts = [mlx / c for c in (w.get("components") or [])]
            for f in parts:
                # A link left pointing at a cache that is gone would make the port skip the fetch.
                if f.is_symlink() and not f.exists():
                    f.unlink()
            if all(f.exists() for f in parts):
                out.append((True, f"{w['repo']} present under {mlx}"))
                continue
            if check_only:
                out.append((not required, f"{w['repo']} missing under {mlx}"))
                continue
            # Khaos's own mirror first; the port's downloader only when the mirror cannot serve the set.
            if _from_mirror(lambda: _mlx_from_mirror(w["repo"], mlx, w.get("components") or [])) and all(f.exists() for f in parts):
                out.append((True, f"{w['repo']} fetched from the Khaos weights mirror"))
                continue
            code = ("import sys; sys.path.insert(0, 'scripts'); from weights import ensure_local; "
                    "[ensure_local(p, verbose=False) for p in sys.argv[1:]]")
            done = _run([python, "-c", code, *(w.get("components") or [])], cwd=mlx, env={**os.environ, "HF_HUB_OFFLINE": "0"})
            whole = done.returncode == 0 and all(f.exists() for f in parts)
            out.append((whole, f"{w['repo']} fetched by the MLX port" if whole
                        else f"{w['repo']} could not be fetched: {done.stderr.strip()[-200:] or 'a file is still missing'}"))
            continue
        files = list(w.get("components") or ["config.json"])
        if cached(w["repo"], files):
            out.append((True, f"{w['repo']} in the Hugging Face cache"))
            continue
        if check_only or not required:
            out.append((not required, f"{w['repo']} missing from the Hugging Face cache" + ("" if required else " (optional)")))
            continue
        if _from_mirror(lambda: mirror.fetch_into_hub_cache(w["repo"], mirror.default_hub_cache(), w.get("patterns"))) \
                and cached(w["repo"], files):
            out.append((True, f"{w['repo']} fetched from the Khaos weights mirror into the Hugging Face cache"))
            continue
        code = ("import json, sys; from huggingface_hub import snapshot_download; "
                "snapshot_download(sys.argv[1], allow_patterns=json.loads(sys.argv[2]) or None)")
        done = _run([python, "-c", code, w["repo"], json.dumps(w.get("patterns") or [])], env={**os.environ, "HF_HUB_OFFLINE": "0"})
        if done.returncode != 0 or not cached(w["repo"], files):
            why = done.stderr.strip().splitlines()[-1:] or ["a file is still missing"]
            note = (f"{w['repo']} could not be downloaded: {why[0][:200]}"
                    + ("; accept its terms on its Hugging Face page, then `hf auth login`" if w.get("gated") else ""))
            out.append((False, note))
        else:
            out.append((True, f"{w['repo']} downloaded to the Hugging Face cache"))
    if runtime == "mlx":
        link = mlx_base_link(root, weights_dir)
        ready = root / MLX_SUBDIR / "models" / "mlx" / "dit_medium-base_f16.npz"
        if link.exists():
            out.append((True, f"MLX base weights at {link}"))
        elif check_only or not ready.exists():
            out.append((False, f"MLX base weights missing at {link}"))
        else:
            weights_dir.mkdir(parents=True, exist_ok=True)
            if link.is_symlink():
                link.unlink()
            # The link keeps the .npz name at every hop: the MLX loader reads the format from the ending it is given.
            link.symlink_to(ready.absolute())
            out.append((True, f"MLX base weights linked at {link}"))
    return out


def installed(root: Path, runtime: str, weights_dir: Path) -> bool:
    """A cheap reading of whether the engine can run here: its own interpreter and, on MLX, its base weights.
    The full check is provision(check_only=True)."""
    python = venv_python(root, runtime)
    if not python.is_file():
        return False
    return runtime != "mlx" or mlx_base_link(root, weights_dir).exists()


def provision(repo_dir=None, weights_dir=None, *, device: str = "auto", check_only: bool = False) -> Report:
    root = find_root(repo_dir)
    runtime = runtime_for(device)
    weights_dir = Path(weights_dir) if weights_dir else root / "checkpoints"
    report = Report()
    ok, note = checkout(root, check_only=check_only)
    (report.ok if ok else report.bad)(note)
    if ok:
        ok, note = environment(root, runtime, check_only=check_only)
        (report.ok if ok else report.bad)(note)
    if ok and runtime == "mlx":
        ok, note = mlx_trainer_patch(root, check_only=check_only)
        (report.ok if ok else report.bad)(note)
    if ok:
        for good, line in weights(root, runtime, weights_dir, check_only=check_only):
            (report.ok if good else report.bad)(line)
    return report


def main(argv=None) -> int:
    import argparse  # noqa: PLC0415
    ap = argparse.ArgumentParser(description="Provision Stable Audio 3 for Khaos on this machine.")
    ap.add_argument("--repo-dir", type=Path, default=None)
    ap.add_argument("--weights-dir", type=Path, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--check", action="store_true", help="report without changing anything")
    a = ap.parse_args(argv)
    report = provision(a.repo_dir, a.weights_dir, device=a.device, check_only=a.check)
    print(report.text())
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
