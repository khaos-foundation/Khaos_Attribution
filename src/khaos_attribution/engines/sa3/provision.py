"""Stable Audio 3's provision slot: Stability's code at the descriptor's pin, its own environment for this
machine's runtime (torch with Flash Attention on CUDA; the MLX port on Apple silicon), the weights the descriptor
lists, and the MLX base weights exported once. --check reports without changing anything. Nothing here handles a
Hugging Face account: every weight set comes from Khaos's own mirror first, the hub only when the mirror cannot serve it."""

from __future__ import annotations

from khaos_attribution import weights as mirror

import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from khaos_attribution.engines.sa3 import DESCRIPTOR
from khaos_attribution.engines.sa3.render import MLX_SUBDIR, WINDOWS, find_root, runtime_for, venv_python
from khaos_attribution.engines.sa3.train import BASE_NPZ
from khaos_attribution.engines.worker_process import host_env

# Stability's CUDA 12.6 build of torch 2.7.1 cannot drive a Blackwell card; its CUDA 12.8 build can.
TORCH = "torch==2.7.1"
TORCHAUDIO = "torchaudio==2.7.1"
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
FLASH_ATTN = "flash-attn==2.8.3"
# Flash Attention's own build for this torch and Python: its installer needs a CUDA compiler even to fetch it.
FLASH_ATTN_WHEEL = ("https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
                    "flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl")
# The release asset's digest as GitHub publishes it; the wheel is installed only once it matches.
FLASH_ATTN_WHEEL_SHA256 = "cd1a45ebfc1731a13e55ad68e0c9ad92390ddfffba306f9222be67c6d5a805af"
# What the trainer and Flash Attention's install need beyond Stability's lock, at the versions a working CUDA
# environment holds, so two builds agree.
EXTRAS = ("setuptools==82.0.1", "wheel==0.48.0", "ninja==1.13.2", "psutil==7.2.2", "matplotlib==3.11.2",
          "wandb==0.31.0")
PYTHON_VERSION = "3.11"
# Beside the interpreter's venv: the pin and the lock an environment was built from.
BUILD_RECORD = ".khaos-built.json"


@dataclass
class Report:
    checks: list = field(default_factory=list)

    def ok(self, what: str) -> None:
        self.checks.append((True, what))

    def bad(self, what: str) -> None:
        self.checks.append((False, what))

    def note(self, what: str) -> None:
        """Said, never failed on."""
        self.checks.append((None, what))

    @property
    def passed(self) -> bool:
        return all(ok is not False for ok, _ in self.checks)

    def text(self) -> str:
        mark = {True: "ok", False: "FAIL", None: "note"}
        return "\n".join(f"[{mark[ok]}] {what}" for ok, what in self.checks) + \
            ("\nCHECK OK" if self.passed else "\nCHECK FAILED")


def uv_path() -> str:
    found = shutil.which("uv") or os.path.expanduser("~/.local/bin/uv")
    if not Path(found).exists():
        raise RuntimeError("uv is not installed; the engine's environment is made with it")
    return found


def _run(cmd: list, cwd: Path | None = None, env: dict | None = None) -> subprocess.CompletedProcess:
    # The engine's interpreters never see the room's PYTHONPATH: its packages are not theirs.
    env = {k: v for k, v in (env if env is not None else host_env()).items() if k != "PYTHONPATH"}
    return subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None, env=env, capture_output=True, text=True)


def make_venv(uv: str, venv: Path, python: Path, version: str, *, fallback: bool = False) -> tuple:
    """A fresh venv at ``venv`` on Python ``version``, or with ``fallback`` on the Python running this: (ok, note).
    The note is empty on ``version``, says why on the fallback, and gives uv's own words when nothing can be made."""
    # A folder left by an earlier failed try holds no interpreter; uv refuses it unless told to replace it.
    env = {**host_env(), "UV_VENV_CLEAR": "1"}
    done = _run([uv, "venv", "-q", "--python", version, venv], env=env)
    note = ""
    if done.returncode != 0:
        why = done.stderr.strip()[:400]
        if not (fallback and (3, 10) <= sys.version_info[:2] <= (3, 12)):
            return False, f"the engine's Python {version} environment could not be made: {why}"
        here = f"{sys.version_info.major}.{sys.version_info.minor}"
        again = _run([uv, "venv", "-q", "--python", sys.executable, venv], env=env)
        if again.returncode != 0:
            return False, (f"the engine's Python {version} environment could not be made: {why}; "
                           f"nor on this Python {here} ({sys.executable}): {again.stderr.strip()[:400]}")
        note = f"Python {version} could not be had ({why}); the environment was made on this Python {here}"
    if not python.is_file():
        return False, f"uv made the environment but left no interpreter at {python}"
    return True, note


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


def remove_tree(path: Path) -> None:
    """``path`` removed, read-only files too (git marks its objects so on Windows); raises when any of it stays."""
    import stat  # noqa: PLC0415

    def writable_then_again(func, p, _exc):
        for q in (p, os.path.dirname(p)):
            try:
                os.chmod(q, os.stat(q).st_mode | stat.S_IWRITE | stat.S_IWUSR)
            except OSError:
                pass
        func(p)
    if path.exists():
        if sys.version_info >= (3, 12):
            shutil.rmtree(path, onexc=writable_then_again)
        else:
            shutil.rmtree(path, onerror=writable_then_again)
    if path.exists():
        raise OSError(f"{path} could not be removed")


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
    old = root.with_name(root.name + ".old")
    # Both are cleared before anything of the checkout moves, so a failure leaves it whole.
    remove_tree(part)
    remove_tree(old)
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
    if root.exists():
        try:
            root.replace(old)
        except OSError:
            # The checkout could not be set aside (a file in it is open): its environments and weights go back.
            for rel in kept:
                if (part / rel).exists():
                    (root / rel).parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(part / rel), str(root / rel))
            raise
    part.replace(root)
    try:
        remove_tree(old)
    except OSError:
        pass  # the checkout is in place; the next move clears what is left


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
    is_git = (root / ".git").is_dir()
    if not is_git and checkout_commit(root) is None:
        if check_only:
            return False, f"Stable Audio 3 code not cloned at {root}"
        root.parent.mkdir(parents=True, exist_ok=True)
        if git_usable():
            done = _run(["git", "clone", "-q", repo, root])
            if done.returncode != 0:
                return False, f"clone failed: {done.stderr.strip()[-300:]}"
            is_git = True
        else:
            try:
                unpack_archive(archive_url(repo, commit), root, commit)
            except (OSError, ValueError) as exc:
                return False, f"the code archive could not be fetched: {exc}"
            return True, f"Stable Audio 3 code unpacked at the pin {commit[:12]}"
    head = (_run(["git", "-C", root, "rev-parse", "HEAD"]).stdout.strip() if is_git and git_usable()
            else (checkout_commit(root) or ""))
    if head == commit:
        return True, f"Stable Audio 3 code at the pin {commit[:12]}"
    if check_only:
        return False, f"Stable Audio 3 code at {head[:12] or 'an unknown commit'}, pin is {commit[:12]}"
    # A clone with no git to move it takes the archive at the pin, as a checkout made without git does.
    if not is_git or not git_usable():
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


def flash_attn_requirement(machine: str | None = None) -> str:
    """The ready-made Flash Attention on Linux x86-64; elsewhere the source package, which builds with a CUDA compiler."""
    import platform  # noqa: PLC0415
    machine = machine or platform.machine()
    return FLASH_ATTN_WHEEL if sys.platform.startswith("linux") and machine in ("x86_64", "AMD64") else FLASH_ATTN


def _lock_file(root: Path, runtime: str) -> Path:
    return root / MLX_SUBDIR / "requirements.txt" if runtime == "mlx" else root / "uv.lock"


def build_wanted(root: Path, runtime: str) -> dict:
    """The pin and the sha256 of the lock an environment built here today would be built from."""
    try:
        lock = hashlib.sha256(_lock_file(root, runtime).read_bytes()).hexdigest()
    except OSError:
        lock = None
    return {"commit": DESCRIPTOR.commit, "lock_sha256": lock}


def _record_path(root: Path, runtime: str) -> Path:
    return venv_python(root, runtime).parent.parent / BUILD_RECORD


def build_record(root: Path, runtime: str) -> dict | None:
    """What the environment says it was built from, or None when it keeps no record."""
    try:
        record = json.loads(_record_path(root, runtime).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def build_stale(root: Path, runtime: str) -> str | None:
    """Why the environment was built from another pin or lock than today's, or None; no record is never stale, so a
    live environment built before the record was kept is left as it is."""
    record = build_record(root, runtime)
    if record is None:
        return None
    wanted = build_wanted(root, runtime)
    if record.get("commit") != wanted["commit"]:
        return f"was built at {str(record.get('commit'))[:12]}, the pin is {wanted['commit'][:12]}"
    if record.get("lock_sha256") != wanted["lock_sha256"]:
        return "was built from another lock than the pin's"
    return None


def _write_build_record(root: Path, runtime: str) -> None:
    _record_path(root, runtime).write_text(json.dumps(build_wanted(root, runtime)) + "\n", encoding="utf-8")


def fetch_flash_attn(work: Path) -> tuple:
    """Flash Attention's ready-made wheel under ``work``, kept only once its sha256 matches: (path, None) or (None, why)."""
    target = Path(work) / FLASH_ATTN_WHEEL.rsplit("/", 1)[1]

    def digest(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()
    if target.is_file() and digest(target) == FLASH_ATTN_WHEEL_SHA256:
        return target, None
    part = target.with_name(target.name + ".part")
    try:
        # The pinned release asset, checked below before anything installs it.
        with urllib.request.urlopen(FLASH_ATTN_WHEEL, timeout=120) as resp, part.open("wb") as out:  # noqa: S310
            shutil.copyfileobj(resp, out)
    except OSError as exc:
        part.unlink(missing_ok=True)
        return None, f"Flash Attention's wheel could not be fetched: {exc}"
    if digest(part) != FLASH_ATTN_WHEEL_SHA256:
        part.unlink(missing_ok=True)
        return None, "Flash Attention's wheel did not match its sha256; nothing was installed"
    part.replace(target)
    return target, None


def environment(root: Path, runtime: str, *, check_only: bool = False) -> tuple:
    """The engine's own interpreter: (ok, note). An environment whose build record names another pin or lock is
    installed again in place; one with no record is left as it is."""
    unrecorded = "; no build record (built before Khaos kept one)"
    if runtime == "mlx":
        mlx = root / MLX_SUBDIR
        python = venv_python(root, "mlx")
        if python.is_file():
            probe = _run([python, "-c", "import mlx.core, safetensors; print('ok')"], cwd=mlx)
            if probe.returncode == 0:
                stale = build_stale(root, "mlx")
                if stale and check_only:
                    return False, f"the MLX environment at {python} {stale}: provision again to rebuild it"
                if not stale:
                    note = "the MLX runtime imports" + ("" if build_record(root, "mlx") else unrecorded)
                    # The MLX measure worker projects its features with the contract, as the torch one does.
                    if contract_present(python):
                        return True, note
                    if check_only:
                        return False, f"the MLX environment at {python} runs the engine but {contract_mismatch(python)}"
                    return contract_into(python, root)
        if check_only:
            return False, f"no MLX environment at {python}"
        done = _run(["bash", "install.sh", "-y"], cwd=mlx, env={**host_env(), "UV": uv_path()})
        if done.returncode != 0:
            return False, f"the MLX installer failed: {done.stderr.strip()[-300:]}"
        _run([uv_path(), "pip", "install", "-q", "--python", python, "safetensors"])
        if not python.is_file():
            return False, "the MLX runtime did not install"
        ok, note = contract_into(python, root)
        if ok:
            _write_build_record(root, "mlx")
        return ok, ("the MLX runtime installed; " + note) if ok else note
    python = venv_python(root, "torch")
    # Flash Attention has no Windows build; Stability's model falls back to PyTorch's own attention without it.
    probe_code = ("import torch, stable_audio_3" + ("" if WINDOWS else ", flash_attn")
                  + "; assert torch.cuda.is_available(); print('ok')")
    probe_note = "torch sees the GPU and stable_audio_3 imports" + ("" if WINDOWS else "; Flash Attention imports")
    if python.is_file():
        probe = _run([python, "-c", probe_code])
        if probe.returncode == 0:
            stale = build_stale(root, "torch")
            if stale and check_only:
                return False, f"the environment at {python} {stale}: provision again to rebuild it"
            if not stale:
                note = probe_note + ("" if build_record(root, "torch") else unrecorded)
                if contract_present(python):
                    return True, note
                if check_only:
                    return False, f"the environment at {python} runs the engine but {contract_mismatch(python)}"
                ok, cnote = contract_into(python, root)
                return ok, (note + "; " + cnote) if ok else cnote
        elif check_only:
            return False, f"the environment at {python} cannot run the engine: {probe.stderr.strip()[-200:]}"
    elif check_only:
        return False, f"no environment at {python}"
    uv = uv_path()
    work = root / "work"
    work.mkdir(exist_ok=True)
    flash = [] if WINDOWS else [flash_attn_requirement()]
    # Fetched and checked before anything installs, so a bad download changes nothing.
    if flash == [FLASH_ATTN_WHEEL]:
        wheel, why = fetch_flash_attn(work)
        if why:
            return False, why
        flash = [wheel]
    made = ""
    # An interpreter that cannot start (its base Python moved or went) is made again, not patched.
    if not python.is_file() or _run([python, "-c", "pass"]).returncode != 0:
        # Windows alone may fall back: it installs no Flash Attention, whose ready-made wheel is for 3.11 only.
        ok, made = make_venv(uv, root / ".venv", python, PYTHON_VERSION, fallback=WINDOWS)
        if not ok:
            return False, made
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
        [uv, "pip", "install", "-q", "--python", python, *EXTRAS],
        *[[uv, "pip", "install", "--python", python, "--no-build-isolation", f] for f in flash],
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
    _write_build_record(root, "torch")
    return True, f"the engine's environment installed; {probe_note}" + (f"; {made}" if made else "")


CONTRACT_SRC_ENV = "KHAOS_CONTRACT_SRC"
CONTRACT_WHEEL_ENV = "KHAOS_CONTRACT_WHEEL"
CONTRACT_REPO = "https://github.com/khaos-foundation/Khaos_Attribution.git"


def contract_spec(root: Path) -> str:
    """Where the contract comes from for the engine's environment: a checkout named in KHAOS_CONTRACT_SRC, a wheel
    named in KHAOS_CONTRACT_WHEEL (the app ships one), the room's sibling checkout while it holds the code this room
    imports, else the tagged release of this room's version."""
    from khaos_attribution import __version__  # noqa: PLC0415
    from khaos_attribution.engines.pins import contract_code_hash  # noqa: PLC0415
    src = os.environ.get(CONTRACT_SRC_ENV, "").strip()
    if src:
        return src
    wheel = os.environ.get(CONTRACT_WHEEL_ENV, "").strip()
    if wheel:
        return wheel
    sibling = root.parent / "Khaos_Attribution"
    package = sibling / "src" / "khaos_attribution"
    if (sibling / "pyproject.toml").is_file() and package.is_dir() and \
            contract_code_hash(package).split()[1] == contract_code_hash().split()[1]:
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
    return mismatch_words(contract_fingerprint(python), contract_fingerprint())


def mismatch_words(theirs: str | None, ours: str) -> str:
    """Why the fingerprint ``theirs`` is not this room's ``ours``, in words."""
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
    if not contract_present(python):
        return False, f"khaos-attribution was installed into the engine's environment, but it {contract_mismatch(python)}"
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


def cached_path(repo: str, name: str) -> str | None:
    try:
        from huggingface_hub import try_to_load_from_cache  # noqa: PLC0415
    except ImportError:
        return None
    found = try_to_load_from_cache(repo, name)
    return found if isinstance(found, str) else None


LICENCE_PATTERNS = ("LICENSE*", "NOTICE*")


def licence_note(w: dict) -> str | None:
    """Said when a cached set whose licence must travel with it holds no licence file in its snapshot: weights fetched
    before the set named its licence files are not fetched again, so the gap is reported, never assumed away."""
    if not DESCRIPTOR.licence.get("obligations") or not set(LICENCE_PATTERNS) <= set(w.get("patterns") or ()):
        return None
    first = (w.get("components") or [None])[0]
    where = cached_path(w["repo"], first) if first else None
    if where is None:
        return None
    snapshot = Path(where).parents[len(Path(first).parts) - 1]
    if any(any(snapshot.glob(p)) for p in LICENCE_PATTERNS):
        return None
    return (f"no licence file beside {w['repo']} in the Hugging Face cache: these weights were fetched before their "
            "licence travelled with them")


def licence_notes(runtime: str) -> list:
    """The licence notes for this runtime's cached sets, kept apart from weights(), whose every line is a check."""
    return [note for w in DESCRIPTOR.weights if w.get("runtime", runtime) == runtime and (note := licence_note(w))]


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
            done = _run([python, "-c", code, *(w.get("components") or [])], cwd=mlx, env={**host_env(), "HF_HUB_OFFLINE": "0"})
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
        done = _run([python, "-c", code, w["repo"], json.dumps(w.get("patterns") or [])], env={**host_env(), "HF_HUB_OFFLINE": "0"})
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
        for note in licence_notes(runtime):
            report.note(note)
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
