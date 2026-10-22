"""mimetic-a0's provision slot: Mick's code at the descriptor's pin, its own environment (his lock, torch from the
CUDA 12.8 index on an NVIDIA machine, stable-audio-tools' training extras the adapters need), the contract, and the
weights the descriptor lists, from Khaos's own mirror first. --check reports without changing anything."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from khaos_attribution import weights as mirror
from khaos_attribution.engines.mimetic import DESCRIPTOR
from khaos_attribution.engines.mimetic.render import find_root, venv_python
from khaos_attribution.engines.sa3.provision import (Report, _from_mirror, _run, archive_url, contract_into, contract_mismatch,
                                                     contract_present, git_usable, unpack_archive, uv_path)

PYTHON_VERSION = "3.11"
# Mick's lock takes torch from the CUDA 12.6 index on Linux, which cannot drive a Blackwell card; 12.8 can.
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
TORCH_FAMILY = ("torch==2.7.1", "torchaudio==2.7.1", "torchvision==0.22.1")
# stable-audio-tools' own train extras at Mick's pin, less what its trainer never imports here.
TRAIN_EXTRAS = ("prefigure==0.0.9", "wandb==0.15.4", "webdataset==0.2.100", "auraloss==0.4.0", "descript-audio-codec==1.0.0",
                "matplotlib==3.11.2")
STRIPPED = ("torch==", "torchaudio==", "torchvision==", "triton==", "nvidia-", "#", "-e")
KEPT_ACROSS_PINS = (".venv",)
SAT_GIT = "stable-audio-tools @ git+"


def runtime_for(device: str = "auto") -> str:
    """"mps" on a Mac, else "cuda"; the engine offers no other."""
    if device not in ("auto", None, ""):
        return "mps" if device == "mps" else "cuda"
    return "mps" if sys.platform == "darwin" else "cuda"


def locked_requirements(export: str) -> str:
    """Mick's lock as requirements: the torch family left to the caller, stable-audio-tools from its archive so no
    git is needed."""
    out = []
    for line in export.splitlines():
        line = line.strip()
        if not line or line.startswith(STRIPPED):
            continue
        if line.startswith(SAT_GIT):
            url, _, commit = line[len(SAT_GIT):].partition("@")
            line = f"stable-audio-tools @ {archive_url(url, commit.split(';')[0].strip())}"
        out.append(line)
    return "\n".join(out) + "\n"


def torch_args(runtime: str) -> list:
    if runtime == "cuda":
        return [f"{p}+cu128" for p in TORCH_FAMILY] + ["--extra-index-url", TORCH_INDEX, "--index-strategy", "unsafe-best-match"]
    return list(TORCH_FAMILY)


def probe_code(runtime: str) -> str:
    check = "torch.cuda.is_available()" if runtime == "cuda" else "torch.backends.mps.is_available()"
    return f"import torch, stable_audio_tools, prefigure, pytorch_lightning; assert {check}; print('ok')"


def checkout(root: Path, *, check_only: bool = False) -> tuple:
    """Mick's code at the descriptor's commit; (ok, note)."""
    from khaos_attribution.engines.pins import checkout_commit  # noqa: PLC0415
    repo, commit = DESCRIPTOR.doc["code"]["repo"], DESCRIPTOR.commit
    is_git = (root / ".git").is_dir()
    head = (_run(["git", "-C", root, "rev-parse", "HEAD"]).stdout.strip() if is_git and git_usable()
            else (checkout_commit(root) or ""))
    if head == commit:
        return True, f"mimetic-a0 code at the pin {commit[:12]}"
    if check_only:
        return False, (f"mimetic-a0 code at {head[:12]}, pin is {commit[:12]}" if head or is_git
                       else f"mimetic-a0 code not present at {root}")
    if is_git and git_usable():
        _run(["git", "-C", root, "fetch", "-q", "origin", commit])
        done = _run(["git", "-C", root, "checkout", "-q", "--detach", commit])
        return (done.returncode == 0), (f"checked out {commit[:12]}" if done.returncode == 0
                                        else f"checkout failed: {done.stderr.strip()[-300:]}")
    if not root.exists() and git_usable():
        root.parent.mkdir(parents=True, exist_ok=True)
        done = _run(["git", "clone", "-q", repo, root])
        if done.returncode != 0:
            return False, f"clone failed: {done.stderr.strip()[-300:]}"
        done = _run(["git", "-C", root, "checkout", "-q", "--detach", commit])
        return (done.returncode == 0), (f"mimetic-a0 code cloned at the pin {commit[:12]}" if done.returncode == 0
                                        else f"checkout failed: {done.stderr.strip()[-300:]}")
    try:
        unpack_archive(archive_url(repo, commit), root, commit, kept=KEPT_ACROSS_PINS)
    except (OSError, ValueError) as exc:
        return False, f"the code archive could not be fetched: {exc}"
    return True, f"mimetic-a0 code unpacked at the pin {commit[:12]}"


def environment(root: Path, runtime: str, *, check_only: bool = False) -> tuple:
    """The engine's own interpreter: (ok, note)."""
    python = venv_python(root)
    code, note = probe_code(runtime), "torch sees the GPU; stable-audio-tools and its trainer import"
    if python.is_file():
        probe = _run([python, "-c", code])
        if probe.returncode == 0:
            if contract_present(python):
                return True, note
            if check_only:
                return False, f"the environment at {python} runs the engine but {contract_mismatch(python)}"
            ok, cnote = contract_into(python, root)
            return ok, (note + "; " + cnote) if ok else cnote
        if check_only:
            return False, f"the environment at {python} cannot run the engine: {probe.stderr.strip()[-200:]}"
    elif check_only:
        return False, f"no environment at {python}"
    uv = uv_path()
    if not python.is_file():
        _run([uv, "venv", "-q", "--python", PYTHON_VERSION, root / ".venv"])
    export = _run([uv, "export", "--frozen", "--no-hashes", "--no-emit-project", "--no-dev"], cwd=root)
    if export.returncode != 0 or not export.stdout.strip():
        return False, f"Mick's lock could not be exported: {export.stderr.strip()[-300:]}"
    work = root / ".venv" / "khaos"
    work.mkdir(parents=True, exist_ok=True)
    (work / "locked.txt").write_text(locked_requirements(export.stdout), encoding="utf-8")
    steps = [
        [uv, "pip", "install", "-q", "--python", python, "-r", work / "locked.txt", *torch_args(runtime)],
        [uv, "pip", "install", "-q", "--python", python, "--no-deps", "-e", root],
        [uv, "pip", "install", "-q", "--python", python, *TRAIN_EXTRAS],
    ]
    for cmd in steps:
        done = _run(cmd, cwd=root)
        if done.returncode != 0:
            return False, f"environment step failed ({' '.join(str(c) for c in cmd[:4])}…): {done.stderr.strip()[-300:]}"
    ok, cnote = contract_into(python, root)
    if not ok:
        return False, f"the engine's environment installed; {cnote}"
    probe = _run([python, "-c", code])
    if probe.returncode != 0:
        return False, f"the environment installed but cannot run the engine: {probe.stderr.strip()[-200:]}"
    return True, f"the engine's environment installed; {note}"


def cached(repo: str, files: list, revision: str | None) -> bool:
    """Whether the Hugging Face cache holds every named file of the repo at the pinned revision."""
    try:
        from huggingface_hub import try_to_load_from_cache  # noqa: PLC0415
    except ImportError:
        return False
    return all(isinstance(try_to_load_from_cache(repo, name, revision=revision), str) for name in files)


def pin_ref(repo: str, revision: str | None, *, check_only: bool = False) -> tuple:
    """(ok, note): the cache's refs/main names the pinned revision, since stable-audio-tools loads T5 by name alone."""
    if not revision:
        return True, ""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE  # noqa: PLC0415
    except ImportError:
        return False, f"{repo}: huggingface_hub is not in this environment"
    ref = Path(HF_HUB_CACHE) / ("models--" + repo.replace("/", "--")) / "refs" / "main"
    try:
        current = ref.read_text(encoding="utf-8").strip()
    except OSError:
        current = ""
    if current == revision:
        return True, ""
    if check_only:
        return False, f"{repo}: the cache's main revision is {current[:12] or 'unset'}, the pin is {revision[:12]}"
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_text(revision, encoding="utf-8")
    return True, f"{repo}: the cache's main revision set to the pin {revision[:12]}"


def weights(root: Path, *, check_only: bool = False) -> list:
    """Every weight set the descriptor lists, into the Hugging Face cache at its pinned revision: the mirror first,
    the hub only when the mirror cannot serve it."""
    python = venv_python(root)
    out = []
    for w in DESCRIPTOR.weights:
        repo, rev, files = w["repo"], w.get("revision"), list(w.get("components") or ["config.json"])
        if cached(repo, files, rev):
            ok, note = pin_ref(repo, rev, check_only=check_only)
            out.append((ok, f"{repo} in the Hugging Face cache" + (f"; {note}" if note else "")))
            continue
        if check_only:
            out.append((False, f"{repo} missing from the Hugging Face cache"))
            continue
        if _from_mirror(lambda: mirror.fetch_into_hub_cache(repo, mirror.default_hub_cache(), w.get("patterns"), revision=rev)) \
                and cached(repo, files, rev):
            out.append((True, f"{repo} fetched from the Khaos weights mirror into the Hugging Face cache"))
            continue
        code = ("import json, sys; from huggingface_hub import snapshot_download; "
                "snapshot_download(sys.argv[1], revision=sys.argv[2] or None, allow_patterns=json.loads(sys.argv[3]) or None)")
        done = _run([python, "-c", code, repo, rev or "", json.dumps(w.get("patterns") or [])], env={**os.environ, "HF_HUB_OFFLINE": "0"})
        if done.returncode != 0 or not cached(repo, files, rev):
            why = done.stderr.strip().splitlines()[-1:] or ["a file is still missing"]
            out.append((False, f"{repo} could not be downloaded: {why[0][:200]}"))
        else:
            ok, note = pin_ref(repo, rev)
            out.append((ok, f"{repo} downloaded to the Hugging Face cache" + (f"; {note}" if note else "")))
    return out


def installed(root: Path) -> bool:
    """A cheap reading of whether the engine can run here; the full check is provision(check_only=True)."""
    return venv_python(root).is_file()


def provision(repo_dir=None, weights_dir=None, *, device: str = "auto", check_only: bool = False) -> Report:
    root = find_root(repo_dir)
    runtime = runtime_for(device)
    report = Report()
    ok, note = checkout(root, check_only=check_only)
    (report.ok if ok else report.bad)(note)
    if ok:
        ok, note = environment(root, runtime, check_only=check_only)
        (report.ok if ok else report.bad)(note)
    if ok:
        for good, line in weights(root, check_only=check_only):
            (report.ok if good else report.bad)(line)
    return report


def main(argv=None) -> int:
    import argparse  # noqa: PLC0415
    ap = argparse.ArgumentParser(description="Provision mimetic-a0 for Khaos on this machine.")
    ap.add_argument("--repo-dir", type=Path, default=None)
    ap.add_argument("--weights-dir", type=Path, default=None, help="unused: the weights live in the Hugging Face cache")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--check", action="store_true", help="report without changing anything")
    a = ap.parse_args(argv)
    report = provision(a.repo_dir, a.weights_dir, device=a.device, check_only=a.check)
    print(report.text())
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
