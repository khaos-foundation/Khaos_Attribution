"""The torch each platform runs, one pin read by both rooms' installers: an index the Workshop builds is scored by the
Space on the same build. ``--stale`` lists another CUDA release's libraries to uninstall first,
``--install-args`` what to hand ``uv pip install``; ``--check`` exits non-zero when the installed torch is not the pin."""

from __future__ import annotations

import sys

CU130_INDEX = "https://download.pytorch.org/whl/cu130"

# A Mac runs the PyPI (Metal) build at an exact version; CUDA machines a minor series from one CUDA index.
TORCH = {
    "darwin": {"torch": "2.11.0", "torchaudio": "2.11.0", "torchvision": "0.26.0", "torchcodec": "0.12.0", "index": None,
               "cuda_tag": None},
    "linux": {"torch": "2.13", "index": CU130_INDEX, "cuda_tag": "cu130"},
    "win32": {"torch": "2.13", "index": CU130_INDEX, "cuda_tag": "cu130"},
}


def pin_for(platform: str | None = None) -> dict:
    """The pin for a platform (``sys.platform`` by default)."""
    platform = platform or sys.platform
    if platform not in TORCH:
        raise KeyError(f"no torch pin for {platform!r}")
    return TORCH[platform]


def matches(version: str, platform: str | None = None) -> bool:
    """Whether an installed torch version is the pin: exact on a Mac, the pinned minor and CUDA tag elsewhere."""
    pin = pin_for(platform)
    local = version.split("+", 1)
    if pin["index"] is None:
        return local[0] == pin["torch"] and len(local) == 1
    return local[0].startswith(pin["torch"] + ".") and len(local) == 2 and local[1] == pin["cuda_tag"]


# The packages that move together: a torchvision or torchcodec built for another torch imports, then fails at its
# first compiled op.
FAMILY = ("torch", "torchvision", "torchaudio", "torchcodec")


def install_args(platform: str | None = None, installed=()) -> list[str]:
    """The ``uv pip install`` arguments that put the torch family in place together. On CUDA the family is reinstalled
    from the pin's index, and with it every CUDA library ``installed``: another CUDA release's copy shares their
    folders, and removing it takes shared files. Nothing else is reinstalled, so the rest of the lock stays as it is."""
    pin = pin_for(platform)
    if pin["index"] is None:
        return [f"torch=={pin['torch']}", f"torchaudio=={pin['torchaudio']}", f"torchvision=={pin['torchvision']}",
                f"torchcodec=={pin['torchcodec']}"]
    installed = list(installed)
    stale = {s.lower() for s in stale_packages(installed, platform)}
    cuda = sorted({n.lower() for n in installed if n.lower().startswith("nvidia-")} - stale)
    again = [a for n in (*FAMILY, *cuda) for a in ("--reinstall-package", n)]
    return [f"torch=={pin['torch']}.*", "torchvision", "torchaudio", "torchcodec", "--index-url", pin["index"], *again]


def stale_packages(installed, platform: str | None = None) -> list[str]:
    """The CUDA libraries of another CUDA release left by an earlier torch (ACE-Step brings its own), to remove
    before the pin is installed: torch would load an old library first."""
    pin = pin_for(platform)
    if pin["cuda_tag"] != "cu130":
        return []
    return sorted(n for n in installed if n.lower().startswith("nvidia-") and n.lower().endswith("-cu12"))


def family_problems(installed: dict, platform: str | None = None) -> list[str]:
    """What is wrong with the rest of the torch family beside torch itself: a version off a Mac's exact pin, or
    another CUDA release's libraries still installed."""
    pin = pin_for(platform)
    if pin["index"] is None:
        return [f"{n} {installed.get(n)} is not {pin[n]}" for n in ("torchaudio", "torchvision", "torchcodec")
                if installed.get(n) != pin[n]]
    stale = stale_packages(installed, platform)
    return [f"another CUDA release's libraries are installed: {' '.join(stale)}"] if stale else []


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args == ["--install-args"]:
        import importlib.metadata  # noqa: PLC0415
        print(" ".join(install_args(installed=[d.metadata["Name"] for d in importlib.metadata.distributions()])))
        return 0
    if args == ["--stale"]:
        import importlib.metadata  # noqa: PLC0415
        print(" ".join(stale_packages(d.metadata["Name"] for d in importlib.metadata.distributions())))
        return 0
    if args == ["--check"]:
        try:
            import torch  # noqa: PLC0415
        except ImportError:
            print("torch is not installed")
            return 1
        import importlib.metadata  # noqa: PLC0415
        installed = {d.metadata["Name"].lower(): d.version for d in importlib.metadata.distributions()}
        problems = family_problems(installed)
        ok = matches(torch.__version__) and not problems
        pin = pin_for()
        print(f"torch {torch.__version__} {'is' if ok else 'is NOT'} the pin ({pin['torch']}"
              f"{' ' + pin['cuda_tag'] if pin['cuda_tag'] else ''}){'; ' + '; '.join(problems) if problems else ''}")
        return 0 if ok else 1
    print("usage: python -m khaos_attribution.torch_pins --install-args | --stale | --check", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
