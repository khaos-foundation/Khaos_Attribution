"""The torch each platform runs, one pin read by both rooms' installers: an index the Workshop builds is scored by the
Space on the same build. ``python -m khaos_attribution.torch_pins --install-args`` prints what to hand ``uv pip
install``; ``--check`` exits non-zero when the installed torch is not the pin."""

from __future__ import annotations

import sys

CU130_INDEX = "https://download.pytorch.org/whl/cu130"

# A Mac runs the PyPI (Metal) build at an exact version; CUDA machines a minor series from one CUDA index.
TORCH = {
    "darwin": {"torch": "2.11.0", "torchaudio": "2.11.0", "torchvision": "0.26.0", "index": None, "cuda_tag": None},
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


def install_args(platform: str | None = None) -> list[str]:
    """The ``uv pip install`` arguments that put torch, torchvision and torchaudio in place together: a torchvision
    built for another torch imports and then fails at its first compiled op."""
    pin = pin_for(platform)
    if pin["index"] is None:
        return [f"torch=={pin['torch']}", f"torchaudio=={pin['torchaudio']}", f"torchvision=={pin['torchvision']}"]
    family = ["torch", "torchvision", "torchaudio"]
    return [f"torch=={pin['torch']}.*", "torchvision", "torchaudio", "--index-url", pin["index"],
            *[a for p in family for a in ("--reinstall-package", p)]]


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args == ["--install-args"]:
        print(" ".join(install_args()))
        return 0
    if args == ["--check"]:
        try:
            import torch  # noqa: PLC0415
        except ImportError:
            print("torch is not installed")
            return 1
        ok = matches(torch.__version__)
        pin = pin_for()
        print(f"torch {torch.__version__} {'is' if ok else 'is NOT'} the pin ({pin['torch']}"
              f"{' ' + pin['cuda_tag'] if pin['cuda_tag'] else ''})")
        return 0 if ok else 1
    print("usage: python -m khaos_attribution.torch_pins --install-args | --check", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
