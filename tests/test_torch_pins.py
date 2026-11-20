"""The torch pin both rooms' installers read."""
import pytest

from khaos_attribution import torch_pins as T


def test_a_mac_runs_one_exact_metal_build_and_a_cuda_machine_one_minor_from_one_index():
    assert T.matches("2.11.0", "darwin") and not T.matches("2.12.1", "darwin") and not T.matches("2.11.0+cpu", "darwin")
    assert T.matches("2.13.0+cu130", "linux") and T.matches("2.13.2+cu130", "win32")
    assert not T.matches("2.10.0+cu128", "linux") and not T.matches("2.13.0+cu128", "linux") and not T.matches("2.13.0", "linux")


def test_the_install_arguments_move_the_whole_torch_family_together():
    mac = T.install_args("darwin")
    assert mac == ["torch==2.11.0", "torchaudio==2.11.0", "torchvision==0.26.0", "torchcodec==0.12.0"]
    gpu = T.install_args("linux")
    assert gpu[:4] == ["torch==2.13.*", "torchvision", "torchaudio", "torchcodec"] and T.CU130_INDEX in gpu


INSTALLED = ["numpy", "setuptools", "pillow", "fsspec", "torch", "nvidia-nccl-cu13", "nvidia-cublas", "nvidia-nccl-cu12",
             "Nvidia-Cudnn-Cu13"]


@pytest.mark.parametrize("platform", ["linux", "win32", "darwin"])
def test_only_the_torch_family_and_its_cuda_libraries_are_reinstalled(platform):
    """A blanket --reinstall would put every locked package back from the torch index alone; on CUDA the family and
    the CUDA libraries that share folders with another release's are reinstalled, and on a Mac nothing is."""
    args = T.install_args(platform, installed=INSTALLED)
    assert "--reinstall" not in args
    again = [args[i + 1] for i, a in enumerate(args) if a == "--reinstall-package"]
    if platform == "darwin":
        assert again == []
    else:
        assert again == ["torch", "torchvision", "torchaudio", "torchcodec", "nvidia-cublas", "nvidia-cudnn-cu13",
                         "nvidia-nccl-cu13"]
        assert not {"numpy", "setuptools", "pillow", "fsspec", "nvidia-nccl-cu12"} & set(again)


def test_the_command_line_reinstalls_the_cuda_libraries_installed_where_it_runs(monkeypatch, capsys):
    import importlib.metadata
    import types
    dists = [types.SimpleNamespace(metadata={"Name": n}) for n in INSTALLED]
    monkeypatch.setattr(T, "pin_for", lambda platform=None: T.TORCH["linux"])
    monkeypatch.setattr(importlib.metadata, "distributions", lambda: dists)
    assert T.main(["--install-args"]) == 0
    out = capsys.readouterr().out.split()
    assert "--reinstall" not in out and "nvidia-cublas" in out and "numpy" not in out


def test_another_cuda_releases_libraries_are_named_for_removal_on_cuda_only():
    names = ["torch", "nvidia-nccl-cu12", "nvidia-cudnn-cu12", "nvidia-nccl-cu13", "nvidia-cublas"]
    assert T.stale_packages(names, "linux") == ["nvidia-cudnn-cu12", "nvidia-nccl-cu12"]
    assert T.stale_packages(names, "darwin") == []


def test_an_unknown_platform_has_no_pin():
    with pytest.raises(KeyError):
        T.pin_for("plan9")


def test_the_family_beside_torch_is_checked_too():
    mac = {"torch": "2.11.0", "torchaudio": "2.11.0", "torchvision": "0.26.0", "torchcodec": "0.16.0"}
    assert T.family_problems(mac, "darwin") == ["torchcodec 0.16.0 is not 0.12.0"]
    assert T.family_problems({**mac, "torchcodec": "0.12.0"}, "darwin") == []
    assert "nvidia-nccl-cu12" in T.family_problems({"nvidia-nccl-cu12": "2.27"}, "linux")[0]
    assert T.family_problems({"nvidia-nccl-cu13": "2.29"}, "linux") == []
