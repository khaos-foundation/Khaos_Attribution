"""The torch pin both rooms' installers read."""
import pytest

from khaos_attribution import torch_pins as T


def test_a_mac_runs_one_exact_metal_build_and_a_cuda_machine_one_minor_from_one_index():
    assert T.matches("2.11.0", "darwin") and not T.matches("2.12.1", "darwin") and not T.matches("2.11.0+cpu", "darwin")
    assert T.matches("2.13.0+cu130", "linux") and T.matches("2.13.2+cu130", "win32")
    assert not T.matches("2.10.0+cu128", "linux") and not T.matches("2.13.0+cu128", "linux") and not T.matches("2.13.0", "linux")


def test_the_install_arguments_move_the_whole_torch_family_together():
    mac = T.install_args("darwin")
    assert mac == ["torch==2.11.0", "torchaudio==2.11.0", "torchvision==0.26.0"]
    gpu = T.install_args("linux")
    assert gpu[:3] == ["torch==2.13.*", "torchvision", "torchaudio"] and T.CU130_INDEX in gpu
    assert gpu.count("--reinstall-package") == 3


def test_an_unknown_platform_has_no_pin():
    with pytest.raises(KeyError):
        T.pin_for("plan9")
