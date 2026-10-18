"""Stability's MLX trainer conditions on an all-ones inpaint mask its renderer never uses: the installer patches it to
the generation mask, a Mac run records which mask it trained under, and neither training nor measuring goes ahead on
the old one."""
import json
import sys

import pytest

from khaos_attribution.engines.sa3 import provision as P
from khaos_attribution.engines.sa3 import train as T

TRAINER = """
    lac_const = mx.array(
        np.concatenate([np.ones((1, 1, crop_len), np.float32),
                        np.zeros((1, 256, crop_len), np.float32)],
                       axis=1).transpose(0, 2, 1)).astype(mx.float16)
"""


@pytest.fixture
def root(tmp_path):
    path = tmp_path / "sa3" / P.MLX_TRAINER
    path.parent.mkdir(parents=True)
    path.write_text(TRAINER)
    return tmp_path / "sa3"


def test_the_installer_patches_the_trainer_to_the_generation_mask_once(root):
    assert P.trainer_mask(root) == "ones"
    ok, note = P.mlx_trainer_patch(root, check_only=True)
    assert not ok and "all-ones" in note
    ok, _ = P.mlx_trainer_patch(root)
    assert ok and P.trainer_mask(root) == "zeros" and "np.ones((1, 1, crop_len)" not in (root / P.MLX_TRAINER).read_text()
    assert P.mlx_trainer_patch(root, check_only=True)[0], "patched once, the check passes and nothing is written again"
    (root / P.MLX_TRAINER).write_text("a trainer this pin rewrote")
    ok, note = P.mlx_trainer_patch(root)
    assert not ok and "needs review" in note, "a trainer whose line moved is never guessed at"


def test_a_mac_run_records_its_mask_and_an_old_one_reads_as_ones(tmp_path):
    run = tmp_path / "run"
    (run / T.TRAINER_DIRNAME).mkdir(parents=True)
    rec = run / T.TRAINER_DIRNAME / "train_command.json"
    rec.write_text(json.dumps({"command": ["py", "scripts/lora_train_mlx.py"]}))
    assert T.trained_mask(run) == "ones", "a Mac run from before the record trained under the trainer's own mask"
    rec.write_text(json.dumps({"command": ["py", "scripts/lora_train_mlx.py"], "mlx_inpaint_mask": "zeros"}))
    assert T.trained_mask(run) == "zeros"
    rec.write_text(json.dumps({"command": ["py", "train_driver.py"]}))
    assert T.trained_mask(run) is None and T.trained_mask(tmp_path / "absent") is None


def test_training_and_measuring_refuse_the_old_mask(root, tmp_path):
    api = T.TrainingApi(root=root, runtime="mlx", python=sys.executable)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / T.BASE_NPZ).write_bytes(b"npz")
    with pytest.raises(T.TrainingUnavailable, match="patch it before training"):
        T.open_session(api, {}, checkpoint_root=ckpt, tensor_dir=tmp_path, output_dir=tmp_path, device="mps",
                       precision="fp16", seed=1)
    run = tmp_path / "run"
    (run / T.TRAINER_DIRNAME).mkdir(parents=True)
    (run / T.TRAINER_DIRNAME / "train_command.json").write_text(json.dumps({"command": ["scripts/lora_train_mlx.py"]}))
    adapter = run / "adapter_model.safetensors"
    adapter.write_bytes(b"w")
    with pytest.raises(T.TrainingUnavailable, match="Retrain it"):
        T.open_module(api, {}, adapter, checkpoint_root=ckpt, output_dir=run)
    P.mlx_trainer_patch(root)
    (run / T.TRAINER_DIRNAME / "train_command.json").write_text(
        json.dumps({"command": ["scripts/lora_train_mlx.py"], "mlx_inpaint_mask": "zeros"}))
    assert T.open_module(api, {}, adapter, checkpoint_root=ckpt, output_dir=run).runtime == "mlx"
