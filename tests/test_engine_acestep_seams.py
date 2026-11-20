"""The canary for the seams Khaos patches in ACE-Step: read from the checkout's source, never imported, so an install
--check can run it anywhere; it passes at the pin and names each seam a revision moved or a source it cannot read."""
from pathlib import Path

import pytest

from khaos_attribution.engines.acestep import seams as S

SIBLING = Path(__file__).resolve().parents[1].parent / "ACE-Step-1.5"

FILES = {
    "acestep/core/generation/handler/service_generate_request.py":
        "class M:\n    def _normalize_service_generate_inputs(self, captions, infer_steps, seed):\n"
        "        return {'infer_steps': infer_steps}\n",
    "acestep/core/generation/handler/memory_utils.py": "class M:\n    def _get_vae_dtype(self, device=None):\n        return 1\n",
    "acestep/core/generation/handler/vae_decode.py": "class M:\n    def decode(self):\n        return self._get_vae_dtype('cpu')\n",
    "acestep/core/generation/handler/init_service_catalog.py": "class M:\n    def is_turbo_model(self):\n        return True\n",
    "acestep/training_v2/fixed_lora_module.py":
        "import torch.nn.functional as F\n\ndef _select_compute_dtype(device_type, requested_precision='auto'):\n    return 1\n\n"
        "class FixedLoRAModule:\n    def training_step(self, batch):\n        return F.mse_loss(batch, batch)\n",
    "acestep/training_v2/preprocess.py":
        "def preprocess_audio_files():\n    from acestep.training_v2.model_loader import (\n        load_vae,\n    )\n"
        "    return load_vae\n",
    "acestep/training/data_module.py":
        "def collate_preprocessed_batch(batch):\n    return batch\n\nclass PreprocessedDataModule:\n"
        "    def train_dataloader(self):\n        return None\n",
}


def _checkout(tmp_path, **changed) -> Path:
    for rel, text in {**FILES, **changed}.items():
        if text is None:
            continue
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return tmp_path


def test_a_checkout_holding_every_seam_passes(tmp_path):
    assert S.problems(_checkout(tmp_path)) == []


@pytest.mark.parametrize("rel, text, words", [
    ("acestep/core/generation/handler/service_generate_request.py",
     "class M:\n    def _normalize_service_generate_inputs(self, captions, steps):\n        return {'steps': steps}\n", "infer_steps"),
    ("acestep/core/generation/handler/memory_utils.py", "class M:\n    pass\n", "_get_vae_dtype"),
    ("acestep/core/generation/handler/vae_decode.py", "class M:\n    pass\n", "_get_vae_dtype"),
    ("acestep/training_v2/fixed_lora_module.py",
     "import torch.nn.functional as F\n\ndef _select_compute_dtype(device_type):\n    return 1\n\n"
     "class FixedLoRAModule:\n    def training_step(self, batch):\n        return F.mse_loss(batch, batch)\n", "requested_precision"),
    ("acestep/training_v2/fixed_lora_module.py",
     "def _select_compute_dtype(device_type, requested_precision='auto'):\n    return 1\n\n"
     "class FixedLoRAModule:\n    def training_step(self, batch):\n        return batch\n", "mse_loss"),
    ("acestep/training_v2/preprocess.py", "from acestep.training_v2.model_loader import load_vae\n", "load_vae"),
    ("acestep/training/data_module.py", "class PreprocessedDataModule:\n    pass\n", "collate_preprocessed_batch"),
    ("acestep/core/generation/handler/init_service_catalog.py", "class M:\n    pass\n", "is_turbo_model"),
])
def test_a_moved_seam_is_named(tmp_path, rel, text, words):
    found = S.problems(_checkout(tmp_path, **{rel: text}))
    assert found and any(words in p for p in found), found


def test_a_source_that_cannot_be_read_fails_closed(tmp_path):
    """A packaged build without readable ACE-Step source cannot be checked, and says so rather than passing."""
    checkout = _checkout(tmp_path, **{"acestep/training_v2/preprocess.py": None})
    found = S.problems(checkout)
    assert found and any("cannot be read" in p and "preprocess.py" in p for p in found)
    (checkout / "acestep/training_v2/fixed_lora_module.py").write_bytes(b"def broken(:\n")
    assert any("cannot be read" in p for p in S.problems(checkout))


def test_the_command_line_reports_and_exits_by_the_verdict(tmp_path, capsys):
    assert S.main([str(_checkout(tmp_path))]) == 0 and "CHECK OK" in capsys.readouterr().out
    assert S.main([str(tmp_path / "nowhere")]) == 1 and "CHECK FAILED" in capsys.readouterr().out


@pytest.mark.skipif(not (SIBLING / "acestep").is_dir(), reason="no ACE-Step checkout beside this one")
def test_the_pinned_checkout_holds_every_seam_khaos_patches():
    assert S.problems(SIBLING) == []
