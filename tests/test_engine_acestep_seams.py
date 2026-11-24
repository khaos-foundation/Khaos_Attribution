"""The canary for the seams Khaos patches in ACE-Step: read from the checkout's source, never imported, so an install
--check can run it anywhere; it passes at the pin and names each seam a revision moved or a source it cannot read."""
from pathlib import Path

import pytest

from khaos_attribution.engines.acestep import seams as S

REPO = Path(__file__).resolve().parents[1]


def _siblings_root(repo: Path) -> Path:
    """The folder holding the main checkout and its siblings: a worktree's .git file names the main checkout."""
    marker = repo / ".git"
    if marker.is_file():
        text = marker.read_text(encoding="utf-8", errors="replace").strip()
        if text.startswith("gitdir:"):
            gitdir = Path(text[len("gitdir:"):].strip())
            gitdir = gitdir if gitdir.is_absolute() else (repo / gitdir).resolve()
            if gitdir.parent.name == "worktrees" and gitdir.parent.parent.name == ".git":
                return gitdir.parents[2].parent
    return repo.parent


SIBLING = _siblings_root(REPO) / "ACE-Step-1.5"

FILES = {
    "acestep/core/generation/handler/service_generate_request.py":
        "class M:\n    def _normalize_service_generate_inputs(self, captions, infer_steps, seed):\n"
        "        return {'infer_steps': infer_steps}\n",
    "acestep/core/generation/handler/memory_utils.py": "class M:\n    def _get_vae_dtype(self, device=None):\n        return 1\n",
    "acestep/core/generation/handler/vae_decode.py": "class M:\n    def decode(self):\n        return self._get_vae_dtype('cpu')\n",
    "acestep/core/generation/handler/init_service_catalog.py": "class M:\n    def is_turbo_model(self):\n        return True\n",
    "acestep/core/generation/handler/init_service_orchestrator.py":
        "class M:\n    def initialize_service(self, device, offload_to_cpu=False, offload_dit_to_cpu=False):\n"
        "        self.dtype = 1\n        return self.dtype\n",
    "acestep/core/generation/handler/init_service_loader_components.py":
        "class M:\n    def _load_text_encoder_and_tokenizer(self, *, checkpoint_dir, device):\n"
        "        self.text_encoder = 1\n",
    "acestep/core/generation/handler/init_service_offload_context.py":
        "class M:\n    def _load_model_context(self, model_name):\n        yield\n",
    "acestep/core/generation/handler/init_service_memory_transfer.py":
        "class M:\n    def _recursive_to_device(self, model, device, dtype=None):\n        return model\n",
    "acestep/core/generation/handler/generate_music_decode.py":
        "import os\n\nclass M:\n    def decode(self):\n        return os.environ.get('ACESTEP_VAE_ON_CPU', '0')\n",
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
    ("acestep/core/generation/handler/init_service_orchestrator.py",
     "class M:\n    def initialize_service(self, device, offload_to_cpu=False, offload_dit_to_cpu=False):\n"
     "        self.precision = 1\n", "self.dtype"),
    ("acestep/core/generation/handler/init_service_orchestrator.py",
     "class M:\n    def initialize_service(self, device, offload_to_cpu=False):\n        self.dtype = 1\n"
     "        return self.dtype\n", "offload_dit_to_cpu"),
    ("acestep/core/generation/handler/init_service_orchestrator.py",
     "class M:\n    def initialize_service(self, device):\n        self.dtype = 1\n        return self.dtype\n",
     "offload_to_cpu"),
    ("acestep/core/generation/handler/init_service_loader_components.py",
     "class M:\n    def _load_text_encoder(self, path):\n        self.text_encoder = 1\n", "_load_text_encoder_and_tokenizer"),
    ("acestep/core/generation/handler/init_service_offload_context.py",
     "class M:\n    def _load_model_context(self, name):\n        yield\n", "_load_model_context"),
    ("acestep/core/generation/handler/init_service_memory_transfer.py", "class M:\n    pass\n", "_recursive_to_device"),
    ("acestep/core/generation/handler/generate_music_decode.py", "class M:\n    pass\n", "ACESTEP_VAE_ON_CPU"),
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


def test_a_worktree_finds_the_siblings_beside_its_main_checkout(tmp_path):
    """From a worktree the pinned checkout is looked for beside the main checkout, so the test above runs there too."""
    main = tmp_path / "Khaos" / "Khaos_Attribution"
    (main / ".git" / "worktrees" / "fix").mkdir(parents=True)
    tree = main / ".claude" / "worktrees" / "fix" / "Khaos_Attribution"
    tree.mkdir(parents=True)
    (tree / ".git").write_text(f"gitdir: {main / '.git' / 'worktrees' / 'fix'}\n")
    assert _siblings_root(tree) == tmp_path / "Khaos" and _siblings_root(main) == tmp_path / "Khaos"
