"""The base the process engines share: one refusal table for research options, one session, one held-out evaluator,
and mimetic's rate, rank and alpha read from one place."""
import subprocess
import sys

import pytest

from khaos_attribution.engines import process_engine as PE
from khaos_attribution.engines.mimetic import DESCRIPTOR as MIMETIC
from khaos_attribution.engines.mimetic import train as MT
from khaos_attribution.engines.sa3 import train as ST
from khaos_attribution.gradient import GradientUnavailable

OPTIONS = {"timestep_norm": True, "param_filter": "attn", "latent_transform": {"kind": "pca"}, "latent_pool": 2,
           "noise_shared": True, "param_groups": "blocks"}


@pytest.mark.parametrize("option, value", sorted(OPTIONS.items()))
def test_a_research_option_no_worker_has_built_is_refused_and_one_it_has_passes(option, value):
    with pytest.raises(GradientUnavailable, match="not built on the toy worker"):
        PE.refuse_unbuilt("toy", built=frozenset(), **{option: value})
    PE.refuse_unbuilt("toy", built=frozenset({option}), **{option: value})


def test_both_process_engines_are_one_session_and_one_evaluator():
    for T in (ST, MT):
        assert issubclass(T.Session, PE.ProcessSession) and T.HeldoutEvaluator is PE.HeldoutEvaluator
        assert T.TrainingApi is PE.TrainingApi


def test_mimetics_rate_comes_from_the_recipe_else_its_descriptor_and_the_driver_has_no_defaults_of_its_own(tmp_path):
    """The rate, rank and alpha live in the descriptor's train block and the recipe; the driver is always told them."""
    api = PE.TrainingApi(root=tmp_path, runtime="torch", python=tmp_path / "python")
    steps = {"total": 4, "checkpoint_every": 2}
    cfg = {"adapter_rank": 8, "adapter_alpha": 8}
    cmd = MT.trainer_command(api, cfg, tensor_dir=tmp_path, save_dir=tmp_path, steps=steps, seed=1, device="cpu")
    assert cmd[cmd.index("--lr") + 1] == str(float(MIMETIC.train["learning_rate"]))
    cmd = MT.trainer_command(api, {**cfg, "learning_rate": 3e-4}, tensor_dir=tmp_path, save_dir=tmp_path, steps=steps,
                             seed=1, device="cpu")
    assert cmd[cmd.index("--lr") + 1] == "0.0003"
    done = subprocess.run([sys.executable, str(MT.TRAIN_DRIVER), "--clips", "c", "--save-dir", "s", "--steps", "1",
                           "--checkpoint-every", "1"], capture_output=True, text=True)
    assert done.returncode != 0 and "--rank" in done.stderr and "--lr" in done.stderr
