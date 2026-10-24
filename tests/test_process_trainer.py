"""The helpers every process-driven trainer shares: steps from epochs, the checkpoint a name carries, a settled
file, the epoch of a step, and clips brought to two channels."""
import numpy as np

from khaos_attribution.engines import process_trainer as PT


def test_epochs_become_steps_with_a_short_last_batch_counted():
    s = PT.steps_for({"batch_size": 4, "max_epochs": 3, "checkpoint_every_n_epochs": 2}, 10)
    assert s == {"per_epoch": 3, "total": 9, "checkpoint_every": 6, "epochs": 3}
    assert PT.steps_for({"max_epochs": 1}, 0)["per_epoch"] == 1, "never zero steps an epoch"


def test_a_checkpoints_step_is_read_from_its_name():
    from pathlib import Path
    assert PT.checkpoint_step(Path("epoch=2-step=1234.ckpt")) == 1234
    assert PT.checkpoint_step(Path("final.safetensors")) is None


def test_a_file_is_settled_once_it_held_still_or_the_trainer_exited(tmp_path):
    f = tmp_path / "ckpt"; f.write_bytes(b"x")
    sizes = {}
    assert PT.settled(sizes, f, final=False) is False, "first sight: nothing to compare with"
    assert PT.settled(sizes, f, final=False) is True
    f.write_bytes(b"xy")
    assert PT.settled(sizes, f, final=False) is False
    assert PT.settled(sizes, f, final=True) is True
    assert PT.settled(sizes, tmp_path / "gone", final=True) is False


def test_the_epoch_of_a_step_counts_from_where_a_resume_started():
    assert PT.epoch_of(0, 10, 10) == 1
    assert PT.epoch_of(0, 10, 25) in (2, 3), "a step mid-epoch rounds to a neighbouring epoch, never beyond"
    assert PT.epoch_of(5, 10, 1) == 6, "a step inside the first epoch of a resume is that epoch, never the one before"


def test_clips_come_to_two_channels_without_a_resampler_when_the_rate_matches():
    class Unavailable(RuntimeError):
        pass
    mono = np.zeros((8, 1), dtype=np.float32)
    assert PT.resample_stereo(mono, 48000, 48000, engine_name="x", unavailable=Unavailable).shape == (8, 2)
    wide = np.zeros((8, 4), dtype=np.float32)
    assert PT.resample_stereo(wide, 48000, 48000, engine_name="x", unavailable=Unavailable).shape == (8, 2)
