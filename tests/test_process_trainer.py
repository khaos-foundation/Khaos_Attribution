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


def test_a_loss_log_is_read_from_where_the_last_read_stopped_and_gives_what_a_whole_read_gives(tmp_path):
    """Each poll reads only what the trainer appended; a half-written last line waits for its end."""
    path = tmp_path / "khaos_losses.csv"
    tail = PT.CsvLossTail(path)
    assert tail() == []
    path.write_text("step,loss\n1,0.5\n2,")
    assert tail() == [(1, 0.5)] and tail.offset == len("step,loss\n1,0.5\n")
    with path.open("a") as fh:
        fh.write("\n3,0.25\n4,0.2")
    assert tail() == [(1, 0.5), (3, 0.25)]
    with path.open("a") as fh:
        fh.write("5\n")
    assert tail() == [(1, 0.5), (3, 0.25), (4, 0.25)] == PT.CsvLossTail(path)()
    read = tail.offset
    tail()
    assert tail.offset == read, "nothing new, nothing read"
    path.write_text("step,loss\n1,0.9\n")
    assert tail() == [(1, 0.9)], "a log written again from the start is read again from the start"
    path.write_text("step,loss\n1,0.7\n" + "".join(f"{s},0.1\n" for s in range(2, 40)))
    assert tail()[:2] == [(1, 0.7), (2, 0.1)] and len(tail()) == 39, "a longer rewrite is told by its first bytes"
    path.unlink()
    path.write_text("step,loss\n1,0.7\n" + "".join(f"{s},0.2\n" for s in range(2, 60)))
    assert tail()[1] == (2, 0.2) and len(tail()) == 59, "another file at the same path is read from its start"


def test_a_binary_loss_log_is_read_by_whole_records(tmp_path):
    import struct
    path = tmp_path / "loss_by_timestep.bin"
    tail = PT.RecordLossTail(path)
    path.write_bytes(struct.pack("Iff", 1, 0.5, 0.75) + struct.pack("Iff", 2, 0.4, 0.5)[:5])
    assert tail() == [(1, 0.75)]
    with path.open("ab") as fh:
        fh.write(struct.pack("Iff", 2, 0.4, 0.5)[5:])
    assert tail() == [(1, 0.75), (2, 0.5)] == PT.RecordLossTail(path)()
