"""CLAP's text models: the one list, the disk check that reads what an offline load reads, and the fetch."""
from __future__ import annotations

import re
import sys
import types
from pathlib import Path

import pytest

from khaos_attribution import embedding as E
from khaos_attribution import weights as W


@pytest.fixture(autouse=True)
def _no_pins(monkeypatch):
    """These tests are about the hub library's own path, taken for a set with no pin."""
    monkeypatch.setattr(W, "PINS", {})


def _home(cache, repo):
    return cache / ("models--" + repo.replace("/", "--"))


def _snapshot(cache, repo, commit, files, ref=True):
    snap = _home(cache, repo) / "snapshots" / commit
    snap.mkdir(parents=True, exist_ok=True)
    for name in files:
        (snap / name.split("|")[0]).write_text("x")
    if ref:
        (_home(cache, repo) / "refs").mkdir(exist_ok=True)
        (_home(cache, repo) / "refs" / "main").write_text(commit)
    return snap


def _fill(cache):
    return [_snapshot(cache, *entry) for entry in E.CLAP_TEXT_SNAPSHOTS]


def _ref(cache, repo):
    return (_home(cache, repo) / "refs" / "main").read_text()


def test_the_list_is_what_the_three_loads_cannot_do_without_each_repo_at_a_commit():
    """Proved against the real library by the Workshop's offline-load test; stated here so a change is a decision."""
    assert {repo: files for repo, _, files in E.CLAP_TEXT_SNAPSHOTS} == {
        "roberta-base": ("config.json", "model.safetensors|pytorch_model.bin", "vocab.json", "merges.txt"),
        "bert-base-uncased": ("vocab.txt",),
        "facebook/bart-base": ("vocab.json", "merges.txt"),
    }
    assert [repo for repo, _, _ in E.CLAP_TEXT_SNAPSHOTS] == ["roberta-base", "bert-base-uncased", "facebook/bart-base"]
    assert all(re.fullmatch(r"[0-9a-f]{40}", pin) for _, pin, _ in E.CLAP_TEXT_SNAPSHOTS)


def test_an_empty_cache_lacks_every_file_and_a_full_one_none(tmp_path):
    everything = [f"{repo}/{name.split('|')[0]}" for repo, _, files in E.CLAP_TEXT_SNAPSHOTS for name in files]
    assert E.clap_text_missing(tmp_path) == everything
    assert E.clap_text_missing(tmp_path / "not" / "made" / "yet") == everything
    snaps = _fill(tmp_path)
    assert E.clap_text_missing(tmp_path) == []
    (snaps[0] / "model.safetensors").unlink()
    (snaps[1] / "vocab.txt").unlink()
    assert E.clap_text_missing(tmp_path) == ["roberta-base/model.safetensors", "bert-base-uncased/vocab.txt"]
    (snaps[0] / "pytorch_model.bin").write_text("x")
    assert E.clap_text_missing(tmp_path) == ["bert-base-uncased/vocab.txt"], "the older spelling of the weights serves the load"
    (snaps[1] / "vocab.txt").symlink_to(tmp_path / "a-blob-that-is-gone")
    assert E.clap_text_missing(tmp_path) == ["bert-base-uncased/vocab.txt"], "a link to nothing is not a file"


def test_the_check_reads_the_snapshot_an_offline_load_reads(tmp_path):
    """transformers resolves through refs/main offline: files at any other commit do not count, the pin included."""
    repo, _, files = E.CLAP_TEXT_SNAPSHOTS[2]
    _fill(tmp_path)
    _snapshot(tmp_path, repo, "0" * 40, files[:1])
    assert E.clap_text_missing(tmp_path) == [f"{repo}/{name}" for name in files[1:]], "the pin is whole, the ref names another"
    (_home(tmp_path, repo) / "refs" / "main").unlink()
    assert E.clap_text_missing(tmp_path) == [f"{repo}/{name}" for name in files], "no ref: nothing an offline load can find"


@pytest.mark.parametrize("ref", [lambda pin: (pin + "\n").encode(), lambda pin: (pin + "\r\n").encode(), lambda pin: (" " + pin).encode(),
                                 lambda pin: ("g" * 40).encode(), lambda pin: b"\xff\xfe" + pin.encode()[:38], lambda pin: b""])
def test_a_ref_the_hub_cannot_read_names_nothing(tmp_path, ref):
    """The hub reads the ref raw, so one written by hand with a newline resolves to no snapshot."""
    repo, pin, files = E.CLAP_TEXT_SNAPSHOTS[1]
    _fill(tmp_path)
    _snapshot(tmp_path, repo, "g" * 40, files, ref=False)
    (_home(tmp_path, repo) / "refs" / "main").write_bytes(ref(pin))
    assert E.clap_text_missing(tmp_path) == [f"{repo}/{name}" for name in files]


FIRST = [(repo, pin, tuple(name.split("|")[0] for name in files)) for repo, pin, files in E.CLAP_TEXT_SNAPSHOTS]
hub_lacks: set[str] = set()


@pytest.fixture
def hub(monkeypatch):
    """A stand-in hub that writes the files it is asked for where it is told, and no ref, as a fetch by commit does."""
    asked = []
    hub_lacks.clear()

    def snapshot_download(repo, *, revision, allow_patterns, cache_dir):
        asked.append((repo, revision, tuple(allow_patterns)))
        if revision in hub_lacks:
            raise OSError(f"404: {revision} is not a commit of {repo}")
        _snapshot(Path(cache_dir), repo, revision, allow_patterns, ref=False)
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=snapshot_download))
    return asked


def test_the_fetch_takes_named_files_at_each_pin_and_makes_the_ref_name_it(hub, tmp_path, monkeypatch):
    monkeypatch.setattr(E, "hub_cache_dir", lambda: tmp_path / "elsewhere")
    said = []
    E.clap_text_prefetch(tmp_path, said.append)
    assert hub == FIRST, "named files at the pinned commit, each by its first spelling, never a whole repository"
    assert said == [f"{repo} @ {pin[:12]}" for repo, pin, _ in E.CLAP_TEXT_SNAPSHOTS]
    assert E.clap_text_missing(tmp_path) == [], "into the cache it was given"
    assert [_ref(tmp_path, repo) for repo, _, _ in E.CLAP_TEXT_SNAPSHOTS] == [pin for _, pin, _ in E.CLAP_TEXT_SNAPSHOTS]
    assert not list(tmp_path.rglob("*.part"))


def test_with_no_cache_named_both_read_the_hub_librarys_own(hub, tmp_path, monkeypatch):
    monkeypatch.setattr(E, "hub_cache_dir", lambda: tmp_path)
    E.clap_text_prefetch()
    assert E.clap_text_missing() == [] and (tmp_path / "models--roberta-base").is_dir()


OVERRIDES = ("TRANSFORMERS_CACHE", "PYTORCH_TRANSFORMERS_CACHE", "PYTORCH_PRETRAINED_BERT_CACHE")


def test_the_cache_is_the_one_transformers_reads(monkeypatch, tmp_path):
    constants = types.SimpleNamespace(HF_HUB_CACHE=str(tmp_path / "hub"))
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(constants=constants))
    for name in OVERRIDES:
        monkeypatch.delenv(name, raising=False)
    assert E.hub_cache_dir() == tmp_path / "hub", "the hub library's own answer"
    for name in reversed(OVERRIDES):
        monkeypatch.setenv(name, str(tmp_path / name))
        assert E.hub_cache_dir() == tmp_path / name, "transformers' old overrides, in its own order"
    monkeypatch.setenv("TRANSFORMERS_CACHE", "")
    assert E.hub_cache_dir() == tmp_path / "PYTORCH_TRANSFORMERS_CACHE"


def test_a_ref_that_cannot_be_put_in_place_leaves_no_half_written_one(hub, tmp_path, monkeypatch):
    import os

    def refuse(src, dst):
        raise OSError("read-only cache")
    monkeypatch.setattr(os, "replace", refuse)
    with pytest.raises(OSError, match="read-only cache"):
        E.clap_text_prefetch(tmp_path)
    assert not list(tmp_path.rglob("*.part"))


def test_a_whole_repo_is_left_alone_and_a_short_one_is_completed_at_the_commit_its_ref_names(hub, tmp_path):
    """Another program's cache may name a later commit: its ref is kept and the files join that snapshot."""
    _fill(tmp_path)
    repo, _, files = E.CLAP_TEXT_SNAPSHOTS[2]
    theirs = "f" * 40
    _snapshot(tmp_path, repo, theirs, files[:1])
    before = _ref(tmp_path, "roberta-base")
    E.clap_text_prefetch(tmp_path)
    assert hub == [(repo, theirs, files)], "only the repo an offline load would find incomplete, at the commit it reads"
    assert _ref(tmp_path, repo) == theirs and _ref(tmp_path, "roberta-base") == before
    assert E.clap_text_missing(tmp_path) == []


def test_a_commit_the_hub_cannot_serve_falls_back_to_the_pin_and_the_ref_follows(hub, tmp_path):
    repo, pin, files = E.CLAP_TEXT_SNAPSHOTS[2]
    _fill(tmp_path)
    gone = "d" * 40
    _snapshot(tmp_path, repo, gone, files[:1])
    hub_lacks.add(gone)
    E.clap_text_prefetch(tmp_path)
    assert hub == [(repo, gone, files), (repo, pin, files)]
    assert _ref(tmp_path, repo) == pin and E.clap_text_missing(tmp_path) == []


def test_a_ref_another_program_moved_during_the_fetch_is_left_as_it_put_it(tmp_path, monkeypatch):
    """Its online load moves the ref to the new head while ours downloads; ours must not move it back."""
    repo, pin, files = E.CLAP_TEXT_SNAPSHOTS[1]
    newer = "c" * 40

    def snapshot_download(repo, *, revision, allow_patterns, cache_dir):
        _snapshot(tmp_path, repo, revision, allow_patterns, ref=False)
        if repo == "bert-base-uncased":
            _snapshot(tmp_path, repo, newer, ())
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=snapshot_download))
    E.clap_text_prefetch(tmp_path)
    assert _ref(tmp_path, repo) == newer
    assert _ref(tmp_path, "roberta-base") == E.CLAP_TEXT_SNAPSHOTS[0][1]


def test_a_ref_the_hub_cannot_read_is_rewritten_to_the_pin(hub, tmp_path):
    repo, pin, files = E.CLAP_TEXT_SNAPSHOTS[1]
    _fill(tmp_path)
    (_home(tmp_path, repo) / "refs" / "main").write_text(pin + "\n")
    E.clap_text_prefetch(tmp_path)
    assert hub == [(repo, pin, files)] and _ref(tmp_path, repo) == pin
    assert E.clap_text_missing(tmp_path) == []


def test_a_fetch_that_lands_nothing_raises_and_leaves_the_ref_alone(tmp_path, monkeypatch):
    """The hub hands back a snapshot folder it could not complete without raising."""
    repo, pin, files = E.CLAP_TEXT_SNAPSHOTS[0]
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(
        snapshot_download=lambda repo, *, revision, allow_patterns, cache_dir: _snapshot(tmp_path, repo, revision, allow_patterns[:1], ref=False)))
    theirs = "a" * 40
    _snapshot(tmp_path, repo, theirs, ())
    with pytest.raises(ConnectionError, match="roberta-base: the fetch did not land model.safetensors, vocab.json, merges.txt"):
        E.clap_text_prefetch(tmp_path)
    assert _ref(tmp_path, repo) == theirs
    (_home(tmp_path, repo) / "refs" / "main").unlink()
    with pytest.raises(ConnectionError):
        E.clap_text_prefetch(tmp_path)
    assert not (_home(tmp_path, repo) / "refs" / "main").exists(), "no ref is written for a snapshot short of a file"
