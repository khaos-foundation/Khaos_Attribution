"""The checklist an engine passes before a room may use it: the descriptor says enough, the three slots carry the
names the rooms call, and (live) the renderer makes the audio it says it makes, the same bytes for the same seed,
and stops when asked. The mock passes it in seconds; a real engine passes it where its weights are."""

from __future__ import annotations

import hashlib
import sys
import wave
from dataclasses import dataclass, field
from pathlib import Path

from khaos_attribution.engines import KNOWN_ENGINES, EngineUnavailable, descriptor, engine
from khaos_attribution.engines.render import RENDERER_CALLS, RENDERER_METHODS, RenderCancelled, RenderRequest, RenderResult

SLOTS = ("train", "render", "measure")
RENDER_NAMES = ("Renderer", "RenderRequest", "RenderResult", "RenderCancelled", "RenderFailed")
TRAIN_NAMES = ("TrainingUnavailable", "training_api", "adapter_kwargs", "preprocess_pairs", "open_module",
               "adapter_file_in", "open_session", "pair_files", "Session")
# What a room does with the session open_session gives it.
SESSION_METHODS = ("install_hooks", "train", "heldout_evaluator", "save_final")
# The memory plan and the features the rooms take: an index, an output's feature, its losses, the adapter's size.
MEASURE_NAMES = ("memory_plan", "scorer_need_gb", "enable_checkpointing", "flush_accelerator", "index_rows",
                 "output_feature", "paired_loss", "output_loss", "parameter_count")
LIVE_SECONDS = 5.0


@dataclass
class Finding:
    name: str
    ok: bool
    note: str = ""


@dataclass
class Checklist:
    engine_id: str
    findings: list = field(default_factory=list)

    def add(self, name: str, ok: bool, note: str = "") -> None:
        self.findings.append(Finding(name, bool(ok), note))

    @property
    def passed(self) -> bool:
        return all(f.ok for f in self.findings)

    @property
    def failed(self) -> list:
        return [f for f in self.findings if not f.ok]

    def report(self) -> str:
        lines = [f"engine {self.engine_id}: {'PASS' if self.passed else 'FAIL'}"]
        for f in self.findings:
            lines.append(f"  [{'ok' if f.ok else 'FAIL'}] {f.name}" + (f": {f.note}" if f.note else ""))
        return "\n".join(lines)


def check(engine_id: str, workdir: Path | str | None = None, *, live: bool = False, repo_dir=None,
          checkpoint_dir=None, device: str = "auto") -> Checklist:
    """Every check the socket asks of an engine; ``live`` also renders, which needs the engine's weights."""
    out = Checklist(engine_id)
    if engine_id not in KNOWN_ENGINES:
        out.add("registered", False, f"{engine_id!r} is not in KNOWN_ENGINES")
        return out
    out.add("registered", True)
    try:
        d = descriptor(engine_id)
    except Exception as exc:  # noqa: BLE001 — the finding carries it
        out.add("descriptor", False, str(exc))
        return out
    out.add("descriptor", True, f"{d.name} {d.doc.get('version')}")
    out.add("descriptor.id", d.id == engine_id, f"descriptor id {d.id!r}")
    out.add("descriptor.platforms", bool(d.platforms) and all(set(v) <= set(SLOTS) for v in d.platforms.values()),
            ", ".join(f"{k}: {'+'.join(v)}" for k, v in d.platforms.items()))
    out.add("descriptor.default_variant", d.default_variant in d.variants, d.default_variant)
    out.add("descriptor.adapter", bool(d.adapter.get("type")) and bool(d.adapter.get("filenames")),
            f"{d.adapter.get('type')} {d.adapter.get('filenames')}")
    try:
        e = engine(engine_id)
    except EngineUnavailable as exc:
        out.add("imports", False, str(exc))
        return out
    out.add("imports", True)
    slots = {}
    for slot in SLOTS:
        try:
            slots[slot] = getattr(e, slot)
            out.add(f"slot.{slot}", True)
        except Exception as exc:  # noqa: BLE001
            out.add(f"slot.{slot}", False, f"{type(exc).__name__}: {exc}")
    if len(slots) < len(SLOTS):
        return out
    _names(out, "render", slots["render"], RENDER_NAMES)
    _names(out, "train", slots["train"], TRAIN_NAMES)
    _names(out, "measure", slots["measure"], MEASURE_NAMES)
    renderer_cls = getattr(slots["render"], "Renderer", None)
    if renderer_cls is not None:
        missing = [m for m in RENDERER_METHODS if not callable(getattr(renderer_cls, m, None))]
        out.add("render.Renderer.methods", not missing, f"missing {missing}" if missing else ", ".join(RENDERER_METHODS))
        if not missing:
            refused = _refused_calls(renderer_cls)
            out.add("render.Renderer.calls", not refused,
                    "; ".join(refused) if refused else "takes every call the rooms make")
    session_cls = getattr(slots["train"], "Session", None)
    if session_cls is not None:
        missing = [m for m in SESSION_METHODS if not callable(getattr(session_cls, m, None))]
        out.add("train.Session.methods", not missing, f"missing {missing}" if missing else ", ".join(SESSION_METHODS))
    out.add("render.words", getattr(slots["render"], "RenderRequest", None) is RenderRequest
            and getattr(slots["render"], "RenderResult", None) is RenderResult,
            "the slot speaks the socket's RenderRequest and RenderResult")
    measured_on = [platform for platform, offered in d.platforms.items() if "measure" in offered]
    measures = bool(measured_on)
    # A plan is asked where the descriptor says the engine measures; an engine that measures nowhere is asked on cpu.
    probe = "cpu" if not measured_on or "cpu" in measured_on else measured_on[0]
    try:
        plan = slots["measure"].memory_plan(probe, 30.0, total_gb=1e6, env={})
        if measures:
            ok = isinstance(plan, dict) and plan.get("fits") == "resident" and "refusal" in plan
        else:
            # An engine that measures nowhere must say so, never hand back a number.
            ok = isinstance(plan, dict) and plan.get("fits") is None and bool(plan.get("refusal"))
        out.add("measure.memory_plan", ok, str(plan)[:120])
    except Exception as exc:  # noqa: BLE001
        out.add("measure.memory_plan", False, f"{type(exc).__name__}: {exc}")
    if live and renderer_cls is not None:
        _live(out, d, renderer_cls, Path(workdir or Path.cwd()) / "checklist", repo_dir, checkpoint_dir, device)
    return out


def _refused_calls(renderer_cls) -> list:
    """Each call the rooms make that the renderer's signatures would refuse, in words."""
    import inspect  # noqa: PLC0415
    refused = []
    for method, calls in RENDERER_CALLS.items():
        sig = inspect.signature(getattr(renderer_cls, method))
        for args, kwargs in calls:
            try:
                sig.bind("self", *args, **kwargs)
            except TypeError as exc:
                refused.append(f"{method}{tuple(args)} {sorted(kwargs)}: {exc}")
    return refused


def _names(out: Checklist, slot: str, module, names: tuple) -> None:
    missing = [n for n in names if not hasattr(module, n)]
    out.add(f"{slot}.names", not missing, f"missing {missing}" if missing else ", ".join(names))


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _live(out: Checklist, d, renderer_cls, workdir: Path, repo_dir, checkpoint_dir, device: str = "auto") -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    try:
        r = renderer_cls(repo_dir=repo_dir, checkpoint_dir=checkpoint_dir, device=device)
        r.load(d.default_variant)
        out.add("live.load", True, d.default_variant)
    except Exception as exc:  # noqa: BLE001
        out.add("live.load", False, f"{type(exc).__name__}: {exc}")
        return
    req = RenderRequest(prompt="a short bright sting", duration_seconds=LIVE_SECONDS, seed=7, variant=d.default_variant)
    try:
        a = r.generate(req, None, workdir / "a.wav")
        b = r.generate(req, None, workdir / "b.wav")
        out.add("live.deterministic", _sha(a.audio_path) == _sha(b.audio_path), "same seed, same bytes")
        c = r.generate(RenderRequest(prompt=req.prompt, duration_seconds=LIVE_SECONDS, seed=8, variant=d.default_variant),
                       None, workdir / "c.wav")
        out.add("live.seed_matters", _sha(a.audio_path) != _sha(c.audio_path), "another seed, other bytes")
        with wave.open(str(a.audio_path), "rb") as fh:
            sr, n = fh.getframerate(), fh.getnframes()
        out.add("live.sample_rate", sr == d.sample_rate, f"{sr} Hz, descriptor {d.sample_rate}")
        out.add("live.duration", abs(n / sr - LIVE_SECONDS) <= 0.5, f"{n / sr:.2f} s for {LIVE_SECONDS:.0f} s asked")
        out.add("live.result", isinstance(a, RenderResult) and a.seed == 7 and a.variant == d.default_variant
                and isinstance(a.params, dict), f"seed {a.seed}, variant {a.variant}")
    except Exception as exc:  # noqa: BLE001
        out.add("live.render", False, f"{type(exc).__name__}: {exc}")
        return
    seen = []
    try:
        r.generate(req, None, workdir / "progress.wav", on_progress=lambda done, total=1.0: seen.append(done / total))
        rising = all(b >= a for a, b in zip(seen, seen[1:]))
        # Rising and ending whole is proven here on the engine run live; another engine's shape is checked by its calls.
        out.add("live.progress", bool(seen) and rising and all(0.0 <= f <= 1.0 + 1e-6 for f in seen)
                and abs(seen[-1] - 1.0) <= 1e-6,
                f"{len(seen)} reports, {seen[:1]} to {seen[-1:]}")
    except Exception as exc:  # noqa: BLE001
        out.add("live.progress", False, f"{type(exc).__name__}: {exc}")
    try:
        r.generate(req, None, workdir / "stopped.wav", should_stop=lambda: True)
        out.add("live.cancel", False, "a render asked to stop finished anyway")
    except RenderCancelled:
        out.add("live.cancel", True, "stops between steps")
    except Exception as exc:  # noqa: BLE001
        out.add("live.cancel", False, f"{type(exc).__name__}: {exc}")


def main(argv=None) -> int:
    import argparse  # noqa: PLC0415
    ap = argparse.ArgumentParser(description="Run an engine through the socket's checklist.")
    ap.add_argument("engine", choices=sorted(KNOWN_ENGINES))
    ap.add_argument("--live", action="store_true", help="also render (needs the engine's weights)")
    ap.add_argument("--workdir", type=Path, default=None)
    ap.add_argument("--repo-dir", type=Path, default=None)
    ap.add_argument("--checkpoint-dir", type=Path, default=None)
    ap.add_argument("--device", default="auto", help="the renderer's device (auto picks this machine's)")
    a = ap.parse_args(argv)
    result = check(a.engine, a.workdir, live=a.live, repo_dir=a.repo_dir, checkpoint_dir=a.checkpoint_dir, device=a.device)
    print(result.report())
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
