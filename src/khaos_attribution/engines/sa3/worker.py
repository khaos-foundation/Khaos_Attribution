"""One Stable Audio 3 render, run inside the engine's own environment: the request as JSON on stdin, the reply as
one JSON line last on stdout, a ``khaos-row`` line at each sampling step before it. "torch" drives Stability's
package; "mlx" drives the MLX port's script in the same checkout. Audio is written as 16-bit PCM WAV at the model's
rate, so the same seed gives the same bytes."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

if __name__ in ("__main__", "__mp_main__"):  # the room's contract, only through engines/launch.py
    import pathlib
    import runpy
    runpy.run_path(str(pathlib.Path(__file__).resolve().parents[1] / "launch.py"), run_name="khaos_contract_check")


REWORKS = ("retake", "audio2audio", "repaint", "extend")
# The prefix the contract's worker loop reads a progress row by; the driver prints it too.
ROW_PREFIX = "khaos-row "


def _row(step: int, of: int) -> None:
    sys.stdout.write(ROW_PREFIX + json.dumps({"step": int(step), "of": int(of)}) + "\n")
    sys.stdout.flush()


def part_path(out) -> Path:
    """Where a render writes until its audio is whole; only a finished render takes the output's name."""
    out = Path(out)
    return out.with_name(f"{out.stem}.part{out.suffix}")


def _fresh_part(out) -> Path:
    # A part left by a render killed before it could tidy up is never taken for this render's audio.
    part = part_path(out)
    part.unlink(missing_ok=True)
    return part


def _loaded_torch(req: dict):
    """Stability's model for the request, its adapter loaded at the request's strength."""
    from khaos_attribution.engines.sa3.loading import load_model  # noqa: PLC0415
    model = load_model(req["model"])
    if req.get("adapter"):
        model.load_lora([str(req["adapter"])])
        model.set_lora_strength(float(req["adapter_scale"]))
    return model


def render_torch(req: dict, model=None) -> dict:
    import soundfile as sf  # noqa: PLC0415
    import torch  # noqa: PLC0415

    out = Path(req["out"])
    part = _fresh_part(out)
    model = model if model is not None else _loaded_torch(req)
    seed = int(req["seed"]) if req.get("seed") is not None else int(torch.randint(0, 2 ** 31 - 1, (1,)).item())
    steps = int(req["steps"])
    kwargs = {"prompt": req["prompt"], "duration": float(req["seconds"]), "steps": steps,
              "cfg_scale": float(req["cfg_scale"]), "apg_scale": float(req.get("apg_scale") or 1.0), "seed": seed,
              # The sampler's own step callback, handed through generate to every sampler Stability ships.
              "callback": lambda info: _row(int(info.get("i", 0)) + 1, steps)}
    if req.get("negative_prompt"):
        kwargs["negative_prompt"] = req["negative_prompt"]
    if req.get("mode") in REWORKS:
        audio, sr = sf.read(str(req["source_audio"]), dtype="float32", always_2d=True)
        tensor = torch.from_numpy(audio.T)
        level = float(req.get("noise_level") or 1.0)
        if level < 1.0:
            kwargs["init_audio"] = (sr, tensor)
            kwargs["init_noise_level"] = level
        if req.get("regions"):
            kwargs["inpaint_audio"] = (sr, tensor)
            kwargs["inpaint_mask_start_seconds"] = [float(s) for s, _ in req["regions"]]
            kwargs["inpaint_mask_end_seconds"] = [float(e) for _, e in req["regions"]]
    audio = model.generate(**kwargs)
    rate = int(model.model.sample_rate)
    sf.write(str(part), audio[0].T.float().cpu().numpy(), rate, subtype="PCM_16")
    part.replace(out)
    return {"seed": seed, "sample_rate": rate, "dropped": [], "runtime": "torch"}


# The variant config whose DiT is the base checkpoint adapters train on; the MLX script ships only the distilled one.
BASE_CONFIG = "medium-base"


def render_mlx(req: dict) -> dict:
    """Stability's MLX script under the contract's driver: on the base checkpoint when the variant names it, with
    every part of a rework masked in one pass."""
    here = Path.cwd() / "optimized" / "mlx"
    script = here / "scripts" / "sa3_mlx.py"
    seed = int(req["seed"]) if req.get("seed") is not None else int.from_bytes(os.urandom(4), "big") % (2 ** 31 - 1)
    dit = "medium"
    out = Path(req["out"])
    part = _fresh_part(out)
    args = ["--prompt", req["prompt"], "--dit", dit,
            "--decoder", "same-l", "--seconds", str(float(req["seconds"])), "--seed", str(seed),
            "--steps", str(int(req["steps"])), "--cfg", str(float(req["cfg_scale"])),
            "--apg", str(float(req.get("apg_scale") or 1.0)), "--out", str(part)]
    if req.get("negative_prompt"):
        args += ["--negative-prompt", req["negative_prompt"]]
    if req.get("adapter"):
        args += ["--lora", str(req["adapter"]), "--lora-strength", str(float(req["adapter_scale"]))]
    regions = req.get("regions") or [] if req.get("mode") in REWORKS else []
    if req.get("mode") in REWORKS:
        args += ["--init-audio", str(req["source_audio"]), "--init-noise-level", str(float(req.get("noise_level") or 1.0))]
    if regions:
        args += ["--inpaint-range", f"{min(s for s, _ in regions)},{max(e for _, e in regions)}"]
    from khaos_attribution.engines.worker_process import command  # noqa: PLC0415
    # The script masks one range: the driver masks every part, and the script is given their span.
    cmd = command(sys.executable, Path(__file__).with_name("render_driver_mlx.py"), script,
                  *(["--khaos-base"] if req.get("model") == BASE_CONFIG else []),
                  *(["--khaos-regions", ";".join(f"{s},{e}" for s, e in regions)] if regions else []), *args)
    try:
        done = _run_forwarding_stop(cmd, cwd=str(here))
        if done.returncode != 0:
            tail = "\n".join(done.stderr.strip().splitlines()[-8:])
            return {"error": f"sa3_mlx.py failed ({done.returncode}): {tail}"}
        if not part.is_file():
            return {"error": "sa3_mlx.py finished and wrote no audio"}
        part.replace(out)
    finally:
        part.unlink(missing_ok=True)
    rate = None
    try:
        import wave  # noqa: PLC0415
        with wave.open(str(out), "rb") as fh:
            rate = fh.getframerate()
    except Exception:  # noqa: BLE001
        pass
    return {"seed": seed, "sample_rate": rate, "dropped": [], "runtime": "mlx"}


def _run_forwarding_stop(cmd: list, *, cwd: str):
    """The render as a child that dies with this worker: a stop asked of the worker is passed on, so no render
    outlives a cancel. Its progress rows are passed on as they come; the rest of its output is kept."""
    import signal  # noqa: PLC0415
    import tempfile  # noqa: PLC0415
    with tempfile.TemporaryFile("w+", encoding="utf-8", errors="replace") as err:
        # stderr to a file: a child that writes much to it would block while stdout is being read.
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=err, text=True, errors="replace")

        def _forward(signum, _frame):
            proc.terminate()
            try:
                proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            sys.exit(128 + signum)

        previous = signal.signal(signal.SIGTERM, _forward)
        try:
            kept = []
            for line in proc.stdout:
                if line.startswith(ROW_PREFIX):
                    sys.stdout.write(line)
                    sys.stdout.flush()
                else:
                    kept.append(line)
            proc.wait()
        finally:
            signal.signal(signal.SIGTERM, previous)
        err.seek(0)
        return subprocess.CompletedProcess(cmd, proc.returncode, "".join(kept), err.read())


class _Resident:
    """The one model and adapter a resident worker holds: loaded at its first request, the strength set again at
    each; a request for another model or adapter is refused, the renderer starting another worker for it."""

    def __init__(self) -> None:
        self.key = None
        self.model = None

    def __call__(self, req: dict) -> dict:
        if req.get("runtime") == "mlx":
            raise RuntimeError("a resident worker renders on torch only")
        key = (req["model"], req.get("adapter"))
        if self.model is None:
            self.model, self.key = _loaded_torch(req), key
        elif key != self.key:
            raise RuntimeError("a resident worker serves the model and adapter it started with")
        elif req.get("adapter"):
            self.model.set_lora_strength(float(req["adapter_scale"]))
        return render_torch(req, self.model)


def main() -> int:
    from khaos_attribution.engines.worker_process import answer, serve  # noqa: PLC0415
    if "--serve" in sys.argv[1:]:
        return serve(_Resident())
    return answer(lambda req: render_mlx(req) if req.get("runtime") == "mlx" else render_torch(req))


if __name__ == "__main__":
    sys.exit(main())
