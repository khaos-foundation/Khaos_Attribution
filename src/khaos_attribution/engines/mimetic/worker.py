"""One mimetic-a0 render inside the engine's own environment: the request as JSON on stdin, the reply as one JSON
line last on stdout. Audio is written as 16-bit PCM WAV at 48 kHz, so the same seed on the same device gives the
same bytes."""

from __future__ import annotations

import sys
from pathlib import Path

if __name__ in ("__main__", "__mp_main__"):  # the room's contract, only through engines/launch.py
    import pathlib
    import runpy
    runpy.run_path(str(pathlib.Path(__file__).resolve().parents[1] / "launch.py"), run_name="khaos_contract_check")


def render(req: dict) -> dict:
    import numpy as np  # noqa: PLC0415
    import soundfile as sf  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from stable_audio_tools.inference.generation import generate_diffusion_cond  # noqa: PLC0415

    from khaos_attribution.engines.mimetic.loading import apply_adapter, load_model  # noqa: PLC0415

    model, config, device = load_model(req.get("device"))
    if req.get("adapter"):
        apply_adapter(model, config, req["adapter"], float(req["adapter_scale"]))
    seed = int(req["seed"]) if req.get("seed") is not None else int(np.random.randint(0, 2 ** 31 - 1))
    rate = int(config["sample_rate"])
    seconds = min(float(req["seconds"]), config["sample_size"] / rate)
    dropped = []
    if req.get("mode") not in (None, "text2music", "retake"):
        dropped.append(f"mode {req['mode']} is not offered on mimetic-a0")
    if req.get("negative_prompt"):
        dropped.append("negative prompt")
    cond = [{"prompt": req["prompt"], "seconds_start": 0, "seconds_total": seconds}]
    with torch.no_grad():
        audio = generate_diffusion_cond(model, steps=int(req["steps"]), cfg_scale=float(req["cfg_scale"]),
                                        conditioning=cond, batch_size=1, sample_size=config["sample_size"],
                                        seed=seed, device=device, apg_scale=float(req.get("apg_scale") or 1.0))
    clip = audio[0, :, : int(round(seconds * rate))].float().cpu().numpy()
    out = Path(req["out"])
    tmp = out.with_name(out.name + ".tmp.wav")
    sf.write(str(tmp), clip.T, rate, subtype="PCM_16")
    tmp.replace(out)
    return {"seed": seed, "sample_rate": rate, "dropped": dropped, "device": device}


def main() -> int:
    from khaos_attribution.engines.worker_process import answer  # noqa: PLC0415
    return answer(lambda req: render(req))


if __name__ == "__main__":
    sys.exit(main())
