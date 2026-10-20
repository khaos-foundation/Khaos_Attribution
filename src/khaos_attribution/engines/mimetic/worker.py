"""One mimetic-a0 render inside the engine's own environment: the request as JSON on stdin, the reply as one JSON
line last on stdout. Audio is written as 16-bit PCM WAV at 48 kHz, so the same seed on the same device gives the
same bytes."""

from __future__ import annotations

import json
import sys
from pathlib import Path


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
    req = json.loads(sys.stdin.read())
    try:
        reply = render(req)
    except Exception as exc:  # noqa: BLE001 — the reply carries it; the caller raises RenderFailed
        reply = {"error": f"{type(exc).__name__}: {exc}"}
    sys.stdout.write("\n" + json.dumps(reply) + "\n")
    sys.stdout.flush()
    return 0 if not reply.get("error") else 1


if __name__ == "__main__":
    sys.exit(main())
