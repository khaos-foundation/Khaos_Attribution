"""The held-out loss on mimetic-a0, inside the engine's environment: the request as JSON on stdin (clips, the
adapter, the timesteps, the seed), the reply as one JSON line of per-sample bucket losses. Each clip is taken as
the trainer takes it (khaos_attribution.engines.mimetic.clip); noise is seeded per clip."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def heldout(req: dict) -> dict:
    import torch  # noqa: PLC0415

    from khaos_attribution import gradient as G  # noqa: PLC0415
    from khaos_attribution.engines.mimetic import clip as C  # noqa: PLC0415
    from khaos_attribution.engines.mimetic.loading import apply_adapter, load_model  # noqa: PLC0415

    model, config, device = load_model(req.get("device"))
    if req.get("adapter"):
        apply_adapter(model, config, req["adapter"], 1.0)
    timesteps = [float(t) for t in req["timesteps"]]
    per_sample = {}
    for wav in sorted(Path(req["clips_dir"]).glob("*.wav")):
        latents, cond, mask, noise = C.prepare(model, config, wav, device, G.pair_noise_seed(int(req["seed"]), wav.name))
        with torch.no_grad():
            per_sample[wav.name] = {f"{t:.2f}": float(C.velocity_mse(C.forward(model, latents, cond, mask, noise, t),
                                                                     latents, noise, mask)) for t in timesteps}
    return {"per_sample": per_sample, "runtime": "torch", "device": device}


def main() -> int:
    req = json.loads(sys.stdin.read())
    try:
        reply = heldout(req)
    except Exception as exc:  # noqa: BLE001 — the reply carries it
        reply = {"error": f"{type(exc).__name__}: {exc}"}
    sys.stdout.write("\n" + json.dumps(reply) + "\n")
    sys.stdout.flush()
    return 0 if not reply.get("error") else 1


if __name__ == "__main__":
    sys.exit(main())
