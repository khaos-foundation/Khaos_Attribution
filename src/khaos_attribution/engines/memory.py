"""What a machine has to score on, and the plan for an engine whose features come from a worker of its own: the
worker's measured peak against the device's memory, less what the machine keeps for itself."""

from __future__ import annotations

import os

__all__ = ["LOW_MEMORY_OVERRIDE_ENV", "SYSTEM_RESERVE_GB", "machine_memory_gb", "physical_memory_gb",
           "process_memory_plan", "process_scorer_need_gb"]

# What the machine keeps for itself whatever is scored, and the override that lets a small machine try anyway.
SYSTEM_RESERVE_GB = 6.0
LOW_MEMORY_OVERRIDE_ENV = "KHAOS_INFLUENCE_LOW_MEMORY"


def physical_memory_gb() -> float | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024 ** 3
    except (ValueError, OSError, AttributeError):
        return None


def machine_memory_gb(device_type: str) -> float | None:
    """The memory the scorer shares: the card's on cuda, the machine's elsewhere."""
    if device_type == "cuda":
        try:
            import torch  # noqa: PLC0415
            return torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        except Exception:  # noqa: BLE001
            return None
    return physical_memory_gb()


def _scorer(descriptor, device_type: str) -> dict:
    """The worker's measured peak on a device type: CUDA's under "scorer", any other under "scorer_<device>"."""
    if device_type == "cuda":
        return descriptor.memory.get("scorer") or {}
    return descriptor.memory.get(f"scorer_{device_type}") or {}


def process_scorer_need_gb(descriptor, device_type: str = "cuda") -> float:
    """What the engine's feature worker needs on the device for any clip up to the longest the Workshop cuts."""
    s = _scorer(descriptor, device_type)
    if not s.get("base_gb"):
        raise NotImplementedError(f"{descriptor.name}'s measure has not been measured on {device_type}")
    return float(s["base_gb"]) + float(s.get("process_gb") or 0.0)


def process_memory_plan(descriptor, device_type: str, *, total_gb: float | None = None, env: dict | None = None) -> dict:
    """The worker runs in its own process on the device, so the plan is one number against the device's memory;
    nothing is checkpointed and nothing of the engine is resident to yield."""
    if not _scorer(descriptor, device_type).get("base_gb"):
        return {"fits": None, "checkpointed": False, "need_gb": None, "total_gb": total_gb,
                "refusal": f"{descriptor.name}'s features have not been measured on {device_type}; this machine "
                           f"scores on {device_type}"}
    total = machine_memory_gb(device_type) if total_gb is None else total_gb
    need = process_scorer_need_gb(descriptor, device_type)
    plan = {"fits": "resident", "checkpointed": False, "need_gb": round(need, 1), "total_gb": total, "refusal": None}
    if total is None or need + SYSTEM_RESERVE_GB <= total or \
            (env if env is not None else os.environ).get(LOW_MEMORY_OVERRIDE_ENV) == "1":
        return plan
    return {**plan, "fits": None, "refusal": (f"measuring an output on {descriptor.name} needs about "
                                              f"{need + SYSTEM_RESERVE_GB:.0f} GB on the device, and this one has "
                                              f"{total:.0f} GB")}
