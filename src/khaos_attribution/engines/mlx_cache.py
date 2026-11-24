"""MLX's buffer cache limit for an engine's MLX process, as the room asks: a small Mac sets 0 to fit. The cache holds
freed buffers for reuse, so its limit changes memory and time, never what is computed."""

from __future__ import annotations

import os

ENV = "KHAOS_MLX_CACHE_LIMIT_GB"


def apply(mx=None) -> float | None:
    """Set MLX's cache limit when the room asked for one; the limit in GB, or None when it asked for none. MLX is
    imported only when a limit is asked for and ``mx`` is not given."""
    raw = os.environ.get(ENV, "").strip()
    if not raw:
        return None
    try:
        gb = float(raw)
    except ValueError:
        raise ValueError(f"{ENV} must be a number of GB, got {raw!r}") from None
    if mx is None:
        import mlx.core as mx  # noqa: PLC0415
    mx.set_cache_limit(int(gb * 2**30))
    return gb
