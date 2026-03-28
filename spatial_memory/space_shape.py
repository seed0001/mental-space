from __future__ import annotations

import math


def _warp_x(x: float, y: float, z: float) -> float:
    # Bean-like asymmetry: slight inward dent on one side plus soft curvature.
    return x + 0.22 * (y * y) - 0.18 * (z * z) - 0.12 * max(0.0, y)


def _inside_bean(x: float, y: float, z: float) -> bool:
    wx = _warp_x(x, y, z)
    return (wx / 1.22) ** 2 + (y / 0.98) ** 2 + (z / 0.88) ** 2 <= 1.0


def constrain_to_bean_space(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Clamp any coordinate into a kidney-bean-like bounded latent volume."""
    if _inside_bean(x, y, z):
        return (x, y, z)
    lo, hi = 0.0, 1.0
    for _ in range(32):
        mid = (lo + hi) * 0.5
        tx, ty, tz = x * mid, y * mid, z * mid
        if _inside_bean(tx, ty, tz):
            lo = mid
        else:
            hi = mid
    s = lo
    return (x * s, y * s, z * s)


def clamp_axis(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def constrain_orientation_full(
    x: float,
    y: float,
    z: float,
    w: float,
    v: float,
) -> tuple[float, float, float, float, float]:
    """Bean (x,y,z) plus independent clamp for auxiliary axes (w,v)."""
    cx, cy, cz = constrain_to_bean_space(x, y, z)
    return (cx, cy, cz, clamp_axis(w), clamp_axis(v))

