from __future__ import annotations

import math

from spatial_memory.constants import EXTRA_AXIS_DIST_WEIGHT
from spatial_memory.models import MemoryNode


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def latent_vector_dist_sq(
    x0: float,
    y0: float,
    z0: float,
    w0: float,
    v0: float,
    x1: float,
    y1: float,
    z1: float,
    w1: float,
    v1: float,
) -> float:
    dx, dy, dz = x0 - x1, y0 - y1, z0 - z1
    dw, dv = w0 - w1, v0 - v1
    return dx * dx + dy * dy + dz * dz + EXTRA_AXIS_DIST_WEIGHT * (dw * dw + dv * dv)


def memory_node_dist_sq(a: MemoryNode, b: MemoryNode) -> float:
    return latent_vector_dist_sq(a.x, a.y, a.z, a.w, a.v, b.x, b.y, b.z, b.w, b.v)
