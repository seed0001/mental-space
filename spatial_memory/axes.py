"""
Latent orientation axes for spatial memory.

All axis values are in [-1.0, 1.0]. X–Z use the kidney-bean volume (see space_shape);
W and V are clamped to the cube only (no bean warp).

Column names in SQLite: x,y,z plus w,v for the extra pair (short names for SQL ergonomics).
"""

from __future__ import annotations

# Ordered keys for vectors, prompts, and momentum (matches Orientation fields).
ORIENTATION_AXIS_KEYS: tuple[str, ...] = (
    "self_other",
    "known_unknown",
    "active_contemplative",
    "abstract_concrete",
    "collaborative_autonomous",
)

AXIS_SPEC: dict[str, dict[str, str]] = {
    "self_other": {
        "letter": "X",
        "negative": "self / listener-inward",
        "positive": "other / world-outward",
    },
    "known_unknown": {
        "letter": "Y",
        "negative": "known / familiar footing",
        "positive": "unknown / thin or new ground",
    },
    "active_contemplative": {
        "letter": "Z",
        "negative": "active / doing-deciding",
        "positive": "contemplative / meaning-making",
    },
    "abstract_concrete": {
        "letter": "W",
        "negative": "abstract / categorical / symbolic",
        "positive": "concrete / sensory / specific",
    },
    "collaborative_autonomous": {
        "letter": "V",
        "negative": "collaborative / dialogic / we-together",
        "positive": "autonomous / self-contained / directive solo tone",
    },
}

__all__ = ["AXIS_SPEC", "ORIENTATION_AXIS_KEYS"]
