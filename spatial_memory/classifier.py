from __future__ import annotations

from spatial_memory.constants import CLASSIFIER_PROMPT_VERSION
from spatial_memory.models import Orientation
from spatial_memory.ollama_client import chat, parse_json_loose


# Pin this string in version control; bump CLASSIFIER_PROMPT_VERSION when it changes.
CLASSIFIER_SYSTEM = """You are orienting an incoming message in a 3D mental space. This is not classification into bins.
Each axis is a felt continuum from -1.0 to +1.0 (float). The message can sit anywhere between the poles.

Axis 1 — self_other (X)
Question: Is this mainly about the agent's own state, identity, capabilities, memory, or corrections to itself (self, -1),
or about the external world, other people, systems, and situations outside the agent (other, +1)?
The middle is meaningful: e.g. "using what you know, comment on X" blends inward knowledge in service of outward attention.

Axis 2 — known_unknown (Y)
Question: Does the agent likely have solid, familiar ground here (known, -1), or is this new, sparse, uncertain territory (unknown, +1)?
Partial or shaky prior knowledge maps between the poles. This axis encodes epistemic stance, not topic difficulty alone.

Axis 3 — active_contemplative (Z)
Question: Does this call primarily for doing, building, fixing, deciding, executing (active, -1),
or for reflection, meaning, frameworks, analysis without immediate action (contemplative, +1)?
Design and architecture questions often sit near the middle (both modes).

Output ONLY valid JSON:
{"self_other": <float>, "known_unknown": <float>, "active_contemplative": <float>}
Each number must be in [-1, 1]."""


def clamp(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _pick_num(d: dict, keys: list[str]) -> float | None:
    for k in keys:
        if k in d:
            try:
                return float(d[k])
            except (TypeError, ValueError):
                continue
    return None


def _orientation_from_dict(data: dict) -> Orientation | None:
    sx = _pick_num(
        data,
        [
            "self_other",
            "selfOther",
            "self-other",
            "x",
            "x_axis",
            "self_other_score",
        ],
    )
    sy = _pick_num(
        data,
        [
            "known_unknown",
            "knownUnknown",
            "known-unknown",
            "y",
            "y_axis",
            "known_unknown_score",
        ],
    )
    sz = _pick_num(
        data,
        [
            "active_contemplative",
            "activeContemplative",
            "active-contemplative",
            "z",
            "z_axis",
            "active_contemplative_score",
        ],
    )
    if sx is None or sy is None or sz is None:
        return None
    return Orientation(
        self_other=clamp(sx),
        known_unknown=clamp(sy),
        active_contemplative=clamp(sz),
        classifier_prompt_version=CLASSIFIER_PROMPT_VERSION,
    )


def classify_message(raw_message: str) -> Orientation:
    # Robust path: never let classifier formatting break the whole turn.
    try:
        raw = chat(
            CLASSIFIER_SYSTEM,
            raw_message,
            temperature=0.0,
            json_mode=True,
        )
        data = parse_json_loose(raw)
        o = _orientation_from_dict(data)
        if o is not None:
            return o
    except Exception:
        pass

    # Retry once without strict JSON mode; many models still output parseable JSON-ish text.
    try:
        raw = chat(
            CLASSIFIER_SYSTEM,
            raw_message,
            temperature=0.0,
            json_mode=False,
        )
        data = parse_json_loose(raw)
        o = _orientation_from_dict(data)
        if o is not None:
            return o
    except Exception:
        pass

    # Safe fallback: neutral center so pipeline continues and still commits memory.
    return Orientation(
        self_other=0.0,
        known_unknown=0.0,
        active_contemplative=0.0,
        classifier_prompt_version=CLASSIFIER_PROMPT_VERSION,
    )


__all__ = ["classify_message", "CLASSIFIER_SYSTEM"]
