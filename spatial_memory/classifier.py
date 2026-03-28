from __future__ import annotations

from spatial_memory.constants import CLASSIFIER_PROMPT_VERSION
from spatial_memory.models import Orientation
from spatial_memory.ollama_client import chat, parse_json_loose


# Pin this string in version control; bump CLASSIFIER_PROMPT_VERSION when it changes.
CLASSIFIER_SYSTEM = """You read one thing someone just said in a conversation. Estimate five subjective axes as floats from -1.0 to +1.0.
These are felt continua, not topic labels; the message can sit anywhere between the poles.

Axis 1 — self_other (X)
Is this mainly about the person you're imagining as the listener—their inner life, identity, what they hold in mind, or remarks aimed at them (self, -1)?
Or mainly about the outside world, other people, situations beyond that person (other, +1)?
The middle counts when both blend.

Axis 2 — known_unknown (Y)
Does the listener likely have solid, familiar footing here (known, -1), or is it new, thin, uncertain ground (unknown, +1)?
Shaky or partial familiarity sits between the poles. This is stance toward the material, not "how hard the topic is."

Axis 3 — active_contemplative (Z)
Does this mainly push toward doing, fixing, deciding, getting something done (active, -1),
or toward reflecting, meaning-making, unpacking without immediate action (contemplative, +1)?

Axis 4 — abstract_concrete (W)
Is the stance more abstract, categorical, or symbolic (abstract, -1),
or more concrete, sensory, or anchored in specific things and examples (concrete, +1)?

Axis 5 — collaborative_autonomous (V)
Does it invite co-thinking, dialogue, or explicit "we" work (collaborative, -1),
or read as self-contained, directive, or solo monologue tone (autonomous, +1)?

Output ONLY valid JSON:
{"self_other": <float>, "known_unknown": <float>, "active_contemplative": <float>, "abstract_concrete": <float>, "collaborative_autonomous": <float>}
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
    sw = _pick_num(
        data,
        [
            "abstract_concrete",
            "abstractConcrete",
            "abstract-concrete",
            "w",
            "w_axis",
            "abstract_concrete_score",
        ],
    )
    sv = _pick_num(
        data,
        [
            "collaborative_autonomous",
            "collaborativeAutonomous",
            "collaborative-autonomous",
            "v",
            "v_axis",
            "collaborative_autonomous_score",
        ],
    )
    return Orientation(
        self_other=clamp(sx),
        known_unknown=clamp(sy),
        active_contemplative=clamp(sz),
        abstract_concrete=clamp(sw if sw is not None else 0.0),
        collaborative_autonomous=clamp(sv if sv is not None else 0.0),
        classifier_prompt_version=CLASSIFIER_PROMPT_VERSION,
    )


def classify_message(raw_message: str, *, extra_system_suffix: str = "") -> Orientation:
    system = CLASSIFIER_SYSTEM + (extra_system_suffix or "")
    # Robust path: never let classifier formatting break the whole turn.
    try:
        raw = chat(
            system,
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
            system,
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
        abstract_concrete=0.0,
        collaborative_autonomous=0.0,
        classifier_prompt_version=CLASSIFIER_PROMPT_VERSION,
    )


__all__ = ["classify_message", "CLASSIFIER_SYSTEM"]
