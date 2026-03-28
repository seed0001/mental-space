"""
Orientation momentum: blend new classifier output with the previous turn's position,
and optional scene-event trail text for the classifier (closed loop with scene_events).
"""

from __future__ import annotations

from spatial_memory import store


def previous_xyzwv_for_momentum(
    *,
    scope: str,
    active_scene_id: str | None,
    db_path: str | None = None,
) -> tuple[float, float, float, float, float] | None:
    scope_l = (scope or "scene").strip().lower()
    if scope_l not in ("global", "scene"):
        scope_l = "scene"
    if scope_l == "global":
        return store.last_global_scene_event_xyzwv(db_path=db_path)
    if active_scene_id:
        return store.last_scene_event_xyzwv(active_scene_id, db_path=db_path)
    return None


def blend_latent_with_previous(
    current: tuple[float, float, float, float, float],
    prev: tuple[float, float, float, float, float] | None,
    prev_weight: float,
) -> tuple[float, float, float, float, float]:
    if prev is None or prev_weight <= 0:
        return current
    w = max(0.0, min(1.0, float(prev_weight)))
    return tuple((1.0 - w) * current[i] + w * prev[i] for i in range(5))


def classifier_scene_trail_suffix(scene_id: str, *, max_events: int, db_path: str | None = None) -> str:
    """Append to classifier system prompt; uses scene_events coordinates + commitment (internal only)."""
    evs = store.recent_scene_events(scene_id, limit=max(1, max_events), db_path=db_path)
    if not evs:
        return ""
    lines: list[str] = []
    for i, e in enumerate(evs, start=1):
        ct = (e.get("commitment_type") or "?").strip()
        ew = float(e["w"]) if "w" in e else 0.0
        ev = float(e["v"]) if "v" in e else 0.0
        lines.append(
            f"  Step {i}: (X={float(e['x']):.2f}, Y={float(e['y']):.2f}, Z={float(e['z']):.2f}, "
            f"W={ew:.2f}, V={ev:.2f}) · stance={ct}"
        )
    return (
        "\n\nOngoing thread: recent listener-orientation path in order "
        "(X=self vs world, Y=familiar vs new, Z=doing vs reflecting, W=abstract vs concrete, "
        "V=collaborative vs autonomous; each in [-1,1]). "
        "Use this for continuity—the new line should land near this trajectory unless it clearly pivots.\n"
        + "\n".join(lines)
        + "\nIf the new line deliberately changes subject, you may jump; otherwise favor smooth continuation.\n"
    )
