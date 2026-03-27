"""
Orientation momentum: blend new classifier output with the previous turn's position,
and optional scene-event trail text for the classifier (closed loop with scene_events).
"""

from __future__ import annotations

from spatial_memory import store


def previous_xyz_for_momentum(
    *,
    scope: str,
    active_scene_id: str | None,
    db_path: str | None = None,
) -> tuple[float, float, float] | None:
    scope_l = (scope or "scene").strip().lower()
    if scope_l not in ("global", "scene"):
        scope_l = "scene"
    if scope_l == "global":
        return store.last_global_scene_event_xyz(db_path=db_path)
    if active_scene_id:
        return store.last_scene_event_xyz(active_scene_id, db_path=db_path)
    return None


def blend_with_previous(
    x: float,
    y: float,
    z: float,
    prev: tuple[float, float, float] | None,
    prev_weight: float,
) -> tuple[float, float, float]:
    if prev is None or prev_weight <= 0:
        return x, y, z
    w = max(0.0, min(1.0, float(prev_weight)))
    return (
        (1.0 - w) * x + w * prev[0],
        (1.0 - w) * y + w * prev[1],
        (1.0 - w) * z + w * prev[2],
    )


def classifier_scene_trail_suffix(scene_id: str, *, max_events: int, db_path: str | None = None) -> str:
    """Append to classifier system prompt; uses scene_events xyz + commitment (internal only)."""
    evs = store.recent_scene_events(scene_id, limit=max(1, max_events), db_path=db_path)
    if not evs:
        return ""
    lines: list[str] = []
    for i, e in enumerate(evs, start=1):
        ct = (e.get("commitment_type") or "?").strip()
        lines.append(
            f"  Step {i}: (X={float(e['x']):.2f}, Y={float(e['y']):.2f}, Z={float(e['z']):.2f}) · stance={ct}"
        )
    return (
        "\n\nOngoing thread: recent listener-orientation path in order "
        "(X=self vs world, Y=familiar vs new, Z=doing vs reflecting; each in [-1,1]). "
        "Use this for continuity—the new line should land near this trajectory unless it clearly pivots.\n"
        + "\n".join(lines)
        + "\nIf the new line deliberately changes subject, you may jump; otherwise favor smooth continuation.\n"
    )
