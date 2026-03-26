from __future__ import annotations

from dataclasses import dataclass

from spatial_memory.classifier import classify_message
from spatial_memory.commit import commit_to_memory
from spatial_memory.config import COMMITMENT_USE_LLM
from spatial_memory.decider import decide_commitment_type, format_global_snippets
from spatial_memory.inspector import compute_resonance, global_memory_snippets, inspect_region
from spatial_memory.lifecycle import apply_post_turn
from spatial_memory.models import CommitmentType, Decision, Orientation
from spatial_memory.responder import generate_response, generate_response_stream
from spatial_memory.scene import resolve_active_scene
from spatial_memory.space_shape import constrain_to_bean_space
from spatial_memory import store


@dataclass
class PipelineResult:
    response: str
    orientation: Orientation
    commitment_type: CommitmentType
    decision: Decision
    coordinate: tuple[float, float, float]


def _scene_state(events: list[dict]) -> str:
    if not events:
        return "active"
    last = (events[-1].get("user_message") or "").lower()
    if any(k in last for k in ("resolved", "thanks", "thank you", "sorry", "apology", "we're good", "all good")):
        return "resolved"
    return "active"


def _scene_trajectory(events: list[dict]) -> str:
    if not events:
        return "unknown"
    start = events[0].get("commitment_type", "founding")
    end = events[-1].get("commitment_type", "deepening")
    return f"{start} -> {end}"


def _scene_memory_snippets(nodes, per_res, *, db_path: str | None = None) -> list[str]:
    ranked = sorted(nodes, key=lambda n: per_res.get(n.id, 0.0), reverse=True)
    out: list[str] = []
    seen_scenes: set[str] = set()
    for n in ranked:
        if per_res.get(n.id, 0.0) < 0.12:
            continue
        scene = store.get_scene_by_node_id(n.id, db_path=db_path)
        if not scene:
            continue
        sid = str(scene["id"])
        if sid in seen_scenes:
            continue
        seen_scenes.add(sid)
        events = store.recent_scene_events(sid, limit=12, db_path=db_path)
        if not events:
            continue
        start = (events[0].get("user_message") or "").strip().replace("\n", " ")
        end = (events[-1].get("user_message") or "").strip().replace("\n", " ")
        peak = max(events, key=lambda e: float(e.get("confidence", 0.0)))
        peak_msg = (peak.get("user_message") or "").strip().replace("\n", " ")
        snippet = (
            f"[scene {sid[:8]}… continuity={per_res.get(n.id, 0.0):.2f}] "
            f"start: {start[:180]} | peak: {peak_msg[:180]} | end: {end[:180]} | "
            f"trajectory: {_scene_trajectory(events)} | state: {_scene_state(events)} | "
            f"events={len(events)}"
        )
        out.append(snippet)
        if len(out) >= 4:
            break
    return out


def _prepare_turn(raw_message: str, db_path: str | None = None):
    store.init_db(db_path)
    scene_resolution = resolve_active_scene(raw_message, db_path=db_path)
    orientation = classify_message(raw_message)
    x, y, z = constrain_to_bean_space(*orientation.as_tuple())
    neighborhood = inspect_region(x, y, z, db_path=db_path)
    if scene_resolution.should_merge and scene_resolution.node_id:
        active_node = store.get_node(scene_resolution.node_id, db_path=db_path)
        if active_node:
            neighborhood.nodes = [active_node]
            neighborhood.dist_sq = {active_node.id: 0.0}
            neighborhood.density = min(1.0, max(0.18, neighborhood.density))
    res_max, per_res, msg_vec = compute_resonance(raw_message, neighborhood.nodes)
    decision = decide_commitment_type(neighborhood, res_max, per_res)
    if scene_resolution.should_merge:
        decision.commitment_type = CommitmentType.DEEPENING
        decision.rule_id = f"scene_merge:{decision.rule_id}"
        decision.rationale = (
            f"Continuing active scene ({scene_resolution.reason}); update timeline of existing scene node."
        )
        if scene_resolution.node_id:
            decision.activated_node_ids = [scene_resolution.node_id]
    if COMMITMENT_USE_LLM:
        from spatial_memory.decider_llm import maybe_refine_with_llm

        decision = maybe_refine_with_llm(
            raw_message,
            neighborhood,
            decision,
            per_node_resonance=per_res,
        )
    if not decision.memory_to_inject:
        gh = global_memory_snippets(msg_vec, db_path=db_path)
        decision.memory_to_inject = format_global_snippets(gh)
    scene_snippets = _scene_memory_snippets(neighborhood.nodes, per_res, db_path=db_path)
    if scene_snippets:
        decision.memory_to_inject = scene_snippets + decision.memory_to_inject[:2]
    return orientation, (x, y, z), neighborhood, decision, per_res, msg_vec, scene_resolution


def process_message(raw_message: str, *, db_path: str | None = None) -> PipelineResult:
    """
    encounter → orient → arrive / inspect → assess → decide → respond → commit → consolidate
    """
    orientation, coord, neighborhood, decision, per_res, _msg_vec, scene_resolution = _prepare_turn(raw_message, db_path)
    x, y, z = coord
    response = generate_response(
        raw_message,
        decision.memory_to_inject,
        decision.confidence_level,
        decision.commitment_type,
        decision.caution_internal_conflict,
        decision.rationale,
    )
    primary = commit_to_memory(
        raw_message,
        response,
        x,
        y,
        z,
        orientation,
        decision,
        neighborhood.nodes,
        per_res,
        force_target_node_id=scene_resolution.node_id if scene_resolution.should_merge else None,
        db_path=db_path,
    )
    if primary:
        if scene_resolution.should_merge and scene_resolution.scene_id:
            store.touch_scene(
                scene_resolution.scene_id,
                continuity_score=scene_resolution.continuity_score,
                db_path=db_path,
            )
            sid = scene_resolution.scene_id
        else:
            sid = store.new_id()
            store.create_scene(sid, primary.id, db_path=db_path)
            store.touch_scene(sid, continuity_score=scene_resolution.continuity_score, db_path=db_path)
        store.append_scene_event(
            sid,
            user_message=raw_message,
            assistant_response=response,
            x=primary.x,
            y=primary.y,
            z=primary.z,
            commitment_type=decision.commitment_type.value,
            confidence=decision.confidence_level,
            caution=decision.caution_internal_conflict,
            db_path=db_path,
        )
    apply_post_turn(
        neighborhood.nodes,
        decision.activated_node_ids,
        decision,
        committed_node_id=primary.id if primary else None,
        db_path=db_path,
    )
    return PipelineResult(
        response=response,
        orientation=orientation,
        commitment_type=decision.commitment_type,
        decision=decision,
        coordinate=coord,
    )


def process_message_stream(raw_message: str, *, db_path: str | None = None):
    """Same pipeline; final generation streams tokens as dict events (NDJSON lines)."""
    orientation, coord, neighborhood, decision, per_res, _msg_vec, scene_resolution = _prepare_turn(raw_message, db_path)
    x, y, z = coord
    yield {
        "event": "meta",
        "data": {
            "x": x,
            "y": y,
            "z": z,
            "commitment": decision.commitment_type.value,
            "confidence": decision.confidence_level,
            "caution": decision.caution_internal_conflict,
            "rule_id": decision.rule_id,
            "rationale": decision.rationale,
            "density": decision.inspection_density,
            "coherence": decision.inspection_coherence,
            "resonance_max": decision.resonance_max,
        },
    }
    parts: list[str] = []
    for token in generate_response_stream(
        raw_message,
        decision.memory_to_inject,
        decision.confidence_level,
        decision.commitment_type,
        decision.caution_internal_conflict,
        decision.rationale,
    ):
        parts.append(token)
        yield {"event": "token", "text": token}
    response = "".join(parts)
    primary = commit_to_memory(
        raw_message,
        response,
        x,
        y,
        z,
        orientation,
        decision,
        neighborhood.nodes,
        per_res,
        force_target_node_id=scene_resolution.node_id if scene_resolution.should_merge else None,
        db_path=db_path,
    )
    if primary:
        if scene_resolution.should_merge and scene_resolution.scene_id:
            store.touch_scene(
                scene_resolution.scene_id,
                continuity_score=scene_resolution.continuity_score,
                db_path=db_path,
            )
            sid = scene_resolution.scene_id
        else:
            sid = store.new_id()
            store.create_scene(sid, primary.id, db_path=db_path)
            store.touch_scene(sid, continuity_score=scene_resolution.continuity_score, db_path=db_path)
        store.append_scene_event(
            sid,
            user_message=raw_message,
            assistant_response=response,
            x=primary.x,
            y=primary.y,
            z=primary.z,
            commitment_type=decision.commitment_type.value,
            confidence=decision.confidence_level,
            caution=decision.caution_internal_conflict,
            db_path=db_path,
        )
    apply_post_turn(
        neighborhood.nodes,
        decision.activated_node_ids,
        decision,
        committed_node_id=primary.id if primary else None,
        db_path=db_path,
    )
    yield {
        "event": "done",
        "data": {
            "reply": response,
            "x": x,
            "y": y,
            "z": z,
            "commitment": decision.commitment_type.value,
            "confidence": decision.confidence_level,
            "caution": decision.caution_internal_conflict,
            "rule_id": decision.rule_id,
            "rationale": decision.rationale,
        },
    }
