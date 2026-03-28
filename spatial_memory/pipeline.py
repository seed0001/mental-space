from __future__ import annotations

from dataclasses import dataclass

from spatial_memory.classifier import classify_message
from spatial_memory.commit import commit_to_memory
from spatial_memory.config import (
    COMMITMENT_USE_LLM,
    ORIENTATION_CLASSIFIER_SCENE_TRAIL,
    ORIENTATION_MOMENTUM_PREV_WEIGHT,
    ORIENTATION_MOMENTUM_SCOPE,
    ORIENTATION_SCENE_TRAIL_EVENTS,
)
from spatial_memory.deep_remember import (
    format_full_field_digest,
    is_deep_remember_trigger,
    weave_memory_field,
    weave_result_to_dict,
)
from spatial_memory.decider import decide_commitment_type, format_global_snippets
from spatial_memory.inspector import compute_resonance, global_memory_snippets, inspect_region
from spatial_memory.lifecycle import apply_post_turn
from spatial_memory.models import CommitmentType, Decision, Orientation
from spatial_memory.inference_options import ResponseInferenceOptions
from spatial_memory.responder import generate_response, generate_response_stream
from spatial_memory.orientation_context import (
    blend_latent_with_previous,
    classifier_scene_trail_suffix,
    previous_xyzwv_for_momentum,
)
from spatial_memory.scene import resolve_active_scene
from spatial_memory.space_shape import constrain_orientation_full
from spatial_memory import store


@dataclass
class PipelineResult:
    response: str
    orientation: Orientation
    commitment_type: CommitmentType
    decision: Decision
    coordinate: tuple[float, float, float, float, float]
    deep_remember: bool = False
    consolidation: dict | None = None


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
            f"Thread in this conversation — how it started: {start[:180]} · "
            f"strongest moment: {peak_msg[:180]} · latest: {end[:180]}"
        )
        out.append(snippet)
        if len(out) >= 4:
            break
    return out


def _prepare_turn(raw_message: str, db_path: str | None = None):
    store.init_db(db_path)
    deep = is_deep_remember_trigger(raw_message)
    weave = weave_memory_field(db_path=db_path) if deep else None
    scene_resolution = resolve_active_scene(raw_message, db_path=db_path, bypass_merge=deep)
    active = store.get_active_scene(db_path=db_path)
    active_sid = str(active["id"]) if active else None

    trail_suffix = ""
    prev5: tuple[float, float, float, float, float] | None = None
    if not deep:
        if ORIENTATION_CLASSIFIER_SCENE_TRAIL and active_sid:
            trail_suffix = classifier_scene_trail_suffix(
                active_sid,
                max_events=ORIENTATION_SCENE_TRAIL_EVENTS,
                db_path=db_path,
            )
        if ORIENTATION_MOMENTUM_PREV_WEIGHT > 0:
            prev5 = previous_xyzwv_for_momentum(
                scope=ORIENTATION_MOMENTUM_SCOPE,
                active_scene_id=active_sid,
                db_path=db_path,
            )

    orientation = classify_message(raw_message, extra_system_suffix=trail_suffix)
    blended = blend_latent_with_previous(
        orientation.as_xyzwv(),
        prev5,
        ORIENTATION_MOMENTUM_PREV_WEIGHT,
    )
    bx, by, bz, bw, bv = blended
    orientation = Orientation(
        self_other=bx,
        known_unknown=by,
        active_contemplative=bz,
        abstract_concrete=bw,
        collaborative_autonomous=bv,
        classifier_prompt_version=orientation.classifier_prompt_version,
    )
    x, y, z, w, v = constrain_orientation_full(bx, by, bz, bw, bv)
    neighborhood = inspect_region(x, y, z, w=w, v=v, db_path=db_path)
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
            f"Continuing the same thread ({scene_resolution.reason}); stay consistent with what came before."
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
    if deep:
        decision.memory_to_inject = format_full_field_digest(db_path=db_path)
        if scene_snippets:
            decision.memory_to_inject = scene_snippets[:2] + decision.memory_to_inject
        decision.rule_id = f"deep_remember:{decision.rule_id}"
        decision.rationale = (
            "They asked you to think it through; more of what you might remember was brought forward for this reply."
        )
    elif scene_snippets:
        decision.memory_to_inject = scene_snippets + decision.memory_to_inject[:2]
    return orientation, (x, y, z, w, v), neighborhood, decision, per_res, msg_vec, scene_resolution, deep, weave


def process_message(
    raw_message: str,
    *,
    db_path: str | None = None,
    inference: ResponseInferenceOptions | None = None,
) -> PipelineResult:
    """
    encounter → orient → arrive / inspect → assess → decide → respond → commit → consolidate
    """
    orientation, coord, neighborhood, decision, per_res, _msg_vec, scene_resolution, deep, weave = _prepare_turn(
        raw_message, db_path
    )
    x, y, z, w, v = coord
    response = generate_response(
        raw_message,
        decision.memory_to_inject,
        decision.confidence_level,
        decision.commitment_type,
        decision.caution_internal_conflict,
        inference=inference,
        deep_memory_scan=deep,
    )
    primary = commit_to_memory(
        raw_message,
        response,
        x,
        y,
        z,
        w,
        v,
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
            w=primary.w,
            v=primary.v,
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
        deep_remember=deep,
        consolidation=weave_result_to_dict(weave) if deep and weave is not None else None,
    )


def process_message_stream(
    raw_message: str,
    *,
    db_path: str | None = None,
    inference: ResponseInferenceOptions | None = None,
):
    """Same pipeline; final generation streams tokens as dict events (NDJSON lines)."""
    orientation, coord, neighborhood, decision, per_res, _msg_vec, scene_resolution, deep, weave = _prepare_turn(
        raw_message, db_path
    )
    x, y, z, w, v = coord
    meta: dict = {
        "x": x,
        "y": y,
        "z": z,
        "w": w,
        "v": v,
        "commitment": decision.commitment_type.value,
        "confidence": decision.confidence_level,
        "caution": decision.caution_internal_conflict,
        "rule_id": decision.rule_id,
        "rationale": decision.rationale,
        "density": decision.inspection_density,
        "coherence": decision.inspection_coherence,
        "resonance_max": decision.resonance_max,
        "deep_remember": deep,
    }
    if deep and weave is not None:
        meta["consolidation"] = weave_result_to_dict(weave)
    yield {"event": "meta", "data": meta}
    parts: list[str] = []
    for token in generate_response_stream(
        raw_message,
        decision.memory_to_inject,
        decision.confidence_level,
        decision.commitment_type,
        decision.caution_internal_conflict,
        inference=inference,
        deep_memory_scan=deep,
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
        w,
        v,
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
            w=primary.w,
            v=primary.v,
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
    done_data: dict = {
        "reply": response,
        "x": x,
        "y": y,
        "z": z,
        "w": w,
        "v": v,
        "commitment": decision.commitment_type.value,
        "confidence": decision.confidence_level,
        "caution": decision.caution_internal_conflict,
        "rule_id": decision.rule_id,
        "rationale": decision.rationale,
        "deep_remember": deep,
    }
    if deep and weave is not None:
        done_data["consolidation"] = weave_result_to_dict(weave)
    yield {"event": "done", "data": done_data}
