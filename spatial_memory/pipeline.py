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
from spatial_memory import store


@dataclass
class PipelineResult:
    response: str
    orientation: Orientation
    commitment_type: CommitmentType
    decision: Decision
    coordinate: tuple[float, float, float]


def _prepare_turn(raw_message: str, db_path: str | None = None):
    store.init_db(db_path)
    orientation = classify_message(raw_message)
    x, y, z = orientation.as_tuple()
    neighborhood = inspect_region(x, y, z, db_path=db_path)
    res_max, per_res, msg_vec = compute_resonance(raw_message, neighborhood.nodes)
    decision = decide_commitment_type(neighborhood, res_max, per_res)
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
    return orientation, (x, y, z), neighborhood, decision, per_res, msg_vec


def process_message(raw_message: str, *, db_path: str | None = None) -> PipelineResult:
    """
    encounter → orient → arrive / inspect → assess → decide → respond → commit → consolidate
    """
    orientation, coord, neighborhood, decision, per_res, _msg_vec = _prepare_turn(raw_message, db_path)
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
    orientation, coord, neighborhood, decision, per_res, _msg_vec = _prepare_turn(raw_message, db_path)
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
