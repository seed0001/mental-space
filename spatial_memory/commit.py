from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Sequence

from spatial_memory.math_util import cosine_similarity
from spatial_memory.models import CommitmentType, Decision, LinkType, MemoryLink, MemoryNode, Orientation, SourceType
from spatial_memory.ollama_client import embed
from spatial_memory import store


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dump_emb(vec: list[float]) -> str:
    return json.dumps(vec)


def _vec_from_node(n: MemoryNode) -> list[float] | None:
    if not n.embedding_json:
        return None
    try:
        return json.loads(n.embedding_json)
    except json.JSONDecodeError:
        return None


def _best_node(
    neighborhood: Sequence[MemoryNode],
    per_res: dict[str, float],
) -> MemoryNode | None:
    if not neighborhood:
        return None
    return max(neighborhood, key=lambda n: per_res.get(n.id, 0.0))


def _second_node(
    neighborhood: Sequence[MemoryNode],
    per_res: dict[str, float],
    exclude_id: str,
) -> MemoryNode | None:
    cand = [n for n in neighborhood if n.id != exclude_id]
    if not cand:
        return None
    return max(cand, key=lambda n: per_res.get(n.id, 0.0))


def _add_or_strengthen_link(node: MemoryNode, target_id: str, ltype: LinkType, delta: float) -> None:
    for L in node.links:
        if L.target_id == target_id and L.link_type == ltype:
            L.strength = min(1.0, L.strength + delta)
            return
    node.links.append(MemoryLink(target_id=target_id, link_type=ltype, strength=min(1.0, delta)))


def _tension_divergent_pair(neighborhood: Sequence[MemoryNode]) -> None:
    """When the region is fragmented, mark the most semantically divergent pair with TENSION."""
    nodes = [n for n in neighborhood]
    if len(nodes) < 2:
        return
    vecs: list[list[float]] = []
    for n in nodes:
        v = _vec_from_node(n)
        vecs.append(v if v is not None else embed(n.understanding[:8000]))
    best = None
    best_sim = 2.0
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            s = cosine_similarity(vecs[i], vecs[j])
            if s < best_sim:
                best_sim = s
                best = (nodes[i], nodes[j])
    if best is None or best_sim > 0.48:
        return
    a, b = best
    _add_or_strengthen_link(a, b.id, LinkType.TENSION, 0.38)
    _add_or_strengthen_link(b, a.id, LinkType.TENSION, 0.38)


def commit_to_memory(
    raw_message: str,
    response_text: str,
    x: float,
    y: float,
    z: float,
    orientation: Orientation,
    decision: Decision,
    neighborhood: Sequence[MemoryNode],
    per_node_resonance: dict[str, float],
    db_path: str | None = None,
) -> MemoryNode | None:
    now = _now()
    ctype = decision.commitment_type

    if decision.caution_internal_conflict:
        for n in neighborhood:
            n.contested = True
        _tension_divergent_pair(neighborhood)
        for n in neighborhood:
            store.update_node(n, db_path=db_path)

    if ctype == CommitmentType.RECOGNITION:
        node = _best_node(neighborhood, per_node_resonance)
        if node is None:
            return _found_new(
                raw_message,
                response_text,
                x,
                y,
                z,
                orientation,
                decision,
                per_node_resonance,
                db_path,
            )
        node.reinforcement_count += 1
        node.last_activation = now
        node.confidence = min(0.99, node.confidence + 0.03)
        node.current_relevance = min(1.0, node.current_relevance + 0.1)
        node.commitment_type = CommitmentType.RECOGNITION
        node.certainty = min(0.99, node.certainty + 0.02)
        store.update_node(node, db_path=db_path)
        return node

    if ctype == CommitmentType.DEEPENING:
        node = _best_node(neighborhood, per_node_resonance)
        if node is None:
            return _found_new(
                raw_message,
                response_text,
                x,
                y,
                z,
                orientation,
                decision,
                per_node_resonance,
                db_path,
            )
        merged = (
            node.understanding.strip()
            + "\n---\nUser message:\n"
            + raw_message.strip()
            + "\n\nAgent understanding:\n"
            + response_text.strip()
        )[:6000]
        node.understanding = merged
        node.original_text = raw_message
        node.embedding_json = _dump_emb(embed(merged[:8000]))
        node.reinforcement_count += 1
        node.last_activation = now
        node.confidence = min(0.98, node.confidence + 0.04)
        node.current_relevance = min(1.0, node.current_relevance + 0.12)
        node.commitment_type = CommitmentType.DEEPENING
        node.self_other_score = orientation.self_other
        node.known_unknown_score = orientation.known_unknown
        node.active_contemplative_score = orientation.active_contemplative
        node.orientation_prompt_version = orientation.classifier_prompt_version or node.orientation_prompt_version
        store.update_node(node, db_path=db_path)
        return node

    if ctype == CommitmentType.BRIDGING:
        a = _best_node(neighborhood, per_node_resonance)
        b = _second_node(neighborhood, per_node_resonance, a.id) if a else None
        if a is None or b is None or per_node_resonance.get(b.id, 0) < 0.18:
            return _found_new(
                raw_message,
                response_text,
                x,
                y,
                z,
                orientation,
                decision,
                per_node_resonance,
                db_path,
            )
        _add_or_strengthen_link(a, b.id, LinkType.BRIDGE, 0.22)
        _add_or_strengthen_link(b, a.id, LinkType.BRIDGE, 0.22)
        a.reinforcement_count += 1
        b.reinforcement_count += 1
        a.last_activation = now
        b.last_activation = now
        a.commitment_type = CommitmentType.BRIDGING
        b.commitment_type = CommitmentType.BRIDGING
        store.update_node(a, db_path=db_path)
        store.update_node(b, db_path=db_path)
        return a

    return _found_new(
        raw_message,
        response_text,
        x,
        y,
        z,
        orientation,
        decision,
        per_node_resonance,
        db_path,
    )


def _found_new(
    raw_message: str,
    response_text: str,
    x: float,
    y: float,
    z: float,
    orientation: Orientation,
    decision: Decision,
    per_node_resonance: dict[str, float],
    db_path: str | None = None,
) -> MemoryNode:
    now = _now()
    understanding = (
        f"User message:\n{raw_message.strip()}\n\nAgent understanding (response summary):\n{response_text.strip()}"
    )[:6000]
    vec = embed(understanding[:8000])
    resonance_max = max(per_node_resonance.values()) if per_node_resonance else 0.0
    node = MemoryNode(
        id=store.new_id(),
        original_text=raw_message,
        understanding=understanding,
        x=x,
        y=y,
        z=z,
        self_other_score=orientation.self_other,
        known_unknown_score=orientation.known_unknown,
        active_contemplative_score=orientation.active_contemplative,
        commitment_type=CommitmentType.FOUNDING,
        confidence=max(0.12, decision.confidence_level * 0.85),
        reinforcement_count=1,
        last_activation=now,
        novelty_at_creation=max(0.0, min(1.0, 1.0 - resonance_max)),
        current_relevance=1.0,
        certainty=max(0.1, decision.confidence_level * 0.8),
        contested=decision.caution_internal_conflict,
        source_type=SourceType.EXPERIENCE,
        links=[],
        embedding_json=_dump_emb(vec),
        orientation_prompt_version=orientation.classifier_prompt_version or "",
    )
    store.insert_node(node, db_path=db_path)
    return node
