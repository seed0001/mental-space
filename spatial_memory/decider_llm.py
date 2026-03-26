from __future__ import annotations

from typing import Sequence

from spatial_memory.models import CommitmentType, Decision, MemoryNode, NeighborhoodStats
from spatial_memory.ollama_client import chat, parse_json_loose


_REFINE_SYSTEM = """You audit a spatial-memory commitment decision. You receive numeric inspection summaries and a proposed commitment.
You may KEEP the proposed type or CHANGE it only if the evidence clearly warrants it.
Valid commitment_type values: recognition, deepening, bridging, founding.

Rules you must respect:
- If internal_conflict_caution is true, you must NOT output recognition. Prefer deepening or founding.
- If density is high but coherence is low, that is contested ground — never high-confidence recognition.
- Bridging requires evidence that multiple distinct memory regions (spatially or semantically) both resonate.

Reply with ONLY JSON:
{"commitment_type":"<string>","confidence":<float 0-1>,"rationale":"<one or two sentences>"}"""


def maybe_refine_with_llm(
    raw_message: str,
    neighborhood: NeighborhoodStats,
    decision: Decision,
    *,
    per_node_resonance: dict[str, float],
) -> Decision:
    """Optional second pass; returns a copy of decision if LLM fails or is disabled."""
    node_lines = _summarize_nodes(neighborhood.nodes, per_node_resonance)
    user_payload = f"""User message (truncated):
{raw_message[:2000]}

Inspection:
- density: {decision.inspection_density:.4f}
- coherence: {decision.inspection_coherence:.4f}
- coherence_pairs_used: {neighborhood.coherence_pairs_used}
- resonance_max: {decision.resonance_max:.4f}
- multi_region_resonance: {decision.multi_region_resonance}
- proposed_rule: {decision.rule_id}
- proposed_commitment: {decision.commitment_type.value}
- proposed_confidence: {decision.confidence_level:.4f}
- internal_conflict_caution: {decision.caution_internal_conflict}

Top neighborhood nodes (id, res, xyz):
{node_lines}

Proposed rationale: {decision.rationale}
"""
    try:
        raw = chat(_REFINE_SYSTEM, user_payload, temperature=0.0, json_mode=True)
        data = parse_json_loose(raw)
        ctype = CommitmentType(str(data["commitment_type"]).strip().lower())
        conf = float(data["confidence"])
        rat = str(data.get("rationale", "")).strip() or decision.rationale
        if decision.caution_internal_conflict and ctype == CommitmentType.RECOGNITION:
            return decision
        conf = max(0.06, min(0.98, conf))
        return Decision(
            commitment_type=ctype,
            memory_to_inject=list(decision.memory_to_inject),
            confidence_level=conf,
            caution_internal_conflict=decision.caution_internal_conflict,
            activated_node_ids=list(decision.activated_node_ids),
            rationale=f"{decision.rationale} [LLM audit: {rat}]",
            rule_id=f"{decision.rule_id}+llm",
            inspection_density=decision.inspection_density,
            inspection_coherence=decision.inspection_coherence,
            resonance_max=decision.resonance_max,
            multi_region_resonance=decision.multi_region_resonance,
        )
    except Exception:
        return decision


def _summarize_nodes(nodes: Sequence[MemoryNode], per: dict[str, float], k: int = 8) -> str:
    ranked = sorted(nodes, key=lambda n: per.get(n.id, 0.0), reverse=True)
    lines = []
    for n in ranked[:k]:
        lines.append(
            f"- {n.id[:8]} res={per.get(n.id, 0):.3f} xyz=({n.x:.2f},{n.y:.2f},{n.z:.2f}) contested={n.contested}"
        )
    return "\n".join(lines) if lines else "(none)"
