from __future__ import annotations

import math
from typing import Sequence

from spatial_memory.models import CommitmentType, Decision, MemoryNode, NeighborhoodStats

# Calibrated for embedding cosine on "understanding" fields (same model family as chat).
DENSITY_HIGH = 0.32
DENSITY_LOW = 0.11
DENSITY_MODERATE_LO = 0.12
DENSITY_MODERATE_HI = 0.36
COHERENCE_HIGH = 0.52
COHERENCE_LOW = 0.34
COHERENCE_MODERATE = 0.40
RESONANCE_HIGH = 0.62
RESONANCE_MODERATE = 0.38
RESONANCE_LOW = 0.28
RESONANCE_BRIDGE_FLOOR = 0.30
MIN_SPATIAL_SPREAD = 0.38


def _spatial_spread(nodes: Sequence[MemoryNode], per_node_res: dict[str, float]) -> float:
    active = [n for n in nodes if per_node_res.get(n.id, 0) >= RESONANCE_BRIDGE_FLOOR]
    if len(active) < 2:
        return 0.0
    best = 0.0
    for i in range(len(active)):
        for j in range(i + 1, len(active)):
            a, b = active[i], active[j]
            d = math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)
            best = max(best, d)
    return best


def _multi_region_resonance(
    nodes: Sequence[MemoryNode],
    per_node_res: dict[str, float],
) -> bool:
    """
    Resonance to multiple regions: split neighborhood along median x; both halves need
    at least one node above threshold and centroids must be meaningfully separated.
    """
    if len(nodes) < 3:
        return False
    thr = RESONANCE_BRIDGE_FLOOR
    sorted_n = sorted(nodes, key=lambda n: n.x)
    mid = max(1, len(sorted_n) // 2)
    A, B = sorted_n[:mid], sorted_n[mid:]
    if not A or not B:
        return False

    def strong_half(half: list[MemoryNode]) -> bool:
        return any(per_node_res.get(n.id, 0) >= thr for n in half)

    if not (strong_half(A) and strong_half(B)):
        return False

    def centroid(half: list[MemoryNode]) -> tuple[float, float, float]:
        sx = sum(n.x for n in half) / len(half)
        sy = sum(n.y for n in half) / len(half)
        sz = sum(n.z for n in half) / len(half)
        return sx, sy, sz

    ca, cb = centroid(A), centroid(B)
    dist = math.sqrt((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2 + (ca[2] - cb[2]) ** 2)
    return dist >= 0.22


def _pick_activated(nodes: Sequence[MemoryNode], per: dict[str, float], k: int = 6) -> list[str]:
    ranked = sorted(per.items(), key=lambda kv: kv[1], reverse=True)
    return [nid for nid, s in ranked[:k] if s > 0.22]


def _memory_snippets(nodes: Sequence[MemoryNode], per: dict[str, float], k: int = 5) -> list[str]:
    ranked = sorted(nodes, key=lambda n: per.get(n.id, 0.0), reverse=True)
    out: list[str] = []
    floor = 0.12
    for n in ranked[:k]:
        if per.get(n.id, 0) < floor:
            continue
        w = _weight_snippet(n, per.get(n.id, 0))
        out.append(f"[memory {n.id[:8]}… w={w:.2f}] {n.understanding[:1400]}")
    return out


def _weight_snippet(n: MemoryNode, res: float) -> float:
    return res * max(0.1, n.current_relevance) * max(0.1, n.certainty)


def format_global_snippets(pairs: list[tuple[MemoryNode, float]], k: int = 5) -> list[str]:
    out: list[str] = []
    for n, s in pairs[:k]:
        out.append(f"[memory {n.id[:8]}… sim={s:.2f}] {n.understanding[:1400]}")
    return out


def decide_commitment_type(
    neighborhood: NeighborhoodStats,
    resonance_max: float,
    per_node_resonance: dict[str, float],
) -> Decision:
    """
    Decision framework (conscious-oriented, rule-first):
    - High density + low coherence → internal conflict / caution; never high-confidence recognition.
    - High density + high coherence + high resonance → recognition.
    - High density + high coherence + moderate resonance → deepening.
    - Moderate density + multi-region resonance → bridging.
    - Low density + low resonance → founding.
    """
    nodes = neighborhood.nodes
    density = neighborhood.density
    coherence = neighborhood.coherence
    spread = _spatial_spread(nodes, per_node_resonance)
    multi = _multi_region_resonance(nodes, per_node_resonance)

    fragmented = density >= DENSITY_HIGH and coherence < COHERENCE_LOW
    multi_bridge_cue = (
        multi
        and DENSITY_MODERATE_LO <= density <= DENSITY_MODERATE_HI + 0.12
        and spread >= MIN_SPATIAL_SPREAD
    )
    alt_bridge = (
        len([n for n in nodes if per_node_resonance.get(n.id, 0) >= RESONANCE_BRIDGE_FLOOR]) >= 2
        and spread >= MIN_SPATIAL_SPREAD
        and DENSITY_MODERATE_LO <= density < DENSITY_HIGH
        and coherence >= COHERENCE_MODERATE
    )

    ctype: CommitmentType
    conf: float
    rule_id: str
    rationale: str

    if fragmented:
        if resonance_max >= RESONANCE_MODERATE:
            ctype = CommitmentType.DEEPENING
            conf = 0.42 + 0.28 * resonance_max
            rule_id = "fragmented_deepen"
            rationale = (
                "Dense neighborhood but low coherence between interpretations — likely contradictory "
                "or loosely related memories; deepen cautiously instead of claiming full recognition."
            )
        else:
            ctype = CommitmentType.FOUNDING
            conf = 0.28 + 0.35 * resonance_max
            rule_id = "fragmented_found"
            rationale = (
                "Dense yet incoherent region with weak resonance — treat as uncertain ground; "
                "found or extend tentatively rather than asserting familiarity."
            )
    elif density >= DENSITY_HIGH and coherence >= COHERENCE_HIGH and resonance_max >= RESONANCE_HIGH:
        ctype = CommitmentType.RECOGNITION
        conf = min(0.96, 0.68 + 0.22 * resonance_max)
        rule_id = "recognition"
        rationale = "High reinforcement-weighted density, coherent local field, strong resonance — re-standing on known ground."
    elif density >= DENSITY_HIGH and coherence >= COHERENCE_HIGH and resonance_max >= RESONANCE_MODERATE:
        ctype = CommitmentType.DEEPENING
        conf = min(0.9, 0.52 + 0.32 * resonance_max)
        rule_id = "deepening"
        rationale = "Familiar, coherent region; message adds nuance or extension relative to prior understanding."
    elif (multi_bridge_cue or alt_bridge) and resonance_max >= RESONANCE_LOW:
        ctype = CommitmentType.BRIDGING
        conf = 0.5 + 0.22 * resonance_max
        rule_id = "bridging"
        rationale = (
            "Input resonates across spatially separated subregions — commitment is linking disparate "
            "patches of memory rather than a single cluster."
        )
    elif density <= DENSITY_LOW and resonance_max < RESONANCE_MODERATE:
        ctype = CommitmentType.FOUNDING
        conf = 0.3 + 0.38 * resonance_max
        rule_id = "founding_sparse"
        rationale = "Sparse field and weak resonance — open territory; found a new node with low initial confidence."
    elif resonance_max >= RESONANCE_MODERATE and not fragmented:
        ctype = CommitmentType.DEEPENING
        conf = 0.48 + 0.3 * resonance_max
        rule_id = "deepening_fallback"
        rationale = "Moderate grounding with usable resonance — extend existing structures where they fit."
    else:
        ctype = CommitmentType.FOUNDING
        conf = 0.36 + 0.32 * resonance_max
        rule_id = "founding_fallback"
        rationale = "No strong match to a stable cluster — default to founding while staying honest about uncertainty."

    caution = fragmented
    if caution:
        conf *= 0.72

    snippets = _memory_snippets(nodes, per_node_resonance)
    activated = _pick_activated(nodes, per_node_resonance)

    return Decision(
        commitment_type=ctype,
        memory_to_inject=snippets,
        confidence_level=max(0.06, min(0.98, conf)),
        caution_internal_conflict=caution,
        activated_node_ids=activated,
        rationale=rationale,
        rule_id=rule_id,
        inspection_density=density,
        inspection_coherence=coherence,
        resonance_max=resonance_max,
        multi_region_resonance=multi,
    )
