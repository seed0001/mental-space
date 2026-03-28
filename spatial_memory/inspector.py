from __future__ import annotations

import json
from typing import Sequence

from spatial_memory.config import (
    INITIAL_RADIUS,
    MIN_NODES_FOR_DENSE,
    RADIUS_EXPAND,
    RADIUS_EXPAND_MAX,
)
from spatial_memory.math_util import cosine_similarity
from spatial_memory.models import LinkType, MemoryNode, NeighborhoodStats
from spatial_memory.ollama_client import embed
from spatial_memory import store


def _vec_from_node(n: MemoryNode) -> list[float] | None:
    if not n.embedding_json:
        return None
    try:
        return json.loads(n.embedding_json)
    except json.JSONDecodeError:
        return None


def _ensure_embedding(text: str, existing: list[float] | None) -> list[float]:
    if existing is not None:
        return existing
    return embed(text)


def _has_tension_link(a: MemoryNode, b: MemoryNode) -> bool:
    for L in a.links:
        if L.target_id == b.id and L.link_type == LinkType.TENSION:
            return True
    for L in b.links:
        if L.target_id == a.id and L.link_type == LinkType.TENSION:
            return True
    return False


def _node_effective_weight(n: MemoryNode) -> float:
    return (
        float(n.reinforcement_count + 1)
        * max(0.05, n.confidence)
        * max(0.08, n.current_relevance)
        * max(0.1, n.certainty)
    )


def inspect_region(
    x: float,
    y: float,
    z: float,
    initial_radius: float | None = None,
    w: float = 0.0,
    v: float = 0.0,
    db_path: str | None = None,
) -> NeighborhoodStats:
    """Arrive: adaptive radius (narrow → wider → widest) until the field is populated enough."""
    r0 = initial_radius if initial_radius is not None else INITIAL_RADIUS
    pairs = store.nodes_within_radius(x, y, z, r0, w=w, v=v, db_path=db_path)
    if len(pairs) < MIN_NODES_FOR_DENSE:
        pairs = store.nodes_within_radius(x, y, z, RADIUS_EXPAND, w=w, v=v, db_path=db_path)
    if len(pairs) < MIN_NODES_FOR_DENSE:
        pairs = store.nodes_within_radius(x, y, z, RADIUS_EXPAND_MAX, w=w, v=v, db_path=db_path)

    nodes = [p[0] for p in pairs]
    dist_sq = {n.id: d for n, d in pairs}

    density = _compute_density(nodes)
    coherence, pairs_used = _compute_coherence(nodes)

    return NeighborhoodStats(
        nodes=nodes,
        density=density,
        coherence=coherence,
        dist_sq=dist_sq,
        coherence_pairs_used=pairs_used,
    )


def _compute_density(nodes: Sequence[MemoryNode]) -> float:
    if not nodes:
        return 0.0
    weight = sum(_node_effective_weight(n) for n in nodes)
    return min(1.0, weight / 14.0)


def _compute_coherence(nodes: Sequence[MemoryNode]) -> tuple[float, int]:
    """Pairwise cosine on understanding embeddings; skip pairs linked by TENSION."""
    if len(nodes) < 2:
        return 1.0, 0
    vecs: list[list[float]] = []
    for n in nodes:
        v = _vec_from_node(n)
        vecs.append(_ensure_embedding(n.understanding[:8000], v))
    sims: list[float] = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            ni, nj = nodes[i], nodes[j]
            if _has_tension_link(ni, nj):
                continue
            sims.append(cosine_similarity(vecs[i], vecs[j]))
    if not sims:
        return 1.0, 0
    return sum(sims) / len(sims), len(sims)


def compute_resonance(
    raw_message: str,
    nodes: Sequence[MemoryNode],
) -> tuple[float, dict[str, float], list[float]]:
    msg_vec = embed(raw_message[:8000])
    if not nodes:
        return 0.0, {}, msg_vec
    per: dict[str, float] = {}
    for n in nodes:
        nv = _vec_from_node(n)
        nv = _ensure_embedding(n.understanding[:8000], nv)
        per[n.id] = cosine_similarity(msg_vec, nv)
    top = max(per.values()) if per else 0.0
    return top, per, msg_vec


def global_memory_snippets(
    msg_vec: list[float],
    *,
    k: int = 5,
    min_sim: float = 0.16,
    db_path: str | None = None,
) -> list[tuple[MemoryNode, float]]:
    nodes = store.all_nodes(db_path=db_path)
    scored: list[tuple[MemoryNode, float]] = []
    for n in nodes:
        nv = _vec_from_node(n)
        nv = _ensure_embedding(n.understanding[:8000], nv)
        s = cosine_similarity(msg_vec, nv)
        if s >= min_sim:
            scored.append((n, s))
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored[:k]
