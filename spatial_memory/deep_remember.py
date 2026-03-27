"""
Deep-remember / full-field scan: trigger phrases, graph weave, and memory digest injection.

When the user asks the system to "think deep" or consolidate memory, we (1) run a pass over
stored nodes to add bridge/reinforcement links between semantically similar pairs, and
(2) inject a wide sample of the entire graph into the reply so retrieval is not limited
to the local neighborhood alone.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from spatial_memory.commit import _add_or_strengthen_link
from spatial_memory.math_util import cosine_similarity
from spatial_memory.models import LinkType, MemoryNode
from spatial_memory.ollama_client import embed
from spatial_memory import store

# Pairwise similarity floor for weaving (embedding space).
PAIR_SIM_MIN = float(os.environ.get("DEEP_REMEMBER_SIM_MIN", "0.38"))
# Squared Euclidean distance in [-1,1]³; above this, similar nodes get BRIDGE; below, REINFORCEMENT.
SPATIAL_DIST_SQ_BRIDGE = float(os.environ.get("DEEP_REMEMBER_BRIDGE_DIST_SQ", "0.22"))
MAX_LINKS_PER_PASS = int(os.environ.get("DEEP_REMEMBER_MAX_LINKS", "72"))
MAX_NODES_FOR_SCAN = int(os.environ.get("DEEP_REMEMBER_MAX_NODES", "240"))
# Digest: how many nodes and max chars per snippet in full-field injection.
DIGEST_MAX_NODES = int(os.environ.get("DEEP_REMEMBER_DIGEST_NODES", "42"))
DIGEST_SNIPPET_CHARS = int(os.environ.get("DEEP_REMEMBER_DIGEST_CHARS", "360"))


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def is_deep_remember_trigger(raw_message: str) -> bool:
    """
    Conservative matching: multi-word phrases, or very short standalone commands.
    Avoids firing on normal sentences like "do you remember when we…".
    """
    t = _norm(raw_message)
    if not t:
        return False

    phrases = (
        "think deep",
        "think deeply",
        "deep think",
        "think hard",
        "think about it",
        "think about this",
        "think this through",
        "remember everything",
        "remember all",
        "scan your memory",
        "scan memory",
        "memory weave",
        "weave memory",
        "consolidate memory",
        "full memory",
        "entire memory",
        "everything you know",
        "everything you remember",
        "deep remember",
    )
    if any(p in t for p in phrases):
        return True

    # Standalone or near-standalone (no long question attached).
    short = re.sub(r"[\s.!?,;:]+$", "", t)
    if len(short) <= 22:
        if short in ("remember", "think deep", "think", "recall", "consolidate"):
            return True

    return False


def _vec_from_node(n: MemoryNode) -> list[float] | None:
    if not n.embedding_json:
        return None
    try:
        return json.loads(n.embedding_json)
    except json.JSONDecodeError:
        return None


def _ensure_vec(n: MemoryNode) -> list[float]:
    v = _vec_from_node(n)
    if v is not None:
        return v
    return embed(n.understanding[:8000])


def _has_tension_link(a: MemoryNode, b: MemoryNode) -> bool:
    for L in a.links:
        if L.target_id == b.id and L.link_type == LinkType.TENSION:
            return True
    for L in b.links:
        if L.target_id == a.id and L.link_type == LinkType.TENSION:
            return True
    return False


def _spatial_dist_sq(a: MemoryNode, b: MemoryNode) -> float:
    dx, dy, dz = a.x - b.x, a.y - b.y, a.z - b.z
    return dx * dx + dy * dy + dz * dz


@dataclass
class WeaveResult:
    bridges_added: int
    reinforcements_added: int
    pairs_considered: int
    pairs_above_threshold: int
    skipped_tension: int
    nodes_in_scan: int
    link_budget_hit: bool


def weave_memory_field(*, db_path: str | None = None) -> WeaveResult:
    """
    Scan embeddings for all (capped) nodes; connect similar pairs with BRIDGE (spatially
    separated) or REINFORCEMENT (nearby), strengthening joint retrieval paths.
    """
    store.init_db(db_path)
    nodes = store.all_nodes(db_path=db_path)
    if len(nodes) < 2:
        return WeaveResult(0, 0, 0, 0, 0, len(nodes), False)

    def score(n: MemoryNode) -> float:
        return float(n.current_relevance) * max(0.05, n.confidence) * max(0.5, float(n.reinforcement_count) ** 0.5)

    ranked = sorted(nodes, key=score, reverse=True)
    scan = ranked[:MAX_NODES_FOR_SCAN]
    vecs = [_ensure_vec(n) for n in scan]

    bridges = reinforcements = 0
    skipped_tension = 0
    pairs_considered = 0
    pairs_above = 0
    links_added = 0
    budget_hit = False

    for i in range(len(scan)):
        for j in range(i + 1, len(scan)):
            pairs_considered += 1
            ni, nj = scan[i], scan[j]
            if _has_tension_link(ni, nj):
                skipped_tension += 1
                continue
            sim = cosine_similarity(vecs[i], vecs[j])
            if sim < PAIR_SIM_MIN:
                continue
            pairs_above += 1
            if links_added >= MAX_LINKS_PER_PASS:
                budget_hit = True
                break

            d_sq = _spatial_dist_sq(ni, nj)
            if d_sq >= SPATIAL_DIST_SQ_BRIDGE:
                delta = min(0.42, 0.12 + (sim - PAIR_SIM_MIN) * 1.1)
                _add_or_strengthen_link(ni, nj.id, LinkType.BRIDGE, delta)
                _add_or_strengthen_link(nj, ni.id, LinkType.BRIDGE, delta)
                bridges += 1
            else:
                delta = min(0.38, 0.08 + (sim - PAIR_SIM_MIN) * 0.85)
                _add_or_strengthen_link(ni, nj.id, LinkType.REINFORCEMENT, delta)
                _add_or_strengthen_link(nj, ni.id, LinkType.REINFORCEMENT, delta)
                reinforcements += 1

            ni.current_relevance = min(1.0, ni.current_relevance + 0.04)
            nj.current_relevance = min(1.0, nj.current_relevance + 0.04)
            store.update_node(ni, db_path=db_path)
            store.update_node(nj, db_path=db_path)
            links_added += 1
        if budget_hit:
            break

    return WeaveResult(
        bridges_added=bridges,
        reinforcements_added=reinforcements,
        pairs_considered=pairs_considered,
        pairs_above_threshold=pairs_above,
        skipped_tension=skipped_tension,
        nodes_in_scan=len(scan),
        link_budget_hit=budget_hit,
    )


def format_full_field_digest(*, db_path: str | None = None) -> list[str]:
    """
    Ranked snippets from the entire node set for injection (not neighborhood-limited).
    """
    nodes = store.all_nodes(db_path=db_path)
    if not nodes:
        return ["(Nothing from before yet.)"]

    def rank_key(n: MemoryNode) -> float:
        return float(n.current_relevance) * max(0.05, n.confidence)

    ranked = sorted(nodes, key=rank_key, reverse=True)

    lines: list[str] = []
    for n in ranked[:DIGEST_MAX_NODES]:
        u = (n.understanding or "").strip().replace("\r\n", "\n")
        if len(u) > DIGEST_SNIPPET_CHARS:
            u = u[: DIGEST_SNIPPET_CHARS - 1] + "…"
        lines.append(f"Earlier note:\n{u}")

    return lines


def weave_result_to_dict(w: WeaveResult) -> dict:
    return {
        "bridges_added": w.bridges_added,
        "reinforcements_added": w.reinforcements_added,
        "pairs_considered": w.pairs_considered,
        "pairs_above_threshold": w.pairs_above_threshold,
        "skipped_tension": w.skipped_tension,
        "nodes_in_scan": w.nodes_in_scan,
        "link_budget_hit": w.link_budget_hit,
    }
