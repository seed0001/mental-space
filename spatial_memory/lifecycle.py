from __future__ import annotations

from typing import Sequence

from spatial_memory.config import COACTIVATION_REINFORCE_DELTA, MEMORY_DECAY_NEIGHBOR
from spatial_memory.models import Decision, LinkType, MemoryLink, MemoryNode
from spatial_memory import store


def apply_post_turn(
    neighborhood: Sequence[MemoryNode],
    activated_ids: Sequence[str],
    decision: Decision,
    *,
    committed_node_id: str | None = None,
    db_path: str | None = None,
) -> None:
    """
    Consolidation pass: passive decay for neighbors not activated; reinforcement links
    between co-activated nodes (reuse-before-generate strengthens joint retrieval paths).
    """
    act = set(activated_ids)
    if committed_node_id:
        act.add(committed_node_id)
    factor = max(0.0, min(0.2, MEMORY_DECAY_NEIGHBOR))
    for n in neighborhood:
        if n.id in act:
            continue
        n.current_relevance = max(0.04, n.current_relevance * (1.0 - factor))
        store.update_node(n, db_path=db_path)

    if len(activated_ids) >= 2:
        _strengthen_coactivation(activated_ids, neighborhood, db_path=db_path)


def _strengthen_coactivation(
    activated_ids: Sequence[str],
    neighborhood: Sequence[MemoryNode],
    *,
    db_path: str | None,
) -> None:
    by_id = {n.id: n for n in neighborhood}
    # First two in pipeline order are strongest by decider ranking
    a_id, b_id = activated_ids[0], activated_ids[1]
    a, b = by_id.get(a_id), by_id.get(b_id)
    if not a or not b or a_id == b_id:
        return
    _add_reinforcement(a, b.id, COACTIVATION_REINFORCE_DELTA)
    _add_reinforcement(b, a.id, COACTIVATION_REINFORCE_DELTA)
    store.update_node(a, db_path=db_path)
    store.update_node(b, db_path=db_path)


def _add_reinforcement(node: MemoryNode, target_id: str, delta: float) -> None:
    for L in node.links:
        if L.target_id == target_id and L.link_type == LinkType.REINFORCEMENT:
            L.strength = min(1.0, L.strength + delta)
            return
    node.links.append(
        MemoryLink(target_id=target_id, link_type=LinkType.REINFORCEMENT, strength=min(1.0, delta))
    )
