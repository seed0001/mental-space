from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CommitmentType(str, Enum):
    RECOGNITION = "recognition"
    DEEPENING = "deepening"
    BRIDGING = "bridging"
    FOUNDING = "founding"


class LinkType(str, Enum):
    REINFORCEMENT = "reinforcement"
    BRIDGE = "bridge"
    TENSION = "tension"
    SEQUENCE = "sequence"


class SourceType(str, Enum):
    EXPERIENCE = "experience"
    TOLD = "told"
    INFERRED = "inferred"
    ASSUMED = "assumed"


@dataclass
class MemoryLink:
    target_id: str
    link_type: LinkType
    strength: float


@dataclass
class MemoryNode:
    id: str
    original_text: str
    understanding: str
    x: float
    y: float
    z: float
    self_other_score: float
    known_unknown_score: float
    active_contemplative_score: float
    commitment_type: CommitmentType
    confidence: float
    reinforcement_count: int
    last_activation: str | None
    novelty_at_creation: float
    current_relevance: float
    certainty: float
    contested: bool
    source_type: SourceType
    links: list[MemoryLink] = field(default_factory=list)
    embedding_json: str | None = None
    orientation_prompt_version: str = ""

    def to_row_dict(self) -> dict[str, Any]:
        import json

        links_ser = [
            {"target_id": L.target_id, "link_type": L.link_type.value, "strength": L.strength}
            for L in self.links
        ]
        return {
            "id": self.id,
            "original_text": self.original_text,
            "understanding": self.understanding,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "self_other_score": self.self_other_score,
            "known_unknown_score": self.known_unknown_score,
            "active_contemplative_score": self.active_contemplative_score,
            "commitment_type": self.commitment_type.value,
            "confidence": self.confidence,
            "reinforcement_count": self.reinforcement_count,
            "last_activation": self.last_activation,
            "novelty_at_creation": self.novelty_at_creation,
            "current_relevance": self.current_relevance,
            "certainty": self.certainty,
            "contested": 1 if self.contested else 0,
            "source_type": self.source_type.value,
            "links_json": json.dumps(links_ser),
            "embedding_json": self.embedding_json,
            "orientation_prompt_version": self.orientation_prompt_version or "",
        }

    @staticmethod
    def from_row(row: dict[str, Any]) -> MemoryNode:
        import json
        from datetime import datetime

        links_raw = json.loads(row.get("links_json") or "[]")
        links = [
            MemoryLink(
                target_id=item["target_id"],
                link_type=LinkType(item["link_type"]),
                strength=float(item["strength"]),
            )
            for item in links_raw
        ]
        la = row.get("last_activation")
        if isinstance(la, datetime):
            la = la.isoformat()
        return MemoryNode(
            id=row["id"],
            original_text=row["original_text"],
            understanding=row["understanding"],
            x=float(row["x"]),
            y=float(row["y"]),
            z=float(row["z"]),
            self_other_score=float(row["self_other_score"]),
            known_unknown_score=float(row["known_unknown_score"]),
            active_contemplative_score=float(row["active_contemplative_score"]),
            commitment_type=CommitmentType(row["commitment_type"]),
            confidence=float(row["confidence"]),
            reinforcement_count=int(row["reinforcement_count"]),
            last_activation=la,
            novelty_at_creation=float(row["novelty_at_creation"]),
            current_relevance=float(row["current_relevance"]),
            certainty=float(row["certainty"]),
            contested=bool(row["contested"]),
            source_type=SourceType(row["source_type"]),
            links=links,
            embedding_json=row.get("embedding_json"),
            orientation_prompt_version=str(row.get("orientation_prompt_version") or ""),
        )


@dataclass
class Orientation:
    self_other: float
    known_unknown: float
    active_contemplative: float
    classifier_prompt_version: str = ""

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.self_other, self.known_unknown, self.active_contemplative)


@dataclass
class NeighborhoodStats:
    nodes: list[MemoryNode]
    density: float
    coherence: float
    dist_sq: dict[str, float]
    """Mean pairwise cosine on understanding embeddings, excluding tension-linked pairs."""
    coherence_pairs_used: int = 0


@dataclass
class Decision:
    commitment_type: CommitmentType
    memory_to_inject: list[str]
    confidence_level: float
    caution_internal_conflict: bool
    activated_node_ids: list[str]
    rationale: str
    rule_id: str
    inspection_density: float = 0.0
    inspection_coherence: float = 0.0
    resonance_max: float = 0.0
    multi_region_resonance: bool = False
