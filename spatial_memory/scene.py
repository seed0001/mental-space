from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone

from spatial_memory.math_util import cosine_similarity
from spatial_memory.ollama_client import embed
from spatial_memory import store

SCENE_TIME_GAP_SECONDS = 45 * 60
MIN_CONTINUITY_FOR_MERGE = 0.56


@dataclass
class SceneResolution:
    should_merge: bool
    scene_id: str | None
    node_id: str | None
    continuity_score: float
    reason: str


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def _emotion_buckets(text: str) -> tuple[bool, bool]:
    t = (text or "").lower()
    neg = any(k in t for k in ("angry", "argument", "hurt", "upset", "conflict", "frustrat", "mad"))
    repair = any(k in t for k in ("sorry", "apolog", "repair", "forgive", "resolved", "understand now"))
    return neg, repair


def _semantic_score(message: str, node_embedding_json: str | None) -> float:
    if not node_embedding_json:
        return 0.0
    try:
        node_vec = json.loads(node_embedding_json)
    except json.JSONDecodeError:
        return 0.0
    try:
        msg_vec = embed(message[:8000])
        return max(-1.0, min(1.0, cosine_similarity(msg_vec, node_vec)))
    except Exception:
        return 0.0


def resolve_active_scene(raw_message: str, *, db_path: str | None = None) -> SceneResolution:
    active = store.get_active_scene(db_path=db_path)
    if not active:
        return SceneResolution(False, None, None, 0.0, "no_active_scene")
    scene_id = str(active["id"])
    node_id = str(active["node_id"])
    node = store.get_node(node_id, db_path=db_path)
    if node is None:
        store.close_scene(scene_id, "missing_node", db_path=db_path)
        return SceneResolution(False, None, None, 0.0, "missing_scene_node")

    now = datetime.now(timezone.utc)
    last_dt = _parse_iso(active.get("last_event_at"))
    if not last_dt:
        last_dt = _parse_iso(node.last_activation)
    gap_seconds = (now - last_dt).total_seconds() if last_dt else 10**9
    if gap_seconds > SCENE_TIME_GAP_SECONDS:
        store.close_scene(scene_id, "time_gap", db_path=db_path)
        return SceneResolution(False, None, None, 0.0, "time_gap")

    sem_raw = _semantic_score(raw_message, node.embedding_json)
    sem = (sem_raw + 1.0) * 0.5
    continuity = 0.15 + 0.55 * sem + 0.30 * max(0.0, 1.0 - (gap_seconds / SCENE_TIME_GAP_SECONDS))

    neg, repair = _emotion_buckets(raw_message)
    prev_neg, prev_repair = _emotion_buckets(node.understanding[-1500:])
    if (neg and prev_neg) or (repair and prev_repair) or (neg and prev_repair) or (repair and prev_neg):
        continuity = min(1.0, continuity + 0.12)

    low = raw_message.lower()
    if any(k in low for k in ("new topic", "switching topics", "different subject", "unrelated question")):
        store.close_scene(scene_id, "topic_shift", db_path=db_path)
        return SceneResolution(False, None, None, continuity, "explicit_topic_shift")

    if continuity >= MIN_CONTINUITY_FOR_MERGE:
        return SceneResolution(True, scene_id, node_id, continuity, "continuity_merge")

    store.close_scene(scene_id, "topic_shift", db_path=db_path)
    return SceneResolution(False, None, None, continuity, "continuity_break")

