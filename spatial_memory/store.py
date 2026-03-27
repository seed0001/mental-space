from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator

from spatial_memory.config import DB_PATH
from spatial_memory.models import LinkType, MemoryLink, MemoryNode


DDL_NODES = """
CREATE TABLE IF NOT EXISTS memory_nodes (
  id TEXT PRIMARY KEY,
  original_text TEXT NOT NULL,
  understanding TEXT NOT NULL,
  x REAL NOT NULL,
  y REAL NOT NULL,
  z REAL NOT NULL,
  self_other_score REAL NOT NULL,
  known_unknown_score REAL NOT NULL,
  active_contemplative_score REAL NOT NULL,
  commitment_type TEXT NOT NULL,
  confidence REAL NOT NULL,
  reinforcement_count INTEGER NOT NULL,
  last_activation TEXT,
  novelty_at_creation REAL NOT NULL,
  current_relevance REAL NOT NULL,
  certainty REAL NOT NULL,
  contested INTEGER NOT NULL DEFAULT 0,
  source_type TEXT NOT NULL,
  links_json TEXT NOT NULL DEFAULT '[]',
  embedding_json TEXT,
  orientation_prompt_version TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_memory_xyz ON memory_nodes (x, y, z);
"""

DDL_LINKS = """
CREATE TABLE IF NOT EXISTS memory_links (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source_id TEXT NOT NULL,
  target_id TEXT NOT NULL,
  link_type TEXT NOT NULL,
  strength REAL NOT NULL,
  UNIQUE(source_id, target_id, link_type)
);
CREATE INDEX IF NOT EXISTS idx_ml_source ON memory_links(source_id);
CREATE INDEX IF NOT EXISTS idx_ml_target ON memory_links(target_id);
"""

DDL_META = """
CREATE TABLE IF NOT EXISTS spatial_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""

DDL_SCENES = """
CREATE TABLE IF NOT EXISTS memory_scenes (
  id TEXT PRIMARY KEY,
  node_id TEXT NOT NULL UNIQUE,
  state TEXT NOT NULL,
  opened_at TEXT NOT NULL,
  last_event_at TEXT NOT NULL,
  closed_at TEXT,
  participants_json TEXT NOT NULL DEFAULT '[]',
  continuity_score REAL NOT NULL DEFAULT 0.0,
  close_reason TEXT NOT NULL DEFAULT '',
  event_count INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_scenes_state ON memory_scenes(state);
CREATE INDEX IF NOT EXISTS idx_scenes_last_event ON memory_scenes(last_event_at);
"""

DDL_SCENE_EVENTS = """
CREATE TABLE IF NOT EXISTS scene_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  scene_id TEXT NOT NULL,
  created_at TEXT NOT NULL,
  user_message TEXT NOT NULL,
  assistant_response TEXT NOT NULL,
  x REAL NOT NULL,
  y REAL NOT NULL,
  z REAL NOT NULL,
  commitment_type TEXT NOT NULL,
  confidence REAL NOT NULL,
  caution INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_scene_events_scene ON scene_events(scene_id, id DESC);
"""


def _ensure_column(conn: sqlite3.Connection, table: str, col: str, decl_sql: str) -> None:
    cur = conn.execute(f"PRAGMA table_info({table})")
    cols = {r[1] for r in cur.fetchall()}
    if col not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl_sql}")


def _meta_get(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM spatial_meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def _meta_set(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO spatial_meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


def _migrate_json_links_to_table(conn: sqlite3.Connection) -> None:
    if _meta_get(conn, "links_migrated_v1"):
        return
    rows = conn.execute("SELECT id, links_json FROM memory_nodes").fetchall()
    for row in rows:
        nid = row["id"]
        try:
            arr = json.loads(row["links_json"] or "[]")
        except json.JSONDecodeError:
            arr = []
        for item in arr:
            try:
                conn.execute(
                    """INSERT OR REPLACE INTO memory_links(source_id, target_id, link_type, strength)
                       VALUES(?,?,?,?)""",
                    (
                        nid,
                        item["target_id"],
                        item["link_type"],
                        float(item.get("strength", 0.5)),
                    ),
                )
            except (KeyError, TypeError, sqlite3.Error):
                continue
        conn.execute("UPDATE memory_nodes SET links_json = '[]' WHERE id = ?", (nid,))
    _meta_set(conn, "links_migrated_v1", "1")


@contextmanager
def connect(db_path: str | None = None) -> Iterator[sqlite3.Connection]:
    path = db_path or DB_PATH
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: str | None = None) -> None:
    with connect(db_path) as conn:
        conn.executescript(DDL_NODES)
        _ensure_column(
            conn,
            "memory_nodes",
            "orientation_prompt_version",
            "TEXT NOT NULL DEFAULT ''",
        )
        conn.executescript(DDL_LINKS)
        conn.executescript(DDL_META)
        conn.executescript(DDL_SCENES)
        conn.executescript(DDL_SCENE_EVENTS)
        _migrate_json_links_to_table(conn)


def clear_all_nodes(db_path: str | None = None) -> int:
    """DELETE all rows; keeps file, schema, indexes."""
    init_db(db_path)
    with connect(db_path) as conn:
        conn.execute("DELETE FROM scene_events")
        conn.execute("DELETE FROM memory_scenes")
        conn.execute("DELETE FROM memory_links")
        cur = conn.execute("DELETE FROM memory_nodes")
        deleted = cur.rowcount
    return int(deleted)


def _row_to_node(row: sqlite3.Row) -> MemoryNode:
    return MemoryNode.from_row(dict(row))


def persist_links_for_node(node: MemoryNode, db_path: str | None = None) -> None:
    """memory_links is canonical; links_json mirrored for ad-hoc SQL readers."""
    with connect(db_path) as conn:
        conn.execute("DELETE FROM memory_links WHERE source_id = ?", (node.id,))
        for L in node.links:
            conn.execute(
                """INSERT INTO memory_links(source_id, target_id, link_type, strength)
                   VALUES(?,?,?,?)""",
                (node.id, L.target_id, L.link_type.value, L.strength),
            )


def hydrate_links(nodes: list[MemoryNode], db_path: str | None = None) -> None:
    if not nodes:
        return
    ids = [n.id for n in nodes]
    placeholders = ",".join("?" * len(ids))
    q = f"SELECT source_id, target_id, link_type, strength FROM memory_links WHERE source_id IN ({placeholders})"
    by: dict[str, list[MemoryLink]] = {nid: [] for nid in ids}
    with connect(db_path) as conn:
        cur = conn.execute(q, ids)
        for row in cur:
            sid = row["source_id"]
            by.setdefault(sid, []).append(
                MemoryLink(
                    target_id=row["target_id"],
                    link_type=LinkType(row["link_type"]),
                    strength=float(row["strength"]),
                )
            )
    for n in nodes:
        db_links = by.get(n.id, [])
        if db_links:
            n.links = db_links


def insert_node(node: MemoryNode, db_path: str | None = None) -> None:
    d = node.to_row_dict()
    cols = ", ".join(d.keys())
    placeholders = ", ".join("?" * len(d))
    with connect(db_path) as conn:
        conn.execute(
            f"INSERT INTO memory_nodes ({cols}) VALUES ({placeholders})",
            tuple(d.values()),
        )
    persist_links_for_node(node, db_path=db_path)


def update_node(node: MemoryNode, db_path: str | None = None) -> None:
    d = node.to_row_dict()
    nid = d.pop("id")
    sets = ", ".join(f"{k} = ?" for k in d)
    vals = list(d.values()) + [nid]
    with connect(db_path) as conn:
        conn.execute(f"UPDATE memory_nodes SET {sets} WHERE id = ?", vals)
    persist_links_for_node(node, db_path=db_path)


def get_node(node_id: str, db_path: str | None = None) -> MemoryNode | None:
    with connect(db_path) as conn:
        cur = conn.execute("SELECT * FROM memory_nodes WHERE id = ?", (node_id,))
        row = cur.fetchone()
    if not row:
        return None
    n = _row_to_node(row)
    hydrate_links([n], db_path=db_path)
    return n


def nodes_within_radius(
    x: float,
    y: float,
    z: float,
    radius: float,
    db_path: str | None = None,
) -> list[tuple[MemoryNode, float]]:
    """Bounding-box prefilter in SQL, exact sphere in Python."""
    r2 = radius * radius
    lo_x, hi_x = x - radius, x + radius
    lo_y, hi_y = y - radius, y + radius
    lo_z, hi_z = z - radius, z + radius
    with connect(db_path) as conn:
        cur = conn.execute(
            """SELECT * FROM memory_nodes
               WHERE x BETWEEN ? AND ? AND y BETWEEN ? AND ? AND z BETWEEN ? AND ?""",
            (lo_x, hi_x, lo_y, hi_y, lo_z, hi_z),
        )
        rows = cur.fetchall()
    out: list[tuple[MemoryNode, float]] = []
    for row in rows:
        d = dict(row)
        nx, ny, nz = float(d["x"]), float(d["y"]), float(d["z"])
        dsq = (nx - x) ** 2 + (ny - y) ** 2 + (nz - z) ** 2
        if dsq <= r2:
            out.append((_row_to_node(row), dsq))
    out.sort(key=lambda t: t[1])
    hydrate_links([t[0] for t in out], db_path=db_path)
    return out


def all_nodes(db_path: str | None = None) -> list[MemoryNode]:
    with connect(db_path) as conn:
        cur = conn.execute("SELECT * FROM memory_nodes")
        rows = cur.fetchall()
    nodes = [_row_to_node(r) for r in rows]
    hydrate_links(nodes, db_path=db_path)
    return nodes


def new_id() -> str:
    return str(uuid.uuid4())


def graph_snapshot(db_path: str | None = None) -> dict:
    """
    Export nodes + links for 3D visualization.
    Coordinates are continuous in [-1, 1].
    """
    init_db(db_path)
    with connect(db_path) as conn:
        nrows = conn.execute(
            """SELECT n.id, n.x, n.y, n.z, n.confidence, n.reinforcement_count, n.commitment_type,
                      n.contested, n.current_relevance, n.certainty, n.understanding,
                      COALESCE(s.id, '') AS scene_id,
                      COALESCE(s.state, 'closed') AS scene_state,
                      COALESCE(s.event_count, 0) AS scene_event_count,
                      COALESCE(s.continuity_score, 0.0) AS scene_continuity,
                      COALESCE(s.close_reason, '') AS scene_close_reason
               FROM memory_nodes n
               LEFT JOIN memory_scenes s ON s.node_id = n.id"""
        ).fetchall()
        lrows = conn.execute(
            """SELECT source_id, target_id, link_type, strength
               FROM memory_links"""
        ).fetchall()
        erows = conn.execute(
            """SELECT scene_id, id, x, y, z
               FROM scene_events
               ORDER BY scene_id, id ASC"""
        ).fetchall()
    nodes = [
        {
            "id": r["id"],
            "x": float(r["x"]),
            "y": float(r["y"]),
            "z": float(r["z"]),
            "confidence": float(r["confidence"]),
            "reinforcement_count": int(r["reinforcement_count"]),
            "commitment_type": r["commitment_type"],
            "contested": bool(r["contested"]),
            "current_relevance": float(r["current_relevance"]),
            "certainty": float(r["certainty"]),
            "label": (r["understanding"] or "")[:160],
            "scene_id": r["scene_id"],
            "scene_state": r["scene_state"],
            "scene_event_count": int(r["scene_event_count"]),
            "scene_continuity": float(r["scene_continuity"]),
            "scene_close_reason": r["scene_close_reason"],
        }
        for r in nrows
    ]
    links = [
        {
            "source_id": r["source_id"],
            "target_id": r["target_id"],
            "link_type": r["link_type"],
            "strength": float(r["strength"]),
        }
        for r in lrows
    ]
    trails_by_scene: dict[str, list[dict]] = {}
    for r in erows:
        sid = str(r["scene_id"] or "")
        if not sid:
            continue
        trails_by_scene.setdefault(sid, []).append(
            {
                "x": float(r["x"]),
                "y": float(r["y"]),
                "z": float(r["z"]),
            }
        )
    return {"nodes": nodes, "links": links, "scene_trails": trails_by_scene}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_scene(scene_id: str, node_id: str, *, participants: list[str] | None = None, db_path: str | None = None) -> None:
    now = _utc_now()
    pj = json.dumps(participants or ["user", "assistant"])
    with connect(db_path) as conn:
        conn.execute(
            """INSERT OR REPLACE INTO memory_scenes
               (id, node_id, state, opened_at, last_event_at, closed_at, participants_json, continuity_score, close_reason, event_count)
               VALUES (?, ?, 'active', ?, ?, NULL, ?, 0.0, '', 0)""",
            (scene_id, node_id, now, now, pj),
        )


def get_active_scene(db_path: str | None = None) -> dict | None:
    init_db(db_path)
    with connect(db_path) as conn:
        row = conn.execute(
            """SELECT * FROM memory_scenes
               WHERE state = 'active'
               ORDER BY last_event_at DESC
               LIMIT 1"""
        ).fetchone()
    if not row:
        return None
    d = dict(row)
    try:
        d["participants"] = json.loads(d.get("participants_json") or "[]")
    except json.JSONDecodeError:
        d["participants"] = ["user", "assistant"]
    return d


def close_scene(scene_id: str, reason: str, *, db_path: str | None = None) -> None:
    now = _utc_now()
    with connect(db_path) as conn:
        conn.execute(
            """UPDATE memory_scenes
               SET state = 'closed', closed_at = ?, close_reason = ?
               WHERE id = ?""",
            (now, reason[:64], scene_id),
        )


def touch_scene(scene_id: str, *, continuity_score: float, db_path: str | None = None) -> None:
    now = _utc_now()
    with connect(db_path) as conn:
        conn.execute(
            """UPDATE memory_scenes
               SET state = 'active',
                   last_event_at = ?,
                   continuity_score = ?,
                   event_count = event_count + 1
               WHERE id = ?""",
            (now, float(continuity_score), scene_id),
        )


def append_scene_event(
    scene_id: str,
    *,
    user_message: str,
    assistant_response: str,
    x: float,
    y: float,
    z: float,
    commitment_type: str,
    confidence: float,
    caution: bool,
    db_path: str | None = None,
) -> None:
    now = _utc_now()
    with connect(db_path) as conn:
        conn.execute(
            """INSERT INTO scene_events
               (scene_id, created_at, user_message, assistant_response, x, y, z, commitment_type, confidence, caution)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                scene_id,
                now,
                user_message,
                assistant_response,
                float(x),
                float(y),
                float(z),
                commitment_type,
                float(confidence),
                1 if caution else 0,
            ),
        )


def recent_scene_events(scene_id: str, *, limit: int = 10, db_path: str | None = None) -> list[dict]:
    with connect(db_path) as conn:
        rows = conn.execute(
            """SELECT * FROM scene_events
               WHERE scene_id = ?
               ORDER BY id DESC
               LIMIT ?""",
            (scene_id, int(max(1, limit))),
        ).fetchall()
    out = [dict(r) for r in rows]
    out.reverse()
    return out


def last_scene_event_xyz(scene_id: str, db_path: str | None = None) -> tuple[float, float, float] | None:
    """Most recent (x,y,z) committed for this scene, or None if no events."""
    init_db(db_path)
    with connect(db_path) as conn:
        row = conn.execute(
            """SELECT x, y, z FROM scene_events
               WHERE scene_id = ?
               ORDER BY id DESC
               LIMIT 1""",
            (scene_id,),
        ).fetchone()
    if not row:
        return None
    return float(row["x"]), float(row["y"]), float(row["z"])


def last_global_scene_event_xyz(db_path: str | None = None) -> tuple[float, float, float] | None:
    """Most recent (x,y,z) in any scene (by event id), or None."""
    init_db(db_path)
    with connect(db_path) as conn:
        row = conn.execute(
            """SELECT x, y, z FROM scene_events
               ORDER BY id DESC
               LIMIT 1"""
        ).fetchone()
    if not row:
        return None
    return float(row["x"]), float(row["y"]), float(row["z"])


def get_scene_by_node_id(node_id: str, db_path: str | None = None) -> dict | None:
    with connect(db_path) as conn:
        row = conn.execute(
            """SELECT * FROM memory_scenes
               WHERE node_id = ?
               LIMIT 1""",
            (node_id,),
        ).fetchone()
    if not row:
        return None
    d = dict(row)
    try:
        d["participants"] = json.loads(d.get("participants_json") or "[]")
    except json.JSONDecodeError:
        d["participants"] = ["user", "assistant"]
    return d
