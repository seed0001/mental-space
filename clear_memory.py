#!/usr/bin/env python3
"""Empty all memory rows. Keeps the database file and schema (DELETE, not rm)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spatial_memory import store
from spatial_memory.config import DB_PATH


def main() -> int:
    db = os.environ.get("SPATIAL_MEMORY_DB")
    path = db or DB_PATH
    n = store.clear_all_nodes(db_path=db)
    print(f"Cleared {n} row(s) from memory_nodes (and all memory_links) in {path!r}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
