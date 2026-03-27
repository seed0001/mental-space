#!/usr/bin/env python3
"""
Start the spatial-memory chat server and open it in your browser.

Usage:
  python launch.py
  python launch.py --no-browser
  python launch.py --port 9000

Requires: Ollama running with llama3.2 (see spatial_memory config).
Persona: edit persona.txt (injected into replies only, not orientation).
Optional: COMMITMENT_USE_LLM=1 for a second-pass LLM audit of commitment; CLASSIFIER_PROMPT_VERSION in spatial_memory/constants.py.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> int:
    os.environ.setdefault("SPATIAL_MEMORY_PROJECT_ROOT", str(ROOT))
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    p = argparse.ArgumentParser(description="Launch spatial memory chat (FastAPI + browser)")
    p.add_argument("--host", default=os.environ.get("CHAT_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("CHAT_PORT", "8765")))
    p.add_argument("--no-browser", action="store_true", help="Only start the server")
    p.add_argument(
        "--clear-memory",
        action="store_true",
        help="DELETE all memory rows before starting (keeps DB file and schema)",
    )
    p.add_argument("--reload", action="store_true", help="Dev: auto-reload on code changes")
    args = p.parse_args()

    if args.clear_memory:
        from spatial_memory import store

        db = os.environ.get("SPATIAL_MEMORY_DB")
        n = store.clear_all_nodes(db_path=db)
        print(f"Cleared {n} memory row(s) before launch.\n")

    import uvicorn

    url_host = "127.0.0.1" if args.host in ("0.0.0.0", "::", "[::]") else args.host
    url = f"http://{url_host}:{args.port}/"

    if not args.no_browser:

        def _open() -> None:
            time.sleep(1.0)
            webbrowser.open(url)

        threading.Thread(target=_open, daemon=True).start()

    print(f"Spatial memory chat → {url}")
    print("Ctrl+C to stop.\n")

    kw: dict = {
        "host": args.host,
        "port": args.port,
        "log_level": "info",
        "reload": args.reload,
    }
    if args.reload:
        kw["reload_dirs"] = [str(ROOT)]

    try:
        uvicorn.run("chat_server:app", **kw)
    except KeyboardInterrupt:
        print("\nStopped.")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
