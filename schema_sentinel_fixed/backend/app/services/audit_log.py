from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def get_log_file_path() -> str:
    """Return audit log path (JSONL). Override with env AUDIT_LOG_PATH."""
    override = (os.getenv("AUDIT_LOG_PATH") or "").strip()
    if override:
        return str(Path(override))

    base_dir = Path(__file__).resolve().parents[2]  # .../backend
    storage_dir = (os.getenv("STORAGE_DIR") or "storage").strip() or "storage"
    return str(base_dir / storage_dir / "audit_log.jsonl")


def append_jsonl(*args, **kwargs) -> None:
    """
    Supports:
      append_jsonl(event_dict)
      append_jsonl(path, event_dict)              # legacy
      append_jsonl(event_dict, path="...")

    Accepts dict/list/str/bytes.
    """
    path = kwargs.get("path")
    event: Any = None

    if len(args) == 1:
        event = args[0]
    elif len(args) >= 2:
        a0, a1 = args[0], args[1]
        if isinstance(a0, (str, Path)) and not isinstance(a1, (str, Path)):
            path = str(a0)
            event = a1
        else:
            event = a0
            if path is None and isinstance(a1, (str, Path)):
                path = str(a1)

    if not path:
        path = get_log_file_path()

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(event, (bytes, bytearray)):
        event = event.decode("utf-8", errors="replace")

    if isinstance(event, str):
        s = event.strip()
        if not s:
            return
        try:
            json.loads(s)
            line = s
        except Exception:
            line = json.dumps({"message": s}, ensure_ascii=False, default=str)
    else:
        line = json.dumps(event, ensure_ascii=False, default=str)

    with p.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def tail_jsonl(limit: int = 200) -> list[dict[str, Any]]:
    limit = max(1, int(limit or 200))
    p = Path(get_log_file_path())
    if not p.exists():
        return []

    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return []

    out: list[dict[str, Any]] = []
    for line in lines[-limit:]:
        s = (line or "").strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
            out.append(obj if isinstance(obj, dict) else {"value": obj})
        except Exception:
            out.append({"raw": s})
    return out