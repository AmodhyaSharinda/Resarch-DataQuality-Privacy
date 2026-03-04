from __future__ import annotations

import csv
import io
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


def get_drift_log_file_path() -> str:
    """
    Default drift log path written by drift pipeline:
      backend/outputs/drift_events.jsonl

    Override with env DRIFT_LOG_PATH if needed.
    """
    override = (os.getenv("DRIFT_LOG_PATH") or "").strip()
    if override:
        return str(Path(override))

    base_dir = Path(__file__).resolve().parents[2]  # .../backend
    return str(base_dir / "outputs" / "drift_events.jsonl")


def _parse_dt(v: Any) -> datetime | None:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v
    s = str(v).strip()
    if not s:
        return None
    # common cases: ISO, "2026-03-03 12:34:56", etc.
    for fmt in (
        None,  # datetime.fromisoformat
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%m/%d/%Y %H:%M:%S",
    ):
        try:
            if fmt is None:
                return datetime.fromisoformat(s.replace("Z", "+00:00"))
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    return None


def read_drift_events_jsonl(dataset: str | None = None) -> list[dict[str, Any]]:
    p = Path(get_drift_log_file_path())
    if not p.exists():
        return []

    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return []

    out: list[dict[str, Any]] = []
    ds = (dataset or "").strip().lower()

    for line in lines:
        s = (line or "").strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
            if not isinstance(obj, dict):
                continue
        except Exception:
            continue

        # dataset may be stored as dataset / dataset_name
        dsn = str(obj.get("dataset") or obj.get("dataset_name") or "").strip().lower()
        if ds and dsn != ds:
            continue

        out.append(obj)

    # sort ascending by time if possible
    def key_fn(e: dict[str, Any]):
        dtv = _parse_dt(e.get("created_at") or e.get("ts") or e.get("timestamp"))
        return dtv or datetime.min

    out.sort(key=key_fn)
    return out


def _safe_list(v: Any) -> list[Any]:
    if v is None:
        return []
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        # "ADD,REMOVE" -> ["ADD","REMOVE"]
        parts = [x.strip() for x in v.split(",") if x.strip()]
        return parts
    return []


def _diff_counts(e: dict[str, Any]) -> dict[str, Any]:
    diff = e.get("diff") or {}
    if isinstance(diff, str):
        try:
            diff = json.loads(diff)
        except Exception:
            diff = {}

    added = diff.get("added") or []
    removed = diff.get("removed") or []
    renames = (e.get("renames") or {}).get("mappings") if isinstance(e.get("renames"), dict) else None
    type_changes = diff.get("type_changes") or []
    nullable_changes = diff.get("nullable_changes") or []

    # renames mappings could be dict old->new
    rename_count = 0
    rename_pairs = ""
    if isinstance(renames, dict) and renames:
        rename_count = len(renames)
        rename_pairs = "; ".join([f"{k}->{v}" for k, v in renames.items()])

    # type_changes may be list of {column,from,to}
    tc_count = len(type_changes) if isinstance(type_changes, list) else 0
    tc_cols = ""
    if isinstance(type_changes, list) and type_changes:
        tc_cols = "; ".join([f"{x.get('column')}:{x.get('from')}->{x.get('to')}" for x in type_changes if isinstance(x, dict)])

    return {
        "added_count": len(added) if isinstance(added, list) else 0,
        "removed_count": len(removed) if isinstance(removed, list) else 0,
        "rename_count": int(rename_count),
        "type_change_count": int(tc_count),
        "nullable_change_count": len(nullable_changes) if isinstance(nullable_changes, list) else 0,
        "added_cols": ", ".join([str(x) for x in added]) if isinstance(added, list) else "",
        "removed_cols": ", ".join([str(x) for x in removed]) if isinstance(removed, list) else "",
        "rename_pairs": rename_pairs,
        "type_changes": tc_cols,
    }


def build_events_rows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for e in events:
        dtv = _parse_dt(e.get("created_at") or e.get("ts") or e.get("timestamp"))
        drift_types = _safe_list(e.get("drift_types"))
        if not drift_types and isinstance(e.get("diff"), dict):
            # sometimes drift_types not stored; infer from diff
            diff = e.get("diff") or {}
            if diff.get("added"):
                drift_types.append("ADD")
            if diff.get("removed"):
                drift_types.append("REMOVE")
            if diff.get("type_changes"):
                drift_types.append("TYPE_CHANGE")
            if diff.get("nullable_changes"):
                drift_types.append("NULLABLE_CHANGE")
            if isinstance(e.get("renames"), dict) and (e["renames"].get("mappings") or {}):
                drift_types.append("RENAME")

        counts = _diff_counts(e)

        rows.append(
            {
                "created_at": (dtv.isoformat() if dtv else str(e.get("created_at") or "")),
                "dataset": e.get("dataset") or e.get("dataset_name") or "",
                "event_id": e.get("event_id") or e.get("id") or "",
                "batch_id": e.get("batch_id") or e.get("batch") or "",
                "route": e.get("route") or "",
                "status": e.get("status") or "",
                "risk_score": e.get("risk_score") if e.get("risk_score") is not None else "",
                "risk_level": e.get("risk_level") or "",
                "drift_types": ",".join([str(x) for x in drift_types]),
                **counts,
            }
        )
    return rows


def build_daily_summary_rows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # aggregate by date (YYYY-MM-DD)
    buckets: dict[str, dict[str, Any]] = {}

    def lvl_to_num(lvl: str) -> int:
        s = str(lvl or "").strip().upper()
        if s == "LOW":
            return 1
        if s == "MEDIUM":
            return 2
        if s == "HIGH":
            return 3
        if s == "CRITICAL":
            return 4
        return 0

    for e in events:
        dtv = _parse_dt(e.get("created_at") or e.get("ts") or e.get("timestamp"))
        day = (dtv.date().isoformat() if dtv else "unknown")

        b = buckets.get(day)
        if b is None:
            b = {
                "date": day,
                "total_events": 0,
                "add_events": 0,
                "remove_events": 0,
                "rename_events": 0,
                "type_change_events": 0,
                "nullable_change_events": 0,
                "avg_risk_score": 0.0,
                "max_risk_score": 0.0,
                "max_risk_level": "",
                "_risk_sum": 0.0,
                "_risk_n": 0,
                "_max_lvl_num": 0,
            }
            buckets[day] = b

        b["total_events"] += 1

        dtypes = _safe_list(e.get("drift_types"))
        dtypes_u = {str(x).strip().upper() for x in dtypes}

        if "ADD" in dtypes_u:
            b["add_events"] += 1
        if "REMOVE" in dtypes_u:
            b["remove_events"] += 1
        if "RENAME" in dtypes_u:
            b["rename_events"] += 1
        if "TYPE_CHANGE" in dtypes_u:
            b["type_change_events"] += 1
        if "NULLABLE_CHANGE" in dtypes_u:
            b["nullable_change_events"] += 1

        rs = e.get("risk_score")
        try:
            rsf = float(rs)
            b["_risk_sum"] += rsf
            b["_risk_n"] += 1
            b["max_risk_score"] = max(float(b["max_risk_score"] or 0.0), rsf)
        except Exception:
            pass

        lvl = str(e.get("risk_level") or "").strip()
        ln = lvl_to_num(lvl)
        if ln >= b["_max_lvl_num"]:
            b["_max_lvl_num"] = ln
            b["max_risk_level"] = lvl

    out = []
    for day in sorted(buckets.keys()):
        b = buckets[day]
        n = int(b["_risk_n"] or 0)
        b["avg_risk_score"] = round((b["_risk_sum"] / n), 6) if n else 0.0
        # cleanup internals
        b.pop("_risk_sum", None)
        b.pop("_risk_n", None)
        b.pop("_max_lvl_num", None)
        out.append(b)

    return out


def rows_to_csv(rows: list[dict[str, Any]]) -> str:
    if not rows:
        # still return header-only CSV
        return "no_data\n"

    # stable column order
    fieldnames = list(rows[0].keys())
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fieldnames})
    return buf.getvalue()