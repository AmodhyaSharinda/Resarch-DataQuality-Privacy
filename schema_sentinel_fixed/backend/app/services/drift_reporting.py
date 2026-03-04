from __future__ import annotations
from typing import Any


def build_drift_explanations(drift: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """
    Convert drift output into a structured, UI-friendly explanation payload.

    IMPORTANT:
    drift["renames"] is an object:
      {
        "mappings": {canonical_col: observed_col, ...},
        "candidates": [...],
        ...
      }
    NOT a flat dict.
    """

    diff = drift.get("diff") or {}
    raw = drift.get("raw_diff") or {}

    ren_obj = drift.get("renames") or {}

    # Extract mappings safely
    mappings: dict[str, str] = {}
    candidates: list[dict[str, Any]] = []

    if isinstance(ren_obj, dict):
        # new format from drift_engine.py
        m = ren_obj.get("mappings") or {}
        if isinstance(m, dict):
            mappings = {str(k): str(v) for k, v in m.items()}
        c = ren_obj.get("candidates") or []
        if isinstance(c, list):
            candidates = [x for x in c if isinstance(x, dict)]
    elif isinstance(ren_obj, dict):
        mappings = {str(k): str(v) for k, v in ren_obj.items()}

    # Build confidence lookup from candidates (canonical/observed/prob)
    conf_by_pair: dict[tuple[str, str], float] = {}
    for c in candidates:
        try:
            old = str(c.get("canonical") or c.get("old") or "")
            new = str(c.get("observed") or c.get("new") or "")
            prob = c.get("prob", c.get("confidence", 0.0))
            conf = float(prob) if prob is not None else 0.0
            if old and new:
                conf_by_pair[(old, new)] = conf
        except Exception:
            continue

    renamed_items: list[dict[str, Any]] = []
    for old, new in (mappings or {}).items():
        conf = conf_by_pair.get((old, new))
        renamed_items.append(
            {
                "old": old,
                "new": new,
                "confidence": conf,
                "reason": "Column rename suggested by model",
            }
        )

    new_cols = [{"column": c, "reason": f"A new column '{c}' appeared in the incoming data."} for c in (diff.get("added") or [])]
    removed_cols = [{"column": c, "reason": f"Column '{c}' is missing from the incoming data."} for c in (diff.get("removed") or [])]

    type_changes: list[dict[str, Any]] = []
    for tc in (diff.get("type_changes") or []):
        if not isinstance(tc, dict):
            continue
        col = tc.get("column")
        if not col:
            continue
        type_changes.append(
            {
                "column": col,
                "from": tc.get("from"),
                "to": tc.get("to"),
                "reason": f"Type changed: {tc.get('from')} → {tc.get('to')}",
            }
        )

    nullable_changes: list[dict[str, Any]] = []
    for nc in (diff.get("nullable_changes") or []):
        if not isinstance(nc, dict):
            continue
        col = nc.get("column")
        if not col:
            continue
        nullable_changes.append(
            {
                "column": col,
                "from": nc.get("from"),
                "to": nc.get("to"),
                "reason": f"Nullability changed: {nc.get('from')} → {nc.get('to')}",
            }
        )

    return {
        "renamed": renamed_items,
        "new_columns": new_cols,
        "removed_columns": removed_cols,
        "type_changes": type_changes,
        "nullable_changes": nullable_changes,
    }


def format_drift_summary(
    *,
    dataset: str,
    batch_id: str,
    observed_columns: list[str],
    drift: dict[str, Any],
    risk: dict[str, Any],
    schema_version: int | None = None,
) -> str:
    lines: list[str] = []
    lines.append(f"BATCH: {batch_id}")
    lines.append(f"Observed columns: {observed_columns}")

    drift_types = drift.get("drift_types") or []
    if drift_types:
        lines.append("🔵 Schema changed → running drift detection...")
    else:
        lines.append("🔵 Schema unchanged → no drift detected.")

    rs = risk.get("risk_score")
    rl = str(risk.get("risk_level", "LOW")).upper()
    route = str(risk.get("route", "staging")).upper()

    lines.append(f"RISK SCORE: {rs}")
    lines.append(f"RISK LEVEL: {rl}")
    lines.append(f"ROUTE: {route}")

    if drift_types:
        ex = build_drift_explanations(drift)
        lines.append("")
        lines.append("DRIFT EXPLANATIONS (AFFECTED COLUMNS):")
        lines.append("")

        if ex["renamed"]:
            lines.append("RENAMED:")
            for r in ex["renamed"]:
                conf = r.get("confidence")
                conf_str = f"{conf:.2f}" if isinstance(conf, (int, float)) else "—"
                lines.append(f" - {r['old']} → {r['new']}: model confidence {conf_str}.")
            lines.append("")

        if ex["new_columns"]:
            lines.append("NEW COLUMNS:")
            for r in ex["new_columns"]:
                lines.append(f" - {r['column']}: {r['reason']}")
            lines.append("")

        if ex["removed_columns"]:
            lines.append("REMOVED COLUMNS:")
            for r in ex["removed_columns"]:
                lines.append(f" - {r['column']}: {r['reason']}")
            lines.append("")

        if ex["type_changes"]:
            lines.append("TYPE CHANGES:")
            for r in ex["type_changes"]:
                lines.append(f" - {r['column']}: {r['reason']}")
            lines.append("")

        if ex["nullable_changes"]:
            lines.append("NULLABILITY CHANGES:")
            for r in ex["nullable_changes"]:
                lines.append(f" - {r['column']}: {r['reason']}")
            lines.append("")

    if schema_version is not None:
        lines.append(f"🔵 Schema version stored: {schema_version}")

    if drift_types:
        lines.append("   🔵 Drift event logged to dbo.drift_events")
        # lines.append("✅ Wrote local drift log: outputs/drift_events.jsonl")

    return "\n".join(lines)


    