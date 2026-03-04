from __future__ import annotations

import ast
import json
from typing import Any

from sqlalchemy.orm import Session

from app.db.models import ProductionRow, StagingRow, DriftEvent
from app.utils.text import norm_col


# -----------------------------
# JSON helpers
# -----------------------------
def _safe_load_json(s: Any) -> Any:
    if isinstance(s, dict):
        return s
    if not s:
        return {}
    try:
        return json.loads(s)
    except Exception:
        try:
            return ast.literal_eval(s)
        except Exception:
            return {}


def _is_packed(obj: Any) -> bool:
    return isinstance(obj, dict) and ("canonical" in obj) and ("raw" in obj) and ("extras" in obj)


# -----------------------------
# Canonical helpers
# -----------------------------
def _canonical_cols(canonical_schema: dict[str, Any]) -> list[str]:
    cols = canonical_schema.get("columns") or []
    out: list[str] = []
    for c in cols:
        if isinstance(c, dict) and c.get("name"):
            out.append(str(c["name"]))
    return out


def _canonical_is_all_null(canon: Any, canonical_schema: dict[str, Any]) -> bool:
    if not isinstance(canon, dict) or not canon:
        return True
    cols = _canonical_cols(canonical_schema)
    if not cols:
        return True
    return all(canon.get(c) is None for c in cols)


def _pick_best_packed_layer(packed: dict[str, Any], canonical_schema: dict[str, Any]) -> dict[str, Any]:
    """
    Fix for the "double-pack" bug:
    Sometimes the real packed row ends up inside packed["extras"] or packed["raw"].
    Prefer a packed candidate whose canonical is NOT all-null.
    """
    candidates: list[dict[str, Any]] = []
    for cand in (packed, packed.get("extras"), packed.get("raw")):
        if _is_packed(cand):
            candidates.append(cand)

    for c in candidates:
        if not _canonical_is_all_null(c.get("canonical"), canonical_schema):
            return c

    return candidates[0] if candidates else packed


def _unwrap_true_raw(obj: Any) -> dict[str, Any]:
    """
    If obj is packed, keep going down obj["raw"] until raw is NOT packed.
    That final dict is the real incoming row (flat keys).
    """
    cur = obj
    guard = 0
    while _is_packed(cur) and isinstance(cur.get("raw"), dict) and guard < 10:
        nxt = cur.get("raw")
        if _is_packed(nxt):
            cur = nxt
            guard += 1
            continue
        return nxt
    return cur if isinstance(cur, dict) else {}


# -----------------------------
# Canonicalizer (IDEMPOTENT)
# -----------------------------
def _canonicalize_row(
    row: dict[str, Any],
    canonical_schema: dict[str, Any],
    renames: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    IMPORTANT: row might already be packed (old bug / wrong call site).
    Make this idempotent by unwrapping to the true raw row first.
    """
    if _is_packed(row):
        row = _unwrap_true_raw(row)

    canonical_cols = _canonical_cols(canonical_schema)
    canonical_set = set(canonical_cols)

    mappings = (renames or {}).get("mappings", {}) if isinstance(renames, dict) else {}
    observed_to_canonical = {norm_col(v): k for k, v in mappings.items() if v}

    norm_map = {norm_col(k): k for k in row.keys()}  # normalized -> original key
    norm_row = {norm_col(k): v for k, v in row.items()}

    canonical_data: dict[str, Any] = {}
    for c in canonical_cols:
        if c in norm_row:
            canonical_data[c] = norm_row.get(c)
        else:
            found = None
            for obs_norm, canon_name in observed_to_canonical.items():
                if canon_name == c and obs_norm in norm_row:
                    found = norm_row.get(obs_norm)
                    break
            canonical_data[c] = found

    extras: dict[str, Any] = {}
    for nk, val in norm_row.items():
        if nk in canonical_set:
            continue
        if nk in observed_to_canonical:
            continue
        extras[nk] = val

    return {
        "canonical": canonical_data,
        "extras": extras,
        "raw": row,
        "normalized_key_map": norm_map,
    }


# -----------------------------
# Writes
# -----------------------------
def write_rows_production(
    db: Session,
    dataset_id: int,
    batch_id: str,
    canonical_schema: dict[str, Any],
    rows: list[dict[str, Any]],
    renames: dict[str, Any] | None,
):
    for r in rows:
        packed = _canonicalize_row(r, canonical_schema, renames)
        db.add(
            ProductionRow(
                dataset_id=dataset_id,
                batch_id=batch_id,
                row_json=json.dumps(packed, ensure_ascii=False, default=str),
            )
        )
    db.commit()


def write_rows_staging(
    db: Session,
    event_id: int,
    canonical_schema: dict[str, Any],
    rows: list[dict[str, Any]],
    renames: dict[str, Any] | None,
):
    for r in rows:
        packed = _canonicalize_row(r, canonical_schema, renames)
        db.add(
            StagingRow(
                event_id=event_id,
                status="PENDING",
                row_json=json.dumps(packed, ensure_ascii=False, default=str),
            )
        )
    db.commit()


# -----------------------------
# Promote staging -> production
# -----------------------------
def promote_ready_rows(
    db: Session,
    event: DriftEvent,
    canonical_schema: dict[str, Any],
    renames: dict[str, Any] | None,
) -> int:
    dataset_id = event.dataset_id
    batch_id = event.batch_id

    count = 0
    ready_rows = [s for s in list(event.staged_rows) if s.status == "READY"]

    for s in ready_rows:
        packed_any = _safe_load_json(s.row_json)
        packed = packed_any if isinstance(packed_any, dict) else {}

        # ✅ fix double-pack: choose best layer
        best = _pick_best_packed_layer(packed, canonical_schema)

        canon = best.get("canonical") if isinstance(best.get("canonical"), dict) else {}
        extras = best.get("extras") if isinstance(best.get("extras"), dict) else {}
        raw = best.get("raw") if isinstance(best.get("raw"), dict) else {}
        nkmap = best.get("normalized_key_map") if isinstance(best.get("normalized_key_map"), dict) else {}

        # ✅ if canonical still broken -> rebuild from true raw
        if _canonical_is_all_null(canon, canonical_schema):
            true_raw = _unwrap_true_raw(best)
            rebuilt = _canonicalize_row(true_raw, canonical_schema, renames)
            canon = rebuilt.get("canonical") or {}
            extras = rebuilt.get("extras") or extras
            raw = rebuilt.get("raw") or true_raw
            nkmap = rebuilt.get("normalized_key_map") or nkmap

        prod_payload = {
            "canonical": canon,
            "extras": extras,
            "raw": raw,
            "normalized_key_map": nkmap,
            "event_id": event.id,
            "batch_id": batch_id,
        }

        db.add(
            ProductionRow(
                dataset_id=dataset_id,
                batch_id=batch_id,
                row_json=json.dumps(prod_payload, ensure_ascii=False, default=str),
            )
        )
        count += 1
        db.delete(s)

    db.commit()
    return count