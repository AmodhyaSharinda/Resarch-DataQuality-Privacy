from __future__ import annotations

import json
import os
import re
import datetime as dt
from pathlib import Path
from typing import Any

from fastapi import UploadFile, HTTPException
from sqlalchemy.orm import Session
from app.db.models import Dataset, RiskConfig

from app.core.config import settings
from app.db.models import Dataset
from app.services.schema_registry import get_active_schema_json
from app.utils.profiling import compute_baseline_profile
from app.utils.text import norm_col


# ----------------------------
# Helpers
# ----------------------------

def _sanitize_name(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[^a-zA-Z0-9_\-]+", "_", name)
    return name


def _storage_paths(dataset_name: str) -> dict[str, str]:
    """Where dataset files are persisted."""
    base = os.path.join(settings.STORAGE_DIR, "datasets", _sanitize_name(dataset_name))
    os.makedirs(base, exist_ok=True)
    return {
        "base": base,
        "canonical": os.path.join(base, "canonical_schema.json"),
        # baseline path chosen dynamically to preserve file extension
        "baseline_base": os.path.join(base, "baseline"),
    }


def _load_json_upload(f: UploadFile) -> Any:
    try:
        try:
            f.file.seek(0)
        except Exception:
            pass

        raw = f.file.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        return json.loads(raw)
    except Exception as e:
        raise HTTPException(400, f"Invalid JSON in '{getattr(f, 'filename', 'upload')}': {e}")


def _to_bool(v: Any, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "1", "yes", "y", "t"):
            return True
        if s in ("false", "0", "no", "n", "f", ""):
            return False
    return default


def _normalize_type(t: Any) -> str:
    """Standardize types to: int, float, bool, datetime, string"""
    if t is None:
        return "string"
    s = str(t).strip().lower()

    if s in ("int", "integer", "int64", "int32", "long", "smallint", "bigint"):
        return "int"
    if s in ("float", "double", "decimal", "number", "numeric", "real", "float64", "float32"):
        return "float"
    if s in ("bool", "boolean"):
        return "bool"
    if s in ("datetime", "timestamp", "date", "time"):
        return "datetime"

    return "string"


def _normalize_canonical_schema(schema: dict[str, Any]) -> dict[str, Any]:
    cols = schema.get("columns")
    if not isinstance(cols, list):
        raise HTTPException(400, "Canonical schema must contain a 'columns' list")

    norm_cols: list[dict[str, Any]] = []
    seen: set[str] = set()

    for c in cols:
        if not isinstance(c, dict):
            continue

        name = c.get("name")
        if not name:
            continue

        norm_name = norm_col(name)
        if not norm_name:
            continue

        if norm_name in seen:
            continue
        seen.add(norm_name)

        norm_cols.append(
            {
                "name": norm_name,
                "type": _normalize_type(c.get("type")),
                "nullable": _to_bool(c.get("nullable"), default=True),
                "primary_key": _to_bool(c.get("primary_key"), default=False),
                "description": c.get("description") or "",
            }
        )

    if not norm_cols:
        raise HTTPException(400, "Canonical schema has no usable columns")

    out = dict(schema)
    out["columns"] = norm_cols
    out["dataset"] = schema.get("dataset") or schema.get("name") or "dataset"
    out["version"] = schema.get("version") or "v0"
    return out


def _choose_baseline_path(paths: dict[str, str], baseline_file: UploadFile) -> str:
    """Preserve baseline extension so profiling can read CSV/Excel."""
    fname = (getattr(baseline_file, "filename", "") or "").strip()
    ext = Path(fname).suffix.lower() if fname else ""

    if ext not in (".csv", ".xlsx", ".xls", ".xlsm"):
        ext = ".csv"

    return paths["baseline_base"] + ext


# ----------------------------
# Public API used by routes
# ----------------------------

async def register_or_update_dataset(
    db: Session,
    dataset_name: str,
    canonical_schema_file: UploadFile,
    baseline_csv_file: UploadFile | None,
) -> dict[str, Any]:
    """
    FIXES:
    - Baseline file is actually saved and profiled
    - If user updates WITHOUT baseline, we DO NOT wipe existing baseline fields
    """
    dataset_name = (dataset_name or "").strip()
    if not dataset_name:
        raise HTTPException(400, "dataset name is required")

    paths = _storage_paths(dataset_name)
    canonical_json = _normalize_canonical_schema(_load_json_upload(canonical_schema_file))

    # Save canonical schema to disk
    with open(paths["canonical"], "w", encoding="utf-8") as f:
        json.dump(canonical_json, f, indent=2)

    # load/create dataset row first (so baseline fields can be preserved)
    d = db.query(Dataset).filter(Dataset.name == dataset_name).first()
    now = dt.datetime.utcnow()
    if not d:
        d = Dataset(name=dataset_name)
        db.add(d)

    d.canonical_schema_json = json.dumps(canonical_json, default=str)
    d.canonical_schema_path = paths["canonical"]

    # --- baseline handling ---
    if baseline_csv_file is not None:
        try:
            raw = await baseline_csv_file.read()
            baseline_path = _choose_baseline_path(paths, baseline_csv_file)

            with open(baseline_path, "wb") as f:
                f.write(raw)

            baseline_profile = compute_baseline_profile(
                baseline_path=baseline_path,
                canonical_schema=canonical_json,
            )

            d.baseline_csv_path = baseline_path
            d.baseline_profile_json = json.dumps(baseline_profile or {}, default=str)

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(400, f"Failed to process baseline file: {e}")

    # timestamps
    if not d.created_at:
        d.created_at = now
    d.updated_at = now

    db.commit()
    db.refresh(d)

    has_baseline = bool(getattr(d, "baseline_csv_path", None)) or bool(getattr(d, "baseline_profile_json", None))

    return {"message": "Dataset registered/updated successfully", "has_baseline": has_baseline}


def get_dataset_config(db: Session, dataset_name: str):
    """Return canonical_schema, risk_config, baseline_profile for dataset."""
    d = db.query(Dataset).filter(Dataset.name == dataset_name).first()
    if not d:
        raise ValueError(f"Dataset '{dataset_name}' not found")

    # Prefer ACTIVE schema dict from registry (if present)
    active_schema = get_active_schema_json(db, d.id)
    if isinstance(active_schema, dict) and active_schema:
        canonical_schema = active_schema
    else:
        try:
            canonical_schema = json.loads(d.canonical_schema_json or "{}")
        except Exception:
            canonical_schema = {}

    # Baseline profile: if path exists but JSON missing, rebuild once (best effort)
    baseline_profile = None
    bp_json = getattr(d, "baseline_profile_json", None)
    bp_path = getattr(d, "baseline_csv_path", None)

    if bp_json:
        try:
            baseline_profile = json.loads(bp_json or "{}")
            if not isinstance(baseline_profile, dict):
                baseline_profile = None
        except Exception:
            baseline_profile = None

    if (baseline_profile is None or baseline_profile == {}) and bp_path:
        try:
            if Path(bp_path).exists():
                baseline_profile = compute_baseline_profile(baseline_path=bp_path, canonical_schema=canonical_schema)
                d.baseline_profile_json = json.dumps(baseline_profile or {}, default=str)
                db.commit()
        except Exception:
            pass

    # Risk config
        # Risk config (prefer risk_configs table if present)
    risk_config: dict[str, Any] = {}

    try:
        rc = db.query(RiskConfig).filter(RiskConfig.dataset_id == d.id).first()
        if rc and rc.config_json:
            risk_config = json.loads(rc.config_json or "{}")
    except Exception:
        risk_config = {}

    if not isinstance(risk_config, dict) or not risk_config:
        # fallback to Dataset.risk_config_json
        if getattr(d, "risk_config_json", None):
            try:
                risk_config = json.loads(d.risk_config_json or "{}")
            except Exception:
                risk_config = {}

    if not isinstance(risk_config, dict) or not risk_config:
        risk_config = {
            "mode": "A",
            "dataset_criticality": "Medium",
            "sensitivity_class": "None",
            "regulation_strictness": "Light",
            "key_fields": [],
        }

    # ensure mode always exists
    if not risk_config.get("mode"):
        risk_config["mode"] = getattr(d, "risk_mode", "A") or "A"

    return canonical_schema, risk_config, baseline_profile


# ----------------------------
# Helpers used by other endpoints
# ----------------------------

def extract_fields_from_canonical(canonical_schema: dict[str, Any]) -> list[str]:
    cols = canonical_schema.get("columns") or []
    out: list[str] = []
    for c in cols:
        if isinstance(c, dict) and c.get("name"):
            out.append(str(c["name"]))
    return sorted(set(out))


def suggest_key_fields(fields: list[str]) -> list[str]:
    patterns = [
        r"(^id$)",
        r"(_id$)",
        r"(customer)",
        r"(user)",
        r"(device)",
        r"(timestamp)",
        r"(time$)",
        r"(date$)",
        r"(amount)",
        r"(price)",
        r"(total)",
        r"(email)",
    ]
    rx = re.compile("|".join(patterns), re.IGNORECASE)
    scored = [f for f in fields if rx.search(f or "")]
    return scored[:12]

def save_risk_config(db: Session, dataset_name: str, risk_config: dict[str, Any]) -> dict[str, Any]:
    d = db.query(Dataset).filter(Dataset.name == dataset_name).first()
    if not d:
        raise ValueError("Dataset not found")

    if not isinstance(risk_config, dict):
        raise ValueError("risk_config must be a dict")

    mode = str(risk_config.get("mode") or "A").strip().upper()
    if mode not in ("A", "B"):
        mode = "A"
        risk_config["mode"] = "A"

    # keep Dataset columns (backward compatible)
    d.risk_mode = mode
    d.risk_config_json = json.dumps(risk_config, default=str)

    # ✅ store in risk_configs table (one row per dataset)
    try:
        rc = db.query(RiskConfig).filter(RiskConfig.dataset_id == d.id).first()
        if rc is None:
            rc = RiskConfig(dataset_id=d.id)
            db.add(rc)

        rc.mode = mode
        rc.config_json = json.dumps(risk_config, default=str)
        db.add(rc)
    except Exception:
        # if table isn't migrated yet, don't crash
        pass

    db.add(d)
    db.commit()
    db.refresh(d)

    return {"dataset": dataset_name, "mode": mode, "risk_config": risk_config}