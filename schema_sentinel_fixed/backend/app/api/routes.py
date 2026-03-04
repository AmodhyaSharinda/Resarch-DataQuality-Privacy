# backend/app/api/routes.py
from __future__ import annotations

import json
import os
import datetime as dt
from pathlib import Path
from typing import Any

import csv
import io
from fastapi.responses import FileResponse, Response
from app.db.models import RiskConfig

import pandas as pd
from confluent_kafka import Producer
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from starlette.responses import StreamingResponse

from app.api.drift_summary import router as drift_summary_router

from app.db.models import Dataset, DriftEvent, StagingRow, StoredBatch, RejectedRow
from app.db.session import get_db, SessionLocal
from app.api.schemas import (
    DatasetRegisterResponse,
    DatasetInfo,
    IngestBatchRequest,
    DriftEventResponse,
    ApproveRequest,
    RejectRequest,
    RollbackRequest,
    StagingRowsResponse,
    RiskConfigRequest,
    RiskConfigResponse,
    DatasetFieldsResponse,
)

from app.services.alerts import send_reject_alert
from app.services.audit_log import append_jsonl, tail_jsonl, get_log_file_path
from app.services.batch_files import save_upload, read_rows_from_file, count_rows
from app.services.datasets import (
    register_or_update_dataset,
    get_dataset_config,
    save_risk_config,
    extract_fields_from_canonical,
    suggest_key_fields,
)
from app.services.drift_engine import run_drift_detection
from app.services.drift_reporting import build_drift_explanations, format_drift_summary
from app.services.risk_engine import score_risk_and_route
from app.services.schema_registry import (
    ensure_canonical_active,
    get_dataset_by_name,
    list_versions,
    activate_version,
    insert_candidate_if_new,
    schema_hash_core,
    find_version_by_hash,
    get_active_schema_json,
)
from app.services.storage import write_rows_production, write_rows_staging, promote_ready_rows
from app.services.stream_hub import STREAM_HUB
from app.utils.time import to_utc_iso

router = APIRouter()

router.include_router(drift_summary_router)

@router.get("/health")
def health():
    return {"ok": True, "time": to_utc_iso(dt.datetime.utcnow()) or dt.datetime.utcnow().isoformat()}


# ----------------------------
# Datasets
# ----------------------------

@router.get("/datasets", response_model=list[DatasetInfo])
def list_datasets(db: Session = Depends(get_db)):
    rows = db.query(Dataset).order_by(Dataset.updated_at.desc()).all()
    return [
        DatasetInfo(
            name=d.name,
            created_at=to_utc_iso(d.created_at) or d.created_at.isoformat(),
            updated_at=to_utc_iso(d.updated_at) or d.updated_at.isoformat(),
            has_baseline=bool(getattr(d, "baseline_csv_path", None) or getattr(d, "baseline_profile_json", None)),
        )
        for d in rows
    ]


@router.get("/datasets/{dataset_name}")
def get_dataset(dataset_name: str, db: Session = Depends(get_db)):
    d = db.query(Dataset).filter(Dataset.name == dataset_name).first()
    if not d:
        raise HTTPException(404, "Dataset not found")

    try:
        risk_cfg = json.loads(getattr(d, "risk_config_json", "{}") or "{}")
        if not isinstance(risk_cfg, dict):
            risk_cfg = {}
    except Exception:
        risk_cfg = {}

    active_schema_json = get_active_schema_json(db, d.id)
    if isinstance(active_schema_json, dict) and active_schema_json:
        canonical_schema_obj = active_schema_json
    else:
        canonical_schema_obj = json.loads((d.canonical_schema_json or "{}") or "{}")

    return {
        "name": d.name,
        "created_at": to_utc_iso(d.created_at) or d.created_at.isoformat(),
        "updated_at": to_utc_iso(d.updated_at) or d.updated_at.isoformat(),
        "has_baseline": bool(getattr(d, "baseline_csv_path", None) or getattr(d, "baseline_profile_json", None)),
        "canonical_schema": canonical_schema_obj,
        "risk_mode": (getattr(d, "risk_mode", "A") or "A"),
        "risk_config": risk_cfg,
        "paths": {
            "canonical_schema_path": d.canonical_schema_path,
            "baseline_csv_path": d.baseline_csv_path,
        },
    }

@router.get("/datasets/{dataset_name}/fields", response_model=DatasetFieldsResponse)
def dataset_fields(dataset_name: str, db: Session = Depends(get_db)):
    d = db.query(Dataset).filter(Dataset.name == dataset_name).first()
    if not d:
        raise HTTPException(404, "Dataset not found")

    active_schema_json = get_active_schema_json(db, d.id)
    if isinstance(active_schema_json, dict) and active_schema_json:
        canonical = active_schema_json
    else:
        canonical = json.loads((d.canonical_schema_json or "{}") or "{}")

    fields = extract_fields_from_canonical(canonical)
    suggested = suggest_key_fields(fields)
    return DatasetFieldsResponse(dataset=dataset_name, fields=fields, suggested_key_fields=suggested)


@router.get("/datasets/{dataset_name}/risk-config", response_model=RiskConfigResponse)
def get_risk_config(dataset_name: str, db: Session = Depends(get_db)):
    d = db.query(Dataset).filter(Dataset.name == dataset_name).first()
    if not d:
        raise HTTPException(404, "Dataset not found")

    try:
        cfg = json.loads(getattr(d, "risk_config_json", "{}") or "{}")
        if not isinstance(cfg, dict):
            cfg = {}
    except Exception:
        cfg = {}

    mode = str(getattr(d, "risk_mode", "A") or "A").upper()
    return RiskConfigResponse(dataset=dataset_name, mode=mode, risk_config=cfg)


@router.post("/datasets/{dataset_name}/risk-config", response_model=RiskConfigResponse)
def set_risk_config(dataset_name: str, payload: RiskConfigRequest, db: Session = Depends(get_db)):
    cfg = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()

    mode = str(cfg.get("mode") or "A").strip().upper()
    if mode not in ("A", "B"):
        mode = "A"
    cfg["mode"] = mode

    if mode == "B":
        if cfg.get("dataset_criticality_num") is not None:
            try:
                cfg["dataset_criticality"] = int(cfg["dataset_criticality_num"])
            except Exception:
                cfg["dataset_criticality"] = 3
        cfg.pop("dataset_criticality_num", None)

    try:
        saved = save_risk_config(db, dataset_name, cfg)
    except ValueError as e:
        raise HTTPException(400, str(e))

    return RiskConfigResponse(dataset=dataset_name, mode=saved["mode"], risk_config=saved["risk_config"])


@router.get("/datasets/{dataset_name}/files")
def dataset_files(dataset_name: str, db: Session = Depends(get_db)):
    d = db.query(Dataset).filter(Dataset.name == dataset_name).first()
    if not d:
        raise HTTPException(404, "Dataset not found")

    mapping = {
        "canonical_schema": d.canonical_schema_path,
        "baseline_csv": d.baseline_csv_path,
    }

    files: list[dict[str, Any]] = []
    for kind, rel_path in mapping.items():
        if not rel_path:
            continue
        p = Path(rel_path)
        if not p.is_absolute():
            p = Path.cwd() / p
        exists = p.exists()
        stat = p.stat() if exists else None
        files.append(
            {
                "kind": kind,
                "filename": p.name,
                "path": str(rel_path),
                "exists": bool(exists),
                "size_bytes": int(stat.st_size) if stat else None,
                "updated_at": to_utc_iso(dt.datetime.utcfromtimestamp(stat.st_mtime)) if stat else None,
            }
        )

    return {"dataset": dataset_name, "files": files}


@router.get("/datasets/{dataset_name}/files/download")
def download_dataset_file(dataset_name: str, kind: str, db: Session = Depends(get_db)):
    d = db.query(Dataset).filter(Dataset.name == dataset_name).first()
    if not d:
        raise HTTPException(404, "Dataset not found")

    kind = (kind or "").strip().lower()
    mapping = {
        "canonical_schema": d.canonical_schema_path,
        "baseline_csv": d.baseline_csv_path,
    }
    rel_path = mapping.get(kind)
    if not rel_path:
        raise HTTPException(404, f"File kind '{kind}' not found for dataset")

    p = Path(rel_path)
    if not p.is_absolute():
        p = Path.cwd() / p
    if not p.exists():
        raise HTTPException(404, "File not found on disk")

    media = "application/octet-stream"
    if p.suffix.lower() == ".json":
        media = "application/json"
    elif p.suffix.lower() == ".csv":
        media = "text/csv"

    return FileResponse(path=str(p), filename=p.name, media_type=media)



@router.post("/datasets/register", response_model=DatasetRegisterResponse)
async def register_dataset(
    dataset: str = Form(...),
    canonical_schema: UploadFile = File(...),
    baseline_csv: UploadFile | str | None = File(None),
    db: Session = Depends(get_db),
):
    # robust: accept any UploadFile-like object
    baseline_file = baseline_csv if hasattr(baseline_csv, "filename") and hasattr(baseline_csv, "read") else None

    res = await register_or_update_dataset(
        db=db,
        dataset_name=dataset,
        canonical_schema_file=canonical_schema,
        baseline_csv_file=baseline_file,
    )

    try:
        d = db.query(Dataset).filter(Dataset.name == dataset).first()
        if d and d.canonical_schema_json:
            ensure_canonical_active(db, d, json.loads(d.canonical_schema_json), note="manual registration")
    except Exception:
        pass

    return DatasetRegisterResponse(
        dataset=dataset,
        ok=True,
        message=res["message"],
        has_baseline=res["has_baseline"],
    )
# ----------------------
# risk config
# ----------------

@router.get("/risk-configs")
def list_risk_configs(db: Session = Depends(get_db)):
    rows = db.query(Dataset).order_by(Dataset.name.asc()).all()

    out = []
    for d in rows:
        cfg = {}
        mode = getattr(d, "risk_mode", "A") or "A"
        updated = d.updated_at

        try:
            rc = db.query(RiskConfig).filter(RiskConfig.dataset_id == d.id).first()
            if rc and rc.config_json:
                cfg = json.loads(rc.config_json or "{}")
                mode = str(rc.mode or cfg.get("mode") or mode).upper()
                updated = rc.updated_at or updated
        except Exception:
            # fallback to dataset JSON if table not available
            try:
                cfg = json.loads(getattr(d, "risk_config_json", "{}") or "{}")
            except Exception:
                cfg = {}

        out.append(
            {
                "dataset": d.name,
                "mode": mode,
                "updated_at": to_utc_iso(updated) if updated else None,
                "dataset_criticality": cfg.get("dataset_criticality"),
                "sensitivity_class": cfg.get("sensitivity_class"),
                "regulation_strictness": cfg.get("regulation_strictness"),
                "key_fields_count": len(cfg.get("key_fields") or []) if isinstance(cfg.get("key_fields"), list) else 0,
                "config": cfg,
            }
        )

    return out

# ----------------------------
# Ingest
# ----------------------------

def _persist_event_and_rows(
    db: Session,
    dataset_name: str,
    batch_id: str,
    rows: list[dict[str, Any]],
    drift: dict[str, Any],
    risk: dict[str, Any],
    canonical_schema: dict[str, Any],
) -> DriftEvent:
    d = db.query(Dataset).filter(Dataset.name == dataset_name).first()
    if not d:
        raise HTTPException(400, f"Dataset '{dataset_name}' is not registered. Register it first.")

    observed_schema = drift.get("observed_schema") or {}
    cols = observed_schema.get("columns") or []
    observed_columns = sorted([c.get("name") for c in cols if isinstance(c, dict) and c.get("name")])

    schema_hash_active = schema_hash_core(canonical_schema or {})
    schema_hash_observed = schema_hash_core(observed_schema or {})
    schema_version_candidate: int | None = None

    if observed_schema and (schema_hash_observed != schema_hash_active):
        try:
            inserted = insert_candidate_if_new(
                db,
                d,
                observed_schema,
                note=f"batch={batch_id} drift={','.join(drift.get('drift_types') or [])}",
            )
            if inserted is not None:
                schema_version_candidate = int(inserted.version)
            else:
                existing = find_version_by_hash(db, d.id, schema_hash_observed)
                if existing is not None:
                    schema_version_candidate = int(existing.version)
        except Exception:
            schema_version_candidate = None

    explanations = build_drift_explanations(drift)

    drift_types = drift.get("drift_types") or []
    has_drift = len(drift_types) > 0

    suggested_route = str(risk.get("route", "production")).lower().strip()
    final_route = "staging" if has_drift else suggested_route
    final_status = "PENDING" if has_drift else "APPROVED"

    summary = format_drift_summary(
        dataset=dataset_name,
        batch_id=batch_id,
        observed_columns=observed_columns,
        drift=drift,
        risk=risk,
        schema_version=schema_version_candidate,
    )

    diff_payload = {
        "diff": drift.get("diff") or {},
        "raw_diff": drift.get("raw_diff") or {},
        "observed_columns": observed_columns,
        "observed_schema": observed_schema,
        "schema_hash_active": schema_hash_active,
        "schema_hash_observed": schema_hash_observed,
        "schema_version_candidate": schema_version_candidate,
        "explanations": explanations,
        "summary": summary,

        # risk/XAI for UI tabs
        "risk_details": risk.get("details"),
        "xai": risk.get("xai"),
        "risk_reasons": risk.get("reasons"),

        # governance transparency
        "route_suggested": suggested_route,
        "route_applied": final_route,
        "approval_required": bool(has_drift),
    }

    ev = DriftEvent(
        dataset_id=d.id,
        batch_id=batch_id,
        drift_types=",".join(drift_types),
        diff_json=json.dumps(diff_payload, default=str),
        rename_json=json.dumps(drift.get("renames", {}) or {}, default=str),
        risk_score=float(risk.get("risk_score", 0.0)),
        risk_level=str(risk.get("risk_level", "LOW")),
        route=final_route,
        status=final_status,
    )
    db.add(ev)
    db.commit()
    db.refresh(ev)

    # Route rows based on FINAL route (not suggested)
    if final_route == "production":
        write_rows_production(
            db=db,
            dataset_id=d.id,
            batch_id=batch_id,
            canonical_schema=canonical_schema,
            rows=rows,
            renames=drift.get("renames") or {},
        )
    else:
        write_rows_staging(
            db=db,
            event_id=ev.id,
            canonical_schema=canonical_schema,
            rows=rows,
            renames=drift.get("renames") or {},
        )

    try:
        append_jsonl(
  {
    "ts": to_utc_iso(ev.detected_at) or ev.detected_at.isoformat(),
    "event_id": ev.id,
    "dataset": dataset_name,
    "batch_id": batch_id,
    "risk_score": risk.get("risk_score"),
    "risk_level": risk.get("risk_level"),
    "route_suggested": suggested_route,
    "route_applied": final_route,
    "status": ev.status,
    "drift_types": drift_types,
    "schema_hash_active": schema_hash_active,
    "schema_hash_observed": schema_hash_observed,
    "schema_version_candidate": schema_version_candidate,
    "summary": summary,

    # ✅ add these 3 for audit/debug
    "diff": drift.get("diff"),
    "raw_diff": drift.get("raw_diff"),
    "renames": drift.get("renames"),
  }
)
    except Exception:
        pass

    return ev


@router.post("/batches/ingest", response_model=DriftEventResponse)
def ingest_batch(payload: IngestBatchRequest, db: Session = Depends(get_db)):
    canonical_schema, risk_config, baseline_profile = get_dataset_config(db, payload.dataset)

    drift = run_drift_detection(rows=payload.rows, canonical_schema=canonical_schema, baseline_profile=baseline_profile)
    risk = score_risk_and_route(drift=drift, risk_config=risk_config)

    ev = _persist_event_and_rows(db, payload.dataset, payload.batch_id, payload.rows, drift, risk, canonical_schema)

    return DriftEventResponse(
        event_id=ev.id,
        dataset=payload.dataset,
        batch_id=payload.batch_id,
        drift_types=drift.get("drift_types", []) or [],
        diff=drift.get("diff") or {},
        renames=drift.get("renames") or {},
        risk_score=risk.get("risk_score", 0.0),
        risk_level=risk.get("risk_level", "LOW"),
        route=ev.route,
        status=ev.status,
    )


@router.post("/batches/ingest_csv", response_model=DriftEventResponse)
async def ingest_batch_csv(
    dataset: str = Form(...),
    batch_id: str = Form(...),
    csv_file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    try:
        content = await csv_file.read()
        from io import BytesIO

        fname = (csv_file.filename or "").lower()
        if fname.endswith(".csv") or not fname:
            df = pd.read_csv(BytesIO(content))
        elif fname.endswith((".xlsx", ".xls", ".xlsm")):
            engines = ["openpyxl", None, "xlrd"] if fname.endswith((".xlsx", ".xlsm")) else ["xlrd", None, "openpyxl"]
            df = None
            last_err: Exception | None = None
            for eng in engines:
                try:
                    df = pd.read_excel(BytesIO(content), engine=eng) if eng else pd.read_excel(BytesIO(content))
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
            if df is None:
                raise ValueError(f"Excel file format cannot be determined: {last_err}")
        else:
            df = pd.read_csv(BytesIO(content))

        df = df.where(pd.notnull(df), None)
        rows = df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(400, f"Failed to read file: {e}")

    canonical_schema, risk_config, baseline_profile = get_dataset_config(db, dataset)
    drift = run_drift_detection(rows=rows, canonical_schema=canonical_schema, baseline_profile=baseline_profile)
    risk = score_risk_and_route(drift=drift, risk_config=risk_config)

    ev = _persist_event_and_rows(db, dataset, batch_id, rows, drift, risk, canonical_schema)

    return DriftEventResponse(
        event_id=ev.id,
        dataset=dataset,
        batch_id=batch_id,
        drift_types=drift.get("drift_types", []) or [],
        diff=drift.get("diff") or {},
        renames=drift.get("renames") or {},
        risk_score=risk.get("risk_score", 0.0),
        risk_level=risk.get("risk_level", "LOW"),
        route=ev.route,
        status=ev.status,
    )


# ----------------------------
# Events
# ----------------------------

@router.get("/events")
def list_events(
    db: Session = Depends(get_db),
    limit: int = 200,
    status: str | None = None,
    route: str | None = None,
    actioned: bool | None = None,  # if true -> returns events with decisions taken
):
    q = db.query(DriftEvent)

    # optional filters
    if status:
        q = q.filter(DriftEvent.status == status.strip().upper())

    if route:
        q = q.filter(DriftEvent.route == route.strip().lower())

    if actioned is True:
        q = q.filter(DriftEvent.status != "PENDING")
    elif actioned is False:
        q = q.filter(DriftEvent.status == "PENDING")

    events = q.order_by(DriftEvent.id.desc()).limit(min(limit, 500)).all()

    # build dataset_id -> name map in one query (faster than querying per row)
    dataset_ids = list({e.dataset_id for e in events})
    ds_map = {d.id: d.name for d in db.query(Dataset).filter(Dataset.id.in_(dataset_ids)).all()} if dataset_ids else {}

    out = []
    for ev in events:
        out.append(
            {
                "id": ev.id,
                "dataset": ds_map.get(ev.dataset_id, str(ev.dataset_id)),
                "batch_id": ev.batch_id,
                "drift_types": ev.drift_types.split(",") if ev.drift_types else [],
                "risk_score": ev.risk_score,
                "risk_level": ev.risk_level,
                "route": ev.route,
                "status": ev.status,
                "detected_at": to_utc_iso(ev.detected_at) or ev.detected_at.isoformat(),
            }
        )
    return out


@router.get("/staging/queue")
def staging_queue(db: Session = Depends(get_db), limit: int = 200):
    q = (
        db.query(DriftEvent)
        .filter(DriftEvent.route == "staging")
        .filter(DriftEvent.status == "PENDING")
        .order_by(DriftEvent.id.desc())
        .limit(min(limit, 500))
    )
    events = q.all()

    dataset_ids = list({e.dataset_id for e in events})
    ds_map = {d.id: d.name for d in db.query(Dataset).filter(Dataset.id.in_(dataset_ids)).all()} if dataset_ids else {}

    return [
        {
            "id": ev.id,
            "dataset": ds_map.get(ev.dataset_id, str(ev.dataset_id)),
            "batch_id": ev.batch_id,
            "drift_types": ev.drift_types.split(",") if ev.drift_types else [],
            "risk_score": ev.risk_score,
            "risk_level": ev.risk_level,
            "route": ev.route,
            "status": ev.status,
            "detected_at": to_utc_iso(ev.detected_at) or ev.detected_at.isoformat(),
        }
        for ev in events
    ]




@router.get("/events/{event_id}")
def get_event(event_id: int, db: Session = Depends(get_db)):
    ev = db.query(DriftEvent).filter(DriftEvent.id == event_id).first()
    if not ev:
        raise HTTPException(404, "Event not found")
    ds = db.query(Dataset).filter(Dataset.id == ev.dataset_id).first()

    diff_payload = json.loads(ev.diff_json or "{}")
    diff_obj = diff_payload.get("diff") if isinstance(diff_payload, dict) and "diff" in diff_payload else diff_payload

    return {
        "id": ev.id,
        "dataset": ds.name if ds else str(ev.dataset_id),
        "batch_id": ev.batch_id,
        "drift_types": ev.drift_types.split(",") if ev.drift_types else [],
        "diff": diff_obj,
        "raw_diff": diff_payload.get("raw_diff") if isinstance(diff_payload, dict) else None,
        "renames": json.loads(ev.rename_json or "{}"),
        "summary": diff_payload.get("summary") if isinstance(diff_payload, dict) else None,
        "explanations": diff_payload.get("explanations") if isinstance(diff_payload, dict) else None,
        "observed_columns": diff_payload.get("observed_columns") if isinstance(diff_payload, dict) else None,
        "observed_schema": diff_payload.get("observed_schema") if isinstance(diff_payload, dict) else None,
        "schema_hash_active": diff_payload.get("schema_hash_active") if isinstance(diff_payload, dict) else None,
        "schema_hash_observed": diff_payload.get("schema_hash_observed") if isinstance(diff_payload, dict) else None,
        "schema_version_candidate": diff_payload.get("schema_version_candidate") if isinstance(diff_payload, dict) else None,
        "schema_version_activated": diff_payload.get("schema_version_activated") if isinstance(diff_payload, dict) else None,
        "risk_details": diff_payload.get("risk_details") if isinstance(diff_payload, dict) else None,
        "xai": diff_payload.get("xai") if isinstance(diff_payload, dict) else None,
        "risk_reasons": diff_payload.get("risk_reasons") if isinstance(diff_payload, dict) else None,
        "risk_score": ev.risk_score,
        "risk_level": ev.risk_level,
        "route": ev.route,
        "status": ev.status,
        "detected_at": to_utc_iso(ev.detected_at) or ev.detected_at.isoformat(),
        "decision_by": ev.decision_by,
        "decision_reason": ev.decision_reason,
        "decision_at": to_utc_iso(ev.decision_at) if ev.decision_at else None,
    }


@router.get("/events/{event_id}/staging", response_model=StagingRowsResponse)
def get_staging_rows(event_id: int, db: Session = Depends(get_db), limit: int = 50, offset: int = 0):
    q = db.query(StagingRow).filter(StagingRow.event_id == event_id)
    total = q.count()
    rows = q.order_by(StagingRow.id.asc()).offset(max(offset, 0)).limit(min(limit, 200)).all()
    parsed = [json.loads(r.row_json or "{}") for r in rows]
    return StagingRowsResponse(event_id=event_id, total=total, rows=parsed)


def _activate_candidate_schema_if_any(db: Session, ev: DriftEvent) -> int | None:
    """Activate schema_version_candidate stored in diff_json. Returns activated version or None."""
    try:
        diff_payload = json.loads(ev.diff_json or "{}")
        cand_v = diff_payload.get("schema_version_candidate") if isinstance(diff_payload, dict) else None
        if cand_v is None:
            return None
        activate_version(db, ev.dataset_id, int(cand_v))
        diff_payload["schema_version_activated"] = int(cand_v)
        ev.diff_json = json.dumps(diff_payload, default=str)
        return int(cand_v)
    except Exception:
        return None


def _activate_previous_schema_for_event(db: Session, ev: DriftEvent) -> int | None:
    """
    Rollback to the schema that was ACTIVE when this event was detected.
    We try to find by schema_hash_active stored in diff_json.
    """
    try:
        ds = db.query(Dataset).filter(Dataset.id == ev.dataset_id).first()
        if not ds:
            return None

        diff_payload = json.loads(ev.diff_json or "{}")
        h = diff_payload.get("schema_hash_active") if isinstance(diff_payload, dict) else None
        if not h:
            return None

        prev = find_version_by_hash(db, ds.id, str(h))
        if prev is None:
            return None

        activate_version(db, ds.id, int(prev.version))
        diff_payload["schema_version_activated"] = int(prev.version)
        diff_payload["rollback_to_hash"] = str(h)
        ev.diff_json = json.dumps(diff_payload, default=str)
        return int(prev.version)
    except Exception:
        return None


@router.post("/events/{event_id}/approve")
def approve_and_promote(event_id: int, payload: ApproveRequest, db: Session = Depends(get_db)):
    """
    ✅ One-click governance:
      Approve = activate candidate schema (if any) + promote staged rows.
    """
    ev = db.query(DriftEvent).filter(DriftEvent.id == event_id).first()
    if not ev:
        raise HTTPException(404, "Event not found")

    ev.status = "APPROVED"
    ev.decision_by = payload.approver
    ev.decision_reason = payload.note
    ev.decision_at = dt.datetime.utcnow()

    # Mark staged rows READY so promote_ready_rows will move them
    for r in ev.staged_rows:
        r.status = "READY"

    activated_version = _activate_candidate_schema_if_any(db, ev)

    # Promote
    ds = db.query(Dataset).filter(Dataset.id == ev.dataset_id).first()
    if not ds:
        raise HTTPException(400, "Dataset config missing")

    canonical_schema, _, _ = get_dataset_config(db, ds.name)
    promoted = promote_ready_rows(
        db,
        event=ev,
        canonical_schema=canonical_schema,
        renames=json.loads(ev.rename_json or "{}"),
    )

    ev.status = "PROMOTED"
    db.commit()

    try:
        append_jsonl(
            {
                "ts": to_utc_iso(dt.datetime.utcnow()) or dt.datetime.utcnow().isoformat(),
                "event_id": event_id,
                "action": "APPROVE_AND_PROMOTE",
                "approver": payload.approver,
                "note": payload.note,
                "activated_schema_version": activated_version,
                "promoted_rows": promoted,
            }
        )
    except Exception:
        pass

    return {
        "ok": True,
        "event_id": event_id,
        "status": ev.status,
        "activated_schema_version": activated_version,
        "promoted_rows": promoted,
    }


@router.post("/events/{event_id}/reject")
def reject_event(event_id: int, payload: RejectRequest, db: Session = Depends(get_db)):
    """Reject a pending staging event and move its staged rows into rejected_rows."""
    ev = db.query(DriftEvent).filter(DriftEvent.id == event_id).first()
    if not ev:
        raise HTTPException(404, "Event not found")

    ev.status = "REJECTED"
    ev.decision_by = payload.approver
    ev.decision_reason = payload.reason or payload.note
    ev.decision_at = dt.datetime.utcnow()

    # Move rows out of staging -> rejected_rows (staging is a queue)
    moved_rows = 0
    for r in list(ev.staged_rows):
        rr = RejectedRow(
            dataset_id=ev.dataset_id,
            event_id=ev.id,
            batch_id=ev.batch_id or "",
            row_json=r.row_json,
            rejected_by=payload.approver,
            rejected_reason=payload.reason or payload.note,
            notify_email=payload.notify_email,
        )
        db.add(rr)
        db.delete(r)
        moved_rows += 1

    db.commit()

    # Best-effort email alert (recipient can be provided by the user)
    try:
        drift_types = ev.drift_types.split(",") if ev.drift_types else []
        alert_res = send_reject_alert(
            dataset_name=ev.dataset.name if ev.dataset else str(ev.dataset_id),
            event_id=ev.id,
            drift_types=drift_types,
            reason=payload.reason or payload.note,
            diff=json.loads(ev.diff_json or "{}") if ev.diff_json else None,
            to_email=payload.notify_email,
        )
    except Exception as e:
        alert_res = {"sent": False, "error": str(e)}

    try:
        append_jsonl(
            {
                "ts": to_utc_iso(dt.datetime.utcnow()) or dt.datetime.utcnow().isoformat(),
                "event_id": event_id,
                "action": "REJECT",
                "approver": payload.approver,
                "reason": payload.reason,
                "note": payload.note,
                "notify_email": payload.notify_email,
                "moved_rows": moved_rows,
                "alert": alert_res,
            }
        )
    except Exception:
        pass

    return {"ok": True, "event_id": event_id, "status": ev.status, "moved_rows": moved_rows, "alert": alert_res}


@router.post("/events/{event_id}/rollback")
def rollback_event(event_id: int, payload: RollbackRequest, db: Session = Depends(get_db)):
    """
    ✅ Rollback:
      - revert ACTIVE schema to what it was when drift was detected
      - then promote staged rows under that old schema shape (best effort)

    UI should explain:
      - ADD columns dropped
      - missing cols become null (risky)
      - rename applied if mapping exists
      - unsafe type casts should be rejected (depends on storage implementation)
    """
    ev = db.query(DriftEvent).filter(DriftEvent.id == event_id).first()
    if not ev:
        raise HTTPException(404, "Event not found")

    ev.status = "ROLLBACK"
    ev.decision_by = payload.approver
    ev.decision_reason = payload.reason or payload.note
    ev.decision_at = dt.datetime.utcnow()

    # Activate previous schema
    activated_version = _activate_previous_schema_for_event(db, ev)

    # Promote under old schema:
    # mark READY and promote
    for r in ev.staged_rows:
        r.status = "READY"

    ds = db.query(Dataset).filter(Dataset.id == ev.dataset_id).first()
    if not ds:
        raise HTTPException(400, "Dataset config missing")

    canonical_schema, _, _ = get_dataset_config(db, ds.name)
    promoted = promote_ready_rows(
        db,
        event=ev,
        canonical_schema=canonical_schema,
        renames=json.loads(ev.rename_json or "{}"),
    )

    ev.status = "ROLLED_BACK"
    db.commit()

    try:
        append_jsonl(
            {
                "ts": to_utc_iso(dt.datetime.utcnow()) or dt.datetime.utcnow().isoformat(),
                "event_id": event_id,
                "action": "ROLLBACK",
                "approver": payload.approver,
                "reason": payload.reason,
                "note": payload.note,
                "activated_schema_version": activated_version,
                "promoted_rows": promoted,
            }
        )
    except Exception:
        pass

    return {
        "ok": True,
        "event_id": event_id,
        "status": ev.status,
        "activated_schema_version": activated_version,
        "promoted_rows": promoted,
    }


# ---------------------------------------------------------------------------
# Schema Registry (unchanged)
# ---------------------------------------------------------------------------

@router.get("/registry/{dataset_name}")
def registry_list(dataset_name: str, db: Session = Depends(get_db)):
    d = get_dataset_by_name(db, dataset_name)
    if not d:
        raise HTTPException(404, "Dataset not found")

    versions = list_versions(db, d.id)
    out = []
    for v in versions:
        out.append(
            {
                "id": v.id,
                "dataset": dataset_name,
                "version": v.version,
                "active": bool(v.active),
                "schema_hash": v.schema_hash,
                "created_at": to_utc_iso(v.created_at) if v.created_at else None,
                "note": v.note,
                "schema": json.loads(v.schema_json or "{}"),
            }
        )
    return {"dataset": dataset_name, "versions": out}


@router.post("/registry/{dataset_name}/activate/{version}")
def registry_activate(dataset_name: str, version: int, db: Session = Depends(get_db)):
    d = get_dataset_by_name(db, dataset_name)
    if not d:
        raise HTTPException(404, "Dataset not found")

    row = activate_version(db, d.id, int(version))
    if not row:
        raise HTTPException(404, "Version not found")
    return {"ok": True, "dataset": dataset_name, "version": row.version, "active": True}


# ---------------------------------------------------------------------------
# Live Stream (unchanged)
# ---------------------------------------------------------------------------

@router.post("/stream/batches/upload")
async def upload_stream_batch(
    background: BackgroundTasks,
    dataset: str = Form(...),
    batch_id: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    d = db.query(Dataset).filter(Dataset.name == dataset).first()
    if not d:
        raise HTTPException(404, "Dataset not found")

    raw = await file.read()
    path = save_upload(dataset, batch_id, file.filename or "batch.csv", raw)

    try:
        canonical_schema, _, _ = get_dataset_config(db, dataset)
        expected_cols = [c["name"] for c in (canonical_schema.get("columns") or []) if isinstance(c, dict) and c.get("name")]
        rc = count_rows(path, expected_cols=expected_cols)
    except Exception:
        rc = count_rows(path)

    sb = StoredBatch(
        dataset_id=d.id,
        batch_id=batch_id,
        filename=file.filename or "batch.csv",
        path=str(path),
        row_count=int(rc) if rc else None,
    )
    db.add(sb)
    db.commit()
    db.refresh(sb)

    await STREAM_HUB.publish(
        {"type": "batch_uploaded", "dataset": dataset, "batch_id": batch_id, "filename": sb.filename, "row_count": sb.row_count}
    )

    return {
        "ok": True,
        "id": sb.id,
        "dataset": dataset,
        "batch_id": batch_id,
        "filename": sb.filename,
        "row_count": sb.row_count,
        "uploaded_at": to_utc_iso(sb.uploaded_at) if sb.uploaded_at else None,
    }


@router.get("/stream/batches")
def list_stream_batches(dataset: str | None = None, db: Session = Depends(get_db)):
    q = db.query(StoredBatch).join(Dataset, Dataset.id == StoredBatch.dataset_id)
    if dataset:
        q = q.filter(Dataset.name == dataset)

    rows = q.order_by(StoredBatch.uploaded_at.desc()).limit(200).all()
    out = []
    for r in rows:
        ds = db.query(Dataset).filter(Dataset.id == r.dataset_id).first()
        out.append(
            {
                "id": r.id,
                "dataset": ds.name if ds else str(r.dataset_id),
                "batch_id": r.batch_id,
                "filename": r.filename,
                "row_count": r.row_count,
                "uploaded_at": to_utc_iso(r.uploaded_at) if r.uploaded_at else None,
            }
        )
    return {"batches": out}


async def _simulate_batch_task(dataset: str, batch_id: str, path: str, topic: str, pace_ms: int = 0):
    await STREAM_HUB.publish({"type": "simulate_start", "dataset": dataset, "batch_id": batch_id, "topic": topic})

    # ✅ IMPORTANT:
    # Do NOT force file rows into the canonical schema here.
    # Drift detection must see the RAW columns from the uploaded file,
    # otherwise ADD/REMOVE/RENAME become invisible.
    try:
        rows = read_rows_from_file(path)  # <-- no expected_cols
    except Exception as e:
        await STREAM_HUB.publish({"type": "simulate_error", "dataset": dataset, "batch_id": batch_id, "error": str(e)})
        return

    bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    producer = Producer({"bootstrap.servers": bootstrap})

    sent = 0
    for row in rows:
        try:
            msg = {"dataset": dataset, "batch_id": batch_id, "payload": row}
            producer.produce(topic, json.dumps(msg).encode("utf-8"))
            producer.poll(0)
            sent += 1
            await STREAM_HUB.publish({"type": "kafka_sent", "dataset": dataset, "batch_id": batch_id, "n": sent, "row": row})
        except Exception as e:
            await STREAM_HUB.publish({"type": "kafka_error", "dataset": dataset, "batch_id": batch_id, "error": str(e)})
            break

        if pace_ms and pace_ms > 0:
            import asyncio
            await asyncio.sleep(pace_ms / 1000.0)

    try:
        end_msg = {"dataset": dataset, "type": "end_batch", "batch_id": batch_id}
        producer.produce(topic, json.dumps(end_msg).encode("utf-8"))
        producer.poll(0)
        await STREAM_HUB.publish({"type": "kafka_sent", "dataset": dataset, "batch_id": batch_id, "n": sent, "message": "end_batch"})
        producer.flush(10)
    except Exception:
        pass

    await STREAM_HUB.publish({"type": "simulate_done", "dataset": dataset, "batch_id": batch_id, "sent": sent})


@router.post("/stream/batches/{stored_batch_id}/simulate")
def simulate_stored_batch(
    stored_batch_id: int,
    background: BackgroundTasks,
    pace_ms: int = 0,
    topic: str | None = None,
    db: Session = Depends(get_db),
):
    sb = db.query(StoredBatch).filter(StoredBatch.id == stored_batch_id).first()
    if not sb:
        raise HTTPException(404, "Stored batch not found")

    ds = db.query(Dataset).filter(Dataset.id == sb.dataset_id).first()
    dataset_name = ds.name if ds else str(sb.dataset_id)

    topic_final = topic or os.getenv("KAFKA_TOPICS", "ingest").split(",")[0].strip() or "ingest"
    background.add_task(_simulate_batch_task, dataset_name, sb.batch_id, sb.path, topic_final, int(pace_ms))
    return {"ok": True, "status": "STARTED", "stored_batch_id": stored_batch_id, "dataset": dataset_name, "batch_id": sb.batch_id, "topic": topic_final}


@router.get("/stream/console")
async def stream_console():
    async def event_gen():
        q, history = await STREAM_HUB.subscribe()
        try:
            for h in history:
                yield f"data: {h}\n\n"
            while True:
                msg = await q.get()
                yield f"data: {msg}\n\n"
        finally:
            await STREAM_HUB.unsubscribe(q)

    return StreamingResponse(event_gen(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Audit Log
# ---------------------------------------------------------------------------

@router.get("/audit/logs")
def audit_logs(limit: int = 200):
    return tail_jsonl(limit=limit)


@router.get("/audit/logs/download")
def audit_logs_download(dataset: str | None = None):
    """
    Download audit log as a clean CSV (Excel-friendly).
    Optional: ?dataset=<name> to export only that dataset.
    """
    p = Path(get_log_file_path())
    if not p.exists():
        raise HTTPException(404, "Audit log file not found")

    # Read + parse JSONL
    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as e:
        raise HTTPException(400, f"Failed to read audit log: {e}")

    events: list[dict[str, Any]] = []
    ds = (dataset or "").strip().lower()

    for line in lines:
        s = (line or "").strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
            if not isinstance(obj, dict):
                obj = {"value": obj}
        except Exception:
            obj = {"raw": s}

        # optional dataset filter
        if ds:
            dsn = str(obj.get("dataset") or obj.get("dataset_name") or "").strip().lower()
            if dsn != ds:
                continue

        events.append(obj)

    # Choose a stable, readable CSV schema
    cols = [
        "ts",
        "dataset",
        "event_id",
        "batch_id",
        "action",
        "approver",
        "status",
        "drift_types",
        "risk_score",
        "risk_level",
        "route_suggested",
        "route_applied",
        "note",
        "reason",
        "summary",
        "schema_version_candidate",
        "schema_hash_active",
        "schema_hash_observed",
        "activated_schema_version",
        "promoted_rows",
        "moved_rows",
        "raw_json",
    ]

    def get_any(o: dict, keys: list[str]):
        for k in keys:
            if k in o and o.get(k) is not None:
                return o.get(k)
        return None

    # Build rows
    rows = []
    for e in events:
        row = {}
        row["ts"] = get_any(e, ["ts", "created_at", "timestamp"]) or ""
        row["dataset"] = get_any(e, ["dataset", "dataset_name"]) or ""
        row["event_id"] = get_any(e, ["event_id", "id"]) or ""
        row["batch_id"] = get_any(e, ["batch_id", "batch"]) or ""
        row["action"] = (get_any(e, ["action"]) or "")
        row["approver"] = (get_any(e, ["approver", "decision_by", "user"]) or "")
        row["status"] = (get_any(e, ["status"]) or "")

        dt = get_any(e, ["drift_types"])
        if isinstance(dt, list):
            row["drift_types"] = ",".join([str(x) for x in dt])
        else:
            row["drift_types"] = str(dt or "")

        row["risk_score"] = get_any(e, ["risk_score"]) or ""
        row["risk_level"] = get_any(e, ["risk_level"]) or ""
        row["route_suggested"] = get_any(e, ["route_suggested"]) or ""
        row["route_applied"] = get_any(e, ["route_applied"]) or ""

        row["note"] = get_any(e, ["note"]) or ""
        row["reason"] = get_any(e, ["reason"]) or ""
        row["summary"] = get_any(e, ["summary"]) or ""

        row["schema_version_candidate"] = get_any(e, ["schema_version_candidate"]) or ""
        row["schema_hash_active"] = get_any(e, ["schema_hash_active"]) or ""
        row["schema_hash_observed"] = get_any(e, ["schema_hash_observed"]) or ""
        row["activated_schema_version"] = get_any(e, ["activated_schema_version"]) or ""
        row["promoted_rows"] = get_any(e, ["promoted_rows"]) or ""
        row["moved_rows"] = get_any(e, ["moved_rows"]) or ""

        # Keep the full original JSON in one column (so nothing is lost)
        row["raw_json"] = json.dumps(e, ensure_ascii=False, default=str)

        rows.append(row)

    # Write CSV with proper quoting (Excel friendly)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore", quoting=csv.QUOTE_MINIMAL)
    w.writeheader()
    for r in rows:
        w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in cols})

    csv_text = buf.getvalue()
    fname = f"audit_logs_{dataset}.csv" if dataset else "audit_logs.csv"

    return Response(
        content=csv_text,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )
