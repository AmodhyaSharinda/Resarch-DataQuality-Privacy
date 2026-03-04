# backend/app/workers/kafka_consumer.py
from __future__ import annotations

import json
import os
import time
from typing import Any

from confluent_kafka import Consumer, KafkaError, KafkaException

from app.db.session import SessionLocal
from app.db.models import Dataset, DriftEvent
from app.services.datasets import get_dataset_config
from app.services.drift_engine import run_drift_detection
from app.services.risk_engine import score_risk_and_route
from app.services.storage import write_rows_production, write_rows_staging
from app.services.drift_reporting import build_drift_explanations, format_drift_summary
from app.services.schema_registry import insert_candidate_if_new, schema_hash_core, find_version_by_hash
from app.services.audit_log import append_jsonl
from app.utils.time import to_utc_iso


def _rename_map_from_payload(renames_payload: Any) -> dict[str, str]:
    """
    Your drift_engine returns either:
      - mapping directly: {old: new}
      - OR details dict: {"mappings": {old:new}, "candidates":[...], ...}

    Old project + storage/frontend typically expect ONLY {old:new}.
    """
    if isinstance(renames_payload, dict):
        m = renames_payload.get("mappings")
        if isinstance(m, dict):
            return {str(k): str(v) for k, v in m.items()}
        # already mapping
        return {str(k): str(v) for k, v in renames_payload.items() if isinstance(v, str)}
    return {}


def _flush_batch(dataset: str, batch_id: str, rows: list[dict[str, Any]]) -> None:
    db = SessionLocal()
    try:
        d = db.query(Dataset).filter(Dataset.name == dataset).first()
        if not d:
            raise ValueError(f"Dataset '{dataset}' not registered")

        canonical_schema, risk_config, baseline_profile = get_dataset_config(db, dataset)

        drift = run_drift_detection(
            rows=rows,
            canonical_schema=canonical_schema,
            baseline_profile=baseline_profile,
        )

        risk = score_risk_and_route(drift=drift, risk_config=risk_config)

        observed_schema = drift.get("observed_schema") or {}
        cols = observed_schema.get("columns") or []
        observed_columns = sorted([c.get("name") for c in cols if isinstance(c, dict) and c.get("name")])

        # --- Schema Registry candidate ---
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
            dataset=dataset,
            batch_id=batch_id,
            observed_columns=observed_columns,
            drift=drift,
            risk=risk,
            schema_version=schema_version_candidate,
        )

        # IMPORTANT: keep full rename payload in DB, but extract mapping for storage
        renames_payload = drift.get("renames") or {}
        renames_map = _rename_map_from_payload(renames_payload)

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
            "risk_details": risk.get("details"),
            "xai": risk.get("xai"),
            "risk_reasons": risk.get("reasons"),
            "route_suggested": suggested_route,
            "route_applied": final_route,
            "approval_required": bool(has_drift),
        }

        ev = DriftEvent(
            dataset_id=d.id,
            batch_id=batch_id,
            drift_types=",".join(drift_types),
            diff_json=json.dumps(diff_payload, default=str),
            rename_json=json.dumps(renames_payload, default=str),  # store full payload
            risk_score=float(risk.get("risk_score", 0.0)),
            risk_level=str(risk.get("risk_level", "LOW")),
            route=final_route,
            status=final_status,
        )
        db.add(ev)
        db.commit()
        db.refresh(ev)

        # Route rows based on FINAL route (use mapping ONLY)
        if final_route == "production":
            write_rows_production(
                db=db,
                dataset_id=d.id,
                batch_id=batch_id,
                canonical_schema=canonical_schema,
                rows=rows,
                renames=renames_map,
            )
        else:
            write_rows_staging(
                db=db,
                event_id=ev.id,
                canonical_schema=canonical_schema,
                rows=rows,
                renames=renames_map,
            )

        # Log (append_jsonl now supports dict safely)
        try:
            append_jsonl(
                {
                    "ts": to_utc_iso(ev.detected_at) or ev.detected_at.isoformat(),
                    "event_id": ev.id,
                    "dataset": dataset,
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
                    "diff": drift.get("diff"),
                    "renames": renames_map,  # mapping for readability
                }
            )
        except Exception:
            pass

        print(f"[{dataset} | {batch_id}] stored event={ev.id} route={final_route} risk={float(risk.get('risk_score', 0.0)):.3f}")

    finally:
        db.close()


def main():
    bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    group_id = os.getenv("KAFKA_GROUP_ID", "schema_sentinel")
    topics = [t.strip() for t in (os.getenv("KAFKA_TOPICS", "ingest").split(",")) if t.strip()]

    conf = {
        "bootstrap.servers": bootstrap,
        "group.id": group_id,
        "auto.offset.reset": "earliest",
        "enable.auto.commit": True,
        "session.timeout.ms": 45000,
    }

    consumer = Consumer(conf)
    consumer.subscribe(topics)
    print(f"Kafka consumer started. Bootstrap={bootstrap}, Topics={topics}")

    buffers: dict[tuple[str, str], list[dict[str, Any]]] = {}
    last_seen: dict[tuple[str, str], float] = {}

    try:
        while True:
            msg = consumer.poll(1.0)

            if msg is None:
                # periodic flush for safety
                now = time.time()
                for key in list(buffers.keys()):
                    if buffers[key] and (now - last_seen.get(key, now) > 8.0):
                        ds, bid = key
                        rows = buffers.pop(key, [])
                        last_seen.pop(key, None)
                        try:
                            _flush_batch(ds, bid, rows)
                        except Exception as e:
                            print(f"[{ds} | {bid}] flush failed: {e}")
                continue

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                raise KafkaException(msg.error())

            try:
                payload = json.loads(msg.value().decode("utf-8"))
            except Exception:
                continue

            dataset = payload.get("dataset")
            batch_id = payload.get("batch_id")

            if payload.get("type") == "end_batch":
                key = (dataset, batch_id)
                rows = buffers.pop(key, [])
                last_seen.pop(key, None)
                if dataset and batch_id and rows:
                    try:
                        _flush_batch(str(dataset), str(batch_id), rows)
                    except Exception as e:
                        print(f"[{dataset} | {batch_id}] flush failed: {e}")
                continue

            row = payload.get("payload")
            if not dataset or not batch_id or not isinstance(row, dict):
                continue

            key = (str(dataset), str(batch_id))
            buffers.setdefault(key, []).append(row)
            last_seen[key] = time.time()

    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()


if __name__ == "__main__":
    main()