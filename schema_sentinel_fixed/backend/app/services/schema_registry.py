from __future__ import annotations

from pathlib import Path
import datetime as dt
import hashlib
import json
from typing import Any

from sqlalchemy.orm import Session

from app.db.models import Dataset, SchemaRegistry
from app.utils.text import norm_col


def schema_hash_core(schema: dict[str, Any]) -> str:
    """Compute a stable hash for a schema.

    This is intentionally *lightweight* ("core") so it can be used as a fast
    schema-equality check:

    - Only uses column name + type (ignores nullable / metadata)
    - Normalizes column names
    - Sorts columns by name
    """

    cols = schema.get("columns", []) if isinstance(schema, dict) else []
    core = []
    for c in cols:
        if not isinstance(c, dict):
            continue
        name = c.get("name")
        typ = c.get("type")
        if not name:
            continue
        core.append({"name": norm_col(str(name)), "type": str(typ or "string").strip().lower()})

    core_sorted = sorted(core, key=lambda x: x["name"])
    payload = json.dumps({"columns": core_sorted}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_dataset_by_name(db: Session, dataset_name: str) -> Dataset | None:
    return db.query(Dataset).filter(Dataset.name == dataset_name).first()


def get_active_schema(db: Session, dataset_id: int) -> SchemaRegistry | None:
    return (
        db.query(SchemaRegistry)
        .filter(SchemaRegistry.dataset_id == dataset_id, SchemaRegistry.active == True)  # noqa: E712
        .order_by(SchemaRegistry.version.desc())
        .first()
    )


def get_active_schema_json(db: Session, dataset_id: int) -> dict[str, Any] | None:
    """Return the active schema as a Python dict (or None)."""

    row = get_active_schema(db, dataset_id)
    if not row:
        return None
    try:
        return json.loads(row.schema_json or "{}")
    except Exception:
        return None


def find_version_by_hash(db: Session, dataset_id: int, schema_hash: str) -> SchemaRegistry | None:
    """Find a schema registry row by hash (used to dedupe candidates)."""

    return (
        db.query(SchemaRegistry)
        .filter(SchemaRegistry.dataset_id == dataset_id, SchemaRegistry.schema_hash == schema_hash)
        .order_by(SchemaRegistry.version.desc())
        .first()
    )


def list_versions(db: Session, dataset_id: int) -> list[SchemaRegistry]:
    return (
        db.query(SchemaRegistry)
        .filter(SchemaRegistry.dataset_id == dataset_id)
        .order_by(SchemaRegistry.version.desc())
        .all()
    )


def insert_version(
    db: Session,
    dataset_id: int,
    schema: dict[str, Any],
    *,
    active: bool,
    note: str | None = None,
) -> SchemaRegistry:
    # next version = max + 1
    last = (
        db.query(SchemaRegistry)
        .filter(SchemaRegistry.dataset_id == dataset_id)
        .order_by(SchemaRegistry.version.desc())
        .first()
    )
    next_ver = int(last.version) + 1 if last else 1

    h = schema_hash_core(schema)
    if active:
        # Deactivate all other versions first (simple, reliable for dev)
        (
            db.query(SchemaRegistry)
            .filter(SchemaRegistry.dataset_id == dataset_id)
            .update({SchemaRegistry.active: False})
        )

    row = SchemaRegistry(
        dataset_id=dataset_id,
        version=next_ver,
        active=bool(active),
        schema_hash=h,
        schema_json=json.dumps(schema, ensure_ascii=False),
        created_at=dt.datetime.utcnow(),
        note=note,
    )
    db.add(row)

    db.commit()
    db.refresh(row)
    return row


def ensure_canonical_active(
    db: Session,
    dataset: Dataset,
    canonical_schema: dict[str, Any],
    *,
    note: str = "canonical",
) -> SchemaRegistry:
    """Ensure there is an active canonical schema version for a dataset.

    If no versions exist, create v1 active.
    If an active version exists with a different hash, create a new active version.
    """

    h = schema_hash_core(canonical_schema)
    active = get_active_schema(db, dataset.id)

    if active is None:
        return insert_version(db, dataset.id, canonical_schema, active=True, note=note)

    if str(active.schema_hash) == h:
        return active

    # new canonical version
    # deactivate old and create new
    active.active = False
    db.add(active)
    db.commit()
    return insert_version(db, dataset.id, canonical_schema, active=True, note=note)


def insert_candidate_if_new(
    db: Session,
    dataset: Dataset,
    observed_schema: dict[str, Any],
    *,
    note: str | None = None,
) -> SchemaRegistry | None:
    """Insert a candidate schema version (active=False) if hash not seen before."""
    h = schema_hash_core(observed_schema)
    existing = (
        db.query(SchemaRegistry)
        .filter(SchemaRegistry.dataset_id == dataset.id, SchemaRegistry.schema_hash == h)
        .first()
    )
    if existing:
        return None
    return insert_version(db, dataset.id, observed_schema, active=False, note=note or "candidate")


def activate_version(db: Session, dataset_id: int, version: int) -> SchemaRegistry | None:
    target = (
        db.query(SchemaRegistry)
        .filter(SchemaRegistry.dataset_id == dataset_id, SchemaRegistry.version == version)
        .first()
    )
    if not target:
        return None

    # Deactivate all, then activate the requested version
    db.query(SchemaRegistry).filter(SchemaRegistry.dataset_id == dataset_id).update(
        {"active": False}
    )
    target.active = True
    db.add(target)

    # IMPORTANT: keep Dataset.canonical_schema_json (and the on-disk canonical schema file) in sync
    # with the ACTIVE registry version so downstream uses the correct schema after approve/rollback.
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if ds is not None:
        ds.canonical_schema_json = target.schema_json
        ds.updated_at = dt.datetime.utcnow()
        db.add(ds)

        # If we have a canonical schema file path, overwrite it with the active schema JSON.
        try:
            if ds.canonical_schema_path:
                p = Path(ds.canonical_schema_path)
                if not p.is_absolute():
                    p = Path.cwd() / p
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(target.schema_json, encoding="utf-8")
        except Exception:
            # File sync is best-effort; DB state is the source of truth.
            pass

    db.commit()
    db.refresh(target)
    return target
