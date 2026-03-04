from __future__ import annotations

import datetime as dt
from typing import Optional

from sqlalchemy import String, Integer, DateTime, Float, Text, ForeignKey, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), unique=True, index=True)

    # User inputs
    canonical_schema_json: Mapped[str] = mapped_column(Text, default="{}")
    baseline_profile_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Risk config (legacy + fallback)
    risk_mode: Mapped[str] = mapped_column(String(10), default="A")
    risk_config_json: Mapped[str] = mapped_column(Text, default="{}")

    canonical_schema_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    baseline_csv_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=lambda: dt.datetime.utcnow())
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime,
        default=lambda: dt.datetime.utcnow(),
        onupdate=lambda: dt.datetime.utcnow(),
    )

    # relationships
    events: Mapped[list["DriftEvent"]] = relationship(back_populates="dataset")
    schema_versions: Mapped[list["SchemaRegistry"]] = relationship(back_populates="dataset")
    stored_batches: Mapped[list["StoredBatch"]] = relationship(back_populates="dataset")
    rejected_rows: Mapped[list["RejectedRow"]] = relationship(back_populates="dataset")

    # ✅ New: 1 row per dataset
    risk_config_row: Mapped[Optional["RiskConfig"]] = relationship(
        back_populates="dataset",
        uselist=False,
        cascade="all, delete-orphan",
    )


class RiskConfig(Base):
    __tablename__ = "risk_configs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"),
        unique=True,
        index=True,
        nullable=False,
    )

    mode: Mapped[str] = mapped_column(String(8), default="A")
    config_json: Mapped[str] = mapped_column(Text, default="{}")

    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=lambda: dt.datetime.utcnow())
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime,
        default=lambda: dt.datetime.utcnow(),
        onupdate=lambda: dt.datetime.utcnow(),
    )

    dataset: Mapped["Dataset"] = relationship(back_populates="risk_config_row")


class SchemaRegistry(Base):
    __tablename__ = "SchemaRegistry"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), index=True)

    version: Mapped[int] = mapped_column(Integer, default=1)
    active: Mapped[bool] = mapped_column(Boolean, default=False)

    schema_hash: Mapped[str] = mapped_column(String(128), index=True)
    schema_json: Mapped[str] = mapped_column(Text, default="{}")

    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=lambda: dt.datetime.utcnow())
    note: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    dataset: Mapped["Dataset"] = relationship(back_populates="schema_versions")


class StoredBatch(Base):
    __tablename__ = "stored_batches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), index=True)

    batch_id: Mapped[str] = mapped_column(String(200), index=True)
    filename: Mapped[str] = mapped_column(String(260))
    path: Mapped[str] = mapped_column(String(800))

    row_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    uploaded_at: Mapped[dt.datetime] = mapped_column(DateTime, default=lambda: dt.datetime.utcnow())

    dataset: Mapped["Dataset"] = relationship(back_populates="stored_batches")


class DriftEvent(Base):
    __tablename__ = "drift_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"))
    batch_id: Mapped[str] = mapped_column(String(200), index=True)

    detected_at: Mapped[dt.datetime] = mapped_column(DateTime, default=lambda: dt.datetime.utcnow())

    drift_types: Mapped[str] = mapped_column(String(200), default="")
    diff_json: Mapped[str] = mapped_column(Text, default="{}")
    rename_json: Mapped[str] = mapped_column(Text, default="{}")

    risk_score: Mapped[float] = mapped_column(Float, default=0.0)
    risk_level: Mapped[str] = mapped_column(String(50), default="low")
    route: Mapped[str] = mapped_column(String(50), default="production")

    status: Mapped[str] = mapped_column(String(50), default="PENDING")
    decision_by: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    decision_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    decision_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime, nullable=True)

    dataset: Mapped["Dataset"] = relationship(back_populates="events")
    staged_rows: Mapped[list["StagingRow"]] = relationship(back_populates="event")
    rejected_rows: Mapped[list["RejectedRow"]] = relationship(back_populates="event")


class ProductionRow(Base):
    __tablename__ = "production_rows"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"))
    batch_id: Mapped[str] = mapped_column(String(200), index=True)
    inserted_at: Mapped[dt.datetime] = mapped_column(DateTime, default=lambda: dt.datetime.utcnow())
    row_json: Mapped[str] = mapped_column(Text, default="{}")


class StagingRow(Base):
    __tablename__ = "staging_rows"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_id: Mapped[int] = mapped_column(ForeignKey("drift_events.id"))
    inserted_at: Mapped[dt.datetime] = mapped_column(DateTime, default=lambda: dt.datetime.utcnow())
    status: Mapped[str] = mapped_column(String(50), default="PENDING")
    row_json: Mapped[str] = mapped_column(Text, default="{}")

    event: Mapped["DriftEvent"] = relationship(back_populates="staged_rows")


class RejectedRow(Base):
    __tablename__ = "rejected_rows"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), index=True)
    event_id: Mapped[int] = mapped_column(ForeignKey("drift_events.id"), index=True)
    batch_id: Mapped[str] = mapped_column(String(200), index=True)

    rejected_at: Mapped[dt.datetime] = mapped_column(DateTime, default=lambda: dt.datetime.utcnow())
    rejected_by: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    rejected_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    notify_email: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    row_json: Mapped[str] = mapped_column(Text, default="{}")

    dataset: Mapped["Dataset"] = relationship(back_populates="rejected_rows")
    event: Mapped["DriftEvent"] = relationship(back_populates="rejected_rows")