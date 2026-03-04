# backend/app/api/schemas.py
from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


# =============================================================================
# Dataset config
# =============================================================================

class DatasetRegisterResponse(BaseModel):
    dataset: str
    ok: bool
    message: str
    has_baseline: bool


class DatasetInfo(BaseModel):
    name: str
    created_at: str
    updated_at: str
    has_baseline: bool


# =============================================================================
# Ingest
# =============================================================================

class IngestBatchRequest(BaseModel):
    dataset: str = Field(..., examples=["Sale"])
    batch_id: str = Field(..., examples=["v1_add_cols"])
    rows: list[dict[str, Any]]


class DriftEventResponse(BaseModel):
    event_id: int
    dataset: str
    batch_id: str
    drift_types: list[str]
    diff: dict[str, Any]
    renames: dict[str, Any]
    risk_score: float
    risk_level: str
    route: str
    status: str


# =============================================================================
# Governance actions
# =============================================================================

class ApproveRequest(BaseModel):
    approver: str = "user"
    note: Optional[str] = None
    approved_renames: dict[str, str] = Field(default_factory=dict)


class RejectRequest(BaseModel):
    approver: str = "user"
    reason: str | None = None
    note: Optional[str] = None
    notify_email: str | None = None


class RollbackRequest(BaseModel):
    approver: str = "user"
    reason: str | None = None
    note: Optional[str] = None


class StagingRowsResponse(BaseModel):
    event_id: int
    total: int
    rows: list[dict[str, Any]]


# =============================================================================
# Risk configuration (Option A / Option B)
# =============================================================================

class RiskConfigRequest(BaseModel):
    mode: str = Field(..., examples=["A", "B"])

    # ---- Option A inputs ----
    dataset_criticality: Optional[str] = Field(default=None, examples=["Low", "Medium", "High"])
    sensitivity_class: Optional[str] = Field(default=None, examples=["None", "Internal", "PII", "Regulated"])
    regulation_strictness: Optional[str] = Field(default=None, examples=["Light", "Strict"])
    key_fields: Optional[list[str]] = None

    # ---- Option B inputs ----
    dataset_criticality_num: Optional[int] = Field(default=None, ge=1, le=5)
    fields: Optional[dict[str, Any]] = None


class RiskConfigResponse(BaseModel):
    dataset: str
    mode: str
    risk_config: dict[str, Any]


class DatasetFieldsResponse(BaseModel):
    dataset: str
    fields: list[str]
    suggested_key_fields: list[str]
