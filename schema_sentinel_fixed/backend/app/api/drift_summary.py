from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from app.utils.drift_log import (
    read_drift_events_jsonl,
    build_events_rows,
    build_daily_summary_rows,
    rows_to_csv,
)

router = APIRouter(prefix="/drift", tags=["drift"])


@router.get("/summary")
def get_drift_summary(
    dataset: str = Query(..., description="Dataset name"),
    mode: str = Query("daily", description="daily | events"),
):
    events = read_drift_events_jsonl(dataset=dataset)
    if (mode or "").lower() == "events":
        return build_events_rows(events)
    return build_daily_summary_rows(events)


@router.get("/summary/download")
def download_drift_summary_csv(
    dataset: str = Query(..., description="Dataset name"),
    mode: str = Query("daily", description="daily | events"),
):
    events = read_drift_events_jsonl(dataset=dataset)

    mode_l = (mode or "daily").strip().lower()
    if mode_l not in ("daily", "events"):
        raise HTTPException(400, "mode must be 'daily' or 'events'")

    rows = build_events_rows(events) if mode_l == "events" else build_daily_summary_rows(events)
    csv_text = rows_to_csv(rows)

    filename = f"drift_{dataset}_{mode_l}_summary.csv"
    return Response(
        content=csv_text,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )