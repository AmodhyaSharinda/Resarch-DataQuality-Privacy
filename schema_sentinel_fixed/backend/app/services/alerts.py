# backend/app/services/alerts.py
from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage
from typing import Any, Optional


def send_reject_alert(
    *,
    dataset: str,
    batch_id: str,
    event_id: int,
    reason: str | None,
    drift_types: list[str] | None,
    diff: dict[str, Any] | None,
    to_email: str | None = None,
) -> dict[str, Any]:
    """
    Best-effort alert.
    - If SMTP env vars are present, sends an email.
    - Otherwise returns a payload that can be logged/audited.

    Env vars (optional):
      ALERT_SMTP_HOST, ALERT_SMTP_PORT, ALERT_SMTP_USER, ALERT_SMTP_PASS
      ALERT_EMAIL_FROM, ALERT_EMAIL_TO
    """
    payload = {
        "type": "ALERT_REJECT",
        "dataset": dataset,
        "batch_id": batch_id,
        "event_id": event_id,
        "reason": reason or "Rejected",
        "drift_types": drift_types or [],
        "diff": diff or {},
    }

    host = os.getenv("ALERT_SMTP_HOST", "").strip()
    to_addr = (to_email or os.getenv("ALERT_EMAIL_TO", "")).strip()
    from_addr = os.getenv("ALERT_EMAIL_FROM", "").strip() or to_addr

    # If not configured, do nothing (but return payload)
    if not host or not to_addr:
        return {"sent": False, "via": "none", "payload": payload}

    port = int(os.getenv("ALERT_SMTP_PORT", "587") or "587")
    user = os.getenv("ALERT_SMTP_USER", "").strip()
    pw = os.getenv("ALERT_SMTP_PASS", "").strip()

    msg = EmailMessage()
    msg["Subject"] = f"[SchemaSentinel] REJECTED drift for {dataset} (event #{event_id})"
    msg["From"] = from_addr
    msg["To"] = to_addr

    body = []
    body.append(f"Dataset: {dataset}")
    body.append(f"Batch: {batch_id}")
    body.append(f"Event: {event_id}")
    body.append(f"Reason: {reason or 'Rejected'}")
    body.append(f"Drift types: {', '.join(drift_types or [])}")
    msg.set_content("\n".join(body))

    try:
        with smtplib.SMTP(host, port, timeout=10) as s:
            s.starttls()
            if user and pw:
                s.login(user, pw)
            s.send_message(msg)
        return {"sent": True, "via": "smtp", "payload": payload}
    except Exception as e:
        return {"sent": False, "via": "smtp_error", "error": str(e), "payload": payload}