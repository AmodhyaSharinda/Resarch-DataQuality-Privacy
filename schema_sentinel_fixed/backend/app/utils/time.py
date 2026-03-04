from __future__ import annotations

import datetime as dt


def to_utc_iso(ts: dt.datetime | None) -> str | None:
    """Return a UTC ISO timestamp with a trailing 'Z'.

    The project stores many timestamps as *naive* datetimes (no tzinfo).
    If we serialize a naive datetime with `.isoformat()`, browsers will treat
    it as *local time* and your UI will show a shifted timestamp.

    We treat naive timestamps as UTC and add a 'Z' suffix.
    """

    if ts is None:
        return None

    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.timezone.utc)
    else:
        ts = ts.astimezone(dt.timezone.utc)

    return ts.isoformat().replace("+00:00", "Z")
