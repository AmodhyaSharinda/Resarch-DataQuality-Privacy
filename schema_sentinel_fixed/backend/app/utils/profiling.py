from __future__ import annotations

from typing import Any
from pathlib import Path
import csv

import pandas as pd

from app.utils.text import norm_col


# ----------------------------
# Type normalization
# ----------------------------

def _normalize_type(t: Any) -> str:
    """Keep one internal type vocabulary across the whole project."""
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


def _infer_series_type(s: pd.Series) -> str:
    """Infer type using the same vocabulary as drift_engine."""
    if pd.api.types.is_bool_dtype(s):
        return "bool"

    if pd.api.types.is_integer_dtype(s):
        return "int"

    if pd.api.types.is_float_dtype(s):
        return "float"

    # numeric coercion from strings like "1,200"
    sn = pd.to_numeric(s.astype(str).str.replace(",", ""), errors="coerce")
    if sn.notna().mean() > 0.8:
        try:
            frac = (sn.dropna() % 1)
            if (frac == 0).mean() > 0.95:
                return "int"
        except Exception:
            pass
        return "float"

    # datetime coercion
    dtv = pd.to_datetime(s, errors="coerce")
    if dtv.notna().mean() > 0.8:
        return "datetime"

    return "string"


def _top_examples(s: pd.Series, k: int = 8) -> list[Any]:
    """Store representative examples for embeddings + rename detection."""
    ss = s.dropna().astype(str).str.strip()
    if ss.empty:
        return []
    vc = ss.value_counts().head(k)
    return vc.index.tolist()


# ----------------------------
# Baseline profiling
# ----------------------------

def _read_wrapped_csv(path: str) -> pd.DataFrame:
    """
    Handles CSV files where each line is wrapped in quotes like:
      "a,b,c"
      "1,2,3"
    This format often gets parsed by pandas as a single column.
    """
    txt = Path(path).read_text(encoding="utf-8", errors="replace")
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]

    cleaned: list[str] = []
    for ln in lines:
        ln = ln.lstrip("\ufeff").strip()
        # remove single/double quotes around whole line
        if (ln.startswith('"') and ln.endswith('"')) or (ln.startswith("'") and ln.endswith("'")):
            ln = ln[1:-1]
        cleaned.append(ln)

    rows = list(csv.reader(cleaned, delimiter=","))
    if not rows or len(rows) < 2:
        return pd.DataFrame()

    header = [h.lstrip("\ufeff").strip() for h in rows[0]]
    data = rows[1:]
    if not header:
        return pd.DataFrame()

    # build df, keep raw strings (profiling converts types later)
    return pd.DataFrame(data, columns=header)


def _read_baseline_table(path: str) -> pd.DataFrame:
    """Read baseline CSV or Excel. UI allows CSV/XLSX/XLS/XLSM."""
    p = Path(path)
    ext = p.suffix.lower()

    if ext in (".xlsx", ".xlsm", ".xls"):
        # Try multiple engines to be robust across environments
        last_err: Exception | None = None
        engines = ["openpyxl", None, "xlrd"] if ext in (".xlsx", ".xlsm") else ["xlrd", None, "openpyxl"]
        for eng in engines:
            try:
                if eng:
                    return pd.read_excel(path, engine=eng)
                return pd.read_excel(path)
            except Exception as e:
                last_err = e
        raise ValueError(f"Failed to read Excel baseline: {last_err}")

    # ---- CSV (tolerant decoding) ----
    # 1) try default
    try:
        df = pd.read_csv(path)
    except UnicodeDecodeError:
        df = None
    except Exception:
        # if it's not a decode error, let it surface
        raise

    # 2) fallback: open with encoding + errors="replace" (THIS is where errors belongs)
    if df is None:
        last_err: Exception | None = None
        for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
            try:
                with open(path, "r", encoding=enc, errors="replace", newline="") as f:
                    df = pd.read_csv(f)
                    break
            except Exception as e:
                last_err = e
        if df is None:
            raise ValueError(f"Failed to read CSV baseline with tolerant decoding: {last_err}")

    # ✅ NEW: auto-repair wrapped CSV that pandas reads as a single column
    try:
        if df.shape[1] == 1:
            col0 = str(df.columns[0] or "")
            # header itself contains commas => likely wrapped line
            if "," in col0:
                df2 = _read_wrapped_csv(path)
                if df2.shape[1] > 1:
                    return df2
    except Exception:
        # best effort only
        pass

    return df


def compute_baseline_profile(*, baseline_path: str, canonical_schema: dict[str, Any]) -> dict[str, Any]:
    """Compute baseline profile aligned with canonical schema.

    - Reads CSV or Excel
    - Normalizes columns using norm_col()
    - Produces stats + examples for each canonical column

    Baseline is *important* for rename detection (value overlap / examples).
    """
    df = _read_baseline_table(baseline_path)

    # normalize columns to match canonical names
    df = df.rename(columns={c: norm_col(c) for c in df.columns})

    profile: dict[str, Any] = {}
    cols = canonical_schema.get("columns", [])
    if not isinstance(cols, list):
        cols = []

    for col in cols:
        if not isinstance(col, dict):
            continue

        name = col.get("name")
        if not name:
            continue

        name = norm_col(name)
        canon_type = _normalize_type(col.get("type", "string"))

        # if column missing in baseline
        if name not in df.columns:
            profile[name] = {
                "present": False,
                "null_rate": 1.0,
                "type_inferred": canon_type,
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "examples": [],
            }
            continue

        s = df[name]
        null_rate = float(s.isna().mean())
        t_inf = _normalize_type(_infer_series_type(s))

        entry: dict[str, Any] = {
            "present": True,
            "null_rate": null_rate,
            "type_inferred": t_inf,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "examples": _top_examples(s, k=8),
        }

        if t_inf in ("int", "float"):
            sn = pd.to_numeric(s.astype(str).str.replace(",", ""), errors="coerce").dropna()
            if len(sn) > 0:
                entry["mean"] = float(sn.mean())
                entry["std"] = float(sn.std(ddof=0))
                entry["min"] = float(sn.min())
                entry["max"] = float(sn.max())

        profile[name] = entry

    return profile