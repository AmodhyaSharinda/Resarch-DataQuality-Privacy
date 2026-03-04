from __future__ import annotations

import os
import pathlib
from typing import Any, Iterable

import pandas as pd

from app.core.config import settings
from app.utils.text import norm_col

_XLS_OLE2_MAGIC = b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1"
_ZIP_MAGIC = b"PK\x03\x04"


def _batches_root() -> pathlib.Path:
    return pathlib.Path(settings.STORAGE_DIR) / "batches"


def ensure_dataset_dir(dataset_name: str) -> pathlib.Path:
    p = _batches_root() / dataset_name
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_upload(dataset_name: str, batch_id: str, filename: str, raw_bytes: bytes) -> pathlib.Path:
    safe_name = filename.replace("\\", "_").replace("/", "_")
    out_dir = ensure_dataset_dir(dataset_name)
    out_path = out_dir / f"{batch_id}__{safe_name}"
    out_path.write_bytes(raw_bytes)
    return out_path


def _detect_excel_kind(p: pathlib.Path) -> str | None:
    head = p.read_bytes()[:8]
    if head.startswith(_ZIP_MAGIC):
        return "xlsx"
    if head.startswith(_XLS_OLE2_MAGIC):
        return "xls"
    return None


def _looks_like_html(p: pathlib.Path) -> bool:
    head = p.read_bytes()[:2048].decode("utf-8", errors="ignore").lower()
    return "<html" in head or "<table" in head


def _choose_delimiter(first_line: str) -> str:
    """
    Deterministic delimiter selection by counts in header line.
    This prevents csv.Sniffer choosing tab because of trailing tabs.
    """
    counts = {
        ",": first_line.count(","),
        ";": first_line.count(";"),
        "\t": first_line.count("\t"),
        "|": first_line.count("|"),
    }
    sep = max(counts, key=counts.get)
    return sep if counts[sep] > 0 else ","


def _read_csv_robust(p: pathlib.Path) -> pd.DataFrame:
    text = p.read_text(encoding="utf-8-sig", errors="ignore")
    first_line = (text.splitlines()[0] if text.splitlines() else "")
    sep = _choose_delimiter(first_line)

    # force sep (do NOT use sep=None / sniffer)
    df = pd.read_csv(p, sep=sep, engine="python", encoding="utf-8-sig")

    # clean column names
    df.columns = [str(c).strip() for c in df.columns]

    # drop useless unnamed columns if they are fully empty
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed")]
    for c in unnamed:
        if df[c].isna().all():
            df = df.drop(columns=[c])

    
    # If we still ended up with one big comma-separated "header" column,
    # split it into proper columns.
    if len(df.columns) == 1 and "," in df.columns[0]:
        big_header = df.columns[0]
        header_parts = [h.strip() for h in big_header.split(",") if h.strip()]

        # split the single string column values by comma
        s = df[big_header].astype("string")
        split_df = s.str.split(",", expand=True)

        if split_df.shape[1] == len(header_parts):
            split_df.columns = header_parts
            df = split_df

    return df


def _norm_set(cols: Iterable[str]) -> set[str]:
    return {norm_col(str(c)) for c in cols if str(c).strip()}


def _sheet_score(cols: Iterable[str], expected_norm: set[str] | None) -> tuple[int, int]:
    colset = _norm_set(cols)
    ncols = len(colset)
    if expected_norm:
        overlap = len(colset & expected_norm)
        return (overlap, ncols)
    return (0, ncols)


def read_rows_from_file(path: str | os.PathLike[str], expected_cols: list[str] | None = None) -> list[dict[str, Any]]:
    """
    CSV: robust delimiter selection + single-column recovery
    Excel: best sheet by normalized overlap with expected columns
    """
    p = pathlib.Path(path)
    ext = p.suffix.lower()
    expected_norm = _norm_set(expected_cols or [])

    try:
        if ext == ".csv":
            df = _read_csv_robust(p)

        elif ext in (".xlsx", ".xls", ".xlsm"):
            # HTML disguised as .xls
            if _looks_like_html(p):
                tables = pd.read_html(p.read_text(encoding="utf-8", errors="ignore"))
                if not tables:
                    raise ValueError("No HTML tables found in file")
                df = tables[0]
            else:
                kind = _detect_excel_kind(p)
                if kind is None:
                    # not real excel => treat as CSV
                    df = _read_csv_robust(p)
                else:
                    xls = pd.ExcelFile(p)
                    best_sheet = None
                    best_score = (-1, -1)

                    for s in xls.sheet_names:
                        try:
                            tmp = pd.read_excel(
                                xls,
                                sheet_name=s,
                                nrows=50,
                                engine="openpyxl" if ext in (".xlsx", ".xlsm") else None,
                            )
                            score = _sheet_score(tmp.columns, expected_norm if expected_norm else None)
                            if score > best_score:
                                best_score = score
                                best_sheet = s
                        except Exception:
                            continue

                    if best_sheet is None:
                        df = pd.read_excel(p, engine="openpyxl" if ext in (".xlsx", ".xlsm") else None)
                    else:
                        df = pd.read_excel(p, sheet_name=best_sheet, engine="openpyxl" if ext in (".xlsx", ".xlsm") else None)

        else:
            raise ValueError(f"Unsupported file type: {ext}")

    except Exception as e:
        raise ValueError(f"Failed to parse '{p.name}'. Root error: {e}")

    # NaN -> None for JSON + drop fully empty unnamed columns again
    df = df.where(pd.notnull(df), None)

    unnamed = [c for c in df.columns if str(c).startswith("Unnamed")]
    for c in unnamed:
        if df[c].isna().all():
            df = df.drop(columns=[c])

    return df.to_dict(orient="records")


def count_rows(path: str | os.PathLike[str], expected_cols: list[str] | None = None) -> int:
    try:
        return len(read_rows_from_file(path, expected_cols=expected_cols))
    except Exception:
        return 0