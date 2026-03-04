from __future__ import annotations

import logging
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.core.config import settings
from app.services.rename_model import RENAME_SCORER
from app.utils.text import norm_col

logger = logging.getLogger(__name__)

ENGINE_VERSION = "2026-03-03-sbert-charhash-v5"


# ----------------------------
# Robust numeric parsing (DON'T accidentally convert dates -> ints)
# ----------------------------

_CURRENCY_WS_RE = re.compile(r"[\s\$\€\£\¥\₹]+")


def _to_numeric_clean(series: pd.Series) -> pd.Series:
    """
    Convert to numeric safely.
    IMPORTANT: Do NOT strip letters/separators in a way that turns datetimes like
    '2020-10-28' or '28 Oct 2020' into '20201028' / '282020' (false int).
    """
    s0 = series.astype(str).str.strip()

    # If value contains letters (month names etc) -> not numeric
    has_alpha = s0.str.contains(r"[A-Za-z]", na=False)

    # If it looks date-ish: multiple '-' or '/' or contains ':' (time) -> not numeric
    has_many_date_seps = (s0.str.count(r"[-/]") >= 2)
    has_time_sep = s0.str.contains(r":", na=False)

    mask_not_numeric = has_alpha | has_many_date_seps | has_time_sep
    s = s0.mask(mask_not_numeric, other=np.nan)

    # Clean common numeric noise only (commas, currency, whitespace, parentheses)
    s = s.str.replace(",", "", regex=False)
    s = s.str.replace(_CURRENCY_WS_RE, "", regex=True)
    s = s.str.replace(r"^\((.*)\)$", r"-\1", regex=True)  # (123) -> -123

    return pd.to_numeric(s, errors="coerce")


# ----------------------------
# Schema extraction
# ----------------------------
def _make_unique(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for n in names:
        if n not in seen:
            seen[n] = 0
            out.append(n)
        else:
            seen[n] += 1
            out.append(f"{n}_{seen[n]}")
    return out


def _safe_to_datetime(series: pd.Series) -> pd.Series:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Could not infer format.*", category=UserWarning)
        try:
            return pd.to_datetime(series, errors="coerce", format="mixed")
        except TypeError:
            return pd.to_datetime(series, errors="coerce")
        except ValueError:
            return pd.to_datetime(series, errors="coerce")


def _infer_logical_type(series: pd.Series, col_name: str = "") -> str:
    cname = (col_name or "").lower()

    if pd.api.types.is_bool_dtype(series):
        return "bool"
    if pd.api.types.is_integer_dtype(series):
        return "int"
    if pd.api.types.is_float_dtype(series):
        return "float"

    # ✅ If the column name looks like date/time, try datetime FIRST
    if any(k in cname for k in ("date", "time", "timestamp", "datetime", "_dt", "_ts")):
        dtv = _safe_to_datetime(series)
        if dtv.notna().mean() > 0.8:
            return "datetime"

    # numeric next (with safe cleaner)
    sn = _to_numeric_clean(series)
    if sn.notna().mean() > 0.8:
        if (sn.dropna() % 1 == 0).mean() > 0.95:
            return "int"
        return "float"

    # datetime fallback for non date-named cols
    dtv = _safe_to_datetime(series)
    if dtv.notna().mean() > 0.8:
        return "datetime"

    return "string"


def _col_profile(series: pd.Series) -> dict[str, Any]:
    null_rate = float(series.isna().mean())
    t = _infer_logical_type(series, col_name=str(getattr(series, "name", "") or ""))

    mean = std = minv = maxv = None
    if t in ("int", "float"):
        sn = _to_numeric_clean(series).dropna()
        if len(sn) > 0:
            mean = float(sn.mean())
            std = float(sn.std(ddof=0))
            minv = float(sn.min())
            maxv = float(sn.max())

    examples: list[str] = []
    try:
        examples = list(series.dropna().astype(str).head(20).values)
    except Exception:
        examples = []

    return {
        "null_rate": null_rate,
        "mean": mean,
        "std": std,
        "min": minv,
        "max": maxv,
        "type_inferred": t,
        "examples": examples,
    }


def _schema_from_rows(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], pd.DataFrame]:
    if not rows:
        return {"columns": [], "original_to_normalized": {}}, pd.DataFrame()

    df_raw = pd.DataFrame(rows)
    orig_cols = list(df_raw.columns)

    norm_cols = [norm_col(c) for c in orig_cols]
    norm_cols = _make_unique(norm_cols)
    mapping = dict(zip(orig_cols, norm_cols))

    df = df_raw.rename(columns=mapping)

    cols = []
    for c in df.columns:
        s = df[c]
        cols.append({"name": c, "type": _infer_logical_type(s, col_name=c), "nullable": bool(s.isna().any())})

    observed_schema = {"columns": cols, "original_to_normalized": mapping}
    return observed_schema, df


# ----------------------------
# Name similarity (no pandas warnings)
# ----------------------------
def _normalize_name(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def name_similarity(a: str, b: str) -> float:
    a0, b0 = _normalize_name(a), _normalize_name(b)
    if a0 == b0:
        return 1.0

    ta, tb = set(a0.split("_")), set(b0.split("_"))
    jacc = len(ta & tb) / max(1, len(ta | tb))

    va = pd.Series(list(a0)).value_counts()
    vb = pd.Series(list(b0)).value_counts()
    idx = va.index.union(vb.index)

    common = float(
        np.minimum(
            va.reindex(idx, fill_value=0).to_numpy(),
            vb.reindex(idx, fill_value=0).to_numpy(),
        ).sum()
    )
    ratio = common / max(1, max(len(a0), len(b0)))
    return float(0.6 * jacc + 0.4 * ratio)


# ✅ upgraded token_jaccard with synonyms (code/ref/no -> id)
def token_jaccard(a: str, b: str) -> float:
    SYN = {
        "code": "id",
        "identifier": "id",
        "ident": "id",
        "pk": "id",
        "key": "id",
        "no": "id",
        "number": "id",
        "ref": "id",
        "reference": "id",
        "uid": "id",
        "guid": "id",
        "uuid": "id",
    }

    def tok(x: str) -> set:
        s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(x))
        parts = re.split(r"[^a-zA-Z0-9]+", s.lower())
        parts = [SYN.get(p, p) for p in parts if p]
        return set(parts)

    ta, tb = tok(a), tok(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return float(len(ta & tb) / len(ta | tb))


def type_compatible(t1: str, t2: str) -> bool:
    if t1 == t2:
        return True
    numeric = {"int", "float"}
    return t1 in numeric and t2 in numeric


# ✅ upgraded semantic_class: treat code/ref/number like identifiers
def semantic_class(name: str, inferred_type: str) -> str:
    n = (name or "").lower()

    if inferred_type == "datetime":
        return "date"

    # treat code/ref/number as identifiers too
    id_tokens = ("id", "uuid", "guid", "pk", "key", "code", "ref", "reference", "number", "no")
    if any(k in n for k in id_tokens):
        return "id"

    if any(k in n for k in ("date", "time", "ts", "timestamp", "datetime")):
        return "date"

    if inferred_type in ("int", "float"):
        return "numeric"
    if inferred_type == "string":
        return "string"
    return "other"


# ----------------------------
# Charhash embedding (fallback)
# ----------------------------
def _charhash_encode(texts: List[str], dim: int = 256, ngram_range: tuple[int, int] = (3, 5)) -> np.ndarray:
    import hashlib

    lo, hi = ngram_range
    X = np.zeros((len(texts), dim), dtype=np.float32)

    for i, t in enumerate(texts):
        s = (t or "").lower()
        s = re.sub(r"\s+", " ", s)
        s = f" {s} "
        for n in range(lo, hi + 1):
            for j in range(0, max(0, len(s) - n + 1)):
                ng = s[j : j + n].encode("utf-8", errors="ignore")
                h = int(hashlib.blake2b(ng, digest_size=4).hexdigest(), 16) % dim
                X[i, h] += 1.0

    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    return X / norms


# ----------------------------
# Embeddings: SBERT local -> charhash fallback
# Supports HuggingFace cache layout: snapshots/<hash>/
# ----------------------------
def _resolve_sbert_dir(p: str) -> Optional[str]:
    pp = Path(p)
    if not pp.exists() or not pp.is_dir():
        return None

    snap = pp / "snapshots"
    if snap.exists() and snap.is_dir():
        candidates = []
        for child in snap.iterdir():
            if not child.is_dir():
                continue
            if (child / "config.json").exists() or (child / "modules.json").exists():
                candidates.append(child)
        if candidates:
            best = sorted(candidates, key=lambda x: x.stat().st_mtime, reverse=True)[0]
            return str(best)

    if (pp / "config.json").exists() or (pp / "modules.json").exists():
        return str(pp)

    return None


class _Embedder:
    def __init__(self):
        self.backend = "charhash"  # "sbert" | "charhash"
        self._st = None

        name = (getattr(settings, "EMB_MODEL_NAME", "") or "").strip()
        resolved = _resolve_sbert_dir(name) if name else None

        if resolved:
            try:
                from sentence_transformers import SentenceTransformer
                self._st = SentenceTransformer(resolved)
                self.backend = "sbert"
                logger.info("Rename embeddings backend: sbert (%s)", resolved)
            except Exception as e:
                logger.warning("SBERT load failed (%s). Using charhash.", e)
                self._st = None
                self.backend = "charhash"

    def encode(self, texts: List[str]) -> np.ndarray:
        if self.backend == "sbert" and self._st is not None:
            arr = self._st.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return np.asarray(arr, dtype=np.float32)
        return _charhash_encode(texts)


_EMB = _Embedder()


def _similarity_matrix(left_texts: List[str], right_texts: List[str]) -> np.ndarray:
    A = _EMB.encode(left_texts)
    B = _EMB.encode(right_texts)
    return A @ B.T


# ----------------------------
# Baseline profile helpers
# ----------------------------
def _bp_entry(profile: dict[str, Any] | None, col: str) -> dict[str, Any]:
    if not isinstance(profile, dict):
        return {}
    v = profile.get(col) or {}
    return v if isinstance(v, dict) else {}


def _bp_examples(entry: dict[str, Any]) -> list[str]:
    for k in ("examples", "samples", "top_values"):
        v = entry.get(k)
        if isinstance(v, list) and v:
            return [str(x) for x in v[:20] if x is not None and str(x) != ""]
    return []


def build_expected_desc(col: str, col_type: str, baseline_profile: dict[str, Any] | None) -> str:
    bp = _bp_entry(baseline_profile, col)
    null_rate = float(bp.get("null_rate", 0.0) or 0.0)
    examples = _bp_examples(bp)[:10]
    return f"column_name={col}\ntype={col_type}\nnull_rate={null_rate:.3f}\nexamples={examples}\n"


def build_observed_desc(df: pd.DataFrame, col: str, col_type: str) -> str:
    s = df[col]
    null_rate = float(s.isna().mean()) if len(s) else 0.0
    ex = s.dropna().astype(str).head(10).tolist()
    return f"column_name={col}\ntype={col_type}\nnull_rate={null_rate:.3f}\nexamples={ex}\n"


# ----------------------------
# Feature dict (model expects 8 features)
# ----------------------------
def build_features(
    old: str,
    new: str,
    exp_type: str,
    obs_type: str,
    emb_sim: float,
    baseline_profile: dict[str, Any] | None,
    observed_profile: dict[str, Any],
) -> dict[str, float]:
    bp = _bp_entry(baseline_profile, old)
    op = observed_profile.get(new) or {}

    tc = 1.0 if type_compatible(exp_type, obs_type) else 0.0

    null_old = float(bp.get("null_rate", 0.0) or 0.0)
    null_new = float(op.get("null_rate", 0.0) or 0.0)

    mean_diff = std_diff = rov = 0.0
    if tc == 1.0 and exp_type in ("int", "float") and obs_type in ("int", "float"):
        try:
            if bp.get("mean") is not None and op.get("mean") is not None:
                mean_diff = float(abs(float(bp["mean"]) - float(op["mean"])))
            if bp.get("std") is not None and op.get("std") is not None:
                std_diff = float(abs(float(bp["std"]) - float(op["std"])))
        except Exception:
            pass

    return {
        "name_sim": float(name_similarity(old, new)),
        "token_jaccard": float(token_jaccard(old, new)),
        "emb_sim": float(emb_sim),
        "type_compat": float(tc),
        "null_rate_diff": float(abs(null_old - null_new)),
        "mean_diff": float(mean_diff),
        "std_diff": float(std_diff),
        "range_overlap": float(rov),
    }


# ----------------------------
# Main drift detection
# ----------------------------
def run_drift_detection(
    rows: list[dict[str, Any]],
    canonical_schema: dict[str, Any],
    baseline_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    observed_schema, df_obs = _schema_from_rows(rows)

    canon_cols = {c["name"]: c for c in canonical_schema.get("columns", []) if isinstance(c, dict) and c.get("name")}
    obs_cols = {c["name"]: c for c in observed_schema.get("columns", []) if isinstance(c, dict) and c.get("name")}

    canon_names = set(canon_cols.keys())
    obs_names = set(obs_cols.keys())

    added_raw = sorted(list(obs_names - canon_names))
    removed_raw = sorted(list(canon_names - obs_names))

    type_changes: list[dict[str, Any]] = []
    nullable_changes: list[dict[str, Any]] = []

    for name in sorted(list(canon_names & obs_names)):
        ct = str(canon_cols[name].get("type", "string"))
        ot = str(obs_cols[name].get("type", "string"))
        if ct != ot:
            type_changes.append({"column": name, "from": ct, "to": ot})

    observed_profile: dict[str, Any] = {col: _col_profile(df_obs[col]) for col in df_obs.columns}

    renames_map: dict[str, str] = {}
    mapping_confidence: dict[str, float] = {}
    candidates_out: list[dict[str, Any]] = []

    added = list(added_raw)
    removed = list(removed_raw)

    if removed and added:
        exp_type = {c["name"]: str(c.get("type", "string")) for c in canonical_schema.get("columns", []) if c.get("name")}
        obs_type = {c["name"]: str(c.get("type", "string")) for c in observed_schema.get("columns", []) if c.get("name")}

        exp_desc = [build_expected_desc(c, exp_type.get(c, "string"), baseline_profile) for c in removed]
        obs_desc = [build_observed_desc(df_obs, c, obs_type.get(c, "string")) for c in added]

        sim = _similarity_matrix(exp_desc, obs_desc)

        top_k = int(getattr(settings, "RENAME_TOPK", 12) or 12)
        proposed: list[tuple[str, str, float]] = []

        for i, old in enumerate(removed):
            idx = np.argsort(sim[i])[::-1][:min(top_k, len(added))]
            for j in idx:
                proposed.append((old, added[j], float(sim[i, j])))

        scored = []
        for old, new, emb_sim in proposed:
            fd = build_features(
                old=old,
                new=new,
                exp_type=exp_type.get(old, "string"),
                obs_type=obs_type.get(new, "string"),
                emb_sim=float(emb_sim),
                baseline_profile=baseline_profile,
                observed_profile=observed_profile,
            )
            prob = float(RENAME_SCORER.score(fd))

            same_sem = float(
                semantic_class(old, exp_type.get(old, "string")) == semantic_class(new, obs_type.get(new, "string"))
            )

            scored.append(
                {
                    "old": old,
                    "new": new,
                    "prob": prob,
                    "emb_sim": float(emb_sim),
                    "name_sim": float(fd["name_sim"]),
                    "token_jaccard": float(fd["token_jaccard"]),
                    "type_compat": float(fd["type_compat"]),
                    "same_semantic_class": float(same_sem),
                }
            )

        scored.sort(key=lambda r: (r["prob"], r["emb_sim"]), reverse=True)

        emb_thr = float(getattr(settings, "RENAME_EMB_ACCEPT_THRESHOLD", 0.55) or 0.55)
        thr = float(RENAME_SCORER.threshold)

        used_old, used_new = set(), set()

        for r in scored:
            old = r["old"]
            new = r["new"]

            if old in used_old or new in used_new:
                continue

            # ✅ semantic guard (relaxed when emb+token are strong)
            if r["same_semantic_class"] < 0.5 and r["prob"] < max(thr, 0.95):
                if not (r["emb_sim"] >= 0.80 and r["token_jaccard"] >= 0.25):
                    continue

            # ✅ reject totally unrelated names
            # BUT allow abbreviation cases if embedding similarity is strong + semantic class matches
            if r["token_jaccard"] < 0.20 and r["name_sim"] < 0.55:
                if not (r["emb_sim"] >= 0.80 and r["same_semantic_class"] >= 0.5):
                    continue

            # ✅ accept if prob OR embedding passes threshold
            if (r["prob"] < thr) and (r["emb_sim"] < emb_thr):
                continue

            renames_map[old] = new

            # confidence rule:
            # - if model is confident, use prob
            # - otherwise use emb_sim (fallback accepted)
            conf = float(r["prob"]) if r["prob"] >= thr else float(r["emb_sim"])
            mapping_confidence[old] = conf

            used_old.add(old)
            used_new.add(new)

        added = [c for c in added if c not in used_new]
        removed = [c for c in removed if c not in used_old]

        for r in scored[:50]:
            candidates_out.append(
                {
                    "old_name": r["old"],
                    "new_name": r["new"],
                    "prob": float(r["prob"]),
                    "emb_sim": float(r["emb_sim"]),
                    "name_sim": float(r["name_sim"]),
                    "token_jaccard": float(r["token_jaccard"]),
                    "type_compat": float(r["type_compat"]),
                    "same_semantic_class": float(r["same_semantic_class"]),
                    "accepted": bool(renames_map.get(r["old"]) == r["new"]),
                    "threshold": float(thr),
                    "emb_accept_threshold": float(emb_thr),
                    "emb_backend": _EMB.backend,
                }
            )

    drift_types: list[str] = []
    if added:
        drift_types.append("ADD")
    if removed:
        drift_types.append("REMOVE")
    if type_changes:
        drift_types.append("TYPE_CHANGE")
    if nullable_changes:
        drift_types.append("NULLABLE_CHANGE")
    if renames_map:
        drift_types.append("RENAME")

    return {
        "drift_types": drift_types,
        "raw_diff": {"added": added_raw, "removed": removed_raw},
        "diff": {
            "added": added,
            "removed": removed,
            "type_changes": type_changes,
            "nullable_changes": nullable_changes,
        },
        "renames": {
            "mappings": renames_map,
            "mapping_confidence": mapping_confidence,
            "candidates": candidates_out,
            "threshold": float(RENAME_SCORER.threshold),
            "using_heuristic": bool(getattr(RENAME_SCORER, "using_heuristic", True)),
            "emb_backend": _EMB.backend,
            "engine_version": ENGINE_VERSION,
        },
        "observed_schema": observed_schema,
    }