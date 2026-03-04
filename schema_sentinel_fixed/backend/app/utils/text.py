from __future__ import annotations

import re
from difflib import SequenceMatcher


# ----------------------------
# Normalization
# ----------------------------

def norm_col(s: str) -> str:
    """
    Normalize a column name into a stable form:
      - lowercase
      - non-alphanum -> underscore
      - collapse underscores
    """
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


# Common short forms seen across datasets/domains
_ABBREV = {
    "qty": "quantity",
    "num": "number",
    "no": "number",
    "cnt": "count",
    "amt": "amount",
    "avg": "average",
    "min": "minimum",
    "max": "maximum",
    "dt": "date",
    "ts": "timestamp",
    "dob": "date_of_birth",
    "id": "id",  # keep as is (important token)
}


def _expand_tokens(tok: str) -> str:
    return _ABBREV.get(tok, tok)


def tokens(s: str) -> set[str]:
    """
    Tokenize by underscores after norm_col.
    Also expands common abbreviations.
    """
    s = norm_col(s)
    toks = [t for t in s.split("_") if t]
    toks = [_expand_tokens(t) for t in toks]
    return set(toks)


def jaccard_tokens(a: str, b: str) -> float:
    A, B = tokens(a), tokens(b)
    if not A and not B:
        return 1.0
    union = len(A | B)
    return (len(A & B) / union) if union else 0.0


# ----------------------------
# Name similarity (for rename model features)
# ----------------------------

def _char_sim(a: str, b: str) -> float:
    a, b = norm_col(a), norm_col(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return float(SequenceMatcher(None, a, b).ratio())


def _prefix_sim(a: str, b: str) -> float:
    """
    Reward shared prefix segments: useful for
    temperature_c vs temp_celsius, sales_person vs salesperson_name, etc.
    """
    a, b = norm_col(a), norm_col(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    # normalize by the longer length
    denom = max(len(a), len(b))
    return float(i / denom) if denom else 0.0


def name_sim(a: str, b: str) -> float:
    """
    A schema-aware name similarity score in [0,1].

    Combines:
      - char similarity (SequenceMatcher)
      - token jaccard (after normalization + abbreviation expansion)
      - prefix overlap

    This is more robust than SequenceMatcher alone and matches the spirit
    of your notebook's feature engineering (token-aware similarity).
    """
    cs = _char_sim(a, b)
    tj = jaccard_tokens(a, b)
    ps = _prefix_sim(a, b)

    # Weighted blend (tuned for column-name matching)
    score = (0.55 * cs) + (0.30 * tj) + (0.15 * ps)

    # clamp just in case
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return float(score)
