from __future__ import annotations

import math
from typing import Any


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# Shared severities
SEVERITY = {
    "deletion": 1.0,
    "type_change": 0.9,
    "rename": 0.7,
    "nullability": 0.5,
    "addition": 0.4,
}


def _extract_change_summary(drift: dict[str, Any]) -> dict[str, Any]:
    """
    Produces:
      - counts by change type
      - list of affected columns (canonical names where possible)
      - max severity
      - n = number of changed items (count all items; not unique)
    drift["diff"] expected resolved (renames removed from add/remove).
    """
    diff = drift.get("diff") or {}
    ren = drift.get("renames") or {}
    mappings = (ren.get("mappings") or {}) if isinstance(ren, dict) else {}

    removed = diff.get("removed") or []
    added = diff.get("added") or []
    type_changes = diff.get("type_changes") or []
    nullable_changes = diff.get("nullable_changes") or []

    # counts (items)
    n_removed = len(removed)
    n_added = len(added)
    n_type = len(type_changes)
    n_nullable = len(nullable_changes)
    n_rename = len(mappings)

    # affected columns (best effort)
    affected_cols: set[str] = set()
    for c in removed:
        if isinstance(c, str) and c:
            affected_cols.add(c)
    for c in added:
        if isinstance(c, str) and c:
            affected_cols.add(c)
    for tc in type_changes:
        if isinstance(tc, dict) and tc.get("column"):
            affected_cols.add(str(tc["column"]))
    for nc in nullable_changes:
        if isinstance(nc, dict) and nc.get("column"):
            affected_cols.add(str(nc["column"]))
    for canon_col, _obs_col in mappings.items():
        if canon_col:
            affected_cols.add(str(canon_col))

    # max severity present
    severities_present: list[float] = []
    if n_removed:
        severities_present.append(SEVERITY["deletion"])
    if n_type:
        severities_present.append(SEVERITY["type_change"])
    if n_rename:
        severities_present.append(SEVERITY["rename"])
    if n_nullable:
        severities_present.append(SEVERITY["nullability"])
    if n_added:
        severities_present.append(SEVERITY["addition"])

    s_max = max(severities_present) if severities_present else 0.0
    n_total = n_removed + n_added + n_type + n_nullable + n_rename

    return {
        "counts": {
            "removed": n_removed,
            "added": n_added,
            "type_changes": n_type,
            "nullable_changes": n_nullable,
            "renames": n_rename,
        },
        "affected_columns": sorted(affected_cols),
        "s_max": float(s_max),
        "n_total": int(n_total),
        "has_deletion": bool(n_removed > 0),
        "has_type_change": bool(n_type > 0),
        "has_rename": bool(n_rename > 0),
        "has_nullable": bool(n_nullable > 0),
        "has_add_only": bool(n_added > 0 and (n_removed + n_type + n_nullable + n_rename) == 0),
    }


def _rename_uncertainty(drift: dict[str, Any]) -> tuple[float, list[dict[str, Any]]]:
    """
    Best-effort read rename confidences from drift["renames"].
    Supported shapes:
      - renames["pairs"] = [{"from":..,"to":..,"confidence":0.82}, ...]
      - renames["scores"] = [{"canon":..,"observed":..,"conf":0.82}, ...]
    If nothing is found, return U=0.
    """
    ren = drift.get("renames") or {}
    if not isinstance(ren, dict):
        return 0.0, []

    details: list[dict[str, Any]] = []
    uncertainties: list[float] = []

    def _try_add(canon: str | None, obs: str | None, conf: Any):
        try:
            c = float(conf)
            c = _clip(c, 0.0, 1.0)
        except Exception:
            return
        u = 1.0 - c
        uncertainties.append(u)
        details.append({"canon": canon, "observed": obs, "confidence": c, "uncertainty": u})

    pairs = ren.get("pairs")
    if isinstance(pairs, list):
        for p in pairs:
            if isinstance(p, dict):
                _try_add(p.get("from"), p.get("to"), p.get("confidence"))

    scores = ren.get("scores")
    if isinstance(scores, list):
        for s in scores:
            if isinstance(s, dict):
                _try_add(s.get("canon"), s.get("observed"), s.get("conf") or s.get("confidence"))

    if not uncertainties:
        return 0.0, details

    return float(max(uncertainties)), details


# ----------------------------
# OPTION A (Normal Risk)
# ----------------------------

def _score_option_a(drift: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    summary = _extract_change_summary(drift)

    # A1: Technical impact T
    s_max = summary["s_max"]
    n = summary["n_total"]
    T = min(1.0, float(s_max) + 0.08 * math.log(1.0 + float(n)))

    # A2: Dataset criticality C
    crit = str(cfg.get("dataset_criticality") or "Medium").strip().lower()
    C_map = {"low": 0.3, "medium": 0.6, "high": 0.9}
    C = float(C_map.get(crit, 0.6))

    # A3: Sensitivity score S + strictness
    sens = str(cfg.get("sensitivity_class") or "None").strip().lower()
    S_map = {"none": 0.0, "internal": 0.2, "pii": 0.5, "regulated": 0.8}
    S = float(S_map.get(sens, 0.0))

    strictness = str(cfg.get("regulation_strictness") or "Light").strip().lower()
    is_regulated = (sens == "regulated")

    # A4: Key-field flag K
    key_fields = cfg.get("key_fields") or []
    if not isinstance(key_fields, list):
        key_fields = []
    key_fields_norm = {str(x).strip().lower() for x in key_fields if str(x).strip()}
    affected_norm = {c.strip().lower() for c in (summary["affected_columns"] or [])}
    K = 1.0 if (key_fields_norm and (key_fields_norm & affected_norm)) else 0.0

    # A5: Rename uncertainty U
    U, rename_details = _rename_uncertainty(drift)

    # A6: Base risk + uncertainty bump
    risk_base = _clip(0.55 * T + 0.20 * C + 0.20 * S + 0.05 * K, 0.0, 1.0)
    risk = _clip(risk_base + 0.10 * U, 0.0, 1.0)

    # A7: Regulatory floors
    floor_applied = None
    if is_regulated:
        has_del_or_type = bool(summary["has_deletion"] or summary["has_type_change"])
        has_ren_or_null = bool(summary["has_rename"] or summary["has_nullable"])
        add_only = bool(summary["has_add_only"])

        if has_del_or_type:
            floor = 0.85 if strictness == "strict" else 0.75
            floor_applied = {"rule": "del_or_type", "floor": floor}
            risk = max(risk, floor)
        elif has_ren_or_null:
            floor = 0.70 if strictness == "strict" else 0.60
            floor_applied = {"rule": "rename_or_nullable", "floor": floor}
            risk = max(risk, floor)
        elif add_only:
            floor = 0.50 if strictness == "strict" else 0.40
            floor_applied = {"rule": "add_only", "floor": floor}
            risk = max(risk, floor)

    # Routing (A)
    if risk >= 0.70:
        level, route = "high", "staging"
    elif risk >= 0.35:
        level, route = "medium", "staging"
    else:
        level, route = "low", "production"

    reasons: list[str] = []
    reasons.append(f"Mode A: Technical impact T={T:.3f} from s_max={s_max:.2f}, n={n}")
    reasons.append(f"Mode A: Dataset criticality C={C:.2f} ({cfg.get('dataset_criticality','Medium')})")
    reasons.append(f"Mode A: Sensitivity S={S:.2f} ({cfg.get('sensitivity_class','None')})")
    if K > 0:
        reasons.append("Mode A: Drift touched a key field (K=1)")
    if U > 0:
        reasons.append(f"Mode A: Rename uncertainty bump U={U:.3f}")
    if floor_applied:
        reasons.append(f"Mode A: Regulatory floor applied ({floor_applied['rule']} ≥ {floor_applied['floor']})")

    details = {
        "mode": "A",
        "inputs": {
            "dataset_criticality": cfg.get("dataset_criticality", "Medium"),
            "sensitivity_class": cfg.get("sensitivity_class", "None"),
            "regulation_strictness": cfg.get("regulation_strictness", "Light"),
            "key_fields": key_fields,
        },
        "components": {
            "T": T,
            "C": C,
            "S": S,
            "K": K,
            "U": U,
            "risk_base": risk_base,
            "risk_final": float(risk),
            "floor_applied": floor_applied,
        },
        "change_counts": summary["counts"],
        "affected_columns": summary["affected_columns"],
        "rename_details": rename_details,
    }

    return {
        "risk_score": float(risk),
        "risk_level": level,
        "route": route,
        "reasons": reasons,
        "details": details,
        "xai": {
            "mode": "A",
            "drivers": [
                {"name": "Technical impact (T)", "value": float(T)},
                {"name": "Criticality (C)", "value": float(C)},
                {"name": "Sensitivity (S)", "value": float(S)},
                {"name": "Key field touched (K)", "value": float(K)},
                {"name": "Rename uncertainty (U)", "value": float(U)},
            ],
            "notes": reasons,
        },
    }


# ----------------------------
# OPTION B (Accurate Risk)
# ----------------------------

def _semantic_weight(sem: str) -> float:
    s = (sem or "").strip().lower()
    if s in ("identifier", "identifier/pk", "pk", "primary key", "primary_key"):
        return 1.00
    if s in ("foreign key", "fk", "foreign_key"):
        return 0.90
    if s in ("timestamp", "time", "datetime", "date"):
        return 0.80
    if s in ("measure", "metric", "amount", "value"):
        return 0.70
    return 0.60


def _sensitivity_multiplier(sens: str, strictness: str | None) -> float:
    s = (sens or "none").strip().lower()
    st = (strictness or "light").strip().lower()
    if s == "none":
        return 1.00
    if s == "internal":
        return 1.05
    if s == "pii":
        return 1.15
    if s == "regulated":
        return 1.35 if st == "strict" else 1.25
    return 1.00


def _get_field_cfg(cfg: dict[str, Any], col: str) -> dict[str, Any]:
    fields = cfg.get("fields") or {}
    if isinstance(fields, dict) and col in fields and isinstance(fields[col], dict):
        return fields[col]
    return {}


def _tol_b(field_cfg: dict[str, Any], key: str, default: bool) -> bool:
    t = field_cfg.get("tolerances") or {}
    if not isinstance(t, dict):
        return default
    v = t.get(key, default)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "1", "yes", "y", "t"):
            return True
        if s in ("false", "0", "no", "n", "f", ""):
            return False
    return bool(v)


def _score_option_b(drift: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    summary = _extract_change_summary(drift)

    try:
        datasetCrit = int(cfg.get("dataset_criticality") or 3)
    except Exception:
        datasetCrit = 3
    datasetCrit = max(1, min(5, datasetCrit))
    d = datasetCrit / 5.0

    sens = str(cfg.get("sensitivity_class") or "None").strip().lower()
    strictness = str(cfg.get("regulation_strictness") or "Light").strip().lower()
    R = _sensitivity_multiplier(sens, strictness)

    U, rename_details = _rename_uncertainty(drift)

    diff = drift.get("diff") or {}
    ren = drift.get("renames") or {}
    mappings = (ren.get("mappings") or {}) if isinstance(ren, dict) else {}

    contributions: list[dict[str, Any]] = []

    def add_contrib(
        change_type: str,
        column: str,
        severity: float,
        P_tol: float,
        P_conf: float,
        from_to: str | None = None,
    ):
        field_cfg = _get_field_cfg(cfg, column)
        try:
            fieldCrit = int(field_cfg.get("field_criticality") or 3)
        except Exception:
            fieldCrit = 3
        fieldCrit = max(1, min(5, fieldCrit))
        f = fieldCrit / 5.0
        I = 0.5 * d + 0.5 * f

        sem = str(field_cfg.get("semantic_type") or "Dimension")
        M = _semantic_weight(sem)

        raw = (severity * I * M * R) + P_tol + P_conf
        r_i = _clip(raw, 0.0, 1.0)

        contributions.append(
            {
                "column": column,
                "change_type": change_type,
                "severity": severity,
                "datasetCrit": datasetCrit,
                "fieldCrit": fieldCrit,
                "I": I,
                "semantic_type": sem,
                "M": M,
                "R": R,
                "P_tol": P_tol,
                "P_conf": P_conf,
                "r_i": r_i,
                "detail": from_to,
            }
        )

    # Removed
    for c in diff.get("removed") or []:
        if not isinstance(c, str) or not c:
            continue
        field_cfg = _get_field_cfg(cfg, c)
        allow = _tol_b(field_cfg, "allow_remove", False)
        P_tol = 0.20 if not allow else 0.0
        add_contrib("deletion", c, SEVERITY["deletion"], P_tol=P_tol, P_conf=0.0)

    # Type changes
    for tc in diff.get("type_changes") or []:
        if not isinstance(tc, dict):
            continue
        c = tc.get("column")
        if not c:
            continue
        c = str(c)
        field_cfg = _get_field_cfg(cfg, c)
        allow = _tol_b(field_cfg, "allow_type_change", False)
        P_tol = 0.18 if not allow else 0.0
        add_contrib(
            "type_change",
            c,
            SEVERITY["type_change"],
            P_tol=P_tol,
            P_conf=0.0,
            from_to=f"{tc.get('from')}→{tc.get('to')}",
        )

    # Nullable changes (only risky when False->True)
    for nc in diff.get("nullable_changes") or []:
        if not isinstance(nc, dict):
            continue
        c = nc.get("column")
        if not c:
            continue
        before = nc.get("from")
        after = nc.get("to")
        if before is False and after is True:
            c = str(c)
            field_cfg = _get_field_cfg(cfg, c)
            allow = _tol_b(field_cfg, "allow_nullable_change", True)
            P_tol = 0.10 if not allow else 0.0
            add_contrib(
                "nullability",
                c,
                SEVERITY["nullability"],
                P_tol=P_tol,
                P_conf=0.0,
                from_to=f"{before}→{after}",
            )

    # Renames
    conf_map: dict[tuple[str, str], float] = {}
    for r in rename_details:
        canon = r.get("canon")
        obs = r.get("observed")
        if canon and obs and r.get("confidence") is not None:
            conf_map[(str(canon), str(obs))] = float(r["confidence"])

    for canon_col, obs_col in mappings.items():
        if not canon_col:
            continue
        canon_col = str(canon_col)
        obs_col = str(obs_col) if obs_col else ""
        field_cfg = _get_field_cfg(cfg, canon_col)
        allow = _tol_b(field_cfg, "allow_rename", True)
        P_tol = 0.12 if not allow else 0.0
        conf = conf_map.get((canon_col, obs_col))
        P_conf = 0.0
        if conf is not None:
            P_conf = 0.15 * (1.0 - _clip(conf, 0.0, 1.0))
        add_contrib(
            "rename",
            canon_col,
            SEVERITY["rename"],
            P_tol=P_tol,
            P_conf=P_conf,
            from_to=f"{canon_col}→{obs_col}",
        )

    # Adds
    for c in diff.get("added") or []:
        if not isinstance(c, str) or not c:
            continue
        add_contrib("addition", c, SEVERITY["addition"], P_tol=0.0, P_conf=0.0)

    # Combine: Risk = 1 - Π(1 - r_i)
    prod = 1.0
    for it in contributions:
        prod *= (1.0 - float(it["r_i"]))
    risk = _clip(1.0 - prod, 0.0, 1.0)

    # Optional strict rule: regulated strict + deletion/type_change >= 0.80
    if sens == "regulated" and strictness == "strict":
        if summary["has_deletion"] or summary["has_type_change"]:
            risk = max(risk, 0.80)

    # Routing (B)
    if risk >= 0.80:
        level, route = "high", "staging"
    elif risk >= 0.40:
        level, route = "medium", "staging"
    elif risk >= 0.20:
        level, route = "low", "production"
    else:
        level, route = "low", "production"

    top = sorted(contributions, key=lambda x: float(x["r_i"]), reverse=True)[:10]
    reasons = [f"Mode B: Combined risk via union-of-risks = {risk:.3f}"]
    if sens == "regulated" and strictness == "strict" and (summary["has_deletion"] or summary["has_type_change"]):
        reasons.append("Mode B: Enforced regulated-strict floor for deletion/type-change (≥0.80)")

    details = {
        "mode": "B",
        "inputs": {
            "dataset_criticality": datasetCrit,
            "sensitivity_class": cfg.get("sensitivity_class", "None"),
            "regulation_strictness": cfg.get("regulation_strictness", "Light"),
        },
        "change_counts": summary["counts"],
        "affected_columns": summary["affected_columns"],
        "rename_uncertainty": U,
        "rename_details": rename_details,
        "per_change": contributions,
        "risk_final": float(risk),
        "combine": "1 - Π(1-r_i)",
    }

    return {
        "risk_score": float(risk),
        "risk_level": level,
        "route": route,
        "reasons": reasons,
        "details": details,
        "xai": {"mode": "B", "top_contributors": top, "notes": reasons},
    }


def score_risk_and_route(drift: dict[str, Any], risk_config: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    risk_config comes from DB (saved per dataset).
    Structure:
      {"mode":"A", ...}
      {"mode":"B", ...}
    Default to A if missing.
    """
    cfg = risk_config or {}
    if not isinstance(cfg, dict):
        cfg = {}

    mode = str(cfg.get("mode") or "A").strip().upper()
    if mode == "B":
        return _score_option_b(drift, cfg)
    return _score_option_a(drift, cfg)