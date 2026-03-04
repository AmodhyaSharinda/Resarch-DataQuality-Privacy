from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
from app.core.config import settings

logger = logging.getLogger(__name__)

DEFAULT_FEATURES_8 = [
    "name_sim",
    "token_jaccard",
    "emb_sim",
    "type_compat",
    "null_rate_diff",
    "mean_diff",
    "std_diff",
    "range_overlap",
]


def _resolve_path(p: str | None) -> str | None:
    if not p:
        return None

    pth = Path(p)

    if pth.is_absolute() and pth.exists():
        return str(pth)

    cwd_candidate = (Path.cwd() / pth).resolve()
    if cwd_candidate.exists():
        return str(cwd_candidate)

    backend_root = Path(__file__).resolve().parents[2]  # backend/
    root_candidate = (backend_root / pth).resolve()
    if root_candidate.exists():
        return str(root_candidate)

    return None


class RenameScorer:
    def __init__(self) -> None:
        self.model: Any = None
        self.features: list[str] = list(DEFAULT_FEATURES_8)
        self.threshold: float = 0.72
        self.using_heuristic: bool = True

        model_path = _resolve_path(getattr(settings, "RENAME_MODEL_PATH", None))
        feats_path = _resolve_path(getattr(settings, "RENAME_FEATURES_PATH", None))
        thresh_path = _resolve_path(getattr(settings, "RENAME_THRESHOLD_PATH", None))

        if model_path:
            try:
                self.model = joblib.load(model_path)
                self.using_heuristic = False
            except Exception as e:
                logger.warning("Failed to load rename model (%s). Using heuristic.", e)
                self.model = None
                self.using_heuristic = True

        if feats_path:
            try:
                feats = joblib.load(feats_path)
                if isinstance(feats, (list, tuple)) and all(isinstance(x, str) for x in feats):
                    self.features = list(feats)
            except Exception as e:
                logger.warning("Failed to load rename features (%s). Using defaults.", e)
                self.features = list(DEFAULT_FEATURES_8)

        if thresh_path:
            try:
                with open(thresh_path, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                if isinstance(obj, dict) and "threshold" in obj:
                    self.threshold = float(obj["threshold"])
            except Exception as e:
                logger.warning("Failed to load threshold (%s).", e)

        if self.using_heuristic:
            # heuristic mode should not be too strict
            self.threshold = min(max(self.threshold, 0.58), 0.66)

        logger.info(
            "RenameScorer loaded. using_heuristic=%s threshold=%.3f model_loaded=%s features=%s",
            self.using_heuristic,
            self.threshold,
            bool(self.model is not None),
            ",".join(self.features),
        )

    def score(self, feat: dict) -> float:
        if not isinstance(feat, dict):
            return 0.0

        if self.model is not None:
            try:
                X = [[float(feat.get(k, 0.0) or 0.0) for k in self.features]]
                return float(self.model.predict_proba(X)[0, 1])
            except Exception as e:
                logger.warning("predict_proba failed (%s). Falling back.", e)

        # heuristic fallback
        ns = float(feat.get("name_sim", 0.0) or 0.0)
        tj = float(feat.get("token_jaccard", 0.0) or 0.0)
        es = float(feat.get("emb_sim", 0.0) or 0.0)
        tc = float(feat.get("type_compat", 0.0) or 0.0)
        if tc <= 0.0:
            return 0.0

        null_diff = float(feat.get("null_rate_diff", 0.0) or 0.0)
        null_sim = max(0.0, 1.0 - min(1.0, null_diff * 2.0))
        mean_diff = float(feat.get("mean_diff", 0.0) or 0.0)
        std_diff = float(feat.get("std_diff", 0.0) or 0.0)
        rov = float(feat.get("range_overlap", 0.0) or 0.0)

        mean_sim = 1.0 / (1.0 + max(0.0, mean_diff))
        std_sim = 1.0 / (1.0 + max(0.0, std_diff))

        s = (
            0.28 * ns
            + 0.18 * tj
            + 0.22 * es
            + 0.10 * tc
            + 0.07 * rov
            + 0.05 * mean_sim
            + 0.05 * std_sim
            + 0.05 * null_sim
        )
        return float(max(0.0, min(1.0, s)))


RENAME_SCORER = RenameScorer()