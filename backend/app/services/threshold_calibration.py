"""
Threshold Calibration Engine for SatQuery AI.

Provides grid-search optimization of spectral thresholds (NDVI, NDWI, NDBI) and
SAR backscatter thresholds against ground-truth reference masks.
"""

from __future__ import annotations

from typing import Optional
import numpy as np

from app.services.sar_processing import ndvi, ndwi, ndbi, sar_water_mask, lee_filter
from app.services.evaluation_metrics import evaluate_building_detections


def calibrate_ndwi_threshold(
    green: np.ndarray,
    nir: np.ndarray,
    reference_water_mask: np.ndarray,
    candidates: Optional[np.ndarray] = None,
) -> dict[str, float]:
    """Finds optimal NDWI threshold maximizing F1 score against reference water mask."""
    if candidates is None:
        candidates = np.linspace(-0.20, 0.40, 31)

    ndwi_img = ndwi(green, nir)
    ref = reference_water_mask.astype(bool)

    best_thresh = 0.00
    best_f1 = 0.0
    best_iou = 0.0

    for t in candidates:
        pred = ndwi_img > t
        tp = np.logical_and(pred, ref).sum()
        fp = np.logical_and(pred, ~ref).sum()
        fn = np.logical_and(~pred, ref).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(t)
            best_iou = float(iou)

    return {
        "optimal_threshold": round(best_thresh, 4),
        "best_f1_score": round(best_f1, 4),
        "best_iou": round(best_iou, 4),
    }


def calibrate_sar_water_threshold(
    vv_db: np.ndarray,
    reference_water_mask: np.ndarray,
    candidates: Optional[np.ndarray] = None,
) -> dict[str, float]:
    """Finds optimal SAR water backscatter threshold (in dB) maximizing F1 score."""
    if candidates is None:
        candidates = np.linspace(-25.0, -10.0, 31)

    filtered_sar = lee_filter(vv_db)
    ref = reference_water_mask.astype(bool)

    best_thresh = -17.0
    best_f1 = 0.0
    best_iou = 0.0

    for t in candidates:
        pred = sar_water_mask(filtered_sar, threshold_db=float(t))
        tp = np.logical_and(pred, ref).sum()
        fp = np.logical_and(pred, ~ref).sum()
        fn = np.logical_and(~pred, ref).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(t)
            best_iou = float(iou)

    return {
        "optimal_threshold_db": round(best_thresh, 2),
        "best_f1_score": round(best_f1, 4),
        "best_iou": round(best_iou, 4),
    }
