"""
Evaluation Metrics & Calibration Harness for SatQuery AI.

Provides standard remote-sensing object detection and segmentation metrics:
  - Precision, Recall, F1 Score
  - Mean Absolute Error (MAE) for building counting
  - Count Error Percentage
  - Intersection over Union (IoU) for bounding boxes / polygons
"""

from __future__ import annotations

from typing import Any
import numpy as np


def box_iou(box1: list[float], box2: list[float]) -> float:
    """Computes IoU between two [x1, y1, x2, y2] bounding boxes."""
    x1, y1, x2, y2 = box1
    bx1, by1, bx2, by2 = box2

    ix1, iy1 = max(x1, bx1), max(y1, by1)
    ix2, iy2 = min(x2, bx2), min(y2, by2)

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area2 = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def evaluate_building_detections(
    pred_boxes: list[list[float]],
    gt_boxes: list[list[float]],
    iou_threshold: float = 0.50,
) -> dict[str, float]:
    """
    Evaluates predicted building boxes against reference ground-truth boxes.

    Returns:
        {
          "tp": int,
          "fp": int,
          "fn": int,
          "precision": float,
          "recall": float,
          "f1_score": float,
          "mean_iou": float,
          "pred_count": int,
          "gt_count": int,
          "count_mae": float,
          "count_error_pct": float
        }
    """
    pred_count = len(pred_boxes)
    gt_count = len(gt_boxes)

    if gt_count == 0 and pred_count == 0:
        return {
            "tp": 0, "fp": 0, "fn": 0,
            "precision": 1.0, "recall": 1.0, "f1_score": 1.0, "mean_iou": 1.0,
            "pred_count": 0, "gt_count": 0, "count_mae": 0.0, "count_error_pct": 0.0,
        }

    matched_gt = set()
    tp, fp = 0, 0
    ious = []

    for pbox in pred_boxes:
        best_iou = 0.0
        best_gt_idx = -1
        for idx, gbox in enumerate(gt_boxes):
            if idx in matched_gt:
                continue
            iou = box_iou(pbox, gbox)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            matched_gt.add(best_gt_idx)
            ious.append(best_iou)
        else:
            fp += 1

    fn = gt_count - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou = float(np.mean(ious)) if ious else 0.0

    count_mae = float(abs(pred_count - gt_count))
    count_error_pct = (count_mae / gt_count * 100.0) if gt_count > 0 else (100.0 if pred_count > 0 else 0.0)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "mean_iou": round(mean_iou, 4),
        "pred_count": pred_count,
        "gt_count": gt_count,
        "count_mae": round(count_mae, 2),
        "count_error_pct": round(count_error_pct, 2),
    }
