"""
ISRO / Research-Grade Reproducible Analysis Report Engine for SatQuery AI.

Generates structured multi-format analysis reports containing:
  1. Dataset & Sensor Information
  2. User Query & Task Intent
  3. Methodology & Model Checkpoints
  4. Quantitative Results & Physical Metrics
  5. Validation & Calibration Status (Reporting 'Not Available' for uncalibrated metrics)
  6. Limitations & Environmental Constraints
  7. Machine-Readable Outputs (GeoJSON, JSON, CSV)
  8. Reproducibility Metadata
"""

from __future__ import annotations

import datetime
import os
from typing import Any, Optional
from app.config import (
    OBJECT_DETECTOR_WEIGHTS,
    DETECTOR_CONF_THRESH,
    DETECTOR_TILE_OVERLAP,
    DETECTOR_NMS_IOU,
    NDVI_THRESHOLD,
    NDWI_THRESHOLD,
    NDBI_THRESHOLD,
    SAR_WATER_THRESHOLD,
)


def generate_research_report(
    query: str,
    tool_result: Any,
    sensor_info: dict[str, Any],
    aoi_bbox: Optional[list[float]] = None,
    session_id: Optional[str] = None,
) -> dict[str, Any]:
    """Generates an ISRO/Researcher-grade analysis report dict."""
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # Extract metrics from tool result
    raw = getattr(tool_result, "raw", {}) or {}
    phys_metrics = getattr(tool_result, "physical_metrics", {}) or {}
    conf = getattr(tool_result, "confidence", 0.0)
    conf_calibrated = getattr(tool_result, "confidence_calibrated", False)

    # Building metrics
    count = phys_metrics.get("building_count")
    density = phys_metrics.get("density_per_km2")
    area_ha = phys_metrics.get("area_ha")
    area_km2 = phys_metrics.get("area_sq_km")

    # Spectral / Fusion metrics
    water_pct = phys_metrics.get("water_pct")
    builtup_pct = phys_metrics.get("builtup_pct")
    veg_pct = phys_metrics.get("vegetation_pct")
    fusion_score = getattr(tool_result, "fusion_agreement_score", None)

    # Section 1: Dataset Information
    dataset_info = {
        "primary_file": sensor_info.get("filename", "Unknown"),
        "modality": sensor_info.get("modality", "Optical"),
        "crs": sensor_info.get("crs", "Pixel Space"),
        "spatial_resolution_m": sensor_info.get("spatial_resolution_m", 10.0),
        "dimensions_px": [sensor_info.get("width", 512), sensor_info.get("height", 512)],
        "aoi_bbox_px": aoi_bbox if aoi_bbox else "Full Scene (No AOI Crop)",
    }

    # Section 3: Methodology
    methodology = {
        "tool_name": getattr(tool_result, "tool_name", "satquery-tool"),
        "detector_weights": os.path.basename(OBJECT_DETECTOR_WEIGHTS),
        "tile_size": "512 x 512",
        "tile_overlap": f"{DETECTOR_TILE_OVERLAP:.0%}",
        "confidence_threshold": DETECTOR_CONF_THRESH,
        "nms_iou_threshold": DETECTOR_NMS_IOU,
        "spectral_thresholds": {
            "ndvi": NDVI_THRESHOLD,
            "ndwi": NDWI_THRESHOLD,
            "ndbi": NDBI_THRESHOLD,
            "sar_water_db": SAR_WATER_THRESHOLD,
        },
    }

    # Section 4: Results
    results_summary = {
        "answer": getattr(tool_result, "output_text", ""),
        "building_count": count,
        "building_density_per_km2": density,
        "analysed_area_ha": area_ha,
        "analysed_area_sq_km": area_km2,
        "water_coverage_pct": water_pct,
        "builtup_coverage_pct": builtup_pct,
        "vegetation_coverage_pct": veg_pct,
        "fusion_agreement_score": fusion_score,
    }

    # Section 5: Validation Status
    detector_status = raw.get("detector_status", "loaded" if conf > 0 else "not_loaded")
    validation_status = {
        "calibration_status": "Calibrated Model Confidence" if conf_calibrated else "Uncalibrated / Integration Score",
        "confidence": {
            "value": conf,
            "type": "model_output" if conf_calibrated else "integration_score",
            "calibrated": conf_calibrated,
        },
        "detector_status": detector_status,
        "detector_weights": raw.get("detector_weights", os.path.basename(OBJECT_DETECTOR_WEIGHTS)),
        "validation": {
            "precision": "Not Available (Ground Truth Required)",
            "recall": "Not Available (Ground Truth Required)",
            "f1": "Not Available (Ground Truth Required)",
            "count_mae": "Not Available (Ground Truth Required)",
        },
    }

    # Section 6: Limitations
    limitations = [
        "Cloud cover or shadow occlusion may impact optical land-cover indices.",
        "SAR speckle noise filtered using Lee filter; small urban corner reflectors may vary.",
        "Building detection uses centroid containment rule for AOI isolation.",
    ]

    # Section 7: Machine-Readable Outputs (GeoJSON / CSV)
    geojson_overlay = getattr(tool_result, "geojson_overlay", None)
    csv_text = "label,count,density_per_km2,area_ha\n"
    if count is not None:
        csv_text += f"building,{count},{density},{area_ha}\n"
    if water_pct is not None:
        csv_text += f"water_coverage,{water_pct}%,N/A,{area_ha}\n"

    report = {
        "title": "SATQUERY AI - REMOTE SENSING RESEARCH ANALYSIS REPORT",
        "timestamp": timestamp,
        "session_id": session_id,
        "1_dataset_information": dataset_info,
        "2_user_query": query,
        "3_methodology": methodology,
        "4_results": results_summary,
        "5_validation_status": validation_status,
        "6_limitations": limitations,
        "7_machine_readable_outputs": {
            "geojson": geojson_overlay,
            "csv": csv_text,
        },
        "8_reproducibility": {
            "system_version": "SatQuery AI v2.0-RS",
            "parameters": {
                "conf_thresh": DETECTOR_CONF_THRESH,
                "tile_overlap": DETECTOR_TILE_OVERLAP,
                "nms_iou": DETECTOR_NMS_IOU,
            },
        },
    }

    return report
