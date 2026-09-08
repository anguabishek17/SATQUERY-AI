from app.schemas import ToolResult, TaskType
from app.ai.evidence_schema import EvidenceModel

def build_evidence_json(query: str, task: TaskType, result: ToolResult) -> EvidenceModel:
    """
    Transforms the deterministic ToolResult into a standardized Evidence JSON format.
    """
    raw = result.raw or {}
    metrics = result.physical_metrics or {}
    
    evidence = EvidenceModel(
        task=task.value,
        status="success",
        confidence=result.confidence,
        source_tool=result.tool_name,
        measurements={},
        detections={},
        spatial={},
        limitations=[]
    )
    
    # 1. Map Object Counting / Building Detection
    if task in (TaskType.BUILDING_COUNT, TaskType.BUILDING_DISTRIBUTION, TaskType.object_counting):
        count = len(result.bounding_boxes) if result.bounding_boxes else 0
        evidence.measurements["building_count"] = count
        if metrics:
            evidence.measurements.update(metrics)
        evidence.building_evidence = {"building_count": count}
        if raw.get("detector_status") == "not_loaded":
            evidence.limitations.append("Building detector neural model is not loaded. Counts are unavailable.")
            evidence.status = "partial"

    # 2. Map Dynamic Analysis (Land Cover, Water, Vegetation)
    elif task in (TaskType.WATER_DETECTION, TaskType.VEGETATION_ANALYSIS, TaskType.BUILT_UP_ANALYSIS, TaskType.LAND_COVER, TaskType.dynamic_analysis, TaskType.GENERAL_SCENE_ANALYSIS):
        evidence.measurements.update(metrics)
        
        if "water_pct" in raw:
            evidence.water_evidence = {"water_percent": raw["water_pct"]}
            if "ndwi_used" not in raw or not raw.get("ndwi_used"):
                evidence.limitations.append("RGB-based water estimate; precise water detection requires NIR/SWIR data.")
        if "vegetation_pct" in raw:
            evidence.landcover_evidence = evidence.landcover_evidence or {}
            evidence.landcover_evidence["vegetation_percent"] = raw["vegetation_pct"]
        if "built_up_pct" in raw:
            evidence.landcover_evidence = evidence.landcover_evidence or {}
            evidence.landcover_evidence["built_up_percent"] = raw["built_up_pct"]

    # 3. Map Bi-Temporal Change Detection
    elif task in (TaskType.CHANGE_DETECTION, TaskType.change_description, TaskType.change_vqa):
        if "pct_changed" in raw:
            evidence.measurements = {
                "changed_area_percent": raw["pct_changed"],
                "changed_area_ha": metrics.get("area_ha", None),
                "changed_regions": raw.get("num_clusters", 0)
            }
            evidence.change_evidence = dict(evidence.measurements)
            evidence.spatial["region"] = raw.get("primary_change_zone", "unknown")
            
    # 4. Map Optical-SAR Fusion
    elif task in (TaskType.OPTICAL_SAR_FUSION, TaskType.SAR_ANALYSIS, TaskType.optical_sar_fusion):
        optical_water = raw.get("optical_water_pct", metrics.get("optical_water_pct", round(metrics.get("water_pct", 0.0), 2)))
        sar_water = raw.get("sar_water_pct", metrics.get("sar_water_pct", round(metrics.get("water_pct", 0.0), 2)))
        water_agree = raw.get("water_agreement_pct", round(metrics.get("water_pct", 0.0), 2))
        built_agree = raw.get("builtup_agreement_pct", round(metrics.get("builtup_pct", 0.0), 2))
        veg_pct = round(metrics.get("vegetation_pct", raw.get("vegetation_pct", 0.0)), 2)
        disagree_pct = raw.get("disagreement_pct", round(metrics.get("disagreement_pct", 0.0), 2))
        agr_score = result.fusion_agreement_score if result.fusion_agreement_score is not None else result.confidence
        agr_pct = round(agr_score * 100, 1) if agr_score is not None else 0.0
        sar_mean_db = raw.get("sar_mean_db", metrics.get("sar_mean_db"))

        evidence.measurements.update(metrics)
        evidence.measurements["optical_water_pct"] = optical_water
        evidence.measurements["sar_water_pct"] = sar_water
        evidence.measurements["optical_vegetation_pct"] = veg_pct
        evidence.measurements["optical_builtup_pct"] = built_agree
        evidence.measurements["sar_builtup_pct"] = built_agree
        evidence.measurements["water_agreement_pct"] = water_agree
        evidence.measurements["builtup_agreement_pct"] = built_agree
        evidence.measurements["disagreement_pct"] = disagree_pct
        evidence.measurements["agreement_score_pct"] = agr_pct
        if sar_mean_db is not None:
            evidence.measurements["sar_mean_db"] = sar_mean_db

        evidence.fusion_evidence = {
            "optical_observations": {
                "water_percent": optical_water,
                "vegetation_percent": veg_pct,
                "built_up_percent": built_agree,
                "features_identified": f"NDWI water ({optical_water}%), NDVI vegetation ({veg_pct}%), spectral built-up ({built_agree}%)"
            },
            "sar_observations": {
                "water_percent": sar_water,
                "built_up_percent": built_agree,
                "mean_backscatter_db": sar_mean_db,
                "features_identified": f"Specular reflection water ({sar_water}%), double-bounce structural backscatter ({built_agree}%)"
            },
            "comparison": {
                "water_agreement_percent": water_agree,
                "builtup_agreement_percent": built_agree,
                "disagreement_percent": disagree_pct,
                "spatial_disagreement_location": "Global radiometric threshold divergence; exact pixel coordinates not localized" if disagree_pct > 0 else "None"
            },
            "fusion_insight": "SAR provides surface roughness and structural reflectivity independent of solar illumination, while optical provides multispectral vegetative identification.",
            "agreement_percent": agr_pct,
            "confidence_percent": int(result.confidence * 100) if result.confidence else int(agr_pct)
        }

    # General spatial capture
    if result.bounding_boxes:
        evidence.detections["bbox_count"] = len(result.bounding_boxes)

    if not result.confidence_calibrated:
        evidence.limitations.append("Confidence score is an uncalibrated heuristic.")

    return evidence
