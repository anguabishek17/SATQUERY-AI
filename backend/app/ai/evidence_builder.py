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
    elif task in (TaskType.OPTICAL_SAR_FUSION, TaskType.SAR_ANALYSIS):
        evidence.fusion_evidence = {
            "fusion_agreement_score": result.fusion_agreement_score
        }
        evidence.measurements.update(metrics)

    # General spatial capture
    if result.bounding_boxes:
        evidence.detections["bbox_count"] = len(result.bounding_boxes)

    if not result.confidence_calibrated:
        evidence.limitations.append("Confidence score is an uncalibrated heuristic.")

    return evidence
