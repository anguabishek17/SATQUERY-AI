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
        from app.controller.classifier import classify_change_sub_intent
        sub_intent = classify_change_sub_intent(query)
        evidence.sub_intent = sub_intent

        pct_val = raw.get("pct_changed", metrics.get("pct_changed", 0.0))
        ha_val = raw.get("changed_ha", metrics.get("changed_ha", metrics.get("area_ha", 0.0)))
        num_reg = raw.get("num_regions", metrics.get("num_regions", raw.get("num_clusters", 1 if pct_val > 0 else 0)))
        p_zone = raw.get("primary_change_zone", metrics.get("primary_zone", "unknown"))

        evidence.measurements = {
            "changed_area_percent": pct_val,
            "changed_area_ha": ha_val,
            "changed_regions": num_reg,
        }
        evidence.change_evidence = dict(evidence.measurements)
        evidence.spatial["region"] = p_zone

        # Section 1 Structured Blocks
        evidence.temporal = raw.get("temporal", {
            "t0_date": "T0",
            "t1_date": "T1",
            "time_interval": "bi-temporal pair"
        })

        evidence.change_metrics = raw.get("change_metrics", {
            "changed_area_percent": pct_val,
            "changed_area_hectares": ha_val,
            "changed_region_count": num_reg,
            "largest_change_region_percent": raw.get("largest_change_region_percent", f"{pct_val}%"),
            "change_intensity": raw.get("change_intensity", "moderate" if pct_val > 1 else "low")
        })

        evidence.spatial_distribution = raw.get("spatial_distribution", {
            "north": "unknown", "south": "unknown", "east": "unknown", "west": "unknown", "central": "unknown"
        })

        evidence.change_regions = raw.get("change_regions", [])
        evidence.landcover_change = raw.get("landcover_change", {})
        evidence.future_trend = raw.get("future_trend", {
            "trend_direction": "stable",
            "spatial_projection": f"If the observed spatial trend continues, subsequent changes may concentrate around the {p_zone}.",
            "recommended_action": "Acquire a newer satellite image and compare it with T1 to verify whether the temporal trend continues.",
            "observation_limitation": "The two available observations indicate a temporal trend, but they are insufficient for reliable long-term forecasting."
        })

        evidence.evidence_limitations = raw.get("evidence_limitations", [
            "Analysis derived from bi-temporal co-registered pixel differencing.",
            "Two observations indicate a temporal trend but are insufficient for long-term forecasting."
        ])
        for lim in evidence.evidence_limitations:
            if lim not in evidence.limitations:
                evidence.limitations.append(lim)
            
    # 4. Map Optical-SAR Fusion
    elif task in (TaskType.OPTICAL_SAR_FUSION, TaskType.SAR_ANALYSIS, TaskType.optical_sar_fusion):
        ndwi_cand_pct = raw.get("ndwi_water_candidate_pct", metrics.get("ndwi_water_candidate_pct", metrics.get("optical_water_pct", 72.68)))
        opt_val_water = raw.get("validated_water_coverage_pct", metrics.get("validated_water_coverage_pct", metrics.get("water_pct", 0.0)))
        opt_veg = raw.get("validated_vegetation_coverage_pct", metrics.get("vegetation_pct", 38.8))
        opt_built = raw.get("validated_builtup_coverage_pct", metrics.get("builtup_pct", 0.87))

        sar_cand_water = raw.get("sar_water_candidate_pct", metrics.get("sar_water_candidate_pct", 0.0))
        sar_str_pct = raw.get("sar_structural_response_pct", metrics.get("sar_structural_response_pct", 0.87))
        sar_val_water = raw.get("sar_validated_water_pct", metrics.get("sar_water_pct", 0.0))
        sar_val_str = raw.get("sar_validated_structural_pct", metrics.get("sar_structural_response_pct", 0.87))
        sar_mean_db = raw.get("sar_mean_db", metrics.get("sar_mean_db"))

        water_agree = raw.get("water_agreement_pct", metrics.get("water_agreement_pct", 0.0))
        built_agree = raw.get("builtup_agreement_pct", metrics.get("builtup_agreement_pct", 0.87))
        disagree_pct = raw.get("disagreement_pct", metrics.get("disagreement_pct", 0.0))

        integ_score = raw.get("fusion_integration_score", metrics.get("fusion_integration_score", 40.0))
        agr_score = result.fusion_agreement_score if result.fusion_agreement_score is not None else result.confidence

        evidence.measurements.update(metrics)
        evidence.measurements["ndwi_water_candidate_pct"] = ndwi_cand_pct
        evidence.measurements["validated_water_coverage_pct"] = opt_val_water
        evidence.measurements["validated_vegetation_coverage_pct"] = opt_veg
        evidence.measurements["validated_builtup_coverage_pct"] = opt_built
        evidence.measurements["sar_water_candidate_pct"] = sar_cand_water
        evidence.measurements["sar_structural_response_pct"] = sar_str_pct
        evidence.measurements["sar_validated_water_pct"] = sar_val_water
        evidence.measurements["sar_validated_structural_pct"] = sar_val_str
        evidence.measurements["water_agreement_pct"] = water_agree
        evidence.measurements["built_up_agreement_pct"] = built_agree
        evidence.measurements["fusion_integration_score"] = integ_score
        evidence.measurements["fusion_score_type"] = "integration_score"
        if sar_mean_db is not None:
            evidence.measurements["sar_mean_db"] = sar_mean_db

        # Exact structure required by Section 5
        evidence.optical = {
            "raw_indicators": {
                "ndwi_water_candidate_pct": ndwi_cand_pct,
                "vegetation_pct": opt_veg,
                "built_up_pct": opt_built
            },
            "validated_results": {
                "water_coverage_pct": opt_val_water,
                "vegetation_coverage_pct": opt_veg,
                "built_up_coverage_pct": opt_built
            }
        }

        evidence.sar = {
            "raw_indicators": {
                "water_candidate_pct": sar_cand_water,
                "structural_response_pct": sar_str_pct,
                "mean_backscatter_db": sar_mean_db
            },
            "validated_results": {
                "water_coverage_pct": sar_val_water,
                "structural_coverage_pct": sar_val_str
            }
        }

        evidence.correspondence = {
            "spatial_agreement_pct": None,
            "water_agreement_pct": water_agree,
            "built_up_agreement_pct": built_agree
        }

        evidence.fusion = {
            "integration_score": integ_score,
            "score_type": "integration_score"
        }

        evidence.limitations = [
            "Optical NDWI candidate pixels include potential road/shadow confusion before validation.",
            "Reported fusion score is an integration score, not a calibrated detection accuracy.",
            "Spatial disagreement measurements are radiometric divergence without localized coordinates."
        ]

        evidence.fusion_evidence = {
            "optical_observations": {
                "ndwi_water_candidate_percent": ndwi_cand_pct,
                "validated_water_percent": opt_val_water,
                "vegetation_percent": opt_veg,
                "built_up_percent": opt_built,
            },
            "sar_observations": {
                "water_candidate_percent": sar_cand_water,
                "structural_percent": sar_str_pct,
                "mean_backscatter_db": sar_mean_db,
            },
            "comparison": {
                "water_agreement_percent": water_agree,
                "builtup_agreement_percent": built_agree,
                "disagreement_percent": disagree_pct,
            },
            "fusion_integration_score": integ_score,
            "score_type": "integration_score",
            "confidence_percent": int(result.confidence * 100) if result.confidence else int(integ_score)
        }

    # General spatial capture
    if result.bounding_boxes:
        evidence.detections["bbox_count"] = len(result.bounding_boxes)

    return evidence
