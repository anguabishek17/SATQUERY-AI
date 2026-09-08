import pytest
from app.ai.evidence_schema import EvidenceModel
from app.ai.evidence_builder import build_evidence_json
from app.ai.ai_reasoner import _fallback_reasoning
from app.controller.classifier import classify_change_sub_intent, classify
from app.schemas import ToolResult, TaskType, InputConfig
from app.services.change_processing import compute_change_mask
import numpy as np

def test_change_mask_nonzero_regions():
    t0 = np.zeros((100, 100))
    t1 = np.zeros((100, 100))
    # Small 4x4 perturbation (area = 16 < min_region_px=25)
    t1[20:24, 20:24] = 1.0
    res = compute_change_mask(t0, t1, min_region_px=25)
    # Must NOT report 0 regions when changed_px > 0
    assert res["changed_px"] > 0
    assert res["num_regions"] >= 1
    assert "spatial_distribution" in res
    assert "change_regions" in res

def test_sub_intent_classification():
    assert classify_change_sub_intent("What major changes occurred between the two images?") == "CHANGE_SUMMARY"
    assert classify_change_sub_intent("Where are the changes concentrated?") == "CHANGE_LOCATION"
    assert classify_change_sub_intent("How much area changed?") == "CHANGE_QUANTITY"
    assert classify_change_sub_intent("Has vegetation increased or decreased?") == "LANDCOVER_CHANGE"
    assert classify_change_sub_intent("Has built-up area increased?") == "HUMAN_ACTIVITY_CHANGE"
    assert classify_change_sub_intent("Is there evidence of new construction?") == "HUMAN_ACTIVITY_CHANGE"
    assert classify_change_sub_intent("Based on these changes, what is the likely future trend?") == "FUTURE_PREDICTION"
    assert classify_change_sub_intent("What should I monitor next?") == "FUTURE_PREDICTION"
    assert classify_change_sub_intent("Compare the two images") == "CHANGE_COMPARISON"

def test_prediction_query_task_classification():
    # In bi-temporal mode, future prediction queries must classify as CHANGE_DETECTION
    task = classify("Predict the future trend.", InputConfig.bi_temporal)
    assert task == TaskType.CHANGE_DETECTION
    task2 = classify("What should I monitor next?", InputConfig.bi_temporal)
    assert task2 == TaskType.CHANGE_DETECTION

def test_evidence_json_bitemporal_structure():
    raw_payload = {
        "pct_changed": 2.72,
        "changed_ha": 10.9,
        "num_regions": 1,
        "primary_change_zone": "northern region",
        "largest_change_region_percent": "2.72%",
        "change_intensity": "moderate",
        "spatial_distribution": {
            "north": "80%", "south": "20%", "east": "50%", "west": "50%", "central": "10%"
        },
        "change_regions": [
            {
                "region_id": 1,
                "bbox": [10, 10, 40, 40],
                "centroid": [25.0, 25.0],
                "area_percent": "2.72%",
                "area_hectares": 10.9
            }
        ],
        "landcover_change": {
            "vegetation_change": "decrease of 1.5 percentage points",
            "builtup_change": "increase of 2.1 percentage points",
            "water_change": "stable",
            "bare_land_change": "-0.6 percentage points"
        },
        "future_trend": {
            "trend_direction": "increasing development",
            "recommended_action": "Compare T1 with a newer satellite image to confirm whether the trend continues."
        }
    }
    tool_result = ToolResult(
        task=TaskType.CHANGE_DETECTION,
        tool_name="change-detector",
        confidence=0.88,
        confidence_calibrated=True,
        raw=raw_payload,
        physical_metrics={"pct_changed": 2.72, "changed_ha": 10.9, "num_regions": 1}
    )

    ev = build_evidence_json("Where are the changes concentrated?", TaskType.CHANGE_DETECTION, tool_result)
    assert ev.sub_intent == "CHANGE_LOCATION"
    assert ev.change_metrics["changed_area_percent"] == 2.72
    assert ev.change_metrics["changed_region_count"] == 1
    assert ev.spatial_distribution["north"] == "80%"
    assert len(ev.change_regions) == 1
    assert ev.landcover_change["builtup_change"] == "increase of 2.1 percentage points"

def test_query_specific_responses_and_future_prediction():
    raw_payload = {
        "pct_changed": 2.72,
        "changed_ha": 10.9,
        "num_regions": 1,
        "primary_change_zone": "northern region",
        "largest_change_region_percent": "2.72%",
        "change_intensity": "moderate",
        "spatial_distribution": {"north": "85%", "south": "15%", "east": "50%", "west": "50%", "central": "10%"},
        "change_regions": [{"region_id": 1, "area_percent": "2.72%", "area_hectares": 10.9}],
        "landcover_change": {
            "vegetation_change": "decrease of 1.5 percentage points (from 45.0% to 43.5%)",
            "builtup_change": "increase of 2.1 percentage points (from 8.0% to 10.1%)",
            "water_change": "stable of 0.0 percentage points",
            "bare_land_change": "-0.6 percentage points"
        },
        "future_trend": {
            "trend_direction": "increasing development",
            "recommended_action": "Compare T1 with a newer satellite image to confirm whether the trend continues."
        }
    }
    tool_result = ToolResult(
        task=TaskType.CHANGE_DETECTION,
        tool_name="change-detector",
        confidence=0.88,
        confidence_calibrated=True,
        raw=raw_payload,
        physical_metrics={"pct_changed": 2.72, "changed_ha": 10.9, "num_regions": 1}
    )

    queries = [
        "What major changes occurred between the two images?",
        "Where are the changes concentrated?",
        "How much area changed?",
        "Has vegetation increased or decreased?",
        "Has built-up area increased?",
        "Is there evidence of new construction?",
        "Based on these changes, what is the likely future trend?",
        "What should I monitor next?",
    ]

    answers = {}
    for q in queries:
        ev = build_evidence_json(q, TaskType.CHANGE_DETECTION, tool_result)
        ans = _fallback_reasoning(ev, q)
        answers[q] = ans

        # Requirement: Every response must contain "Future Prediction:"
        assert "Future Prediction:" in ans, f"Missing 'Future Prediction:' in answer for '{q}'"

        # Requirement: No confidence percentage in response text
        assert "Confidence is" not in ans, f"Leaked confidence in answer for '{q}'"
        assert "Confidence:" not in ans, f"Leaked confidence in answer for '{q}'"

        # Requirement: No "across 0 distinct regions"
        assert "across 0 distinct regions" not in ans

    # Requirement: Answers must NOT all be identical
    unique_answers = set(answers.values())
    assert len(unique_answers) >= 6, f"Expected distinct query answers, got only {len(unique_answers)} unique answers"
