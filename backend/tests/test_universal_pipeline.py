import pytest
from app.ai.evidence_schema import EvidenceModel
from app.ai.evidence_builder import build_evidence_json
from app.ai.evidence_validator import validate_reasoning
from app.schemas import ToolResult, TaskType

def test_evidence_json_generation():
    # Mock a ToolResult from building detector
    tool_result = ToolResult(
        task=TaskType.BUILDING_COUNT,
        tool_name="building-detector",
        confidence=0.91,
        confidence_calibrated=True,
        bounding_boxes=[[0,0,10,10], [20,20,30,30]],
        physical_metrics={"area_sq_km": 0.5}
    )
    
    evidence = build_evidence_json("how many buildings?", TaskType.BUILDING_COUNT, tool_result)
    
    assert evidence.task == "BUILDING_COUNT"
    assert evidence.status == "success"
    assert evidence.confidence == 0.91
    assert evidence.measurements["building_count"] == 2
    assert evidence.measurements["area_sq_km"] == 0.5
    assert evidence.building_evidence["building_count"] == 2

def test_evidence_json_change_detection():
    # Mock a ToolResult from change detector
    tool_result = ToolResult(
        task=TaskType.CHANGE_DETECTION,
        tool_name="change-detector",
        confidence=0.88,
        confidence_calibrated=True,
        raw={"pct_changed": 2.72, "num_clusters": 1, "primary_change_zone": "northern region"},
        physical_metrics={"area_ha": 10.89}
    )
    
    evidence = build_evidence_json("what changed?", TaskType.CHANGE_DETECTION, tool_result)
    
    assert evidence.measurements["changed_area_percent"] == 2.72
    assert evidence.measurements["changed_area_ha"] == 10.89
    assert evidence.spatial["region"] == "northern region"
    assert evidence.change_evidence["changed_area_percent"] == 2.72

def test_validator_pass_through():
    # Test that valid reasoning is passed through unchanged
    # (Without a real Gemini API key, it passes through by default)
    evidence = EvidenceModel(
        task="WATER_DETECTION",
        status="success",
        confidence=0.9,
        source_tool="test",
        water_evidence={"water_percent": 10.0}
    )
    reasoning = "There is 10% water."
    validated = validate_reasoning(reasoning, evidence)
    corrected = validated["corrected_answer"] if isinstance(validated, dict) else validated
    assert corrected == reasoning
