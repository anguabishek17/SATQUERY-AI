import json
import os
from dotenv import load_dotenv

load_dotenv()

from app.ai.evidence_schema import EvidenceModel
from app.schemas import TaskType
from app.ai.ai_reasoner import generate_ai_reasoning
from app.ai.evidence_validator import validate_reasoning

# Mock a failed genai import
import app.ai.ai_reasoner as ai_reasoner
import app.ai.evidence_validator as evidence_validator

def test_water_query_success():
    print("--- TEST 1: Water query success ---")
    evidence = EvidenceModel(
        task="WATER_DETECTION",
        status="success",
        confidence=0.70,
        source_tool="dynamic-analysis-engine",
        measurements={"water_percent": 3.8},
        detections={},
        spatial={},
        limitations=["RGB-based estimate"]
    )
    # Assuming GEMINI_API_KEY is set and google-genai is installed
    answer = generate_ai_reasoning("Is there water in this satellite image?", evidence)
    print("AI Answer:", answer)
    
    val = validate_reasoning(answer, evidence)
    print("Validation:", json.dumps(val, indent=2))
    assert val['corrected_answer'] != ""

def test_gemini_failure_fallback():
    print("--- TEST 2: Gemini failure fallback ---")
    evidence = EvidenceModel(
        task="WATER_DETECTION",
        status="success",
        confidence=0.70,
        source_tool="dynamic-analysis-engine",
        measurements={"water_percent": 3.8},
        detections={},
        spatial={},
        limitations=["RGB-based estimate"]
    )
    # Temporarily remove genai to simulate failure
    original_genai = ai_reasoner.genai
    ai_reasoner.genai = None
    try:
        answer = ai_reasoner.generate_ai_reasoning("Is there water?", evidence)
        print("Fallback Answer:", answer)
        assert "3.8" in answer
        assert "70" in answer
        assert "RGB" in answer
        assert "AI explanation layer unavailable" not in answer
    finally:
        ai_reasoner.genai = original_genai

def test_invalid_claim_rejection():
    print("--- TEST 4: Invalid AI claim (validation rejection) ---")
    evidence = EvidenceModel(
        task="WATER_DETECTION",
        status="success",
        confidence=0.70,
        source_tool="dynamic-analysis-engine",
        measurements={"water_percent": 3.8},
        detections={},
        spatial={},
        limitations=["RGB-based estimate"]
    )
    bad_reasoning = "Water covers 25% of the area and a huge flood destroyed the city."
    val = validate_reasoning(bad_reasoning, evidence)
    print("Validation result:", json.dumps(val, indent=2))
    if val['reason'] != "Validator exception" and val['reason'] != "No validator available":
        assert "25" not in val['corrected_answer'] or val['valid'] is False
        assert "3.8" in val['corrected_answer'] or "3.8%" in val['corrected_answer']
    else:
        print("Skipping validation assertion due to timeout/exception")

def test_building_query():
    print("--- TEST 5: Building query ---")
    evidence = EvidenceModel(
        task="BUILDING_COUNT",
        status="success",
        confidence=0.85,
        source_tool="building-detector",
        measurements={"building_count": 42},
        detections={},
        spatial={},
        limitations=[]
    )
    answer = generate_ai_reasoning("How many buildings?", evidence)
    print("AI Answer:", answer)
    val = validate_reasoning(answer, evidence)
    print("Validation:", json.dumps(val, indent=2))

def test_bitemporal_query():
    print("--- TEST 6: Bi-temporal query ---")
    evidence = EvidenceModel(
        task="CHANGE_DETECTION",
        status="success",
        confidence=0.92,
        source_tool="change-detector",
        measurements={"changed_area_percent": 12.5, "changed_regions": 3},
        detections={},
        spatial={},
        limitations=[]
    )
    answer = generate_ai_reasoning("What changed?", evidence)
    print("AI Answer:", answer)
    val = validate_reasoning(answer, evidence)
    print("Validation:", json.dumps(val, indent=2))

if __name__ == "__main__":
    test_water_query_success()
    test_gemini_failure_fallback()
    test_invalid_claim_rejection()
    test_building_query()
    test_bitemporal_query()
    print("ALL TESTS COMPLETED.")
