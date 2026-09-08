import pytest
from app.ai.evidence_schema import EvidenceModel
from app.ai.evidence_builder import build_evidence_json
from app.ai.ai_reasoner import _fallback_reasoning
from app.schemas import ToolResult, TaskType

def test_optical_sar_evidence_separation():
    """1 & 2: Verify Raw NDWI != Validated Water and clean Optical/SAR separation."""
    tool_result = ToolResult(
        task=TaskType.optical_sar_fusion,
        tool_name="optical-sar-fusion",
        output_text="Test optical and SAR fusion output",
        confidence=0.40,
        confidence_calibrated=False,
        physical_metrics={
            "ndwi_water_candidate_pct": 84.59,
            "validated_water_coverage_pct": 0.0,
            "water_pct": 0.0,
            "optical_vegetation_pct": 2.0,
            "optical_builtup_pct": 2.1,
            "sar_water_candidate_pct": 0.06,
            "sar_water_pct": 0.06,
            "sar_structural_response_pct": 2.1,
            "sar_mean_db": 19.92,
            "water_agreement_pct": 0.0,
            "builtup_agreement_pct": 2.1,
            "disagreement_pct": 84.65,
            "fusion_integration_score": 40.0,
            "fusion_score_type": "integration_score"
        },
        raw={
            "ndwi_water_candidate_pct": 84.59,
            "validated_water_coverage_pct": 0.0,
            "validated_vegetation_coverage_pct": 2.0,
            "validated_builtup_coverage_pct": 2.1,
            "sar_water_candidate_pct": 0.06,
            "sar_structural_response_pct": 2.1,
            "sar_validated_water_pct": 0.06,
            "sar_validated_structural_pct": 2.1,
            "sar_mean_db": 19.92,
            "water_agreement_pct": 0.0,
            "builtup_agreement_pct": 2.1,
            "disagreement_pct": 84.65,
            "fusion_integration_score": 40.0,
            "fusion_score_type": "integration_score"
        }
    )

    evidence = build_evidence_json("Compare the optical and SAR images", TaskType.optical_sar_fusion, tool_result)

    # Check 1: Raw NDWI != Validated water
    assert evidence.optical["raw_indicators"]["ndwi_water_candidate_pct"] == 84.59
    assert evidence.optical["validated_results"]["water_coverage_pct"] == 0.0
    assert evidence.optical["raw_indicators"]["ndwi_water_candidate_pct"] != evidence.optical["validated_results"]["water_coverage_pct"]

    # Check 2: Optical/SAR separation
    assert evidence.optical["validated_results"]["vegetation_coverage_pct"] == 2.0
    assert evidence.optical["validated_results"]["built_up_coverage_pct"] == 2.1

    assert evidence.sar["raw_indicators"]["water_candidate_pct"] == 0.06
    assert evidence.sar["raw_indicators"]["structural_response_pct"] == 2.1
    assert evidence.sar["raw_indicators"]["mean_backscatter_db"] == 19.92
    assert evidence.sar["validated_results"]["water_coverage_pct"] == 0.06
    assert evidence.sar["validated_results"]["structural_coverage_pct"] == 2.1

    # Check 3 & 4: Correspondence & Integration score correctly labelled
    assert evidence.correspondence["spatial_agreement_pct"] is None
    assert evidence.correspondence["water_agreement_pct"] == 0.0
    assert evidence.correspondence["built_up_agreement_pct"] == 2.1
    assert evidence.fusion["integration_score"] == 40.0
    assert evidence.fusion["score_type"] == "integration_score"

def test_optical_sar_reasoning_structure_and_grounding():
    """3, 4, 5, 6: Structure, no fabricated disagreement, integration score disclaimer."""
    evidence = EvidenceModel(
        task="OPTICAL_SAR_FUSION",
        status="success",
        confidence=0.40,
        source_tool="optical-sar-fusion",
        optical={
            "raw_indicators": {
                "ndwi_water_candidate_pct": 84.59,
                "vegetation_pct": 2.0,
                "built_up_pct": 2.1
            },
            "validated_results": {
                "water_coverage_pct": 0.0,
                "vegetation_coverage_pct": 2.0,
                "built_up_coverage_pct": 2.1
            }
        },
        sar={
            "raw_indicators": {
                "water_candidate_pct": 0.06,
                "structural_response_pct": 2.1,
                "mean_backscatter_db": 19.92
            },
            "validated_results": {
                "water_coverage_pct": 0.06,
                "structural_coverage_pct": 2.1
            }
        },
        correspondence={
            "spatial_agreement_pct": None,
            "water_agreement_pct": 0.0,
            "built_up_agreement_pct": 2.1
        },
        fusion={
            "integration_score": 40.0,
            "score_type": "integration_score"
        },
        measurements={
            "ndwi_water_candidate_pct": 84.59,
            "validated_water_coverage_pct": 0.0,
            "validated_vegetation_coverage_pct": 2.0,
            "validated_builtup_coverage_pct": 2.1,
            "sar_water_candidate_pct": 0.06,
            "sar_structural_response_pct": 2.1,
            "sar_validated_water_pct": 0.06,
            "sar_validated_structural_pct": 2.1,
            "sar_mean_db": 19.92,
            "water_agreement_pct": 0.0,
            "built_up_agreement_pct": 2.1,
            "fusion_integration_score": 40.0,
            "fusion_score_type": "integration_score"
        }
    )

    # Test 1: Compare query
    ans_compare = _fallback_reasoning(evidence, "Compare the optical and SAR images")
    assert "Optical:" in ans_compare
    assert "SAR:" in ans_compare
    assert "Comparison:" in ans_compare
    assert "Fusion insight:" in ans_compare
    assert "Agreement/Confidence:" in ans_compare
    # Must report validated water (0.0%), not candidate (84.59%) as confirmed
    assert "0.0% water coverage" in ans_compare or "0.0% confirmed water" in ans_compare or "84.59% initial NDWI candidate" in ans_compare
    assert "84.59% water coverage" not in ans_compare
    # Integration score correctly labelled
    assert "40.0%" in ans_compare or "40%" in ans_compare
    assert "integration score, not a calibrated accuracy measure" in ans_compare

    # Test 2: Disagreement query when spatial correspondence is missing
    ans_disagree = _fallback_reasoning(evidence, "Where do optical and SAR disagree?")
    assert "The available evidence is insufficient to determine this reliably." in ans_disagree
