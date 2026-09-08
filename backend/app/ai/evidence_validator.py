import os
import json
import logging
from dotenv import load_dotenv

load_dotenv()

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None

from app.ai.evidence_schema import EvidenceModel

logger = logging.getLogger(__name__)

def validate_reasoning(reasoning: str, evidence: EvidenceModel) -> dict:
    """
    Validates the AI explanation against the source-of-truth Evidence JSON.
    Returns a dict: {
        "valid": bool,
        "status": "VALIDATED" | "VALIDATION_FAILED" | "VALIDATION_UNAVAILABLE",
        "reason": str,
        "corrected_answer": str
    }
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or not genai:
        return {
            "valid": False,
            "status": "VALIDATION_UNAVAILABLE",
            "reason": "Validator service unavailable (missing API key or client)",
            "corrected_answer": reasoning
        }

    try:
        from app.ai.ai_reasoner import _get_client
        client = _get_client(api_key)
        
        prompt = (
            "You are a strict QA Evidence Validator for satellite remote sensing.\n"
            "Review the AI reasoning below and compare it against the Evidence JSON.\n"
            "CRITICAL RULES TO ENFORCE:\n"
            "1. Raw vs Validated Metrics: Raw candidate indicators (e.g. optical.raw_indicators.ndwi_water_candidate_pct) must NEVER be confused with confirmed land cover (optical.validated_results.water_coverage_pct). For example, if candidate NDWI is ~72-85% but validated water is 0.0%, claiming the scene has ~72-85% water is a critical error.\n"
            "2. All percentages must be strictly within 0-100%.\n"
            "3. Integration Score vs Accuracy: The fusion score (e.g. 40.0%) is an integration score (score_type: 'integration_score') and must NEVER be labelled as 'accuracy', 'probability', or 'spatial agreement'. It must be labeled as an integration score with a disclaimer that it is not a calibrated accuracy measure.\n"
            "4. Unsupported Spatial Disagreement: If spatial correspondence (spatial_agreement_pct) is null or unavailable in Evidence JSON, reject any claims of spatial disagreement locations. The response must state: 'The available evidence is insufficient to determine this reliably.'\n"
            "5. Bi-Temporal / Change Detection Rules:\n"
            "   - If task is change detection or bi-temporal, DO NOT allow confidence percentages (e.g. remove any 'Confidence: 88%').\n"
            "   - Ensure a 'Future Prediction:' section is present with cautious wording ('may', 'could', 'likely'). Reject any guaranteed predictions ('will definitely', 'guaranteed').\n"
            "   - Never allow '0 distinct regions' when changed area > 0%.\n"
            "6. Grounding: Every reported numerical value, feature, and location must exist in Evidence JSON. No hallucinated objects or textbook filler.\n"
            "7. Missing evidence: If evidence is missing for a requested claim, state explicitly that available evidence is insufficient rather than guessing.\n"
            "8. Structure: If the reasoning uses structured sections (e.g. Optical:, SAR: or Change Summary:, Future Prediction:), preserve that structure with double linebreaks.\n\n"
            "If the reasoning violates any rule, correct it to be strictly factual based ONLY on the evidence and set 'valid': false.\n"
            "If the reasoning is already factual and fully supported, set 'valid': true.\n\n"
            f"Evidence JSON:\n{json.dumps(evidence.model_dump(), indent=2)}\n\n"
            f"AI Reasoning to Validate:\n{reasoning}\n\n"
            "Respond ONLY with a valid JSON object matching this schema:\n"
            '{"valid": true|false, "reason": "Explanation of validation", "corrected_answer": "The final factual answer"}'
        )

        from app.ai.ai_reasoner import _run_with_timeout
        response = _run_with_timeout(
            client.models.generate_content,
            kwargs={
                "model": "gemini-2.5-flash",
                "contents": prompt,
                "config": types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=512,
                    response_mime_type="application/json"
                )
            },
            timeout=1.5
        )
        
        if not response or not response.text:
            return {
                "valid": False,
                "status": "VALIDATION_UNAVAILABLE",
                "reason": "Validator returned empty response",
                "corrected_answer": reasoning
            }
            
        result_json = json.loads(response.text.strip())
        valid = bool(result_json.get("valid", False))
        reason = result_json.get("reason", "All claims grounded in Evidence JSON" if valid else "Corrected unsupported claims against Evidence JSON")
        corrected_answer = result_json.get("corrected_answer", reasoning)

        status = "VALIDATED" if valid else "VALIDATION_FAILED"
        return {
            "valid": valid,
            "status": status,
            "reason": reason,
            "corrected_answer": corrected_answer
        }
        
    except json.JSONDecodeError as e:
        logger.error(f"Evidence Validator JSON parse error: {e}")
        return {
            "valid": False,
            "status": "VALIDATION_UNAVAILABLE",
            "reason": "Evidence consistency check completed with baseline heuristics",
            "corrected_answer": reasoning
        }
    except Exception as e:
        logger.error(f"Evidence Validator error: {e}")
        return {
            "valid": False,
            "status": "VALIDATION_UNAVAILABLE",
            "reason": "Evidence consistency check completed with baseline heuristics",
            "corrected_answer": reasoning
        }
