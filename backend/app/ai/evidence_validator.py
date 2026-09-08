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
            "You are a strict QA Evidence Validator.\n"
            "Review the AI reasoning below and compare it against the Evidence JSON.\n"
            "Rules to verify:\n"
            "1. Every numerical value in the AI response must exist in the Evidence JSON.\n"
            "2. Every reported location must exist in the Evidence JSON.\n"
            "3. Every reported detected feature must be evidence-supported.\n"
            "4. No unsupported modality claims were introduced.\n"
            "5. Confidence and agreement must match the Evidence JSON.\n"
            "6. No hallucinated spatial disagreement locations exist (if Evidence JSON does not contain exact coordinates, no locations must be named).\n\n"
            "If the reasoning contains unsupported claims, correct it to be strictly factual based ONLY on the evidence.\n"
            "If the reasoning follows the 5-part structure (Optical:, SAR:, Comparison:, Fusion insight:, Agreement/Confidence:), preserve that structure with double linebreaks between sections.\n"
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
            "reason": f"Validator output parse failed: {e}",
            "corrected_answer": reasoning
        }
    except Exception as e:
        logger.error(f"Evidence Validator error: {e}")
        return {
            "valid": False,
            "status": "VALIDATION_UNAVAILABLE",
            "reason": f"Validator exception: {e}",
            "corrected_answer": reasoning
        }
