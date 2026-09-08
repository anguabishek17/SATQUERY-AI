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
    Returns a dict: {"valid": bool, "reason": str, "corrected_answer": str}
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or not genai:
        return {"valid": True, "reason": "No validator available", "corrected_answer": reasoning}

    try:
        from app.ai.ai_reasoner import _get_client
        client = _get_client(api_key)
        
        prompt = (
            "You are a strict QA Evidence Validator.\n"
            "Review the AI reasoning below and compare it against the Evidence JSON.\n"
            "If the reasoning contains unsupported claims (e.g. claiming 'flooding' without water evidence, "
            "fabricating numbers, or inventing locations), correct it to be strictly factual based ONLY on the evidence.\n"
            "If the reasoning is already factual, return it exactly as is for the corrected_answer.\n\n"
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
            return {"valid": True, "reason": "Validator returned empty", "corrected_answer": reasoning}
            
        result_json = json.loads(response.text.strip())
        
        # Ensure schema compliance
        valid = result_json.get("valid", True)
        reason = result_json.get("reason", "")
        corrected_answer = result_json.get("corrected_answer", reasoning)
        
        return {"valid": valid, "reason": reason, "corrected_answer": corrected_answer}
        
    except json.JSONDecodeError as e:
        logger.error(f"Evidence Validator JSON parse error: {e}. Raw response: {response.text if response else 'None'}")
        return {"valid": True, "reason": "Validator JSON parse failed", "corrected_answer": reasoning}
    except Exception as e:
        logger.error(f"Evidence Validator error: {e}")
        return {"valid": True, "reason": "Validator exception", "corrected_answer": reasoning}
