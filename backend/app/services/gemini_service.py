import os
import logging
from typing import Optional, Dict, Any, List
from PIL import Image

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None

logger = logging.getLogger(__name__)

def generate_visual_explanation(
    query: str, 
    image_paths: List[str], 
    computational_evidence: Dict[str, Any]
) -> Optional[str]:
    """
    Calls the Gemini API to provide a visual reasoning layer on top of the deterministic analysis.
    If the API key is missing or an error occurs, it returns None.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or not genai:
        return None

    try:
        client = genai.Client(api_key=api_key)
        
        # Build prompt grounding instructions
        prompt_parts = []
        prompt_parts.append(
            "You are an Earth-observation visual analysis assistant. "
            "Analyze ONLY the provided imagery and the specific computational evidence supplied below. "
            "Do NOT invent measurements, coordinates, spectral bands, percentages, objects, or geographic facts. "
            "Clearly distinguish your visual observations from the computed metrics. "
            "If the image does not provide sufficient visual evidence to answer the query, explicitly say so. "
            "Do NOT claim NDVI, NDWI, or NDBI values unless those values are explicitly supplied by the analysis pipeline."
        )
        prompt_parts.append(f"\nUser Query: {query}")
        
        # Add computational evidence
        prompt_parts.append("\n=== COMPUTATIONAL EVIDENCE (AUTHORITATIVE) ===")
        if computational_evidence.get("physical_metrics"):
            prompt_parts.append(f"Metrics: {computational_evidence['physical_metrics']}")
        if computational_evidence.get("fusion_agreement_score") is not None:
            prompt_parts.append(f"Fusion Agreement Score: {computational_evidence['fusion_agreement_score']}")
        if computational_evidence.get("detector_status") == "loaded":
            prompt_parts.append(f"Detected Object Count: {computational_evidence.get('object_count', 'unknown')}")
        if computational_evidence.get("change_stats"):
            prompt_parts.append(f"Change Statistics: {computational_evidence['change_stats']}")
            
        prompt_parts.append("\n=== TASK ===")
        prompt_parts.append("Provide a concise, qualitative visual interpretation that complements the above computational evidence. Answer the user query based ONLY on what you can clearly see in the image and the provided metrics.")

        # Load images
        contents = []
        for path in image_paths:
            if os.path.exists(path):
                # Using PIL to open image for Gemini SDK
                contents.append(Image.open(path))
            else:
                logger.warning(f"Gemini service: Image path not found: {path}")

        contents.append("\n".join(prompt_parts))

        # Use gemini-2.5-flash which is ideal for multimodal reasoning
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=512,
            )
        )
        return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini Vision API error: {e}")
        return None
