from typing import Optional, Dict, Any, List
from pydantic import BaseModel

class EvidenceModel(BaseModel):
    """
    Standardized Evidence JSON schema.
    This structured payload is the ONLY ground-truth data passed to the AI Reasoning layer.
    """
    task: str
    status: str
    measurements: Dict[str, Any] = {}
    detections: Dict[str, Any] = {}
    spatial: Dict[str, Any] = {}
    
    # Specific task evidence blocks
    water_evidence: Optional[Dict[str, Any]] = None
    building_evidence: Optional[Dict[str, Any]] = None
    landcover_evidence: Optional[Dict[str, Any]] = None
    change_evidence: Optional[Dict[str, Any]] = None
    fusion_evidence: Optional[Dict[str, Any]] = None
    
    confidence: float
    limitations: List[str] = []
    source_tool: str
