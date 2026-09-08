from typing import Optional, Dict, Any, List
from pydantic import BaseModel

class EvidenceModel(BaseModel):
    """
    Standardized Evidence JSON schema.
    This structured payload is the ONLY ground-truth data passed to the AI Reasoning layer.
    """
    task: str
    sub_intent: Optional[str] = None
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

    # Modality-separated evidence blocks for cross-modal fusion
    optical: Optional[Dict[str, Any]] = None
    sar: Optional[Dict[str, Any]] = None
    correspondence: Optional[Dict[str, Any]] = None
    fusion: Optional[Dict[str, Any]] = None
    
    # Bi-temporal structured blocks (Section 1)
    temporal: Optional[Dict[str, Any]] = None
    change_metrics: Optional[Dict[str, Any]] = None
    spatial_distribution: Optional[Dict[str, Any]] = None
    change_regions: Optional[List[Dict[str, Any]]] = None
    landcover_change: Optional[Dict[str, Any]] = None
    future_trend: Optional[Dict[str, Any]] = None
    evidence_limitations: Optional[List[str]] = None

    confidence: float
    validation_status: Optional[str] = None
    limitations: List[str] = []
    source_tool: str

