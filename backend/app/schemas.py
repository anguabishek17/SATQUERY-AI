"""
Shared request/response contracts. Every specialist tool speaks this schema so the
controller can combine outputs without caring which model produced them.
"""
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class InputConfig(str, Enum):
    single = "single"
    cross_modal = "cross_modal"      # co-registered optical + SAR
    bi_temporal = "bi_temporal"      # two dates, same area
    compound = "compound"            # 3+ images for a multi-step chain (e.g. t0/t1 optical + SAR)


class TaskType(str, Enum):
    vqa = "vqa"
    captioning = "captioning"
    grounding = "grounding"
    object_counting = "object_counting"
    change_vqa = "change_vqa"
    change_description = "change_description"
    optical_sar_fusion = "optical_sar_fusion"


class ImageRef(BaseModel):
    file_id: str
    modality: str  # "optical" | "sar"
    acquisition_date: Optional[str] = None


class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural-language question from the user")
    images: list[ImageRef] = Field(default_factory=list)
    session_id: Optional[str] = Field(None, description="Carries multi-turn context memory across requests")
    aoi_bbox: Optional[list[float]] = Field(
        None, description="Optional [x1,y1,x2,y2] pixel-space AOI drawn on the map. "
                           "When present, tools scope analysis to this region instead of the full image."
    )


class ObjectCount(BaseModel):
    label: str
    count: int


class ToolResult(BaseModel):
    task: TaskType
    tool_name: str
    output_text: Optional[str] = None
    bounding_boxes: Optional[list[list[float]]] = None  # [x1,y1,x2,y2]
    object_counts: Optional[list[ObjectCount]] = None
    mask_path: Optional[str] = None
    confidence: float
    # False when `confidence` is a placeholder/integration signal rather than
    # a genuine calibrated probability (e.g. an HTTP-based model server that
    # doesn't expose token log-probs). The frontend must not render a bare
    # percentage in that case — see ConfidenceBadge.jsx.
    confidence_calibrated: bool = True
    fusion_agreement_score: Optional[float] = None
    geojson_overlay: Optional[dict[str, Any]] = None
    visualization_type: Optional[str] = None
    physical_metrics: Optional[dict[str, Any]] = None
    raw: Optional[dict[str, Any]] = None


class ExecutionStep(BaseModel):
    step: str
    detail: str


class ChangeStatsModel(BaseModel):
    changed_area_ha: float
    pct_changed: float
    primary_change_zone: str


class ChainStepModel(BaseModel):
    step_id: int
    action: str
    description: str
    depends_on: list[int] = []


class QueryResponse(BaseModel):
    task: TaskType
    input_config: InputConfig
    answer: str
    confidence: float
    confidence_calibrated: bool = True
    fusion_agreement_score: Optional[float] = None
    geojson_overlay: Optional[dict[str, Any]] = None
    visualization_type: Optional[str] = "overlay"
    physical_metrics: Optional[dict[str, Any]] = None
    sensor_info: Optional[dict[str, Any]] = None
    research_report: Optional[dict[str, Any]] = None
    low_confidence: bool
    evidence_image_url: Optional[str] = None
    bounding_boxes: Optional[list[list[float]]] = None
    object_counts: Optional[list[ObjectCount]] = None
    execution_trace: list[ExecutionStep]
    tools_used: list[str]
    report_id: str
    session_id: str
    turn_count: int
    chain: Optional[list[ChainStepModel]] = None
    change_stats: Optional[ChangeStatsModel] = None
    detector_status: Optional[str] = None


