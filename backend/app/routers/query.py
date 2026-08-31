from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.controller.agent_controller import handle_query
from app.controller.validator import ValidationError
from app.schemas import QueryRequest, QueryResponse
from app.services.audit_log import get_execution
from app.services.report import build_pdf_report

router = APIRouter()


@router.post("", response_model=QueryResponse)
def run_query(payload: QueryRequest) -> QueryResponse:
    try:
        return handle_query(
            payload.query, payload.images,
            session_id=payload.session_id, aoi_bbox=payload.aoi_bbox,
        )
    except ValidationError as e:
        raise HTTPException(400, str(e))


@router.get("/{report_id}/trace")
def get_trace(report_id: str) -> dict:
    execution = get_execution(report_id)
    if not execution:
        raise HTTPException(404, "report not found")
    return execution


@router.get("/{report_id}/report.pdf")
def download_report(report_id: str):
    execution = get_execution(report_id)
    if not execution:
        raise HTTPException(404, "report not found")

    import json
    pdf_path = build_pdf_report(
        report_id=report_id,
        query=execution["query"],
        answer=execution["answer"],
        confidence=execution["confidence"],
        tools_used=json.loads(execution["tools_used"]),
        execution_trace=json.loads(execution["execution_trace"]),
    )
    return FileResponse(pdf_path, filename=f"satquery_report_{report_id[:8]}.pdf")
