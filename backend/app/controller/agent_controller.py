"""
The Agentic Remote-Sensing Controller.

Visibly performs:
  1. Sensor & Imagery Intelligence Inspection (modality, resolution, CRS)
  2. Automated multi-image workflow classification (single, optical_sar, bi_temporal)
  3. Query intent classification & tool selection
  4. Defensive Tool Execution with universal AOI scoping (prevents 500 crashes / empty JSON)
  5. Truthful Research-grade analysis report generation (ISRO / researcher multi-format output)
  6. Auditable execution trace persistence
"""

from app.config import LOW_CONFIDENCE_THRESHOLD
from app.controller import context_memory
from app.controller.chain_executor import execute_chain
from app.controller.classifier import classify
from app.controller.query_decomposer import decompose, is_compound
from app.schemas import ChainStepModel, ChangeStatsModel, ExecutionStep, ImageRef, InputConfig, QueryResponse, TaskType
from app.services.audit_log import log_execution
from app.services.change_stats import compute_stats_from_result, compute_stats_stub
from app.services.gemini_service import generate_visual_explanation
from app.services.image_io import saved_path
from app.services.report_generator import generate_research_report
from app.services.sensor_intelligence import inspect_image_sensor, classify_multi_image_workflow
from app.tools.change_detection import ChangeDetectionTool
from app.tools.grounding import GroundingTool
from app.tools.object_counting import ObjectCountingTool
from app.tools.sar_fusion import SARFusionTool
from app.tools.vqa_caption import VQACaptionTool
from app.tools.dynamic_analysis_tool import DynamicAnalysisTool

_REGISTRY = {
    TaskType.vqa: VQACaptionTool(),
    TaskType.captioning: VQACaptionTool(),
    TaskType.grounding: GroundingTool(),
    TaskType.object_counting: ObjectCountingTool(),
    TaskType.change_vqa: ChangeDetectionTool(),
    TaskType.change_description: ChangeDetectionTool(),
    TaskType.optical_sar_fusion: SARFusionTool(),
    TaskType.dynamic_analysis: DynamicAnalysisTool(),
    TaskType.BUILDING_COUNT: ObjectCountingTool(),
    TaskType.BUILDING_DISTRIBUTION: ObjectCountingTool(),
    TaskType.WATER_DETECTION: DynamicAnalysisTool(),
    TaskType.VEGETATION_ANALYSIS: DynamicAnalysisTool(),
    TaskType.BUILT_UP_ANALYSIS: DynamicAnalysisTool(),
    TaskType.LAND_COVER: DynamicAnalysisTool(),
    TaskType.SAR_ANALYSIS: SARFusionTool(),
    TaskType.OPTICAL_SAR_FUSION: SARFusionTool(),
    TaskType.CHANGE_DETECTION: ChangeDetectionTool(),
    TaskType.GENERAL_SCENE_ANALYSIS: DynamicAnalysisTool(),
}

_COMPAT_MAPPING = {
    TaskType.BUILDING_COUNT: TaskType.object_counting,
    TaskType.BUILDING_DISTRIBUTION: TaskType.object_counting,
    TaskType.WATER_DETECTION: TaskType.dynamic_analysis,
    TaskType.VEGETATION_ANALYSIS: TaskType.dynamic_analysis,
    TaskType.BUILT_UP_ANALYSIS: TaskType.dynamic_analysis,
    TaskType.LAND_COVER: TaskType.dynamic_analysis,
    TaskType.SAR_ANALYSIS: TaskType.optical_sar_fusion,
    TaskType.OPTICAL_SAR_FUSION: TaskType.optical_sar_fusion,
    TaskType.CHANGE_DETECTION: TaskType.change_vqa,
    TaskType.GENERAL_SCENE_ANALYSIS: TaskType.dynamic_analysis,
}



def _handle_compound_query(query: str, images: list[ImageRef], chain_steps, chain,
                             trace: list[ExecutionStep], session_id: str, sensor_info: dict) -> QueryResponse:
    input_config = classify_multi_image_workflow(images)
    trace.append(ExecutionStep(
        step="input_validation",
        detail=f"{len(images)} image(s) supplied for chain execution "
               f"({'/'.join(sorted({i.modality for i in images}))})",
    ))

    try:
        answer, chain_trace_entries, results = execute_chain(chain_steps, query, images)
        for entry in chain_trace_entries:
            trace.append(ExecutionStep(**entry))
    except Exception as exc:
        trace.append(ExecutionStep(step="chain_error", detail=f"Chain execution error: {exc}"))
        answer = f"[Chain Execution Error]: {exc}"
        results = {}

    tools_invoked = []
    if any(s.action == "ground" for s in chain_steps):
        tools_invoked.append(GroundingTool().name)
    if any(s.action == "change_detect" for s in chain_steps):
        tools_invoked.append(ChangeDetectionTool().name)
    if any(s.action == "sar_evidence" for s in chain_steps):
        tools_invoked.append(SARFusionTool().name)

    confidences = [r.get("confidence") for r in results.values() if isinstance(r.get("confidence"), (int, float))]
    confidence = sum(confidences) / len(confidences) if confidences else 0.65
    low_confidence = confidence < LOW_CONFIDENCE_THRESHOLD

    change_stats = None
    change_result = next((r for r in results.values() if r.get("raw", {}).get("pct_changed") is not None), None)
    if change_result:
        stats = compute_stats_from_result(change_result["raw"])
        change_stats = ChangeStatsModel(**stats.__dict__)

    bounding_boxes = None
    ground_result = next((r for r in results.values() if r.get("bbox")), None)
    if ground_result:
        bounding_boxes = [ground_result["bbox"]]

    trace.append(ExecutionStep(
        step="output_combination",
        detail=f"combined {len(results)} chain step result(s), mean confidence={confidence:.2f}, "
               f"low_confidence={low_confidence}",
    ))

    entity = {"summary": f"chain result from '{query}'", "bounding_boxes": bounding_boxes} if bounding_boxes else None
    context_memory.record_turn(session_id, query, "compound_chain", entity)

    report_id = log_execution(
        query=query,
        input_config=input_config.value,
        task="compound_chain",
        tools_used=tools_invoked,
        confidence=confidence,
        execution_trace=[s.model_dump() for s in trace],
        answer=answer,
    )

    return QueryResponse(
        task=TaskType.change_vqa if change_result else TaskType.vqa,
        input_config=input_config,
        answer=answer,
        confidence=confidence,
        low_confidence=low_confidence,
        evidence_image_url=None,
        bounding_boxes=bounding_boxes,
        execution_trace=trace,
        tools_used=tools_invoked,
        report_id=report_id,
        session_id=session_id,
        turn_count=context_memory.turn_count(session_id),
        chain=chain,
        change_stats=change_stats,
        sensor_info=sensor_info,
    )


def handle_query(
    query: str,
    images: list[ImageRef],
    session_id: str | None = None,
    aoi_bbox: list[float] | None = None,
) -> QueryResponse:
    trace: list[ExecutionStep] = []
    session_id = session_id or context_memory.new_session()

    # 1. Automated Sensor Intelligence Inspection
    primary_img_path = str(saved_path(images[0].file_id)) if images else ""
    sensor_info = inspect_image_sensor(primary_img_path) if primary_img_path else {}

    trace.append(ExecutionStep(
        step="sensor_intelligence",
        detail=f"sensor inspection: modality={sensor_info.get('modality')}, "
               f"crs={sensor_info.get('crs')}, res={sensor_info.get('spatial_resolution_m')}m/px",
    ))

    if aoi_bbox:
        trace.append(ExecutionStep(
            step="aoi_selection",
            detail=f"analysis scoped to user-drawn AOI bbox={[round(v, 1) for v in aoi_bbox]}",
        ))

    # 2. Context memory — resolve referents
    last_entity = context_memory.get_last_entity(session_id)
    if context_memory.has_referent(query) and last_entity:
        trace.append(ExecutionStep(
            step="context_resolution",
            detail=f"resolved referent in query against prior turn: {last_entity.get('summary', last_entity)}",
        ))

    # 3. Compound query decomposition ("SatQuery Chain")
    chain = None
    if is_compound(query):
        chain_steps = decompose(query)
        chain = [ChainStepModel(**s.__dict__) for s in chain_steps]
        trace.append(ExecutionStep(
            step="compound_query_decomposition",
            detail=f"query decomposed into {len(chain_steps)} chain step(s): "
                   f"{' -> '.join(s.action for s in chain_steps)}",
        ))
        return _handle_compound_query(query, images, chain_steps, chain, trace, session_id, sensor_info)

    # 4. Automated Workflow Classification
    input_config = classify_multi_image_workflow(images)
    trace.append(ExecutionStep(
        step="input_validation",
        detail=f"{len(images)} image(s) -> input_config={input_config.value}",
    ))

    # 5. Classify task from query + input config
    task = classify(query, input_config)
    trace.append(ExecutionStep(
        step="task_classification",
        detail=f"query classified as task={task.value}",
    ))

    # 6. Select tool from registry
    tool = _REGISTRY[task]
    trace.append(ExecutionStep(
        step="tool_selection",
        detail=f"selected tool={tool.name} for task={task.value}",
    ))

    # 7. Defensive Tool Execution with Universal AOI Scoping
    try:
        result = tool.run(query, images, aoi_bbox=aoi_bbox)
        detector_status = result.raw.get("detector_status") if result.raw else None
        if detector_status == "not_loaded":
            trace_detail = "Building detector NOT LOADED — Checkpoint not available. Skipping count and vector footprints."
        elif detector_status == "loaded":
            det_count = len(result.bounding_boxes or [])
            det_type = result.raw.get("detector_type", "Building Detector") if result.raw else "Building Detector"
            trace_detail = f"{det_type} executed successfully (LOADED [OK]). Detections: {det_count}"
        else:
            trace_detail = f"tool={tool.name} returned confidence={result.confidence:.2f}"

        trace.append(ExecutionStep(
            step="tool_execution",
            detail=trace_detail,
        ))
    except Exception as exc:
        trace.append(ExecutionStep(
            step="tool_execution_error",
            detail=f"defensive error catch: tool={tool.name} raised exception: {exc}",
        ))
        # Produce valid fallback QueryResponse instead of crashing backend
        legacy_task = _COMPAT_MAPPING.get(task, task)
        research_report = generate_research_report(query=query, tool_result=None, sensor_info=sensor_info, aoi_bbox=aoi_bbox, session_id=session_id)
        report_id = log_execution(query=query, input_config=input_config.value, task=legacy_task.value, tools_used=[tool.name], confidence=0.0, execution_trace=[s.model_dump() for s in trace], answer=f"[Tool Error]: {exc}")
        return QueryResponse(
            task=legacy_task,
            input_config=input_config,
            answer=f"Tool execution encountered an error: {exc}. Scoped analysis area: {aoi_bbox or 'Full Scene'}.",
            confidence=0.0,
            confidence_calibrated=False,
            low_confidence=True,
            execution_trace=trace,
            tools_used=[tool.name],
            report_id=report_id,
            session_id=session_id,
            turn_count=context_memory.turn_count(session_id),
            sensor_info=sensor_info,
            research_report=research_report,
        )

    change_stats = None
    legacy_task = _COMPAT_MAPPING.get(task, task)
    if legacy_task in (TaskType.change_vqa, TaskType.change_description):
        stats = compute_stats_from_result(result.raw) if result.raw else compute_stats_stub()
        change_stats = ChangeStatsModel(**stats.__dict__)

    # 8. Generate ISRO / Research-Grade Analysis Report
    research_report = generate_research_report(
        query=query,
        tool_result=result,
        sensor_info=sensor_info,
        aoi_bbox=aoi_bbox,
        session_id=session_id,
    )

    low_confidence = (not result.confidence_calibrated) or (result.confidence < LOW_CONFIDENCE_THRESHOLD)

    # 9. Update context memory
    entity = None
    if result.bounding_boxes:
        entity = {"summary": f"region from '{query}'", "bounding_boxes": result.bounding_boxes}
    context_memory.record_turn(session_id, query, legacy_task.value, entity)

    # 10. Persist auditable execution trace
    report_id = log_execution(
        query=query,
        input_config=input_config.value,
        task=legacy_task.value,
        tools_used=[tool.name],
        confidence=result.confidence,
        execution_trace=[s.model_dump() for s in trace],
        answer=result.output_text or "",
    )

    # 11. AI Visual Intelligence Layer (Gemini)
    # Only invoke for visual interpretation if images are available
    final_answer = result.output_text or ""
    
    # Collect computational evidence to ground Gemini
    comp_evidence = {
        "physical_metrics": result.physical_metrics,
        "fusion_agreement_score": result.fusion_agreement_score,
        "detector_status": detector_status,
        "object_count": len(result.bounding_boxes or []),
        "change_stats": change_stats.model_dump() if change_stats else None,
    }
    
    image_paths = [str(saved_path(img.file_id)) for img in images if img.file_id]
    
    # Do not call Gemini for simple building counting unless strictly requested, to save quota.
    # We call it for most qualitative/analytical tasks.
    if legacy_task not in (TaskType.object_counting, TaskType.BUILDING_COUNT) or "describe" in query.lower() or "visual" in query.lower():
        visual_explanation = generate_visual_explanation(query, image_paths, comp_evidence)
        
        if visual_explanation:
            final_answer = (
                f"**AI ANALYSIS**\n"
                f"{result.output_text}\n\n"
                f"**VISUAL INTERPRETATION**\n"
                f"{visual_explanation}"
            )
            
    return QueryResponse(
        task=legacy_task,
        input_config=input_config,
        answer=final_answer,
        confidence=result.confidence,
        confidence_calibrated=result.confidence_calibrated,
        fusion_agreement_score=result.fusion_agreement_score,
        geojson_overlay=result.geojson_overlay,
        visualization_type=result.visualization_type or "overlay",
        physical_metrics=result.physical_metrics,
        sensor_info=sensor_info,
        research_report=research_report,
        low_confidence=low_confidence,
        evidence_image_url=result.evidence_image_url,
        bounding_boxes=result.bounding_boxes,
        object_counts=result.object_counts,
        execution_trace=trace,
        tools_used=[tool.name],
        report_id=report_id,
        session_id=session_id,
        turn_count=context_memory.turn_count(session_id),
        chain=chain,
        change_stats=change_stats,
        detector_status=detector_status,
    )
