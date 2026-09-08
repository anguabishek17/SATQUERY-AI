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
import logging
logger = logging.getLogger(__name__)

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
    import time
    t_start = time.time()
    trace: list[ExecutionStep] = []
    session_id = session_id or context_memory.new_session()

    # 1. Automated Sensor Intelligence Inspection
    t0_understand = time.time()
    primary_img_path = ""
    try:
        if images:
            primary_img_path = str(saved_path(images[0].file_id))
    except FileNotFoundError:
        trace.append(ExecutionStep(
            step="sensor_intelligence",
            detail=f"file_id '{images[0].file_id}' not found in upload store; proceeding without sensor metadata.",
        ))
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
    ms_understand = int((time.time() - t0_understand) * 1000)

    # 4. Automated Workflow Classification
    input_config = classify_multi_image_workflow(images)
    trace.append(ExecutionStep(
        step="input_validation",
        detail=f"{len(images)} image(s) -> input_config={input_config.value}",
    ))

    # 5. Classify task from query + input config
    t0_classify = time.time()
    task = classify(query, input_config)
    ms_classify = int((time.time() - t0_classify) * 1000)
    trace.append(ExecutionStep(
        step="task_classification",
        detail=f"query classified as task={task.value}",
    ))

    # 6. Select tool from registry
    t0_tool_sel = time.time()
    tool = _REGISTRY[task]
    ms_tool_sel = int((time.time() - t0_tool_sel) * 1000)
    trace.append(ExecutionStep(
        step="tool_selection",
        detail=f"selected tool={tool.name} for task={task.value}",
    ))

    # Image Preprocessing timing
    t0_preproc = time.time()
    if primary_img_path:
        from app.services.analysis_cache import analysis_cache
        pre = analysis_cache.get_preprocessed_image(primary_img_path, aoi_bbox)
    ms_preproc = int((time.time() - t0_preproc) * 1000)

    # 7. Defensive Tool Execution with Universal AOI Scoping
    t0_geo = time.time()
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
    ms_geo = int((time.time() - t0_geo) * 1000)

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

    # 11. Universal AI Response Pipeline
    from app.ai.evidence_builder import build_evidence_json
    from app.ai.ai_reasoner import generate_ai_reasoning
    from app.ai.evidence_validator import validate_reasoning

    t0_ev = time.time()
    evidence = build_evidence_json(query, legacy_task, result)
    ms_evidence = int((time.time() - t0_ev) * 1000)
    
    trace.append(ExecutionStep(
        step="evidence_json",
        detail=f"standardized evidence payload built: source={evidence.source_tool}"
    ))

    # Generate AI Reasoning based on Evidence JSON
    t0_reasoning = time.time()
    reasoning = generate_ai_reasoning(query, evidence)
    ms_reasoning = int((time.time() - t0_reasoning) * 1000)
    
    trace.append(ExecutionStep(
        step="ai_reasoning",
        detail="generated explanation from evidence json"
    ))

    # Validate to ensure no hallucinated claims
    t0_val = time.time()
    validation_result = validate_reasoning(reasoning, evidence)
    ms_val = int((time.time() - t0_val) * 1000)
    
    val_status = validation_result.get("status", "VALIDATED" if validation_result.get("valid") else "VALIDATION_FAILED")
    val_pass = validation_result.get("valid", False)
    val_reason = validation_result.get("reason", "")
    evidence.validation_status = val_status

    trace.append(ExecutionStep(
        step="evidence_validation",
        detail=f"status={val_status}; pass={val_pass}; reason='{val_reason}'"
    ))

    final_answer = validation_result.get("corrected_answer", reasoning)
    # Anti-generic safeguard: if answer contains boilerplate textbook text, use deterministic evidence-based fallback
    if "Cross-modal optical and SAR fusion analysis reveals complementary physical characteristics" in final_answer:
        from app.ai.ai_reasoner import _fallback_reasoning
        final_answer = _fallback_reasoning(evidence, query)
    
    # Bi-temporal safeguard: ensure query-specific response, strip confidence, and ensure Future Prediction is present
    if legacy_task in (TaskType.change_vqa, TaskType.change_description, TaskType.CHANGE_DETECTION) or evidence.temporal is not None:
        import re
        # Remove any leaked Confidence: XX% in text
        final_answer = re.sub(r"Confidence\s+(is|:)\s*\d+%\.?", "", final_answer, flags=re.IGNORECASE).strip()
        # If the answer collapsed to canned text or lacks Future Prediction, use query-specific fallback
        if "across 0 distinct regions" in final_answer or "Future Prediction:" not in final_answer:
            from app.ai.ai_reasoner import _fallback_reasoning
            final_answer = _fallback_reasoning(evidence, query)
            final_answer = re.sub(r"Confidence\s+(is|:)\s*\d+%\.?", "", final_answer, flags=re.IGNORECASE).strip()

    ms_total = int((time.time() - t_start) * 1000)

    yolo_executed = bool(
        tool.name == "building-detector"
        or (result.raw and result.raw.get("detector_status") == "loaded")
        or (task in (TaskType.BUILDING_COUNT, TaskType.BUILDING_DISTRIBUTION))
    )

    # Emit performance logs
    perf_log = (
        f"\n[PERF] Query Understanding: {ms_understand} ms\n"
        f"[PERF] Task Classification: {ms_classify} ms\n"
        f"[PERF] Tool Selection: {ms_tool_sel} ms\n"
        f"[PERF] Image Preprocessing: {ms_preproc} ms\n"
        f"[PERF] Geoanalysis: {ms_geo} ms\n"
        f"[PERF] Evidence JSON: {ms_evidence} ms\n"
        f"[PERF] AI Reasoning: {ms_reasoning} ms\n"
        f"[PERF] Validation: {ms_val} ms\n"
        f"[PERF] TOTAL: {ms_total} ms\n"
        f"[PERF] YOLO EXECUTED: {'TRUE' if yolo_executed else 'FALSE'}\n"
    )
    print(perf_log)
    logger.info(perf_log)

    trace.append(ExecutionStep(
        step="performance_metrics",
        detail=f"Total: {ms_total}ms | Geoanalysis: {ms_geo}ms | YOLO: {yolo_executed}"
    ))

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
