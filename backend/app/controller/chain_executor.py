"""
Executes a SatQuery Chain step-by-step against the REAL tools, threading each
step's actual output into the next step's input. This is what turns the
chain from "a plan we display" into "a plan we run" — the distinction the
problem statement's agentic-orchestration requirement cares about.

Each step function takes (step, context) and returns a result dict that gets
stored in context['results'][step.step_id] for later steps (and 'intersect'/
'summarize') to consume. Every number in the final summary traces back to a
real tool call, not a hardcoded string.
"""
import re

from app.controller.query_decomposer import ChainStep
from app.schemas import ImageRef
from app.tools.aoi_tools import build_aoi
from app.tools.change_detection import _load_pair, ChangeDetectionTool
from app.tools.grounding import GroundingTool
from app.tools.sar_fusion import _load_bands
from app.services.change_processing import compute_change_mask, pixels_to_hectares
from app.services.sar_processing import lee_filter


def _step_ground(step: ChainStep, query: str, images: list[ImageRef], context: dict) -> dict:
    optical = next((i for i in images if i.modality == "optical"), None)
    if not optical:
        return {"skipped": True, "reason": "no optical image available to ground against"}
    result = GroundingTool().run(query, [optical])
    return {
        "bbox": result.bounding_boxes[0] if result.bounding_boxes else None,
        "confidence": result.confidence,
        "note": result.output_text,
    }


def _step_buffer(step: ChainStep, query: str, images: list[ImageRef], context: dict) -> dict:
    src = context["results"].get(step.depends_on[0]) if step.depends_on else None
    bbox = src.get("bbox") if src else None
    if not bbox:
        return {"skipped": True, "reason": "no grounded region to buffer around"}
    m = re.search(r"(\d+)m", step.description)
    meters = float(m.group(1)) if m else 500.0
    aoi = build_aoi(bbox, meters)
    return {"aoi_bbox": aoi.aoi_bbox_pixel, "buffer_meters": meters}


def _step_change_detect(step: ChainStep, query: str, images: list[ImageRef], context: dict) -> dict:
    optical_images = [i for i in images if i.modality == "optical"]
    if len(optical_images) < 2:
        return {"skipped": True, "reason": "need two same-modality images at different dates for change detection"}
    result = ChangeDetectionTool().run(query, optical_images[:2])
    return {"raw": result.raw, "confidence": result.confidence, "answer": result.output_text}


def _bbox_iou_pixel(a: list[float], b: list[float]) -> float:
    """Real intersection-over-union between two [x1,y1,x2,y2]-style boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _step_intersect(step: ChainStep, query: str, images: list[ImageRef], context: dict) -> dict:
    buffer_result = next((context["results"].get(sid) for sid in step.depends_on
                           if context["results"].get(sid, {}).get("aoi_bbox")), None)
    change_result = next((context["results"].get(sid) for sid in step.depends_on
                           if context["results"].get(sid, {}).get("raw", {}).get("largest_region_bbox")), None)

    if not buffer_result or not change_result:
        return {"skipped": True, "reason": "need both a buffered AOI and a change region to intersect"}

    row_min, col_min, row_max, col_max = change_result["raw"]["largest_region_bbox"]
    change_bbox_xy = [col_min, row_min, col_max, row_max]
    iou = _bbox_iou_pixel(buffer_result["aoi_bbox"], change_bbox_xy)

    return {
        "iou": iou,
        "within_aoi": iou > 0.0,
        "change_bbox": change_bbox_xy,
        "aoi_bbox": buffer_result["aoi_bbox"],
    }


def _step_sar_evidence(step: ChainStep, query: str, images: list[ImageRef], context: dict) -> dict:
    optical = next((i for i in images if i.modality == "optical"), None)
    sar = next((i for i in images if i.modality == "sar"), None)
    if not optical or not sar:
        return {"skipped": True, "reason": "no SAR image provided — chain proceeds on optical evidence only"}

    _, _, _, vv_db = _load_bands(optical, sar)
    filtered = lee_filter(vv_db)

    intersect_result = next((context["results"].get(sid) for sid in step.depends_on
                              if context["results"].get(sid, {}).get("change_bbox")), None)
    if intersect_result:
        x1, y1, x2, y2 = [int(v) for v in intersect_result["change_bbox"]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(filtered.shape[1], x2), min(filtered.shape[0], y2)
        region = filtered[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else filtered
    else:
        region = filtered

    mean_backscatter = float(region.mean()) if region.size else float(filtered.mean())
    scene_mean = float(filtered.mean())
    strength = "strong" if mean_backscatter > scene_mean + 2 else "moderate" if mean_backscatter > scene_mean else "weak"

    return {"mean_backscatter_db": mean_backscatter, "scene_mean_db": scene_mean, "evidence_strength": strength}


def _step_summarize(step: ChainStep, query: str, images: list[ImageRef], context: dict) -> str:
    parts = []
    for sid, r in context["results"].items():
        if r.get("skipped"):
            continue
        if "answer" in r:
            parts.append(r["answer"])
        elif "iou" in r:
            verdict = "falls within" if r["within_aoi"] else "does not fall within"
            parts.append(f"The detected change region {verdict} the specified buffer zone (spatial overlap: {r['iou']:.0%}).")
        elif "evidence_strength" in r:
            parts.append(
                f"SAR backscatter over the candidate region averages {r['mean_backscatter_db']:.1f}dB "
                f"against a scene mean of {r['scene_mean_db']:.1f}dB — {r['evidence_strength']} SAR evidence."
            )
        elif "note" in r:
            parts.append(r["note"])

    return " ".join(parts) if parts else "Chain executed but no step produced a conclusive result — check the execution trace for skipped steps."


_STEP_RUNNERS = {
    "ground": _step_ground,
    "buffer": _step_buffer,
    "change_detect": _step_change_detect,
    "intersect": _step_intersect,
    "sar_evidence": _step_sar_evidence,
}


def execute_chain(steps: list[ChainStep], query: str, images: list[ImageRef]) -> tuple[str, list[dict], dict]:
    """
    Runs every step in order (steps are already topologically sorted by the
    decomposer), threading outputs forward via `context['results']`.
    Returns (final_answer, trace_entries, results_by_step_id).
    """
    context = {"results": {}}
    trace_entries = []

    for step in steps:
        if step.action == "summarize":
            continue
        runner = _STEP_RUNNERS.get(step.action)
        if not runner:
            context["results"][step.step_id] = {"skipped": True, "reason": f"no executor for action '{step.action}'"}
            continue
        result = runner(step, query, images, context)
        context["results"][step.step_id] = result
        status = "skipped" if result.get("skipped") else "ok"
        detail = result.get("reason") if result.get("skipped") else _summarize_step_result(step.action, result)
        trace_entries.append({"step": f"chain_step_{step.step_id}_{step.action}", "detail": f"[{status}] {detail}"})

    final_step = next((s for s in steps if s.action == "summarize"), None)
    answer = _step_summarize(final_step, query, images, context) if final_step else _step_summarize(None, query, images, context)

    return answer, trace_entries, context["results"]


def _summarize_step_result(action: str, result: dict) -> str:
    if action == "ground":
        return f"grounded region bbox={result.get('bbox')} confidence={result.get('confidence', 0):.2f}"
    if action == "buffer":
        return f"AOI bbox={result.get('aoi_bbox')} (+{result.get('buffer_meters')}m)"
    if action == "change_detect":
        raw = result.get("raw", {})
        return f"{raw.get('pct_changed', 0):.1f}% changed, {raw.get('num_regions', 0)} region(s)"
    if action == "intersect":
        return f"IoU={result.get('iou', 0):.2f}, within_aoi={result.get('within_aoi')}"
    if action == "sar_evidence":
        return f"{result.get('evidence_strength')} evidence, {result.get('mean_backscatter_db', 0):.1f}dB"
    return str(result)
