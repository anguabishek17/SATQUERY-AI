"""
DynamicAnalysisTool — intent-first evidence-aware reasoning engine.

Flow:
  query + images
    → determine_evidence_required()   (dynamic_planner)
    → _gather_evidence()              (optical processing + building detector)
    → _classify_intent()              (semantic goal of the question)
    → _synthesize_answer()            (evidence → reasoning → natural-language answer)
    → ToolResult

Key design rules:
  - Answer the user's actual question, not the nearest metric dump.
  - Lead with the answer; put limitations last.
  - Never claim a spectral analysis that did not occur.
  - Never invent roads, land-cover classes, or tool outputs.
  - Qualitative visual inferences must be phrased as observations, not facts.
  - Confidence is derived from evidence strength, not hardcoded.
"""

import re
from typing import Any, Dict, Optional, Tuple

import numpy as np

from app.schemas import ImageRef, TaskType, ToolResult
from app.tools.base import BaseTool
from app.controller.dynamic_planner import determine_evidence_required
from app.services.image_io import saved_path


# ---------------------------------------------------------------------------
# Intent categories (semantic goal of the question, not question phrasing)
# ---------------------------------------------------------------------------

INTENT_DIRECT_COUNT      = "direct_count"        # "how many buildings?"
INTENT_SPATIAL_DIST      = "spatial_distribution" # "where are buildings concentrated?"
INTENT_SPATIAL_COMPARE   = "spatial_comparison"   # "compare north vs south"
INTENT_DOMINANCE         = "dominance_comparison" # "built-up vs vegetation — which is larger?"
INTENT_SCENE_INFERENCE   = "scene_inference"      # "what does this landscape suggest?"
INTENT_MULTI_OBSERVATION = "multi_observation"    # "give me three evidence-backed observations"
INTENT_CAPABILITY        = "capability"           # "can you calculate NDVI?"
INTENT_DESCRIPTION       = "description"          # "describe the scene / major features"


def _classify_intent(query: str, plan: Dict[str, Any]) -> str:
    """
    Classify the *semantic goal* of the question, independent of exact wording.
    Returns one of the INTENT_* constants above.
    """
    q = query.lower()

    # Capability questions come first — they don't need evidence collected
    if re.search(r'\b(can you|is it possible|do you have|available|calculate ndvi|calculate ndwi|calculate ndbi)\b', q):
        return INTENT_CAPABILITY

    # Multi-observation: explicitly asking for a list of evidence-backed facts
    if re.search(r'\b(three|3|multiple|list|give me\b.*\bobservation|observations.*evidence|evidence.*observations)\b', q):
        return INTENT_MULTI_OBSERVATION

    # Spatial comparison: compare two named regions (north/south/east/west)
    if re.search(r'\b(north|south|east|west|upper|lower|left|right|northern|southern|eastern|western)\b', q) \
            and re.search(r'\b(compare|vs|versus|difference|more|which|between)\b', q):
        return INTENT_SPATIAL_COMPARE

    # Spatial distribution: where are things concentrated, not how many
    if re.search(r'\b(where are|concentrated|distribution|spread|cluster|most developed|which part|which (area|region|portion|side))\b', q):
        return INTENT_SPATIAL_DIST

    # Dominance comparison: which coverage type is larger
    if re.search(r'\b(more dominant|dominant|which is more|vegetation or built|built.?up or veg|vs|versus)\b', q) \
            and re.search(r'\b(vegetation|greenery|built|cover|land)\b', q):
        return INTENT_DOMINANCE

    # Scene inference: what does the scene imply about human activity / land-use / modification
    if re.search(r'\b(suggest|impl[yi]|infer|human (activity|modification|presence)|land.?use|what (does|can|could)|what evidence|settlement|strongest evidence|stands? out|major features?|what is visible)\b', q):
        return INTENT_SCENE_INFERENCE

    # Direct count: measurement-style question
    if re.search(r'\b(how many|count|number of|total)\b', q):
        return INTENT_DIRECT_COUNT

    # Broad description
    return INTENT_DESCRIPTION


def _fmt_area(area_ha: float, area_km2: float) -> str:
    """Format area with correct unit relationship (1 km² = 100 ha)."""
    # Recompute km2 from ha to prevent floating-point inconsistency
    km2 = area_ha / 100.0
    return f"{area_ha:.2f} ha ({km2:.3f} km²)"


def _building_summary(b_count, b_density, area_ha) -> str:
    if b_count is None:
        return "building count unavailable"
    area_km2 = area_ha / 100.0
    density = round(b_count / area_km2, 1) if area_km2 > 0 else 0.0
    return f"{b_count} building{'s' if b_count != 1 else ''} detected ({density:.1f}/km² across {area_ha:.2f} ha)"


def _spectral_note(veg_pct, water_pct, builtup_pct, include_when_none=True) -> str:
    """Return a compact spectral summary, or an honest note if NIR unavailable."""
    parts = []
    if veg_pct is not None:
        parts.append(f"vegetation {veg_pct:.1f}%")
    if water_pct is not None:
        parts.append(f"water {water_pct:.1f}%")
    if builtup_pct is not None:
        parts.append(f"built-up index {builtup_pct:.1f}%")
    if parts:
        return "Spectral indices (NIR/SWIR): " + ", ".join(parts) + "."
    if include_when_none:
        return "Spectral indices (NDVI/NDWI/NDBI) are not available because this image lacks NIR/SWIR bands."
    return ""


def _synthesize_answer(
    intent: str,
    query: str,
    metrics: Dict[str, Any],
    plan: Dict[str, Any],
    spatial: Optional[Dict[str, Any]] = None,
) -> Tuple[str, float, bool, list]:
    """
    Convert intent + evidence metrics into a natural-language answer.

    Returns: (answer_text, confidence, confidence_calibrated, limitations)
    """
    q = query.lower()
    b_count   = metrics.get("building_count")
    area_ha   = metrics.get("area_ha") or 0.0
    area_km2  = area_ha / 100.0
    veg_pct   = metrics.get("vegetation_pct")
    water_pct = metrics.get("water_pct")
    builtup_pct = metrics.get("builtup_pct")
    
    has_spectral = (veg_pct is not None or water_pct is not None or builtup_pct is not None)
    has_buildings = (b_count is not None and b_count > 0)
    
    # Recompute density from actual counts + area to prevent stale values
    b_density = round(b_count / area_km2, 1) if (b_count is not None and area_km2 > 0) else 0.0
    limitations = []

    # -----------------------------------------------------------------------
    # Capability question
    # -----------------------------------------------------------------------
    if intent == INTENT_CAPABILITY:
        if re.search(r'\bndvi\b', q):
            if veg_pct is not None:
                return f"Yes — NDVI was calculated. Vegetation coverage measures {veg_pct:.1f}%.", 0.85, True, []
            return "NDVI cannot be calculated from this image because it contains only RGB bands. NIR data would be required.", 0.90, True, []
        elif re.search(r'\bndwi\b', q):
            if water_pct is not None:
                return f"Yes — NDWI was calculated. Water coverage measures {water_pct:.1f}%.", 0.85, True, []
            return "NDWI cannot be calculated from this image — it lacks the required NIR spectral band.", 0.90, True, []
        return "This capability requires multi-spectral (NIR/SWIR) imagery which is not present in the current input.", 0.80, True, []

    # -----------------------------------------------------------------------
    # Multi-observation (give me three observations)
    # -----------------------------------------------------------------------
    if intent == INTENT_MULTI_OBSERVATION:
        obs = []
        if b_count is not None:
            if b_count > 0:
                obs.append(f"1. {b_count} building{'s' if b_count != 1 else ''} were detected.")
                obs.append(f"2. Building density is {b_density:.1f} buildings/km² across {_fmt_area(area_ha, area_km2)}.")
            else:
                obs.append(f"1. No buildings were detected across {_fmt_area(area_ha, area_km2)}.")
        
        if spatial:
            r1 = spatial["r1_m"].get("building_count", 0) or 0
            r2 = spatial["r2_m"].get("building_count", 0) or 0
            more_display = spatial["r1_display"] if r1 >= r2 else spatial["r2_display"]
            less_display = spatial["r2_display"] if r1 >= r2 else spatial["r1_display"]
            idx = len(obs) + 1
            obs.append(f"{idx}. The {more_display} portion contains more detected buildings than the {less_display} portion ({max(r1,r2)} vs {min(r1,r2)}).")
        
        if has_spectral:
            idx = len(obs) + 1
            spec_parts = []
            if veg_pct is not None: spec_parts.append(f"vegetation {veg_pct:.1f}%")
            if water_pct is not None: spec_parts.append(f"water {water_pct:.1f}%")
            if builtup_pct is not None: spec_parts.append(f"built-up {builtup_pct:.1f}%")
            obs.append(f"{idx}. Spectral indices: " + ", ".join(spec_parts) + ".")
        else:
            idx = len(obs) + 1
            obs.append(f"{idx}. Spectral land-cover indices are unavailable for this RGB image.")
            limitations.append("NIR/SWIR data would be required to extract precise multi-spectral observations.")
            
        while len(obs) < 3:
            obs.append(f"{len(obs)+1}. The analyzed spatial footprint covers {_fmt_area(area_ha, area_km2)}.")
            
        return "\n".join(obs[:3]), 0.80, False, limitations

    # -----------------------------------------------------------------------
    # Abstract Component Builder for General Questions
    # -----------------------------------------------------------------------
    core_answer = ""
    conf = 0.65
    calibrated = False

    if intent == INTENT_DIRECT_COUNT:
        if b_count is not None:
            if b_count == 0:
                core_answer = f"No buildings were detected in the analyzed area ({_fmt_area(area_ha, area_km2)})."
            else:
                core_answer = f"{b_count} building{'s' if b_count != 1 else ''} were detected across the analyzed area, giving a density of {b_density:.1f} buildings/km²."
            conf = 0.95
        else:
            core_answer = "Building count could not be obtained from the detector."
            conf = 0.40

    elif intent in (INTENT_SPATIAL_COMPARE, INTENT_SPATIAL_DIST):
        if spatial:
            r1 = spatial["r1_m"].get("building_count", 0) or 0
            r2 = spatial["r2_m"].get("building_count", 0) or 0
            max_b, min_b = max(r1, r2), min(r1, r2)
            more_display = spatial["r1_display"] if r1 >= r2 else spatial["r2_display"]
            less_display = spatial["r2_display"] if r1 >= r2 else spatial["r1_display"]
            
            if r1 == 0 and r2 == 0:
                core_answer = "No buildings were detected in either portion of the scene, making spatial structural comparison inconclusive."
                conf = 0.70
            elif intent == INTENT_SPATIAL_COMPARE:
                core_answer = f"The {more_display} portion appears more developed because it contains more detected buildings than the {less_display} portion ({max_b} vs {min_b}), indicating a somewhat higher concentration of constructed structures in that region."
                conf = 0.85
            else:
                core_answer = f"Buildings are more concentrated in the {more_display} portion ({max_b} vs {min_b} in the {less_display} portion), indicating a higher spatial distribution of constructed structures there."
                conf = 0.85
        else:
            core_answer = f"Spatial comparison could not be completed. However, {b_count if b_count else 0} buildings were detected overall."
            
        if not has_spectral:
            limitations.append("A quantitative spectral land-cover comparison cannot be made as NIR/SWIR bands are unavailable in this RGB image.")

    elif intent == INTENT_DOMINANCE:
        if has_spectral and veg_pct is not None and builtup_pct is not None:
            dom = "Vegetation is" if veg_pct > builtup_pct else "Built-up surfaces are"
            core_answer = f"{dom} more dominant in this scene (vegetation {veg_pct:.1f}% vs built-up {builtup_pct:.1f}%)."
            if has_buildings:
                core_answer += f" Additionally, {b_count} building structures were detected."
            conf = 0.85
            calibrated = True
        else:
            if has_buildings:
                desc = "sparse structures" if b_density < 10 else "moderate to dense structures"
                core_answer = f"Based on the available visual and structural evidence, the scene appears more structurally developed/open depending on the observed evidence. {b_count} buildings are detected across {area_ha:.2f} ha, indicating relatively {desc}."
            else:
                core_answer = "Based on available evidence, the scene appears predominantly open or vegetated as no buildings were detected."
            limitations.append("A quantitative vegetation-vs-built-up comparison using NDVI/NDBI cannot be confirmed because NIR/SWIR bands are unavailable.")
            conf = 0.70

    elif intent in (INTENT_SCENE_INFERENCE, INTENT_DESCRIPTION):
        if has_buildings:
            density_str = "sparse" if b_density < 10 else "moderate" if b_density < 50 else "dense"
            core_answer = f"The strongest evidence of development is the presence of {b_count} detected building structure{'s' if b_count != 1 else ''}. Buildings are direct indicators of constructed infrastructure, and their measured density of {b_density:.1f} buildings/km² provides quantitative support for human modification of the area."
            
            # Contextual structural distribution if spatial data was gathered
            if spatial:
                r1 = spatial["r1_m"].get("building_count", 0) or 0
                r2 = spatial["r2_m"].get("building_count", 0) or 0
                if r1 != r2:
                    more_display = spatial["r1_display"] if r1 > r2 else spatial["r2_display"]
                    core_answer += f" The scene appears to have a mixed pattern, with detected structures distributed across the area and somewhat greater structural concentration toward the {more_display}."
                    
            if has_spectral and veg_pct is not None:
                core_answer += f" Vegetation covers {veg_pct:.1f}% of the scene (NDVI-derived)."
            conf = 0.80
        else:
            core_answer = f"The scene appears to have a relatively open pattern. No building structures were detected across the {_fmt_area(area_ha, area_km2)}, suggesting an absence of permanent human settlement or constructed infrastructure."
            conf = 0.65
            
        if not has_spectral:
            limitations.append("Formal land-cover classification is not available from the current RGB-only imagery.")

    return core_answer, conf, calibrated, limitations


class DynamicAnalysisTool(BaseTool):
    name = "dynamic-analysis-engine"

    def _gather_evidence(
        self,
        plan: Dict[str, Any],
        image_path: str,
        images: list[ImageRef],
        aoi_bbox: Optional[list[float]],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Run optical processing and/or building detector per the plan."""
        evidence: Dict[str, Any] = {}
        physical_metrics: Dict[str, Any] = {}

        # 1. Optical processing for water, vegetation, built_up
        if "optical_processing" in plan["tools_required"]:
            try:
                import rasterio
                from app.services.optical_processing import create_optical_masks, optical_statistics, normalize_band

                with rasterio.open(image_path) as src:
                    # Require at least 4 bands for NIR-based spectral indices (NDWI, NDVI)
                    if src.count >= 4:
                        red   = src.read(3)
                        green = src.read(2)
                        nir   = src.read(4)
                        swir  = src.read(5) if src.count >= 5 else None

                        if aoi_bbox:
                            ax1, ay1, ax2, ay2 = [max(0, int(v)) for v in aoi_bbox]
                            ax2 = min(ax2, red.shape[1])
                            ay2 = min(ay2, red.shape[0])
                            red   = red[ay1:ay2, ax1:ax2]
                            green = green[ay1:ay2, ax1:ax2]
                            nir   = nir[ay1:ay2, ax1:ax2]
                            if swir is not None:
                                swir = swir[ay1:ay2, ax1:ax2]

                        masks = create_optical_masks(
                            normalize_band(red), normalize_band(green), normalize_band(nir),
                            normalize_band(swir) if swir is not None else None,
                        )
                        stats = optical_statistics(masks)

                        if "water" in plan["evidence_required"]:
                            evidence["water_evidence"] = {"water_pct": stats.get("water_pct", 0)}
                            physical_metrics["water_pct"] = stats.get("water_pct", 0)
                        if "vegetation" in plan["evidence_required"]:
                            evidence["vegetation_evidence"] = {"vegetation_pct": stats.get("vegetation_pct", 0)}
                            physical_metrics["vegetation_pct"] = stats.get("vegetation_pct", 0)
                        if "built_up" in plan["evidence_required"]:
                            evidence["built_up_evidence"] = {"builtup_pct": stats.get("builtup_pct", 0)}
                            physical_metrics["builtup_pct"] = stats.get("builtup_pct", 0)

            except Exception as e:
                evidence["optical_error"] = str(e)

        # 2. Building detector (only if required)
        if "object_counting" in plan["tools_required"]:
            try:
                from app.tools.object_counting import ObjectCountingTool
                counter = ObjectCountingTool()
                count_res = counter.run("building", images, aoi_bbox=aoi_bbox)

                evidence["building_evidence"] = {
                    "count":   count_res.physical_metrics.get("building_count"),
                    "density": count_res.physical_metrics.get("density_per_km2"),
                    "area_ha": count_res.physical_metrics.get("area_ha"),
                }
                physical_metrics["building_count"]  = count_res.physical_metrics.get("building_count")
                physical_metrics["density_per_km2"] = count_res.physical_metrics.get("density_per_km2")
                physical_metrics["area_ha"]         = count_res.physical_metrics.get("area_ha")
                physical_metrics["area_sq_km"]      = count_res.physical_metrics.get("area_sq_km")

            except Exception as e:
                evidence["building_error"] = str(e)

        return evidence, physical_metrics

    def run(
        self,
        query: str,
        images: list[ImageRef],
        aoi_bbox: Optional[list[float]] = None,
    ) -> ToolResult:
        if not images:
            return ToolResult(
                task=TaskType.dynamic_analysis,
                tool_name=self.name,
                output_text="Error: No image provided.",
                confidence=0.0,
            )

        plan = determine_evidence_required(query)

        # Temporal prerequisite guard
        if plan.get("temporal_comparison_required") and len(images) == 1:
            return ToolResult(
                task=TaskType.dynamic_analysis,
                tool_name=self.name,
                output_text=(
                    "I can't determine change over time from the current input because only one image "
                    "is available. Please provide a second image of the same area from another date "
                    "for bi-temporal comparison."
                ),
                confidence=0.30,
                confidence_calibrated=False,
            )

        optical_image = next((img for img in images if img.modality == "optical"), images[0])
        try:
            image_path = str(saved_path(optical_image.file_id))
        except Exception as e:
            return ToolResult(
                task=TaskType.dynamic_analysis,
                tool_name=self.name,
                output_text=f"Error reading image: {e}",
                confidence=0.0,
            )

        q_lower = query.lower()
        intent  = _classify_intent(query, plan)
        evidence: Dict[str, Any] = {}
        physical_metrics: Dict[str, Any] = {}
        spatial_ctx = None

        # ------------------------------------------------------------------
        # Spatial branch: split image into two regions and gather evidence
        # Only fire for truly spatial intents — SCENE_INFERENCE and DESCRIPTION
        # must always use full-image evidence regardless of planner hints.
        # ------------------------------------------------------------------
        do_spatial = (
            intent in (INTENT_SPATIAL_COMPARE, INTENT_SPATIAL_DIST)
            or (
                "spatial_comparison" in plan["reasoning_tasks"]
                and intent not in (INTENT_SCENE_INFERENCE, INTENT_DESCRIPTION,
                                   INTENT_MULTI_OBSERVATION, INTENT_CAPABILITY)
            )
        )

        if do_spatial:
            try:
                import rasterio
                with rasterio.open(image_path) as src:
                    w, h = src.width, src.height

                bx1, by1, bx2, by2 = aoi_bbox if aoi_bbox else [0, 0, w, h]
                is_left_right = re.search(r'\b(east|west|left|right)\b', q_lower) is not None

                if is_left_right:
                    mid = bx1 + (bx2 - bx1) / 2
                    r1_bbox, r2_bbox  = [bx1, by1, mid, by2], [mid, by1, bx2, by2]
                    r1_name, r2_name  = "west", "east"
                    r1_disp, r2_disp  = "western/left", "eastern/right"
                else:
                    mid = by1 + (by2 - by1) / 2
                    r1_bbox, r2_bbox  = [bx1, by1, bx2, mid], [bx1, mid, bx2, by2]
                    r1_name, r2_name  = "north", "south"
                    r1_disp, r2_disp  = "northern/upper", "southern/lower"

                r1_ev, r1_m = self._gather_evidence(plan, image_path, images, r1_bbox)
                r2_ev, r2_m = self._gather_evidence(plan, image_path, images, r2_bbox)

                evidence = {r1_name: r1_m, r2_name: r2_m}
                physical_metrics = {r1_name: r1_m, r2_name: r2_m}

                spatial_ctx = {
                    "r1_name": r1_name, "r2_name": r2_name,
                    "r1_display": r1_disp, "r2_display": r2_disp,
                    "r1_m": r1_m, "r2_m": r2_m,
                }

                # Use the region with more buildings as the primary metrics summary
                r1_b = r1_m.get("building_count", 0) or 0
                r2_b = r2_m.get("building_count", 0) or 0
                primary_m = r1_m if r1_b >= r2_b else r2_m
                # Merge region metrics for confidence calculation
                merged = {}
                for k in ["building_count", "density_per_km2", "area_ha", "vegetation_pct", "water_pct", "builtup_pct"]:
                    v1 = r1_m.get(k)
                    v2 = r2_m.get(k)
                    if v1 is not None and v2 is not None:
                        merged[k] = v1 + v2 if k == "building_count" else (v1 + v2) / 2
                    elif v1 is not None:
                        merged[k] = v1
                    elif v2 is not None:
                        merged[k] = v2
                synthesis_metrics = merged

            except Exception as e:
                evidence = {"spatial_error": str(e)}
                physical_metrics = {}
                do_spatial = False
                evidence, physical_metrics = self._gather_evidence(plan, image_path, images, aoi_bbox)
                synthesis_metrics = physical_metrics
        else:
            evidence, physical_metrics = self._gather_evidence(plan, image_path, images, aoi_bbox)
            synthesis_metrics = physical_metrics

        # ------------------------------------------------------------------
        # Standard reasoning tasks (assess_urban_rural, assess_flooding, etc.)
        # handled by synthesize when intent is DESCRIPTION / SCENE_INFERENCE
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Run synthesis
        # ------------------------------------------------------------------
        final_answer, confidence, confidence_calibrated, limitations = _synthesize_answer(
            intent=intent,
            query=query,
            metrics=synthesis_metrics,
            plan=plan,
            spatial=spatial_ctx,
        )

        # Append limitations as a separate paragraph only when they add value
        if limitations:
            final_answer = final_answer.rstrip() + "\n\nNote: " + " ".join(limitations)

        return ToolResult(
            task=TaskType.dynamic_analysis,
            tool_name=self.name,
            output_text=final_answer,
            confidence=confidence,
            confidence_calibrated=confidence_calibrated,
            physical_metrics=physical_metrics,
            raw={"plan": plan, "evidence": evidence, "intent": intent},
        )
