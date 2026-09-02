import traceback
from typing import Any, Optional, Tuple, Dict
import numpy as np

from app.schemas import ImageRef, TaskType, ToolResult
from app.tools.base import BaseTool
from app.controller.dynamic_planner import determine_evidence_required
from app.services.image_io import saved_path

class DynamicAnalysisTool(BaseTool):
    name = "dynamic-analysis-engine"
    
    def _gather_evidence(self, plan: Dict[str, Any], image_path: str, images: list[ImageRef], aoi_bbox: Optional[list[float]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        evidence = {}
        physical_metrics = {}
        
        # 1. Optical processing for water, vegetation, and built_up
        if "optical_processing" in plan["tools_required"]:
            try:
                import rasterio
                from app.services.optical_processing import create_optical_masks, optical_statistics, normalize_band
                
                with rasterio.open(image_path) as src:
                    # Require at least 4 bands for NIR-based spectral indices (NDWI, NDVI)
                    if src.count >= 4:
                        red = src.read(3)
                        green = src.read(2)
                        nir = src.read(4)
                        swir = src.read(5) if src.count >= 5 else None

                        if aoi_bbox:
                            ax1, ay1, ax2, ay2 = [max(0, int(v)) for v in aoi_bbox]
                            # Bounds check
                            ax2 = min(ax2, red.shape[1])
                            ay2 = min(ay2, red.shape[0])
                            red = red[ay1:ay2, ax1:ax2]
                            green = green[ay1:ay2, ax1:ax2]
                            nir = nir[ay1:ay2, ax1:ax2]
                            if swir is not None:
                                swir = swir[ay1:ay2, ax1:ax2]
                        
                        masks = create_optical_masks(normalize_band(red), normalize_band(green), normalize_band(nir), normalize_band(swir) if swir is not None else None)
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

        # 2. Object counting for buildings (only if required)
        if "object_counting" in plan["tools_required"]:
            try:
                from app.tools.object_counting import ObjectCountingTool
                counter = ObjectCountingTool()
                count_res = counter.run("building", images, aoi_bbox=aoi_bbox)
                
                evidence["building_evidence"] = {
                    "count": count_res.physical_metrics.get("building_count"),
                    "density": count_res.physical_metrics.get("density_per_km2"),
                    "area_ha": count_res.physical_metrics.get("area_ha")
                }
                physical_metrics["building_count"] = count_res.physical_metrics.get("building_count")
                physical_metrics["density_per_km2"] = count_res.physical_metrics.get("density_per_km2")
                physical_metrics["area_ha"] = count_res.physical_metrics.get("area_ha")
                
            except Exception as e:
                evidence["building_error"] = str(e)
                
        return evidence, physical_metrics

    def run(self, query: str, images: list[ImageRef], aoi_bbox: Optional[list[float]] = None) -> ToolResult:
        if not images:
            return ToolResult(
                task=TaskType.dynamic_analysis,
                tool_name=self.name,
                output_text="Error: No image provided.",
                confidence=0.0
            )

        plan = determine_evidence_required(query)
        
        optical_image = next((img for img in images if img.modality == "optical"), images[0])
        try:
            image_path = str(saved_path(optical_image.file_id))
        except Exception as e:
            return ToolResult(
                task=TaskType.dynamic_analysis,
                tool_name=self.name,
                output_text=f"Error reading image: {e}",
                confidence=0.0
            )

        evidence = {}
        physical_metrics = {}
        answer_parts = []
        limitations = []

        if "spatial_comparison" in plan["reasoning_tasks"]:
            try:
                import rasterio
                with rasterio.open(image_path) as src:
                    w, h = src.width, src.height
                
                # Default to full image if no aoi_bbox
                bx1, by1, bx2, by2 = aoi_bbox if aoi_bbox else [0, 0, w, h]
                
                is_left_right = "east" in query.lower() or "west" in query.lower() or "left" in query.lower() or "right" in query.lower()
                
                if is_left_right:
                    mid_x = bx1 + (bx2 - bx1) / 2
                    region1_bbox = [bx1, by1, mid_x, by2]
                    region2_bbox = [mid_x, by1, bx2, by2]
                    r1_name, r2_name = "west", "east"
                    r1_display, r2_display = "western/left", "eastern/right"
                else:
                    mid_y = by1 + (by2 - by1) / 2
                    region1_bbox = [bx1, by1, bx2, mid_y]
                    region2_bbox = [bx1, mid_y, bx2, by2]
                    r1_name, r2_name = "north", "south"
                    r1_display, r2_display = "northern/upper", "southern/lower"

                r1_evidence, r1_metrics = self._gather_evidence(plan, image_path, images, region1_bbox)
                r2_evidence, r2_metrics = self._gather_evidence(plan, image_path, images, region2_bbox)
                
                def extract_summary(mets):
                    return {
                        "buildings": mets.get("building_count"),
                        "vegetation": mets.get("vegetation_pct"),
                        "water": mets.get("water_pct"),
                        "built_up": mets.get("builtup_pct")
                    }

                evidence = {
                    r1_name: extract_summary(r1_metrics),
                    r2_name: extract_summary(r2_metrics)
                }
                physical_metrics = {
                    r1_name: r1_metrics,
                    r2_name: r2_metrics
                }
                
                # Comparison logic
                comp_parts = []
                r1_b = r1_metrics.get("building_count", 0) or 0
                r2_b = r2_metrics.get("building_count", 0) or 0
                if r1_b > 0 or r2_b > 0:
                    more_b = r1_display if r1_b > r2_b else r2_display
                    comp_parts.append(f"the {more_b} portion has a higher concentration of buildings ({r1_b} vs {r2_b})")
                    
                r1_v = r1_metrics.get("vegetation_pct", 0) or 0
                r2_v = r2_metrics.get("vegetation_pct", 0) or 0
                if r1_v > 0 or r2_v > 0:
                    more_v = r1_display if r1_v > r2_v else r2_display
                    comp_parts.append(f"the {more_v} portion shows greater vegetation coverage ({max(r1_v, r2_v):.1f}% vs {min(r1_v, r2_v):.1f}%)")
                    
                r1_w = r1_metrics.get("water_pct", 0) or 0
                r2_w = r2_metrics.get("water_pct", 0) or 0
                if r1_w > 0 or r2_w > 0:
                    more_w = r1_display if r1_w > r2_w else r2_display
                    comp_parts.append(f"the {more_w} portion contains more water ({max(r1_w, r2_w):.1f}% vs {min(r1_w, r2_w):.1f}%)")
                    
                if comp_parts:
                    answer_parts.append("Spatial comparison: " + ", while ".join(comp_parts) + ".")
                else:
                    answer_parts.append("Spatial comparison: No significant differences found in the requested evidence across the regions.")
                    
            except Exception as e:
                limitations.append(f"Spatial subdivision failed: {e}")
                answer_parts.append("Spatial comparison could not be completed.")
        else:
            # Normal single-region extraction
            evidence, physical_metrics = self._gather_evidence(plan, image_path, images, aoi_bbox)
            
            water_pct = physical_metrics.get("water_pct")
            veg_pct = physical_metrics.get("vegetation_pct")
            b_count = physical_metrics.get("building_count")
            b_density = physical_metrics.get("density_per_km2")
            builtup_pct = physical_metrics.get("builtup_pct")

            if "assess_urban_rural" in plan["reasoning_tasks"]:
                answer = "The scene appears predominantly "
                is_urban = False
                reasons = []
                
                if builtup_pct is not None and builtup_pct > 20:
                    is_urban = True
                    reasons.append(f"built-up surfaces covering {builtup_pct:.1f}% of the area")
                if b_density is not None and b_density > 100:
                    is_urban = True
                    reasons.append(f"a high density of detected structures ({b_count} buildings, {b_density:.1f} per km²)")
                if veg_pct is not None and veg_pct > 60:
                    is_urban = False
                    reasons.append(f"extensive vegetation coverage ({veg_pct:.1f}%)")
                
                if is_urban:
                    answer += "urban/built-up. This conclusion is based on " + " and ".join(reasons) + "."
                else:
                    if not reasons:
                        reasons.append(f"a lack of significant built structures (detected {b_count or 0})")
                    answer += "rural or natural. This conclusion is based on " + " and ".join(reasons) + "."
                    
                answer += " However, this is an image-based inference combining physical metrics, rather than an official land-use classification."
                answer_parts.append(answer)

            elif "assess_built_environment" in plan["reasoning_tasks"]:
                if b_count is not None and b_density is not None:
                    if b_density > 1000:
                        answer_parts.append(f"There is a dense urban presence, with {b_count} detected structures ({b_density:.1f} buildings/km²).")
                    elif b_density > 100:
                        answer_parts.append(f"There is a moderate built-up presence, with {b_count} structures detected ({b_density:.1f} buildings/km²).")
                    else:
                        answer_parts.append(f"Only {b_count} building structures were detected in this area.")

            if "assess_water_presence" in plan["reasoning_tasks"]:
                if water_pct is not None:
                    if water_pct > 15:
                        answer_parts.append(f"There is significant water present, covering approximately {water_pct:.1f}% of the analyzed area based on spectral indices.")
                    elif water_pct > 2:
                        answer_parts.append(f"There are indications of minor water bodies or moisture, covering roughly {water_pct:.1f}% of the area.")
                    else:
                        answer_parts.append("There is no significant water visible in this region.")
                else:
                    answer_parts.append("Definitive NDWI-based water detection is unavailable because the current image does not contain the required spectral bands. A second input containing the necessary spectral bands would be required for spectral water assessment.")
                    limitations.append("Water assessment failed due to missing spectral bands.")

            if "assess_vegetation" in plan["reasoning_tasks"]:
                if veg_pct is not None:
                    if veg_pct > 50:
                        answer_parts.append(f"The area is predominantly vegetated, with {veg_pct:.1f}% covered by plant life (based on NDVI).")
                    elif veg_pct > 15:
                        answer_parts.append(f"There are moderate vegetation patterns, covering {veg_pct:.1f}% of the area.")
                    else:
                        answer_parts.append(f"Vegetation is sparse in this area (coverage: {veg_pct:.1f}%).")
                        
            if "assess_flooding" in plan["reasoning_tasks"]:
                if not plan.get("temporal_comparison_required") or "change" not in plan["evidence_required"]:
                    answer_parts.append(f"While water coverage is {water_pct:.1f}%" if water_pct is not None else "While water could be assessed")
                    answer_parts[-1] += ", flooding cannot be reliably confirmed from a single image without contextual evidence such as temporal change or pre-event baseline data."
                    limitations.append("Flooding assessment requires bi-temporal data, which was not available in this single-image query.")
                else:
                    answer_parts.append("I am analyzing the temporal data for flood expansion...")

            if "broad_scene_summary" in plan["reasoning_tasks"]:
                summary = "Structured Scene Summary:\n"
                if water_pct is not None:
                    summary += f"- Water: {water_pct:.1f}% coverage\n"
                if veg_pct is not None:
                    summary += f"- Vegetation: {veg_pct:.1f}% coverage\n"
                if builtup_pct is not None:
                    summary += f"- Built-up (NDBI): {builtup_pct:.1f}% coverage\n"
                if b_count is not None:
                    summary += f"- Structures: {b_count} buildings detected"
                    
                if summary.strip() == "Structured Scene Summary:":
                    summary += "No definitive spectral or structural metrics could be extracted."
                answer_parts.append(summary)

        # Temporal override for single image or Fallback
        if plan.get("temporal_comparison_required") and len(images) == 1:
            answer_parts = ["I can't determine change over time from the current input because only one image is available. Please provide a second image of the same area from another date for bi-temporal comparison."]
        elif not answer_parts:
            answer_parts.append("The tools executed but no significant evidence was found to answer the query specifically.")

        if limitations:
            answer_parts.append("\nLimitations: " + " ".join(limitations))

        final_answer = "\n".join(answer_parts) if "Structured Scene Summary" in "\n".join(answer_parts) else " ".join(answer_parts)
        
        return ToolResult(
            task=TaskType.dynamic_analysis,
            tool_name=self.name,
            output_text=final_answer,
            confidence=0.85,
            confidence_calibrated=False,
            physical_metrics=physical_metrics,
            raw={"plan": plan, "evidence": evidence}
        )
