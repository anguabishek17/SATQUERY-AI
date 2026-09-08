"""
Dynamic analysis planner.

Maps natural language questions to required evidence and tools
compositionally.
"""

import re
from typing import Any, Dict

def determine_evidence_required(query: str) -> Dict[str, Any]:
    query_lower = query.lower()
    
    plan = {
        "goal": query,
        "scope": "full_scene", # Will be updated if AOI is provided
        "evidence_required": set(),
        "tools_required": set(),
        "reasoning_tasks": [],
        "numeric_metrics_required": False,
        "temporal_comparison_required": False,
        "visual_reasoning_required": False
    }

    # Compositional Evidence Selection
    needs_water = bool(re.search(r'\b(water|river|lake|flood|flooding|wet)\b', query_lower))
    needs_veg = bool(re.search(r'\b(vegetation|greenery|forest|trees|crop|agriculture|rural)\b', query_lower))
    needs_built = bool(re.search(r'\b(urban|built-up|road|settlement|human (activity|presence))\b', query_lower))
    needs_building = bool(re.search(r'\b(building|buildings|structure|structures|house|houses)\b', query_lower))
    needs_change = bool(re.search(r'\b(change|changed|difference|compare.*time|before.*after)\b', query_lower))
    needs_broad = bool(re.search(r'\b(summary|summarize|what can you identify|describe|major features?|stands? out|landscape|land.?use|strongest evidence|what (is|are|does) this|give me)\b', query_lower))
    needs_spatial = bool(re.search(r'\b(compare.*(north|south|east|west|upper|lower|side)|which side|where are|where is|more.*than|concentrated|distribution|most developed|which part|which (area|region|portion))\b', query_lower))

    is_counting = bool(re.search(r'\b(how many|count|number of|total)\b', query_lower))
    if is_counting:
        plan["numeric_metrics_required"] = True

    # 1. Water evidence
    if needs_water:
        plan["evidence_required"].add("water")
        plan["tools_required"].add("optical_processing")
        plan["reasoning_tasks"].append("assess_water_presence")

    # 2. Vegetation evidence
    if needs_veg:
        plan["evidence_required"].add("vegetation")
        plan["tools_required"].add("optical_processing")
        plan["reasoning_tasks"].append("assess_vegetation")

    # 3. Built-up / Urban land-cover evidence (spectral/RGB landcover, lightweight)
    if needs_built or (needs_building and not is_counting and not re.search(r'\bwhere are buildings\b', query_lower)):
        plan["evidence_required"].add("built_up")
        plan["tools_required"].add("optical_processing")
        plan["reasoning_tasks"].append("assess_built_environment")

    # 4. Building detection (HEAVY YOLO - strictly for building count / footprint queries)
    if needs_building and (is_counting or re.search(r'\b(where are buildings|building.*concentrated|building count|detect.*building)\b', query_lower)):
        plan["evidence_required"].add("buildings")
        plan["tools_required"].add("object_counting")
        plan["reasoning_tasks"].append("building_count")

    # 5. Change detection
    if needs_change:
        plan["evidence_required"].add("change")
        plan["tools_required"].add("change_detection")
        plan["temporal_comparison_required"] = True

    if re.search(r'\b(flood|flooding)\b', query_lower):
        plan["reasoning_tasks"].append("assess_flooding")

    # 6. Spatial comparison (scope only to requested modality, NEVER trigger YOLO unless building-specific)
    if needs_spatial:
        plan["evidence_required"].add("spatial_distribution")
        plan["reasoning_tasks"].append("spatial_comparison")
        if needs_water:
            plan["evidence_required"].add("water")
            plan["tools_required"].add("optical_processing")
        elif needs_veg:
            plan["evidence_required"].add("vegetation")
            plan["tools_required"].add("optical_processing")
        elif needs_building:
            plan["evidence_required"].add("buildings")
            plan["tools_required"].add("object_counting")
        else:
            plan["evidence_required"].update(["vegetation", "water", "built_up"])
            plan["tools_required"].add("optical_processing")

    # 7. Broad / Scene-level description (LIGHTWEIGHT: vegetation + water + built_up, NEVER YOLO)
    if needs_broad or not plan["evidence_required"]:
        plan["evidence_required"].update(["water", "vegetation", "built_up"])
        plan["tools_required"].add("optical_processing")
        plan["reasoning_tasks"].append("broad_scene_summary")

    # Convert sets to lists for JSON serialization
    plan["evidence_required"] = list(plan["evidence_required"])
    plan["tools_required"] = list(plan["tools_required"])
    
    return plan
