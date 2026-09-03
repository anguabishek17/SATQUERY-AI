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
    needs_water = re.search(r'\b(water|river|lake|flood|flooding|wet)\b', query_lower)
    needs_veg = re.search(r'\b(vegetation|greenery|forest|trees|crop|agriculture|rural)\b', query_lower)
    needs_built = re.search(r'\b(building|buildings|urban|built-up|structure|house|road|settlement)\b', query_lower)
    needs_change = re.search(r'\b(change|changed|difference|compare.*time|before.*after|flood|flooding)\b', query_lower)
    needs_broad = re.search(r'\b(summary|summarize|what can you identify|describe|major features?|stands? out|landscape|land.?use|human (activity|modification)|strongest evidence|what (is|are|does) this|give me)\b', query_lower)
    needs_spatial = re.search(r'\b(compare.*(north|south|east|west|upper|lower|side)|which side|where are.*concentrated|more.*than|concentrated|distribution|most developed|which part|which (area|region|portion))\b', query_lower)

    if re.search(r'\b(how many|count|number of)\b', query_lower):
        plan["numeric_metrics_required"] = True

    # Urban/rural questions need comprehensive evidence, not just buildings
    if re.search(r'\b(urban|rural)\b', query_lower):
        needs_built = True
        needs_veg = True
        needs_water = True
        plan["reasoning_tasks"].append("assess_urban_rural")
    elif needs_built:
        plan["reasoning_tasks"].append("assess_built_environment")

    if needs_water:
        plan["evidence_required"].add("water")
        plan["tools_required"].add("optical_processing")
        if not re.search(r'\b(urban|rural)\b', query_lower):
            plan["reasoning_tasks"].append("assess_water_presence")

    if needs_veg:
        plan["evidence_required"].add("vegetation")
        plan["tools_required"].add("optical_processing")
        if not re.search(r'\b(urban|rural)\b', query_lower):
            plan["reasoning_tasks"].append("assess_vegetation")
            
    if needs_built:
        plan["evidence_required"].update(["buildings", "built_up"])
        plan["tools_required"].update(["object_counting", "optical_processing"])
        
    if needs_change:
        plan["evidence_required"].add("change")
        plan["tools_required"].add("change_detection")
        plan["temporal_comparison_required"] = True
        
    if re.search(r'\b(flood|flooding)\b', query_lower):
        plan["reasoning_tasks"].append("assess_flooding")
        
    if needs_spatial:
        plan["evidence_required"].update(["spatial_distribution", "buildings", "vegetation", "water", "built_up"])
        plan["tools_required"].update(["optical_processing", "object_counting"])
        plan["reasoning_tasks"].append("spatial_comparison")
        
    if needs_broad or not plan["evidence_required"]:
        # Broad/scene-level questions: gather all available evidence
        plan["evidence_required"].update(["water", "vegetation", "buildings", "built_up"])
        plan["tools_required"].update(["optical_processing", "object_counting"])
        if needs_change:
            plan["evidence_required"].add("change")
        plan["reasoning_tasks"].append("broad_scene_summary")
        # Also trigger spatial only for directional/comparative questions, NOT for pure scene-description
        if re.search(r'\b(most developed|concentrated|distribution|which.*more|which (part|portion|area|region))\b', query_lower):
            plan["evidence_required"].add("spatial_distribution")
            plan["reasoning_tasks"].append("spatial_comparison")

    # Convert sets to lists for JSON serialization
    plan["evidence_required"] = list(plan["evidence_required"])
    plan["tools_required"] = list(plan["tools_required"])
    
    return plan
