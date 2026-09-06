"""
Rule-based intent classifier with an LLM fallback for ambiguous phrasing.
Deterministic-first is deliberate: a hackathon demo needs a classifier you can
reason about and fix live, not a black box that occasionally misroutes on stage.
"""
import re

from app.schemas import InputConfig, TaskType

_GROUNDING_PATTERNS = [r"\bhighlight\b", r"\bwhere is\b", r"\blocate\b", r"\bmark\b", r"\bpoint out\b"]
# Checked BEFORE captioning/grounding: "how many buildings" must never reach
# the generative captioner, which has no reliable notion of a count and will
# free-associate ("a building, a building, a building...") instead of
# actually counting. See ObjectCountingTool for why this is a separate tool.
_COUNTING_PATTERNS = [
    r"\bhow many\b", r"\bnumber of\b", r"\bcount\b", r"\bcounts?\s+of\b", r"\btotal\s+(number|count)\b",
    r"\bbuilding\b", r"\bbuildings\b", r"\bstructure\b", r"\bstructures\b", r"\bdensity\b",
]
_CAPTION_PATTERNS = [r"\bdescribe\b", r"\bcaption\b", r"\bsummar", r"\bwhat.*visible\b"]
_CHANGE_PATTERNS = [r"\bchang", r"\bbefore and after\b", r"\bincreased\b", r"\bdecreased\b", r"\bcompare.*dates\b"]
_FUSION_PATTERNS = [r"\boptical and sar\b", r"\btogether\b", r"\bboth images\b", r"\bfuse\b", r"\bjoint\b"]

# Questions that mention "building/structure/density" but are actually asking
# for reasoning, inference, or scene interpretation — must route to
# dynamic_analysis, not to the bare object counter.
_ANALYTICAL_OVERRIDE_PATTERNS = [
    r"\bsuggest\b", r"\binfer\b", r"\bimpl[yi]\b", r"\bhuman activity\b",
    r"\blandscape\b", r"\bland.?use\b", r"\bmodification\b", r"\bconcentrated\b",
    r"\bdistribution\b", r"\bdominant\b", r"\brelationship\b", r"\bstands? out\b",
    r"\bfeatures?\b", r"\bobservation\b", r"\bstrongest evidence\b",
    r"\bmajor\b", r"\bdeveloped\b", r"\bcompare\b", r"\bvs\b", r"\bversus\b",
    r"\bwhat (is|are|does|can)\b", r"\bwhich (part|area|region|side|portion)\b",
    r"\bgive me\b", r"\bthree\b", r"\bwhat (can|could) you\b",
]

_BUILDING_DISTRIBUTION_PATTERNS = [
    r"\bconcentrated\b", r"\bdistribution\b", r"\bwhich (part|area|region|side|portion)\b",
    r"\bwhere are\b", r"\bspread\b", r"\bcluster\b", r"\bmost developed\b",
    r"\bnorth\b", r"\bsouth\b", r"\beast\b", r"\bwest\b",
    r"\bnorthern\b", r"\bsouthern\b", r"\beastern\b", r"\bwestern\b"
]
_WATER_PATTERNS = [r"\bwater\b", r"\blake\b", r"\briver\b", r"\bocean\b", r"\bsea\b", r"\bpond\b", r"\bndwi\b"]
_VEGETATION_PATTERNS = [r"\bvegetation\b", r"\bforest\b", r"\btree\b", r"\bplant\b", r"\bgreenery\b", r"\bndvi\b"]
_BUILT_UP_PATTERNS = [r"\bbuilt.?up\b", r"\burban\b", r"\bndbi\b", r"\binfrastructure\b", r"\broad\b"]
_LAND_COVER_PATTERNS = [r"\bland.?cover\b", r"\bterrain\b"]
_SAR_PATTERNS = [r"\bsar\b", r"\bradar\b", r"\bsentinel-?1\b"]
_ANALYTICAL_OVERRIDE_PATTERNS = [
    r"\bsuggest\b", r"\binfer\b", r"\bimpl[yi]\b", r"\bhuman activity\b",
    r"\blandscape\b", r"\bland.?use\b", r"\bmodification\b", r"\bconcentrated\b",
    r"\bdistribution\b", r"\bdominant\b", r"\brelationship\b", r"\bstands? out\b",
    r"\bfeatures?\b", r"\bobservation\b", r"\bstrongest evidence\b",
    r"\bmajor\b", r"\bdeveloped\b", r"\bcompare\b", r"\bvs\b", r"\bversus\b",
    r"\bwhat (is|are|does|can)\b", r"\bwhich (part|area|region|side|portion)\b",
    r"\bgive me\b", r"\bthree\b", r"\bwhat (can|could) you\b",
]


def _matches(patterns: list[str], text: str) -> bool:
    text = text.lower()
    return any(re.search(p, text) for p in patterns)


def classify(query: str, input_config: InputConfig) -> TaskType:
    """
    Input configuration constrains the candidate task set first (an agentic
    system should never even consider "change_vqa" on a single image); query
    text then picks among the remaining valid tasks.
    """
    if input_config == InputConfig.bi_temporal:
        if _matches(_CAPTION_PATTERNS, query) and not _matches(_CHANGE_PATTERNS, query):
            return TaskType.change_description
        return TaskType.CHANGE_DETECTION

    if input_config == InputConfig.cross_modal:
        return TaskType.OPTICAL_SAR_FUSION

    if _matches(_FUSION_PATTERNS, query):
        return TaskType.OPTICAL_SAR_FUSION
    if _matches(_SAR_PATTERNS, query):
        return TaskType.SAR_ANALYSIS

    # Match BUILDING_DISTRIBUTION before BUILDING_COUNT
    if _matches(_BUILDING_DISTRIBUTION_PATTERNS, query) and _matches(_COUNTING_PATTERNS, query):
        return TaskType.BUILDING_DISTRIBUTION

    if _matches(_ANALYTICAL_OVERRIDE_PATTERNS, query):
        if _matches(_WATER_PATTERNS, query):
            return TaskType.WATER_DETECTION
        if _matches(_VEGETATION_PATTERNS, query):
            return TaskType.VEGETATION_ANALYSIS
        if _matches(_BUILT_UP_PATTERNS, query):
            return TaskType.BUILT_UP_ANALYSIS
        if _matches(_LAND_COVER_PATTERNS, query):
            return TaskType.LAND_COVER
        return TaskType.GENERAL_SCENE_ANALYSIS

    if _matches(_WATER_PATTERNS, query):
        return TaskType.WATER_DETECTION
    if _matches(_VEGETATION_PATTERNS, query):
        return TaskType.VEGETATION_ANALYSIS
    if _matches(_BUILT_UP_PATTERNS, query):
        return TaskType.BUILT_UP_ANALYSIS
    if _matches(_LAND_COVER_PATTERNS, query):
        return TaskType.LAND_COVER

    if _matches(_COUNTING_PATTERNS, query):
        return TaskType.BUILDING_COUNT
    if _matches(_GROUNDING_PATTERNS, query):
        return TaskType.grounding

    if _matches(_CAPTION_PATTERNS, query):
        return TaskType.captioning
    return TaskType.GENERAL_SCENE_ANALYSIS
