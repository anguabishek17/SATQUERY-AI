"""
Compound query decomposition — "SatQuery Chain".

This is the differentiator called out against GeoPilot/EarthGPT/TEOChat: those
systems select ONE tool per query. SatQuery Chain detects when a query
implies a *sequence* of dependent steps (e.g. "find new built-up areas within
500m of water and show the strongest SAR evidence") and decomposes it into an
ordered plan before any tool runs, then hands each step to the controller in
order, threading each step's output into the next step's input.

Deliberately rule-based, not a free-form LLM planner: a fixed vocabulary of
step types keeps the plan auditable and demoable, which matters more than
generality in a 4-day build.
"""
import re
from dataclasses import dataclass, field


@dataclass
class ChainStep:
    step_id: int
    action: str          # e.g. "ground", "buffer", "change_detect", "intersect", "sar_evidence"
    description: str
    depends_on: list[int] = field(default_factory=list)


_COMPOUND_MARKERS = [
    r"\bwithin\s+\d+\s*m(eters)?\b",       # spatial buffer
    r"\bnear\b.*\band\b",                    # "near X and Y"
    r"\bshow\b.*\b(and then|also|next)\b",   # multi-clause request with sequential actions
    r"\b(then|after that|followed by)\b",
]


def is_compound(query: str) -> bool:
    text = query.lower()
    return any(re.search(p, text) for p in _COMPOUND_MARKERS)


def decompose(query: str) -> list[ChainStep]:
    """
    Builds an ordered step plan from a compound query. This is intentionally
    a shallow pattern-matcher — extend _COMPOUND_MARKERS and the rules below
    together as you see real judge/demo queries during testing.
    """
    text = query.lower()
    steps: list[ChainStep] = []
    sid = 1

    if "water" in text or "river" in text:
        steps.append(ChainStep(sid, "ground", "Locate the referenced water body / river"))
        sid += 1

    buffer_match = re.search(r"within\s+(\d+)\s*m", text)
    if buffer_match:
        steps.append(ChainStep(sid, "buffer", f"Create a {buffer_match.group(1)}m buffer around the grounded region", depends_on=[sid - 1]))
        sid += 1

    if "chang" in text or "new" in text or "built-up" in text or "construct" in text:
        steps.append(ChainStep(sid, "change_detect", "Run bi-temporal change detection", depends_on=[]))
        sid += 1

    if buffer_match and any(s.action == "change_detect" for s in steps):
        steps.append(ChainStep(sid, "intersect", "Spatially intersect the change map with the buffer zone",
                                depends_on=[s.step_id for s in steps if s.action in ("buffer", "change_detect")]))
        sid += 1

    if "sar" in text:
        steps.append(ChainStep(sid, "sar_evidence", "Cross-check candidate regions against SAR backscatter", depends_on=[sid - 1] if steps else []))
        sid += 1

    steps.append(ChainStep(sid, "summarize", "Calculate statistics and compose the final evidence-grounded answer",
                            depends_on=[s.step_id for s in steps]))
    return steps
