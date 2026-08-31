# Positioning — read this before writing the pitch deck

## The honest starting point

SatQuery AI is **not** the first system to talk to satellite images, and "agentic
orchestration" alone is **not** a unique claim — GeoPilot already does
tool-augmented RS orchestration; GeoChat, EarthGPT, and TEOChat each cover
pieces of VQA, multimodal, and temporal reasoning individually. Don't build a
pitch around "nobody has done this before" — a judge who knows the literature
will call it out, and it isn't true.

## What's actually true and defensible

Existing systems demonstrate strong **individual** capabilities. SIH26167 asks
for a system that integrates **all of them together** — single-image,
cross-modal, bi-temporal, and agentic routing — behind one auditable,
query-driven interface, evaluated against a specific benchmark suite and an
ISRO/SAC eval set. That integration, with explicit input validation and an
auditable execution trace, is the actual ask. Say exactly that:

> "Existing remote-sensing VLMs and assistants demonstrate individual
> capabilities such as VQA, grounding, temporal reasoning, and multisensor
> analysis. SatQuery integrates these into a query-driven agentic workflow with
> explicit input validation, dynamic specialist-tool selection, evidence
> fusion, and an auditable execution trace aligned with the SIH26167
> specification."

## Where SatQuery goes further than a GeoPilot-style router

Single-tool-per-query routing (GeoPilot's core pattern) is table stakes, not
the differentiator. Three things in this build go past it — build these for
real, don't just claim them:

1. **Compound query decomposition ("SatQuery Chain")** —
   `backend/app/controller/query_decomposer.py`. Most systems pick one tool per
   query. A query like *"find newly built-up areas within 500m of the water
   body and show the strongest SAR evidence"* gets decomposed into an ordered,
   dependency-tracked step plan (ground → buffer → change-detect → intersect →
   SAR-evidence → summarize) before any tool runs. Render this plan in the UI —
   it's the single most demo-friendly artifact in the repo.

2. **Multi-turn spatial context memory** —
   `backend/app/controller/context_memory.py`. "Highlight the water body" →
   "how large is it?" → "what changed there?" resolves referents against the
   previous turn instead of treating each query as independent. This is what
   turns image Q&A into a spatial conversation.

3. **Quantified, auditable evidence, not prose claims** —
   `backend/app/services/change_stats.py` + the audit log. "Built-up area
   increased" is a claim; "23.4% changed, 12.8 ha, concentrated in the
   northern region, confidence 91%, full execution trace attached" is
   evidence. Judges evaluating against a benchmark suite respond to the
   second one.

## What NOT to claim or build

- Don't claim to have invented a new VLM, a new change-detection architecture,
  or a new fusion network — you didn't, and claiming so under questioning is
  where teams lose credibility. Say plainly that the specialist models are
  proven open-source components (GeoChat/LLaVA+LoRA, GroundingDINO, BIT-CD)
  and that SatQuery's contribution is the orchestration layer around them.
- Don't spend build time on voice interfaces, blockchain, face recognition,
  a custom-trained foundation model, or autonomous satellite tasking — none of
  it strengthens the SIH26167 score, and all of it eats days you don't have.
- Don't put a feature-by-feature "SatQuery beats GeoChat/EarthGPT/TEOChat/
  GeoPilot" comparison table in the deck unless every cell is verified against
  each system's actual paper/repo — an unverified competitive claim is worse
  than no claim if a judge checks it live.

## The one-line pitch

> "SatQuery AI turns satellite-image analysis from a model-driven workflow
> into a query-driven workflow — the user asks what they want, and the agent
> decides how to analyze the imagery, decomposing compound questions into an
> auditable chain of specialist steps."

## If asked "what's novel here?" — the five-point answer

1. Compound query decomposition into a dependency-tracked step chain, not
   single-tool routing.
2. Multi-turn spatial context memory across the conversation.
3. Quantified change evidence (area, %, zone) backed by a full audit trail,
   not just a text answer.
4. One natural-language interface across single-image, cross-modal, and
   bi-temporal inputs — the user never selects a model or GIS tool.
5. Built specifically to the SIH26167 evaluation spec: input compatibility
   checking, confidence flagging, and an execution trace that maps directly
   onto the stated evaluation criteria.
