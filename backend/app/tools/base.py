"""
Every specialist tool implements this interface. Keeping the contract identical
across VQA, grounding, object counting, change detection, and SAR fusion is
what makes the controller a thin router instead of a pile of special cases.

`aoi_bbox` is part of the contract for every tool, not just the ones that
currently use it: a tool that ignores it (grounding, change detection, SAR
fusion — not yet AOI-aware) still declares the parameter so the controller
can call every tool the same way instead of branching per-task on whether
AOI scoping is supported.
"""
from abc import ABC, abstractmethod

from app.schemas import ImageRef, ToolResult


class BaseTool(ABC):
    name: str = "base_tool"

    @abstractmethod
    def run(self, query: str, images: list[ImageRef], aoi_bbox: list[float] | None = None) -> ToolResult:
        """Execute the tool and return a populated ToolResult."""
        raise NotImplementedError
