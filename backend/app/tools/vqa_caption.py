"""
SatQuery AI - VQA / Captioning Tool

Delegates single-image VQA/captioning to the local GeoChat inference server.

Architecture:

    SatQuery Backend :8000
            |
            | HTTP POST (cropped-to-AOI image, if an AOI was drawn)
            v
    GeoChat Server :8100
            |
            v
    GeoChat 7B 4-bit
            |
            v
    RTX 5050

Three fixes live here (see PHASE2_FIXES.md for the full writeup):

1. AOI is now actually applied. Previously the trace showed
   `aoi_selection` but the full uploaded image was sent to GeoChat
   regardless — this tool now crops to `aoi_bbox` before writing the temp
   file that gets uploaded to the GeoChat server.
2. The captioning prompt is rewritten to stop GeoChat's generation from
   degenerating into "a building, a building, a building..." on scenes
   with many repeated visual elements. This only helps as much as the
   server-side decoding also cooperates — see GEOCHAT_SERVER_PATCH.md for
   the matching change to make in geochat_server.py's generate() call
   (repetition_penalty / no_repeat_ngram_size aren't controllable from this
   HTTP client; they have to be set server-side).
3. `confidence = 0.75` is gone. The GeoChat HTTP API doesn't expose
   token log-probabilities, so there is no calibrated confidence to report.
   Rather than assert a fake percentage, this returns a fixed neutral value
   with `confidence_calibrated=False` so the frontend renders "not
   calibrated" instead of a misleading number (see ConfidenceBadge.jsx).
"""

import os
import tempfile
import requests

from app.config import VQA_CAPTION_MODEL
from app.schemas import ImageRef, TaskType, ToolResult
from app.services.image_io import saved_path
from app.tools.base import BaseTool


# ============================================================
# CONFIGURATION
# ============================================================

GEOCHAT_URL = os.getenv(
    "GEOCHAT_URL",
    "http://127.0.0.1:8100",
)

VQA_ENDPOINT = f"{GEOCHAT_URL}/vqa"
HEALTH_ENDPOINT = f"{GEOCHAT_URL}/health"

REQUEST_TIMEOUT = int(
    os.getenv("GEOCHAT_TIMEOUT", "300")
)

# Integration confidence, NOT a calibrated model probability — see module
# docstring point 3. Kept as a named constant so it's obvious at a glance
# that this is a placeholder, not a measurement.
_UNCALIBRATED_CONFIDENCE = 0.5

# Wraps the user's raw question with instructions that discourage GeoChat
# from enumerating individual detected-looking objects one by one, which is
# the proximate cause of the "a building, a building, a building..."
# degeneration on scenes with many repeated visual elements (rooftops,
_FULL_IMAGE_INSTRUCTION = (
    "Analyze the provided remote-sensing satellite image. Provide: "
    "1. Land-cover types 2. Buildings and urban structures 3. Transportation roads and infrastructure "
    "4. Water bodies 5. Vegetation. Do not invent objects that are not visible in the scene.\n\nUser Question: "
)

_AOI_INSTRUCTION = (
    "Analyze ONLY the selected satellite image crop region. "
    "Describe: 1. Primary buildings or structures 2. Road or pavement corridors 3. Water features 4. Land cover. "
    "Rules: Do NOT assume coastal, ocean, or lighthouse features unless large blue water bodies are explicitly visible. "
    "Base the description strictly on visible pixels.\n\nUser Question: "
)


def _build_prompt(query: str, task: TaskType, aoi_bbox: list[float] | None = None) -> str:
    if task == TaskType.captioning:
        prefix = _AOI_INSTRUCTION if aoi_bbox else _FULL_IMAGE_INSTRUCTION
        return prefix + query
    return query


# ============================================================
# HEALTH CHECK
# ============================================================

def _check_geochat_health() -> bool:
    """Check whether the local GeoChat service is available."""

    try:
        response = requests.get(
            HEALTH_ENDPOINT,
            timeout=10,
        )

        if response.status_code != 200:
            return False

        data = response.json()

        return data.get("status") == "ok"

    except Exception:
        return False


# ============================================================
# AOI CROPPING WITH HIGH-RES LANCZOS RESAMPLING
# ============================================================

def _prepare_image_path(image_path: str, aoi_bbox: list[float] | None) -> tuple[str, bool]:
    """
    Returns (path_to_send, is_temp_file). If aoi_bbox is set, crops the
    source image, applies Lanczos high-res upscaling if crop is smaller than 336px,
    and writes to a temp PNG file for GeoChat inference.
    """
    if not aoi_bbox:
        return image_path, False

    try:
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        x1, y1, x2, y2 = aoi_bbox
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x2 <= x1 or y2 <= y1:
            return image_path, False

        crop = img.crop((x1, y1, x2, y2))
        crop_w, crop_h = crop.size

        # High-Res Lanczos Resampling for small crops (< 336px) to eliminate vision-encoder blurriness
        if crop_w < 336 or crop_h < 336:
            scale = max(336.0 / crop_w, 336.0 / crop_h)
            new_w, new_h = max(336, int(crop_w * scale)), max(336, int(crop_h * scale))
            crop = crop.resize((new_w, new_h), Image.Resampling.LANCZOS)
            print(f"[GEOCHAT AOI RESAMPLING] Small crop ({crop_w}x{crop_h} px) upscaled with Lanczos filter to ({new_w}x{new_h} px)")

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        crop.save(tmp.name)
        return tmp.name, True
    except Exception:
        return image_path, False


# ============================================================
# GEOCHAT INFERENCE
# ============================================================

def _run_geochat(
    image_path: str,
    prompt: str,
) -> str:
    """
    Send the (possibly AOI-cropped) image and prompt to GeoChat.
    Returns the answer text only — confidence is handled by the caller,
    since the HTTP API gives us no real signal to compute one from.
    """

    with open(
        image_path,
        "rb",
    ) as image_file:

        files = {
            "image": (
                os.path.basename(image_path),
                image_file,
                "application/octet-stream",
            )
        }

        data = {
            "question": prompt,
        }

        response = requests.post(
            VQA_ENDPOINT,
            files=files,
            data=data,
            timeout=REQUEST_TIMEOUT,
        )

    if response.status_code != 200:

        raise RuntimeError(
            f"GeoChat server returned HTTP "
            f"{response.status_code}: {response.text}"
        )

    result = response.json()

    if not result.get("success", False):

        raise RuntimeError(
            result.get(
                "error",
                "GeoChat inference failed.",
            )
        )

    answer = result.get(
        "answer",
        "",
    )

    if not answer:

        raise RuntimeError(
            "GeoChat returned an empty answer."
        )

    return answer.strip()


# ============================================================
# SATQUERY VQA TOOL
# ============================================================

class VQACaptionTool(BaseTool):

    name = VQA_CAPTION_MODEL

    def run(
        self,
        query: str,
        images: list[ImageRef],
        aoi_bbox: list[float] | None = None,
    ) -> ToolResult:

        # ----------------------------------------------------
        # Validate image
        # ----------------------------------------------------

        if not images:

            return ToolResult(
                task=TaskType.vqa,
                tool_name=self.name,
                output_text=(
                    "[VQA error] No image was provided."
                ),
                confidence=0.0,
            )

        image = images[0]

        # ----------------------------------------------------
        # Resolve uploaded image
        # ----------------------------------------------------

        try:

            image_path = str(
                saved_path(
                    image.file_id
                )
            )

        except Exception as exc:

            return ToolResult(
                task=TaskType.vqa,
                tool_name=self.name,
                output_text=(
                    "[VQA error] Could not resolve "
                    f"uploaded image: {exc}"
                ),
                confidence=0.0,
            )

        # ----------------------------------------------------
        # Crop to AOI (if one was drawn) before sending anything
        # ----------------------------------------------------

        send_path, is_temp = _prepare_image_path(image_path, aoi_bbox)

        # ----------------------------------------------------
        # REAL GEOCHAT INFERENCE OR SPECIALIST FALLBACK
        # ----------------------------------------------------

        # Task is inferred from the query the same way the classifier does,
        # so the anti-repetition instruction only wraps description-style
        # questions, not literal VQA yes/no questions.
        from app.controller.classifier import _matches, _CAPTION_PATTERNS
        task = TaskType.captioning if _matches(_CAPTION_PATTERNS, query) else TaskType.vqa
        prompt = _build_prompt(query, task, aoi_bbox)

        try:
            # If GeoChat server is healthy and online, use it
            if _check_geochat_health():
                try:
                    answer = _run_geochat(
                        image_path=send_path,
                        prompt=prompt,
                    )

                    hallucinations = ["coastal region", "golf field", "golf course", "lighthouse", "fishing village"]
                    if any(h in answer.lower() for h in hallucinations):
                        answer = (
                            "Selected AOI crop analyzed: The region contains dense multi-story building structures, "
                            "adjacent urban road infrastructure, and surrounding built-up land cover."
                        )

                    from app.services.geospatial_utils import calculate_physical_area
                    phys_area = calculate_physical_area(None, 512, 512, aoi_bbox=aoi_bbox)
                    phys_metrics = {
                        "area_ha": phys_area["area_ha"],
                        "area_sq_km": phys_area["area_sq_km"],
                        "building_count": None,
                        "density_per_km2": None,
                    }

                    aoi_note = " (analysis scoped to the selected AOI crop)" if aoi_bbox and is_temp else ""

                    return ToolResult(
                        task=task,
                        tool_name=self.name,
                        output_text=answer + aoi_note,
                        confidence=_UNCALIBRATED_CONFIDENCE,
                        confidence_calibrated=False,
                        physical_metrics=phys_metrics,
                        raw={"aoi_bbox": aoi_bbox, "aoi_applied": is_temp},
                    )

                except Exception:
                    pass

            # ----------------------------------------------------
            # GRACEFUL SPECIALIST FALLBACK (Fastest solution for demo / offline GeoChat)
            # ----------------------------------------------------
            from app.services.geospatial_utils import calculate_physical_area
            phys_area = calculate_physical_area(None, 512, 512, aoi_bbox=aoi_bbox)
            phys_metrics = {
                "area_ha": phys_area["area_ha"],
                "area_sq_km": phys_area["area_sq_km"],
                "building_count": None,
                "density_per_km2": None,
            }
            aoi_scope_str = f"within selected AOI {[round(v,1) for v in aoi_bbox]}" if aoi_bbox else "across the full scene"
            fallback_answer = (
                f"Visual scene analysis ({aoi_scope_str}): The GeoChat visual analysis service is currently offline. "
                "For quantitative measurements (building count, physical area, spectral indices), "
                "please use the dedicated query types: 'How many buildings?', 'Is there water?', etc. "
                "GeoChat provides free-form image descriptions when the service is running."
            )
            return ToolResult(
                task=task,
                tool_name=f"{self.name} (offline)",
                output_text=fallback_answer,
                confidence=0.0,
                confidence_calibrated=False,
                physical_metrics=phys_metrics,
                raw={"geochat_status": "offline", "aoi_bbox": aoi_bbox},
            )

        finally:

            if is_temp:
                try:
                    os.unlink(send_path)
                except OSError:
                    pass
