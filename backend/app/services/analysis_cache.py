"""
analysis_cache.py — In-memory caching engine for SatQuery AI.

Provides two-tier caching:
1. Image Preprocessing Cache:
   Caches decoded image arrays (BGR, HSV, Gray), dimensions, and spatial metadata
   so subsequent queries on the same image do not re-decode or re-convert color spaces.

2. Task Result Cache:
   Caches computed GeoAnalysis metrics and evidence for specific tasks
   (water, vegetation, built_up, buildings, land_cover, fusion, scene)
   so follow-up queries on the same image return instantaneously.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np

class AnalysisCache:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AnalysisCache, cls).__new__(cls)
            cls._instance._image_cache: Dict[str, Dict[str, Any]] = {}
            cls._instance._task_cache: Dict[str, Dict[str, Any]] = {}
        return cls._instance

    @staticmethod
    def _make_aoi_key(aoi_bbox: Optional[list]) -> str:
        if not aoi_bbox:
            return "full_scene"
        return f"{round(aoi_bbox[0], 1)}_{round(aoi_bbox[1], 1)}_{round(aoi_bbox[2], 1)}_{round(aoi_bbox[3], 1)}"

    # -------------------------------------------------------------------------
    # 1. Image Preprocessing Cache
    # -------------------------------------------------------------------------
    def get_preprocessed_image(self, image_id: str, aoi_bbox: Optional[list] = None) -> Optional[Dict[str, Any]]:
        """
        Returns cached preprocessed image data:
        {
            "img": np.ndarray (BGR),
            "hsv": np.ndarray,
            "gray": np.ndarray,
            "shape": (h, w, c),
            "spatial_meta": dict
        }
        """
        aoi_key = self._make_aoi_key(aoi_bbox)
        cache_key = f"{image_id}:{aoi_key}"
        return self._image_cache.get(cache_key)

    def cache_preprocessed_image(
        self,
        image_id: str,
        bgr_img: np.ndarray,
        hsv_img: np.ndarray,
        gray_img: np.ndarray,
        spatial_meta: Optional[Dict[str, Any]] = None,
        aoi_bbox: Optional[list] = None
    ) -> None:
        aoi_key = self._make_aoi_key(aoi_bbox)
        cache_key = f"{image_id}:{aoi_key}"
        self._image_cache[cache_key] = {
            "img": bgr_img,
            "hsv": hsv_img,
            "gray": gray_img,
            "shape": bgr_img.shape,
            "spatial_meta": spatial_meta or {},
            "masks": {}
        }

    # -------------------------------------------------------------------------
    # 2. Task Result Cache
    # -------------------------------------------------------------------------
    def get_task_result(self, image_id: str, task_category: str, aoi_bbox: Optional[list] = None) -> Optional[Dict[str, Any]]:
        """
        Retrieves cached analysis result for a given image and task category.
        Categories: "water", "vegetation", "built_up", "buildings", "land_cover", "scene", "fusion"
        """
        aoi_key = self._make_aoi_key(aoi_bbox)
        cache_key = f"{image_id}:{aoi_key}:{task_category}"
        return self._task_cache.get(cache_key)

    def cache_task_result(
        self,
        image_id: str,
        task_category: str,
        result_data: Dict[str, Any],
        aoi_bbox: Optional[list] = None
    ) -> None:
        aoi_key = self._make_aoi_key(aoi_bbox)
        cache_key = f"{image_id}:{aoi_key}:{task_category}"
        self._task_cache[cache_key] = result_data

    def clear(self) -> None:
        """Clears all in-memory caches."""
        self._image_cache.clear()
        self._task_cache.clear()


analysis_cache = AnalysisCache()
