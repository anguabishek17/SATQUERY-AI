"""
Automated Image & Sensor Intelligence Layer for SatQuery AI.

Inspects uploaded satellite imagery metadata:
  - Band count, data type, band roles
  - Image width, height, spatial resolution (m/pixel)
  - Geographic CRS and geotransform matrix
  - Modality classification (optical_rgb, optical_multispectral, sar_vv_vh)
  - Automatic multi-image workflow inference (single, cross_modal, bi_temporal, compound)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional
from PIL import Image

from app.schemas import ImageRef, InputConfig

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


def inspect_image_sensor(image_path: str | Path) -> dict[str, Any]:
    """Inspects a raster file and returns comprehensive sensor & geospatial metadata."""
    path = str(image_path)
    filename = os.path.basename(path)
    ext = os.path.splitext(path)[1].lower()

    if HAS_RASTERIO and ext in (".tif", ".tiff"):
        try:
            with rasterio.open(path) as src:
                count = src.count
                crs_str = str(src.crs) if src.crs else "Unreferenced (Pixel Space)"
                transform = [src.transform.a, src.transform.b, src.transform.c,
                             src.transform.d, src.transform.e, src.transform.f]
                px_size_x = abs(src.transform.a)
                px_size_y = abs(src.transform.e)

                # Determine Modality
                if count == 1:
                    modality = "sar_vv_vh" if "sar" in filename.lower() or "sentinel1" in filename.lower() else "panchromatic"
                elif count == 3:
                    modality = "optical_rgb"
                elif count >= 4:
                    modality = "optical_multispectral"
                else:
                    modality = "optical"

                return {
                    "filename": filename,
                    "is_geotiff": True,
                    "bands_count": count,
                    "width": src.width,
                    "height": src.height,
                    "crs": crs_str,
                    "transform": transform,
                    "spatial_resolution_m": round(max(px_size_x, px_size_y), 2),
                    "modality": modality,
                    "driver": src.driver,
                }
        except Exception:
            pass

    # Non-GeoTIFF (PNG/JPEG benchmark fallback)
    try:
        with Image.open(path) as img:
            w, h = img.size
            mode = img.mode
            modality = "sar_vv_vh" if "sar" in filename.lower() else "optical_rgb"
            return {
                "filename": filename,
                "is_geotiff": False,
                "bands_count": len(mode),
                "width": w,
                "height": h,
                "crs": "Pixel Space (No Georeferencing)",
                "transform": [10.0, 0, 0, 0, -10.0, 0],
                "spatial_resolution_m": 10.0,
                "modality": modality,
                "driver": ext.replace(".", "").upper(),
            }
    except Exception as exc:
        return {
            "filename": filename,
            "error": str(exc),
            "modality": "optical_rgb",
            "spatial_resolution_m": 10.0,
        }


def classify_multi_image_workflow(images: list[ImageRef]) -> InputConfig:
    """Automatically determines analysis workflow configuration from uploaded imagery list."""
    if not images:
        return InputConfig.single

    if len(images) == 1:
        return InputConfig.single

    if len(images) == 2:
        modalities = {img.modality for img in images}
        if modalities == {"optical", "sar"}:
            return InputConfig.cross_modal
        if modalities == {"optical"} or modalities == {"sar"}:
            return InputConfig.bi_temporal

    return InputConfig.compound
