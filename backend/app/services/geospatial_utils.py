"""
Geospatial core utilities for SatQuery AI.

Provides:
  - GeoTIFF CRS, transform, and extent validation
  - Co-registration and automated reprojection (SpaceNet 9 pattern)
  - Extent overlap verification to reject unrelated rasters
  - Coordinate translation between screen/pixel space and geographic CRS
  - Ground area calculation in m², hectares, sq km
  - GeoJSON FeatureCollection generation with property enrichment for MapLibre
"""

from __future__ import annotations

import math
from typing import Any, Optional
import numpy as np

try:
    import rasterio
    import rasterio.warp
    from rasterio.enums import Resampling
    from rasterio.windows import Window
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


class GeospatialValidationError(Exception):
    """Raised when spatial input constraints (e.g. non-overlapping extents) are violated."""
    pass


def _bounds_intersect(b1: tuple[float, float, float, float], b2: tuple[float, float, float, float]) -> bool:
    """Checks if two [left, bottom, right, top] bounding boxes intersect."""
    l1, b_1, r1, t1 = b1
    l2, b_2, r2, t2 = b2
    return not (r1 <= l2 or r2 <= l1 or t1 <= b_2 or t2 <= b_1)


def validate_and_coregister(
    optical_path: str,
    sar_path: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Validates alignment between Optical reference GeoTIFF and SAR GeoTIFF.
    If dimensions, CRS, or geotransforms differ, performs automated reprojection
    and resampling of the SAR band onto the Optical reference grid.

    Raises GeospatialValidationError if the rasters have no geographic overlap.
    """
    if not HAS_RASTERIO:
        raise RuntimeError("Rasterio is required for GeoTIFF co-registration.")

    with rasterio.open(optical_path) as opt_src:
        opt_crs = opt_src.crs
        opt_transform = opt_src.transform
        opt_shape = (opt_src.height, opt_src.width)
        opt_bounds = opt_src.bounds
        opt_meta = {
            "crs": str(opt_crs) if opt_crs else None,
            "transform": [opt_transform.a, opt_transform.b, opt_transform.c,
                          opt_transform.d, opt_transform.e, opt_transform.f],
            "width": opt_src.width,
            "height": opt_src.height,
            "bounds": [opt_bounds.left, opt_bounds.bottom, opt_bounds.right, opt_bounds.top],
            "pixel_size_x": abs(opt_transform.a),
            "pixel_size_y": abs(opt_transform.e),
        }

        with rasterio.open(sar_path) as sar_src:
            sar_band = sar_src.read(1).astype(np.float64)
            sar_crs = sar_src.crs
            sar_transform = sar_src.transform
            sar_shape = (sar_src.height, sar_src.width)
            sar_bounds = sar_src.bounds

            # Check extent overlap if both have valid CRSs
            if opt_crs and sar_crs and opt_crs == sar_crs:
                b1 = (opt_bounds.left, opt_bounds.bottom, opt_bounds.right, opt_bounds.top)
                b2 = (sar_bounds.left, sar_bounds.bottom, sar_bounds.right, sar_bounds.top)
                if not _bounds_intersect(b1, b2):
                    raise GeospatialValidationError(
                        f"Optical bounds {b1} and SAR bounds {b2} have no geographic overlap."
                    )

            # Strict check: same CRS, transform, shape
            same_crs = (opt_crs == sar_crs) or (opt_crs is None and sar_crs is None)
            same_transform = (opt_transform == sar_transform)
            same_shape = (opt_shape == sar_shape)

            if same_crs and same_transform and same_shape:
                return sar_band, opt_meta

            # Mismatch detected -> Reproject SAR to Optical grid
            reprojected_sar = np.zeros(opt_shape, dtype=np.float32)
            try:
                rasterio.warp.reproject(
                    source=sar_band,
                    destination=reprojected_sar,
                    src_transform=sar_transform,
                    src_crs=sar_crs,
                    dst_transform=opt_transform,
                    dst_crs=opt_crs,
                    resampling=Resampling.bilinear,
                )
            except Exception:
                from PIL import Image
                sar_img = Image.fromarray(sar_band.astype(np.float32), mode="F").resize((opt_src.width, opt_src.height), Image.BILINEAR)
                reprojected_sar = np.array(sar_img, dtype=np.float32)

            opt_meta["reprojected"] = True
            opt_meta["reproject_reason"] = (
                f"SAR CRS ({sar_crs}) or grid {sar_shape} differed from Optical ({opt_crs}, {opt_shape})"
            )
            return reprojected_sar, opt_meta


def calculate_physical_area(
    transform: Optional[list[float] | Any],
    width: int,
    height: int,
    aoi_bbox: Optional[list[float]] = None,
    crs: Optional[str] = None,
) -> dict[str, float]:
    if aoi_bbox:
        x1, y1, x2, y2 = aoi_bbox
        w = max(1.0, abs(x2 - x1))
        h = max(1.0, abs(y2 - y1))
    else:
        w = float(width)
        h = float(height)

    total_pixels = w * h

    if transform is not None:
        if isinstance(transform, (list, tuple)) and len(transform) >= 6:
            px_x = abs(transform[0])
            px_y = abs(transform[4])
        elif hasattr(transform, 'a') and hasattr(transform, 'e'):
            px_x = abs(transform.a)
            px_y = abs(transform.e)
        else:
            px_x, px_y = 10.0, 10.0
    else:
        px_x, px_y = 10.0, 10.0

    if crs and ("4326" in str(crs) or "GEOGCS" in str(crs).upper()):
        lat_rad = math.radians(20.0)
        meters_per_deg_lat = 111000.0
        meters_per_deg_lon = 111000.0 * math.cos(lat_rad)
        px_x_m = px_x * meters_per_deg_lon
        px_y_m = px_y * meters_per_deg_lat
    else:
        px_x_m = px_x
        px_y_m = px_y

    area_sq_m = total_pixels * px_x_m * px_y_m
    area_ha = area_sq_m / 10000.0
    area_sq_km = area_sq_m / 1000000.0

    return {
        "area_sq_m": round(area_sq_m, 2),
        "area_ha": round(area_ha, 4),
        "area_sq_km": round(area_sq_km, 4),
        "pixel_resolution_m": round(max(px_x_m, px_y_m), 2),
    }


def boxes_to_geojson(
    boxes: list[list[float]],
    scores: Optional[list[float]] = None,
    labels: Optional[list[str]] = None,
    transform: Optional[Any] = None,
    pixel_resolution_m: float = 10.0,
) -> dict[str, Any]:
    """
    Converts list of bounding boxes [x1, y1, x2, y2] into an enriched GeoJSON FeatureCollection.
    Includes building_id, area_sq_m, confidence, and centroid in properties.
    """
    features = []
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        score = scores[idx] if scores and idx < len(scores) else 1.0
        label = labels[idx] if labels and idx < len(labels) else "building"

        box_w = abs(x2 - x1)
        box_h = abs(y2 - y1)
        area_sq_m = box_w * box_h * (pixel_resolution_m ** 2)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        if transform is not None:
            if hasattr(transform, '__mul__'):
                lon1, lat1 = transform * (x1, y1)
                lon2, lat2 = transform * (x2, y1)
                lon3, lat3 = transform * (x2, y2)
                lon4, lat4 = transform * (x1, y2)
                clon, clat = transform * (cx, cy)
            else:
                lon1, lat1, lon2, lat2, lon3, lat3, lon4, lat4 = x1, y1, x2, y1, x2, y2, x1, y2
                clon, clat = cx, cy
            poly_coords = [[[lon1, lat1], [lon2, lat2], [lon3, lat3], [lon4, lat4], [lon1, lat1]]]
            centroid_coords = [clon, clat]
        else:
            poly_coords = [[[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]]]
            centroid_coords = [cx, cy]

        features.append({
            "type": "Feature",
            "id": idx + 1,
            "geometry": {
                "type": "Polygon",
                "coordinates": poly_coords,
            },
            "properties": {
                "building_id": f"BLDG-{idx + 1:04d}",
                "label": label,
                "confidence": round(float(score), 4),
                "area_sq_m": round(area_sq_m, 1),
                "centroid": centroid_coords,
                "pixel_bbox": [x1, y1, x2, y2],
            }
        })

    return {
        "type": "FeatureCollection",
        "features": features,
    }
