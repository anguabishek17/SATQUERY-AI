"""
Natural-language region-of-interest construction — "analyze the area within
500m of the river". Combines a grounded region (from GroundingTool) with a
metric buffer to produce an AOI polygon that downstream tools (change
detection, SAR fusion) can be scoped to.

--- Wiring in real geometry (stretch goal, Day 3-4) ---
    from shapely.geometry import box
    from shapely.ops import transform
    import pyproj

    # 1. Convert the grounded pixel bbox to a geographic polygon using the
    #    image's affine transform (rasterio dataset.transform).
    # 2. Reproject to a metric CRS (UTM zone for the scene) before buffering
    #    — buffering in degrees gives you a non-uniform, wrong-sized buffer.
    # 3. Buffer by the requested distance, reproject back to the display CRS.
"""
from dataclasses import dataclass


@dataclass
class AOIResult:
    source_bbox: list[float]     # the grounded region this AOI is built from
    buffer_meters: float
    aoi_bbox_pixel: list[float]  # stub: naive pixel-space expansion until real geo transform is wired in


def build_aoi(source_bbox: list[float], buffer_meters: float, meters_per_pixel: float = 10.0) -> AOIResult:
    """
    Stub implementation: expands the bbox by buffer_meters converted to pixels
    using an assumed ground sample distance. Replace meters_per_pixel with the
    real GSD read from the GeoTIFF's affine transform, and replace the pixel
    expansion with a proper geographic buffer (see module docstring) once
    shapely/pyproj are wired in.
    """
    px = buffer_meters / meters_per_pixel
    x1, y1, x2, y2 = source_bbox
    expanded = [max(0, x1 - px), max(0, y1 - px), x2 + px, y2 + px]
    return AOIResult(source_bbox=source_bbox, buffer_meters=buffer_meters, aoi_bbox_pixel=expanded)
