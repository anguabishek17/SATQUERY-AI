"""
GeoTIFF/TIFF read helpers. Keep all rasterio/GDAL band math here so tool
modules stay focused on model logic, not I/O plumbing.
"""
from pathlib import Path

from app.config import UPLOAD_DIR


def saved_path(file_id: str) -> Path:
    matches = list(UPLOAD_DIR.glob(f"{file_id}.*"))
    if not matches:
        raise FileNotFoundError(f"No uploaded file found for id {file_id}")
    return matches[0]


def read_preview_png(file_id: str) -> bytes:
    """
    Returns a browser-viewable PNG preview of an uploaded GeoTIFF/TIFF for the
    dashboard's image viewer. For real GeoTIFFs:

        import rasterio
        from rasterio.plot import reshape_as_image
        import numpy as np
        from PIL import Image
        with rasterio.open(path) as src:
            arr = src.read([1, 2, 3])  # adjust band order per sensor
            arr = (arr / arr.max(axis=(1, 2), keepdims=True) * 255).astype("uint8")
            img = Image.fromarray(reshape_as_image(arr))
            buf = io.BytesIO(); img.save(buf, format="PNG")
            return buf.getvalue()

    PNG/JPEG benchmark uploads can just be read directly.
    """
    path = saved_path(file_id)
    return path.read_bytes()
