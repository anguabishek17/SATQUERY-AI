"""
Real classical remote-sensing signal processing — no model weights required,
runs on CPU. Lee-filter despeckling, spectral indices (NDVI/NDWI/NDBI), and
calibrated backscatter-based water/built-up thresholding.

References:
  - Lee, J.S. (1980). Digital image enhancement and noise filtering by use
    of local statistics. IEEE PAMI.
  - McFeeters, S.K. (1996). NDWI. Int. J. Remote Sensing.
  - Zha et al. (2003). NDBI. Int. J. Remote Sensing.
"""
from typing import Optional
import numpy as np
from scipy.ndimage import uniform_filter

from app.config import (
    NDVI_THRESHOLD,
    NDWI_THRESHOLD,
    NDBI_THRESHOLD,
    SAR_WATER_THRESHOLD,
    SAR_BUILTUP_PERCENTILE,
    SAR_TEXTURE_THRESHOLD,
)


def lee_filter(img: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Adaptive speckle filter for SAR imagery. Smooths homogeneous regions
    while preserving edges, using local-statistics-based weighting.
    """
    img = img.astype(np.float64)
    img_mean = uniform_filter(img, size=window)
    img_sqr_mean = uniform_filter(img ** 2, size=window)
    img_var = np.maximum(img_sqr_mean - img_mean ** 2, 0)

    overall_var = np.var(img)
    weights = img_var / (img_var + overall_var + 1e-9)
    return img_mean + weights * (img - img_mean)


def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Normalized Difference Vegetation Index. Positive values -> vegetation."""
    nir, red = nir.astype(np.float64), red.astype(np.float64)
    return (nir - red) / (nir + red + 1e-9)


def ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Normalized Difference Water Index. Positive values -> water."""
    green, nir = green.astype(np.float64), nir.astype(np.float64)
    return (green - nir) / (green + nir + 1e-9)


def ndbi(swir: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Normalized Difference Built-up Index. Positive values -> built-up."""
    swir, nir = swir.astype(np.float64), nir.astype(np.float64)
    return (swir - nir) / (swir + nir + 1e-9)


def sar_water_mask(vv_db: np.ndarray, threshold_db: float = SAR_WATER_THRESHOLD) -> np.ndarray:
    """Water surfaces are smooth and specular -> low VV backscatter in dB."""
    return vv_db < threshold_db


def sar_builtup_mask(
    vv_db: np.ndarray,
    window: int = 7,
    percentile: float = SAR_BUILTUP_PERCENTILE,
    texture_threshold: float = SAR_TEXTURE_THRESHOLD,
) -> np.ndarray:
    """
    Built-up surfaces are rough and corner reflectors -> high VV backscatter AND high local texture.
    """
    local_mean = uniform_filter(vv_db, size=window)
    local_sqr_mean = uniform_filter(vv_db ** 2, size=window)
    local_var = np.maximum(local_sqr_mean - local_mean ** 2, 0)
    high_backscatter = vv_db > np.percentile(vv_db, percentile)
    high_texture = local_var > texture_threshold
    return high_backscatter & high_texture


from PIL import Image


def fuse_optical_sar(
    green: np.ndarray,
    nir: np.ndarray,
    swir: Optional[np.ndarray],
    vv_db: np.ndarray,
    red: Optional[np.ndarray] = None,
) -> dict:
    """
    Runs full Optical + SAR fusion pipeline using calibrated configurable thresholds.
    """
    ref_h, ref_w = green.shape[:2]

    def _align(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if arr is None:
            return None
        if arr.shape[:2] == (ref_h, ref_w):
            return arr
        img = Image.fromarray(arr.astype(np.float32), mode="F").resize((ref_w, ref_h), Image.BILINEAR)
        return np.array(img, dtype=np.float64)

    nir = _align(nir)
    if swir is not None:
        swir = _align(swir)
    if red is not None:
        red = _align(red)
    vv_db = _align(vv_db)

    vv_filtered = lee_filter(vv_db)

    water_optical = ndwi(green, nir) > NDWI_THRESHOLD
    water_sar = sar_water_mask(vv_filtered, threshold_db=SAR_WATER_THRESHOLD)
    water_agree = water_optical & water_sar

    builtup_sar = sar_builtup_mask(
        vv_filtered, percentile=SAR_BUILTUP_PERCENTILE, texture_threshold=SAR_TEXTURE_THRESHOLD
    )
    builtup_optical = ndbi(swir, nir) > NDBI_THRESHOLD if swir is not None else np.zeros_like(nir, dtype=bool)
    builtup_agree = builtup_sar & builtup_optical if swir is not None else builtup_sar

    veg_optical = ndvi(nir, red) > NDVI_THRESHOLD if red is not None else np.zeros_like(nir, dtype=bool)

    total_px = max(1, green.size)
    return {
        "water_pct": float(100 * water_agree.sum() / total_px),
        "water_optical_only_pct": float(100 * (water_optical & ~water_sar).sum() / total_px),
        "water_sar_only_pct": float(100 * (water_sar & ~water_optical).sum() / total_px),
        "builtup_pct": float(100 * builtup_agree.sum() / total_px),
        "vegetation_pct": float(100 * veg_optical.sum() / total_px) if red is not None else 0.0,
        "masks": {
            "water_optical": water_optical,
            "water_sar": water_sar,
            "water_agree": water_agree,
            "builtup_sar": builtup_sar,
            "builtup_agree": builtup_agree,
            "veg_optical": veg_optical,
        },
    }
