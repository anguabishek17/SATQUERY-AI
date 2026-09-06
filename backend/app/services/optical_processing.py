import numpy as np


def normalize_band(band: np.ndarray) -> np.ndarray:
    """Normalize a satellite band to 0-1."""
    band = band.astype(np.float32)

    minimum = np.nanpercentile(band, 2)
    maximum = np.nanpercentile(band, 98)

    if maximum <= minimum:
        return np.zeros_like(band, dtype=np.float32)

    result = (band - minimum) / (maximum - minimum)

    return np.clip(result, 0.0, 1.0)


def ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    Normalized Difference Vegetation Index.

    Higher values generally indicate vegetation.
    """
    red = red.astype(np.float32)
    nir = nir.astype(np.float32)

    return (nir - red) / (nir + red + 1e-8)


def ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    Normalized Difference Water Index.

    Positive values can indicate water.
    """
    green = green.astype(np.float32)
    nir = nir.astype(np.float32)

    return (green - nir) / (green + nir + 1e-8)


def ndbi(swir: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    Normalized Difference Built-up Index.

    Higher values can indicate built-up surfaces.
    """
    swir = swir.astype(np.float32)
    nir = nir.astype(np.float32)

    return (swir - nir) / (swir + nir + 1e-8)


def create_optical_masks(
    red: np.ndarray,
    green: np.ndarray,
    nir: np.ndarray,
    swir: np.ndarray | None = None,
) -> dict:
    """
    Create basic optical masks for vegetation, water and built-up areas.
    """

    vegetation_index = ndvi(red, nir)
    water_index = ndwi(green, nir)

    vegetation_mask = vegetation_index > 0.3
    water_mask = water_index > 0.0

    result = {
        "ndvi": vegetation_index,
        "ndwi": water_index,
        "vegetation_mask": vegetation_mask,
        "water_mask": water_mask,
    }

    if swir is not None:
        builtup_index = ndbi(swir, nir)
        builtup_mask = builtup_index > 0.0

        result["ndbi"] = builtup_index
        result["builtup_mask"] = builtup_mask

    return result


def optical_statistics(masks: dict) -> dict:
    """Calculate simple scene-level percentages."""

    statistics = {}

    if "vegetation_mask" in masks:
        mask = masks["vegetation_mask"]
        statistics["vegetation_pct"] = float(
            100.0 * np.mean(mask)
        )

    if "water_mask" in masks:
        mask = masks["water_mask"]
        statistics["water_pct"] = float(
            100.0 * np.mean(mask)
        )

    if "builtup_mask" in masks:
        mask = masks["builtup_mask"]
        statistics["builtup_pct"] = float(
            100.0 * np.mean(mask)
        )

    return statistics