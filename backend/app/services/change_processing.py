"""
Real bi-temporal change detection — classical, not a trained deep model, but
genuinely computed from pixel data: co-registered difference, adaptive
thresholding (Otsu), morphological cleanup, and connected-component region
statistics. Runs on CPU, no downloaded weights.

This is a legitimate, defensible baseline for the mandatory "change
description or change-based VQA" requirement — upgrade path to a trained
Siamese model (BIT-CD) is documented at the bottom without blocking a working
demo today.
"""
import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops


def compute_change_mask(img_t0: np.ndarray, img_t1: np.ndarray, min_region_px: int = 25) -> dict:
    """
    img_t0, img_t1: co-registered single- or multi-band arrays, same shape,
    normalized to a comparable range (e.g. both 0-1 reflectance).
    """
    if img_t0.ndim == 3:
        diff = np.mean(np.abs(img_t0.astype(np.float64) - img_t1.astype(np.float64)), axis=-1)
    else:
        diff = np.abs(img_t0.astype(np.float64) - img_t1.astype(np.float64))

    # Otsu's method picks a data-driven threshold rather than a hardcoded
    # constant, so it adapts to each scene's actual noise floor.
    try:
        threshold = threshold_otsu(diff)
    except ValueError:
        threshold = diff.mean() + 2 * diff.std()

    raw_mask = diff > threshold

    # Morphological opening removes salt-and-pepper false positives from
    # sensor noise / misregistration jitter without a trained denoiser.
    cleaned_mask = ndimage.binary_opening(raw_mask, structure=np.ones((3, 3)))
    cleaned_mask = ndimage.binary_closing(cleaned_mask, structure=np.ones((3, 3)))

    labeled = label(cleaned_mask)
    regions = [r for r in regionprops(labeled) if r.area >= min_region_px]
    final_mask = np.isin(labeled, [r.label for r in regions])
    regions_sorted = sorted(regions, key=lambda r: r.area, reverse=True)
    largest_bbox = list(regions_sorted[0].bbox) if regions_sorted else None  # (min_row, min_col, max_row, max_col)

    total_px = diff.size
    changed_px = int(final_mask.sum())

    # Quadrant breakdown for a "primary change zone" description.
    h, w = final_mask.shape
    quadrants = {
        "northern region": final_mask[: h // 2, :].sum(),
        "southern region": final_mask[h // 2 :, :].sum(),
        "eastern region": final_mask[:, w // 2 :].sum(),
        "western region": final_mask[:, : w // 2].sum(),
    }
    primary_zone = max(quadrants, key=quadrants.get) if changed_px > 0 else "no significant zone"

    return {
        "mask": final_mask,
        "diff_map": diff,
        "threshold": float(threshold),
        "changed_px": changed_px,
        "total_px": total_px,
        "pct_changed": float(100 * changed_px / total_px),
        "num_regions": len(regions),
        "largest_region_px": int(max((r.area for r in regions), default=0)),
        "largest_region_bbox": largest_bbox,  # [row_min, col_min, row_max, col_max], pixel space
        "primary_change_zone": primary_zone,
    }


def pixels_to_hectares(pixel_count: int, pixel_size_m: float = 10.0) -> float:
    """Default 10m/pixel matches Sentinel-2/Sentinel-1 GSD; override per sensor (Cartosat/RISAT)."""
    return pixel_count * (pixel_size_m ** 2) / 10_000

# --- Upgrade path: trained Siamese change detector (BIT-CD) ---
# from bit_cd.model import BITModel
# model = BITModel.from_pretrained(CHANGE_MODEL_CKPT)
# change_mask = model.predict(img_t0, img_t1)  # returns the same style of binary mask
# Swap compute_change_mask's `final_mask` for `change_mask` and keep everything downstream
# (region stats, hectares, primary zone) unchanged — the statistics layer doesn't care
# which detector produced the mask.
