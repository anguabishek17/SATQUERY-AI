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
    
    # Critical consistency fix: If cleaned_mask has changed pixels but all were < min_region_px,
    # do not report 0 regions. Include the top labeled components.
    if not regions and cleaned_mask.sum() > 0:
        all_regions = regionprops(labeled)
        if all_regions:
            regions = sorted(all_regions, key=lambda r: r.area, reverse=True)[:3]

    final_mask = np.isin(labeled, [r.label for r in regions]) if regions else cleaned_mask
    regions_sorted = sorted(regions, key=lambda r: r.area, reverse=True)
    largest_bbox = [int(v) for v in regions_sorted[0].bbox] if regions_sorted else None  # (min_row, min_col, max_row, max_col)

    total_px = diff.size
    changed_px = int(final_mask.sum())
    pct_changed = float(100 * changed_px / total_px) if total_px > 0 else 0.0

    # Sector / Quadrant breakdown for spatial distribution
    h, w = final_mask.shape
    h_mid, w_mid = h // 2, w // 2
    h_third, w_third = h // 3, w // 3

    north_px = int(final_mask[:h_mid, :].sum())
    south_px = int(final_mask[h_mid:, :].sum())
    west_px = int(final_mask[:, :w_mid].sum())
    east_px = int(final_mask[:, w_mid:].sum())
    central_px = int(final_mask[h_third : 2 * h_third, w_third : 2 * w_third].sum())

    denom = max(changed_px, 1)
    spatial_distribution = {
        "north": f"{round(100 * north_px / denom, 1)}%",
        "south": f"{round(100 * south_px / denom, 1)}%",
        "east": f"{round(100 * east_px / denom, 1)}%",
        "west": f"{round(100 * west_px / denom, 1)}%",
        "central": f"{round(100 * central_px / denom, 1)}%",
    }

    quadrants = {
        "northern region": north_px,
        "southern region": south_px,
        "eastern region": east_px,
        "western region": west_px,
        "central sector": central_px,
    }
    primary_zone = max(quadrants, key=quadrants.get) if changed_px > 0 else "no significant zone"

    # Extract structured change regions (Section 1 schema)
    change_regions_list = []
    largest_region_pct = 0.0
    for idx, r in enumerate(regions_sorted[:10], start=1):
        r_area_pct = round(100 * r.area / total_px, 3)
        if idx == 1:
            largest_region_pct = r_area_pct
        r_ha = round(pixels_to_hectares(r.area), 2)
        change_regions_list.append({
            "region_id": idx,
            "bbox": [int(v) for v in r.bbox],
            "centroid": [round(float(c), 1) for c in r.centroid],
            "area_percent": f"{r_area_pct}%",
            "area_hectares": r_ha,
        })

    # Change intensity assessment
    if pct_changed < 1.0:
        change_intensity = "low"
    elif pct_changed < 5.0:
        change_intensity = "moderate"
    else:
        change_intensity = "high"

    return {
        "mask": final_mask,
        "diff_map": diff,
        "threshold": float(threshold),
        "changed_px": changed_px,
        "total_px": total_px,
        "pct_changed": pct_changed,
        "num_regions": len(regions),
        "largest_region_px": int(max((r.area for r in regions), default=0)),
        "largest_region_bbox": largest_bbox,  # [row_min, col_min, row_max, col_max], pixel space
        "largest_change_region_percent": f"{largest_region_pct}%",
        "change_intensity": change_intensity,
        "spatial_distribution": spatial_distribution,
        "change_regions": change_regions_list,
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
