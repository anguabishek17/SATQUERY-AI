import cv2
import numpy as np
import base64
import time
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURABLE THRESHOLDS & CONSTANTS
# =============================================================================
ROAD_LIKELIHOOD_SUPPRESS_THRESH = 0.70  # Road likelihood >= 0.70 -> suppress candidate
ROAD_LIKELIHOOD_REDUCE_THRESH   = 0.50  # Road likelihood >= 0.50 -> strongly reduce water confidence
ROAD_PENALTY_WEIGHT             = 0.50  # Penalty applied to final_water_score based on road_likelihood
WATER_ACCEPT_THRESH_HIGH        = 0.75  # High confidence water score cutoff
WATER_ACCEPT_THRESH_MOD         = 0.55  # Moderate confidence water score cutoff (minimum acceptance)
MIN_CANDIDATE_AREA              = 40    # Ignore tiny pixel noise below 40px


def _calculate_width_consistency(roi_mask: np.ndarray) -> float:
    """
    Calculate width consistency using L2 distance transform on local ROI.
    Returns score in [0.0, 1.0] where 1.0 indicates high width uniformity (road-like)
    and 0.0 indicates variable width (natural lake/river expansion).
    """
    dist = cv2.distanceTransform(roi_mask, cv2.DIST_L2, 5)
    non_zero = dist[dist > 0]
    if len(non_zero) < 10:
        return 0.5
    thresh = np.max(non_zero) * 0.35
    core_dists = dist[dist >= thresh]
    if len(core_dists) < 5:
        return 0.5
    mean_val = float(np.mean(core_dists))
    std_val = float(np.std(core_dists))
    cv_val = std_val / max(mean_val, 1e-5)
    score = max(0.0, 1.0 - cv_val * 2.0)
    return float(np.clip(score, 0.0, 1.0))


def _calculate_skeleton_junctions(roi_mask: np.ndarray) -> tuple[float, int]:
    """
    Morphological skeletonization on local ROI to count endpoints and junction branch points.
    Returns (linearity_score [0,1], junction_count).
    """
    h, w = roi_mask.shape
    # Maximum iterations bounded by local ROI dimension
    max_iters = max(3, min(18, min(h, w) // 2))
    skel = np.zeros_like(roi_mask)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    temp = roi_mask.copy()
    
    for _ in range(max_iters):
        eroded = cv2.erode(temp, element)
        opened = cv2.dilate(eroded, element)
        sub = cv2.subtract(temp, opened)
        skel = cv2.bitwise_or(skel, sub)
        temp = eroded
        if cv2.countNonZero(temp) == 0:
            break
            
    skel_pixels = np.argwhere(skel > 0)
    if len(skel_pixels) < 5:
        return 0.5, 0

    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    skel_bin = (skel > 0).astype(np.uint8)
    neighbor_count = cv2.filter2D(skel_bin, -1, kernel) * skel_bin

    junctions = int(np.sum(neighbor_count >= 3))

    total_skel_length = len(skel_pixels)
    ep_pts = skel_pixels[neighbor_count[skel_pixels[:, 0], skel_pixels[:, 1]] == 1]
    if len(ep_pts) >= 2:
        max_dist = 0.0
        n_pts = len(ep_pts)
        for idx1 in range(min(5, n_pts)):
            for idx2 in range(idx1 + 1, min(10, n_pts)):
                d = float(np.linalg.norm(ep_pts[idx1] - ep_pts[idx2]))
                if d > max_dist:
                    max_dist = d
        linearity = max_dist / max(total_skel_length, 1.0)
    else:
        linearity = 0.5

    return float(np.clip(linearity, 0.0, 1.0)), junctions


def _calculate_edge_structure(roi_gray: np.ndarray, roi_mask: np.ndarray) -> float:
    """
    Calculate Sobel/Canny edge density along region boundary on local ROI.
    Returns score in [0.0, 1.0].
    """
    edges = cv2.Canny(roi_gray, 40, 120)
    mask_dilated = cv2.dilate(roi_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    boundary_mask = cv2.subtract(mask_dilated, cv2.erode(roi_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))))
    
    boundary_pixels = cv2.countNonZero(boundary_mask)
    if boundary_pixels == 0:
        return 0.0
        
    edge_in_boundary = cv2.countNonZero(cv2.bitwise_and(edges, boundary_mask))
    edge_density = edge_in_boundary / boundary_pixels
    return float(np.clip(edge_density * 2.0, 0.0, 1.0))


def _calculate_texture_variance(roi_gray: np.ndarray, roi_mask: np.ndarray) -> float:
    """
    Calculate pixel intensity standard deviation inside the component on local ROI.
    Smooth water has low variance (< 10.0), while asphalt/roof/shadow texture is higher.
    Returns score in [0.0, 1.0].
    """
    pixels = roi_gray[roi_mask > 0]
    if len(pixels) < 5:
        return 0.0
    std_val = float(np.std(pixels))
    score = (std_val - 6.0) / 20.0
    return float(np.clip(score, 0.0, 1.0))


def calculate_road_likelihood(
    area: int,
    width: int,
    height: int,
    region_mask: np.ndarray,
    img_bgr: np.ndarray,
    hsv: np.ndarray,
    img_gray: np.ndarray,
    is_adjacent_to_road: bool = False
) -> tuple[float, str]:
    """
    Compute multi-feature road likelihood score in [0.0, 1.0] and return (score, primary_reason).
    Maintains compatibility with tests and callers that pass full image arrays.
    """
    # Auto-crop to local ROI if region_mask is larger than needed
    pts = np.argwhere(region_mask > 0)
    if len(pts) == 0:
        return 0.0, "empty region"
        
    y_min, x_min = pts.min(axis=0)
    y_max, x_max = pts.max(axis=0) + 1
    
    pad = 6
    y0, x0 = max(0, y_min - pad), max(0, x_min - pad)
    y1, x1 = min(region_mask.shape[0], y_max + pad), min(region_mask.shape[1], x_max + pad)
    
    roi_mask = region_mask[y0:y1, x0:x1]
    roi_gray = img_gray[y0:y1, x0:x1]
    roi_hsv = hsv[y0:y1, x0:x1]
    
    # 1. Elongation
    pts_roi = np.argwhere(roi_mask > 0)
    if len(pts_roi) >= 5:
        pts_xy = np.column_stack((pts_roi[:, 1], pts_roi[:, 0])).astype(np.float32)
        rect = cv2.minAreaRect(pts_xy)
        (_, _), (w_rot, h_rot), _ = rect
        aspect_ratio = max(w_rot, h_rot) / max(min(w_rot, h_rot), 1.0)
    else:
        aspect_ratio = max(width, height) / max(min(width, height), 1.0)

    f_elongation = float(np.clip((aspect_ratio - 1.5) / 4.0, 0.0, 1.0))

    # 2. Width Consistency
    f_width_consistency = _calculate_width_consistency(roi_mask)

    # 3. Linearity & Junctions
    linearity, junctions = _calculate_skeleton_junctions(roi_mask)
    if junctions > 0:
        f_linearity_junctions = min(1.0, 0.5 + junctions * 0.25)
    else:
        f_linearity_junctions = float(np.clip(linearity, 0.0, 1.0))

    # 4. Edge Structure
    f_edge_structure = _calculate_edge_structure(roi_gray, roi_mask)

    # 5. Texture Variance
    f_texture = _calculate_texture_variance(roi_gray, roi_mask)

    # 6. Saturation & Hue
    region_hsv = roi_hsv[roi_mask > 0]
    if len(region_hsv) > 0:
        mean_sat = float(np.mean(region_hsv[:, 1]))
        mean_val = float(np.mean(region_hsv[:, 2]))
        mean_hue = float(np.mean(region_hsv[:, 0]))
    else:
        mean_sat, mean_val, mean_hue = 50.0, 100.0, 110.0

    if mean_sat < 40:
        f_color_sat = min(1.0, (40.0 - mean_sat) / 30.0 + 0.3)
    else:
        f_color_sat = 0.0

    if mean_sat < 30 and mean_val > 130:
        f_color_sat = min(1.0, f_color_sat + 0.3)

    # 7. Connectivity Network
    f_connectivity = 0.30 if is_adjacent_to_road else 0.0

    # Multi-feature weighted road-likelihood score calculation (weights sum to 1.00)
    road_likelihood = (
        0.22 * f_elongation +
        0.18 * f_width_consistency +
        0.16 * f_linearity_junctions +
        0.18 * f_color_sat +
        0.12 * f_edge_structure +
        0.09 * f_texture +
        0.05 * f_connectivity
    )

    # Multi-feature agreement dampening for genuine water (e.g. rivers/canals):
    if mean_sat > 45 and 95 <= mean_hue <= 130 and f_texture < 0.35:
        road_likelihood *= 0.25

    road_likelihood = float(np.clip(road_likelihood, 0.0, 1.0))

    reasons = []
    if f_elongation > 0.6: reasons.append("elongated spatial ratio")
    if f_width_consistency > 0.6: reasons.append("uniform width profile")
    if f_linearity_junctions > 0.6: reasons.append("linear corridor or junction structure")
    if f_color_sat > 0.5: reasons.append("low-saturation grey/asphalt color")
    if f_edge_structure > 0.5: reasons.append("parallel boundary edge density")
    if f_texture > 0.5: reasons.append("high surface texture variance")
    if junctions > 0: reasons.append("intersection junction detected")
    if is_adjacent_to_road: reasons.append("connected road network candidate")

    primary_reason = ", ".join(reasons) if reasons else "general spectral/spatial non-water heuristic"
    return road_likelihood, primary_reason


def calculate_water_validation_score(
    area: int,
    region_mask: np.ndarray,
    hsv: np.ndarray,
    img_gray: np.ndarray,
    road_likelihood: float,
    precalc_stats: dict | None = None
) -> tuple[float, str]:
    """
    Calculate water region validation score in [0.0, 1.0] and return (final_score, confidence).
    """
    if precalc_stats is not None:
        mean_hue = precalc_stats.get("mean_hue", 110.0)
        mean_sat = precalc_stats.get("mean_sat", 50.0)
        mean_val = precalc_stats.get("mean_val", 100.0)
        std_val = precalc_stats.get("std_val", 0.0)
    else:
        region_hsv = hsv[region_mask > 0]
        if len(region_hsv) == 0:
            return 0.0, "Rejected"

        mean_hue = float(np.mean(region_hsv[:, 0]))
        mean_sat = float(np.mean(region_hsv[:, 1]))
        mean_val = float(np.mean(region_hsv[:, 2]))
        pixels = img_gray[region_mask > 0]
        std_val = float(np.std(pixels)) if len(pixels) > 0 else 0.0

    base_score = 0.50

    # Vibrant blue water alignment: H in [95, 130], S >= 45
    if 95 <= mean_hue <= 130 and mean_sat >= 45:
        base_score += 0.35
    elif mean_val < 35 and mean_sat >= 30:  # Dark water with blue/green hue
        base_score += 0.15

    # Smoothness boost (water is homogeneous)
    if std_val < 10.0:
        base_score += 0.15
    elif std_val < 15.0:
        base_score += 0.08

    # Size boost for large contiguous water bodies
    if area > 1000:
        base_score += 0.10

    # Low saturation penalty: neutral desaturated grey/black (S < 35) is non-water
    if mean_sat < 35:
        base_score -= 0.15

    # Apply Road Penalty
    final_score = base_score - (road_likelihood * ROAD_PENALTY_WEIGHT)

    # Road Suppression Threshold rules:
    if road_likelihood >= ROAD_LIKELIHOOD_SUPPRESS_THRESH:
        final_score = min(final_score, 0.35)  # Force rejection
    elif road_likelihood >= ROAD_LIKELIHOOD_REDUCE_THRESH:
        final_score = min(final_score, 0.50)  # Cap at moderate/low

    final_score = float(np.clip(final_score, 0.0, 1.0))

    if final_score >= WATER_ACCEPT_THRESH_HIGH:
        confidence = "High"
    elif final_score >= WATER_ACCEPT_THRESH_MOD:
        confidence = "Moderate"
    else:
        confidence = "Rejected"

    return final_score, confidence


def rgb_landcover_estimation(image_path: str, aoi_bbox=None) -> dict:
    """
    Optimized fallback land-cover estimation for 3-band RGB imagery using HSV color heuristics
    and staged cheap-first local-ROI road likelihood suppression with image-level caching.
    """
    from app.services.analysis_cache import analysis_cache
    
    # 1. Check Image Preprocessing Cache
    t_start_total = time.perf_counter()
    cached_img = analysis_cache.get_preprocessed_image(image_path, aoi_bbox)
    if cached_img is not None:
        img = cached_img["img"]
        hsv = cached_img["hsv"]
        img_gray = cached_img["gray"]
    else:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image for RGB analysis: {image_path}")

        if aoi_bbox:
            ax1, ay1, ax2, ay2 = [max(0, int(v)) for v in aoi_bbox]
            ax2 = min(ax2, img.shape[1])
            ay2 = min(ay2, img.shape[0])
            img = img[ay1:ay2, ax1:ax2]

        if img.size == 0:
            raise ValueError("AOI resulted in empty image crop.")

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        analysis_cache.cache_preprocessed_image(image_path, img, hsv, img_gray, aoi_bbox=aoi_bbox)
    
    img_h, img_w = img.shape[:2]

    # Vegetation: Greens
    lower_veg = np.array([35, 40, 40])
    upper_veg = np.array([85, 255, 255])
    mask_veg = cv2.inRange(hsv, lower_veg, upper_veg)

    # -------------------------------------------------------------------------
    # MULTI-FEATURE WATER DETECTION WITH LOCAL-ROI ROAD SUPPRESSION
    # -------------------------------------------------------------------------
    t_w0 = time.perf_counter()
    # 1. Base color mask (relaxed)
    lower_water = np.array([90, 40, 20])
    upper_water = np.array([130, 255, 255])
    mask_water_blue = cv2.inRange(hsv, lower_water, upper_water)
    
    # Dark water
    lower_water_dark = np.array([0, 0, 0])
    upper_water_dark = np.array([180, 255, 30])
    mask_water_dark = cv2.inRange(hsv, lower_water_dark, upper_water_dark)
    
    base_water_mask = cv2.bitwise_or(mask_water_blue, mask_water_dark)

    # 2. Connected Component Analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(base_water_mask, connectivity=8)
    t_cand_extract = (time.perf_counter() - t_w0) * 1000

    debug_info = {
        "candidate_regions": num_labels - 1,
        "road_like_candidates": 0,
        "suppressed_road_candidates": 0,
        "accepted_water_regions": 0,
        "road_likelihoods": [],
        "water_confidences": [],
        "suppression_reasons": [],
        "note": "RGB-based water estimate after spatial and road-structure suppression."
    }

    # Profiling accumulators
    t_cheap_acc = 0.0
    t_width_acc = 0.0
    t_skel_acc = 0.0
    t_edge_tex_acc = 0.0
    t_conn_acc = 0.0
    t_score_acc = 0.0

    cheap_rejected = 0
    moderate_count = 0
    expensive_count = 0

    pad = 6
    candidates = []

    # -------------------------------------------------------------------------
    # PASS 1: LOCAL ROI FEATURE EXTRACTION WITH CHEAP-FIRST CASCADE
    # -------------------------------------------------------------------------
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < MIN_CANDIDATE_AREA:
            continue

        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        width = int(stats[i, cv2.CC_STAT_WIDTH])
        height = int(stats[i, cv2.CC_STAT_HEIGHT])

        # Bounding box with margin
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(img_w, x + width + pad), min(img_h, y + height + pad)

        # Local ROI crops (orders of magnitude faster than full image canvas)
        roi_mask = (labels[y0:y1, x0:x1] == i).astype(np.uint8) * 255
        roi_gray = img_gray[y0:y1, x0:x1]
        roi_hsv = hsv[y0:y1, x0:x1]

        # --- STAGE 1: CHEAP FEATURES ---
        t0_sub = time.perf_counter()
        pts_local = np.argwhere(roi_mask > 0)
        if len(pts_local) >= 5:
            pts_xy = np.column_stack((pts_local[:, 1], pts_local[:, 0])).astype(np.float32)
            rect = cv2.minAreaRect(pts_xy)
            (_, _), (w_rot, h_rot), _ = rect
            aspect_ratio = max(w_rot, h_rot) / max(min(w_rot, h_rot), 1.0)
        else:
            aspect_ratio = max(width, height) / max(min(width, height), 1.0)

        f_elongation = float(np.clip((aspect_ratio - 1.5) / 4.0, 0.0, 1.0))

        # Color & Saturation on ROI
        region_hsv = roi_hsv[roi_mask > 0]
        if len(region_hsv) > 0:
            mean_sat = float(np.mean(region_hsv[:, 1]))
            mean_val = float(np.mean(region_hsv[:, 2]))
            mean_hue = float(np.mean(region_hsv[:, 0]))
        else:
            mean_sat, mean_val, mean_hue = 50.0, 100.0, 110.0

        if mean_sat < 40:
            f_color_sat = min(1.0, (40.0 - mean_sat) / 30.0 + 0.3)
        else:
            f_color_sat = 0.0

        if mean_sat < 30 and mean_val > 130:
            f_color_sat = min(1.0, f_color_sat + 0.3)

        t_cheap_acc += (time.perf_counter() - t0_sub) * 1000

        # Cheap cascade filter:
        # Obvious non-road: compact shape (low elongation) and normal saturation
        is_obvious_non_road = (f_elongation < 0.15 and mean_sat > 35) or (f_elongation < 0.10 and area > 500)
        
        # --- STAGE 2: MODERATE FEATURES ---
        t0_sub = time.perf_counter()
        f_texture = _calculate_texture_variance(roi_gray, roi_mask)
        t_edge_tex_acc += (time.perf_counter() - t0_sub) * 1000

        if is_obvious_non_road:
            cheap_rejected += 1
            f_width_consistency = 0.10
            f_linearity_junctions = 0.10
            f_edge_structure = 0.10
            junctions = 0
        else:
            moderate_count += 1
            # Width consistency on local ROI
            t0_sub = time.perf_counter()
            f_width_consistency = _calculate_width_consistency(roi_mask)
            t_width_acc += (time.perf_counter() - t0_sub) * 1000

            # Edge structure on local ROI
            t0_sub = time.perf_counter()
            f_edge_structure = _calculate_edge_structure(roi_gray, roi_mask)
            t_edge_tex_acc += (time.perf_counter() - t0_sub) * 1000

            # --- STAGE 3: EXPENSIVE FEATURES (Skeleton / Junctions) ---
            # Only run skeletonization if candidate is elongated or low-saturation grey
            needs_skeleton = (f_elongation > 0.25 or f_color_sat > 0.35 or f_width_consistency > 0.40)
            if needs_skeleton:
                expensive_count += 1
                t0_sub = time.perf_counter()
                linearity, junctions = _calculate_skeleton_junctions(roi_mask)
                t_skel_acc += (time.perf_counter() - t0_sub) * 1000
                if junctions > 0:
                    f_linearity_junctions = min(1.0, 0.5 + junctions * 0.25)
                else:
                    f_linearity_junctions = float(np.clip(linearity, 0.0, 1.0))
            else:
                f_linearity_junctions = 0.20
                junctions = 0

        # Base 6-feature road likelihood
        t0_sub = time.perf_counter()
        base_road_lk = (
            0.22 * f_elongation +
            0.18 * f_width_consistency +
            0.16 * f_linearity_junctions +
            0.18 * f_color_sat +
            0.12 * f_edge_structure +
            0.09 * f_texture
        )

        # Multi-feature agreement dampening for genuine water
        is_blue_water = (mean_sat > 45 and 95 <= mean_hue <= 130 and f_texture < 0.35)
        
        candidates.append({
            "label": i,
            "area": area,
            "bbox": (x, y, width, height),
            "bbox_padded": (x0, y0, x1, y1),
            "roi_crop": (x0, y0, x1, y1),
            "roi_mask": roi_mask,
            "roi_gray": roi_gray,
            "roi_hsv": roi_hsv,
            "base_road_lk": base_road_lk,
            "is_blue_water": is_blue_water,
            "f_elongation": f_elongation,
            "f_width_consistency": f_width_consistency,
            "f_linearity_junctions": f_linearity_junctions,
            "f_color_sat": f_color_sat,
            "f_edge_structure": f_edge_structure,
            "f_texture": f_texture,
            "junctions": junctions,
            "mean_sat": mean_sat,
            "mean_val": mean_val,
            "mean_hue": mean_hue,
            "std_val": f_texture * 20.0 + 6.0,
        })
        t_score_acc += (time.perf_counter() - t0_sub) * 1000

    # -------------------------------------------------------------------------
    # PASS 2: CHEAP BOUNDING-BOX NETWORK CONNECTIVITY & WATER SCORING
    # -------------------------------------------------------------------------
    t0_conn = time.perf_counter()
    # Road-like candidate bounding boxes (evaluated without re-processing masks)
    road_candidates = [c for c in candidates if (c["base_road_lk"] + 0.0) >= ROAD_LIKELIHOOD_REDUCE_THRESH]
    
    final_water_mask = np.zeros((img_h, img_w), dtype=np.uint8)

    for c in candidates:
        is_adjacent_to_road = False
        if len(road_candidates) > 1:
            bx0, by0, bx1, by1 = c["bbox_padded"]
            for r in road_candidates:
                if r["label"] == c["label"]:
                    continue
                rx0, ry0, rx1, ry1 = r["bbox_padded"]
                # Fast AABB intersection check
                if not (bx1 < rx0 or bx0 > rx1 or by1 < ry0 or by0 > ry1):
                    is_adjacent_to_road = True
                    break

        f_connectivity = 0.30 if is_adjacent_to_road else 0.0
        final_road_lk = c["base_road_lk"] + (0.05 * f_connectivity)
        if c["is_blue_water"]:
            final_road_lk *= 0.25
        final_road_lk = float(np.clip(final_road_lk, 0.0, 1.0))

        # Primary reasons
        reasons = []
        if c["f_elongation"] > 0.6: reasons.append("elongated spatial ratio")
        if c["f_width_consistency"] > 0.6: reasons.append("uniform width profile")
        if c["f_linearity_junctions"] > 0.6: reasons.append("linear corridor or junction structure")
        if c["f_color_sat"] > 0.5: reasons.append("low-saturation grey/asphalt color")
        if c["f_edge_structure"] > 0.5: reasons.append("parallel boundary edge density")
        if c["f_texture"] > 0.5: reasons.append("high surface texture variance")
        if c["junctions"] > 0: reasons.append("intersection junction detected")
        if is_adjacent_to_road: reasons.append("connected road network candidate")
        reason = ", ".join(reasons) if reasons else "general spectral/spatial non-water heuristic"

        precalc = {
            "mean_sat": c["mean_sat"],
            "mean_val": c["mean_val"],
            "mean_hue": c["mean_hue"],
            "std_val": c["std_val"]
        }
        water_score, conf = calculate_water_validation_score(
            c["area"], c["roi_mask"], c["roi_hsv"], c["roi_gray"], final_road_lk, precalc_stats=precalc
        )

        debug_info["road_likelihoods"].append(round(final_road_lk, 3))
        debug_info["water_confidences"].append(round(water_score, 3))

        if final_road_lk >= ROAD_LIKELIHOOD_REDUCE_THRESH:
            debug_info["road_like_candidates"] += 1

        rx0, ry0, rx1, ry1 = c["roi_crop"]
        if conf in ("High", "Moderate"):
            # Paint accepted candidate on final water mask using fast ROI slicing
            final_water_mask[ry0:ry1, rx0:rx1] = cv2.bitwise_or(final_water_mask[ry0:ry1, rx0:rx1], c["roi_mask"])
            debug_info["accepted_water_regions"] += 1
        else:
            debug_info["suppressed_road_candidates"] += 1
            debug_info["suppression_reasons"].append(f"Region area {c['area']}px suppressed (road_likelihood={final_road_lk:.2f}): {reason}")

    t_conn_acc = (time.perf_counter() - t0_conn) * 1000
    t_road_total = t_cand_extract + t_cheap_acc + t_width_acc + t_skel_acc + t_edge_tex_acc + t_conn_acc + t_score_acc

    # Road Suppression Profiling Logs
    perf_road_log = (
        f"[PERF] road_candidate_extraction = {t_cand_extract:.2f} ms\n"
        f"[PERF] road_cheap_features = {t_cheap_acc:.2f} ms\n"
        f"[PERF] road_width_consistency = {t_width_acc:.2f} ms\n"
        f"[PERF] road_skeleton = {t_skel_acc:.2f} ms\n"
        f"[PERF] road_edge_texture = {t_edge_tex_acc:.2f} ms\n"
        f"[PERF] road_connectivity = {t_conn_acc:.2f} ms\n"
        f"[PERF] road_scoring = {t_score_acc:.2f} ms\n"
        f"[PERF] road_total = {t_road_total:.2f} ms\n"
        f"candidate_count = {len(candidates)}\n"
        f"cheap_rejected = {cheap_rejected}\n"
        f"moderate_candidates = {moderate_count}\n"
        f"expensive_candidates = {expensive_count}"
    )
    print(perf_road_log)
    logger.info(perf_road_log)

    mask_water = final_water_mask

    # Built-up: Grays / bright / low saturation
    lower_built = np.array([0, 0, 80])
    upper_built = np.array([180, 40, 255])
    mask_built = cv2.inRange(hsv, lower_built, upper_built)

    # Bare land: Browns / Tans / Yellows
    lower_bare = np.array([10, 30, 80])
    upper_bare = np.array([35, 255, 255])
    mask_bare = cv2.inRange(hsv, lower_bare, upper_bare)

    total_pixels = img.shape[0] * img.shape[1]

    veg_pct = (np.count_nonzero(mask_veg) / total_pixels) * 100
    water_pct = (np.count_nonzero(mask_water) / total_pixels) * 100
    builtup_pct = (np.count_nonzero(mask_built) / total_pixels) * 100
    bare_pct = (np.count_nonzero(mask_bare) / total_pixels) * 100

    # RGBA overlay
    overlay = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
    
    color_veg = [34, 139, 34, 180]      # Forest Green
    color_water = [190, 119, 0, 180]    # Ocean Blue
    color_built = [169, 169, 169, 180]  # Dark Gray
    color_bare = [140, 180, 210, 180]   # Tan

    overlay[mask_bare > 0] = color_bare
    overlay[mask_built > 0] = color_built
    overlay[mask_veg > 0] = color_veg
    overlay[mask_water > 0] = color_water

    _, buffer = cv2.imencode('.png', overlay)
    base64_img = base64.b64encode(buffer).decode('utf-8')
    data_url = f"data:image/png;base64,{base64_img}"

    res = {
        "vegetation_pct": veg_pct,
        "water_pct": water_pct,
        "builtup_pct": builtup_pct,
        "bare_pct": bare_pct,
        "overlay_url": data_url,
        "debug_water": debug_info
    }

    # Store in Task Result Cache
    analysis_cache.cache_task_result(image_path, "land_cover", res, aoi_bbox)
    return res
