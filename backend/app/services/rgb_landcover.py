import cv2
import numpy as np
import base64

def rgb_landcover_estimation(image_path: str, aoi_bbox=None) -> dict:
    """
    Fallback land-cover estimation for 3-band RGB imagery using HSV color heuristics.
    Returns percentages for vegetation, water, built-up, bare land, and a base64 overlay mask.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image for RGB analysis: {image_path}")

    # Crop to AOI if provided
    if aoi_bbox:
        ax1, ay1, ax2, ay2 = [max(0, int(v)) for v in aoi_bbox]
        ax2 = min(ax2, img.shape[1])
        ay2 = min(ay2, img.shape[0])
        img = img[ay1:ay2, ax1:ax2]

    if img.size == 0:
        raise ValueError("AOI resulted in empty image crop.")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Vegetation: Greens
    # Hue ~ 35 to 85, Saturation > 40, Value > 40
    lower_veg = np.array([35, 40, 40])
    upper_veg = np.array([85, 255, 255])
    mask_veg = cv2.inRange(hsv, lower_veg, upper_veg)

    # Water: Dark blues/blacks
    # Satellite water is often very dark, low value. Or blueish hue.
    # Hue ~ 90 to 130, Sat > 40
    lower_water = np.array([90, 40, 20])
    upper_water = np.array([130, 255, 150])
    mask_water_blue = cv2.inRange(hsv, lower_water, upper_water)
    
    # Dark water (low value, low sat)
    lower_water_dark = np.array([0, 0, 0])
    upper_water_dark = np.array([180, 255, 30]) # Very dark
    mask_water_dark = cv2.inRange(hsv, lower_water_dark, upper_water_dark)
    mask_water = cv2.bitwise_or(mask_water_blue, mask_water_dark)

    # Built-up: Grays / bright / low saturation
    # Sat < 40, Val > 80 (not dark water)
    lower_built = np.array([0, 0, 80])
    upper_built = np.array([180, 40, 255])
    mask_built = cv2.inRange(hsv, lower_built, upper_built)

    # Bare land: Browns / Tans / Yellows
    # Hue ~ 10 to 35, Sat > 30, Val > 80
    lower_bare = np.array([10, 30, 80])
    upper_bare = np.array([35, 255, 255])
    mask_bare = cv2.inRange(hsv, lower_bare, upper_bare)

    total_pixels = img.shape[0] * img.shape[1]

    veg_pct = (np.count_nonzero(mask_veg) / total_pixels) * 100
    water_pct = (np.count_nonzero(mask_water) / total_pixels) * 100
    builtup_pct = (np.count_nonzero(mask_built) / total_pixels) * 100
    bare_pct = (np.count_nonzero(mask_bare) / total_pixels) * 100

    # Create visualization mask
    # RGBA overlay
    overlay = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
    
    # Colors: B, G, R, A (OpenCV uses BGRA)
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

    return {
        "vegetation_pct": veg_pct,
        "water_pct": water_pct,
        "builtup_pct": builtup_pct,
        "bare_pct": bare_pct,
        "overlay_url": data_url
    }
