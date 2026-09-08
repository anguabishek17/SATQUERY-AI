import os
import sys
from pathlib import Path

# Ensure backend root is on sys.path
backend_dir = str(Path(__file__).resolve().parent.parent)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import cv2
import numpy as np
from app.services.rgb_landcover import rgb_landcover_estimation


def _validate_debug_payload(dbg: dict):
    """Validate debug_water schema matches requirements."""
    required_keys = [
        "candidate_regions",
        "road_like_candidates",
        "suppressed_road_candidates",
        "accepted_water_regions",
        "road_likelihoods",
        "water_confidences",
        "suppression_reasons",
        "note"
    ]
    for k in required_keys:
        assert k in dbg, f"Missing key '{k}' in debug_water payload: {dbg}"
    assert isinstance(dbg["road_likelihoods"], list)
    assert isinstance(dbg["water_confidences"], list)
    assert all(isinstance(x, (int, float)) for x in dbg["water_confidences"]), "water_confidences must be a list of floats"
    assert dbg["note"] == "RGB-based water estimate after spatial and road-structure suppression."


def create_base_canvas():
    """Create a 500x500 neutral land terrain canvas (BGR)."""
    img = np.ones((500, 500, 3), dtype=np.uint8) * 160
    # Add subtle land noise
    noise = np.random.randint(-15, 15, (500, 500, 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def test_1_thin_straight_road():
    """TEST 1: Thin straight road -> suppressed."""
    img = create_base_canvas()
    # Thin straight asphalt road (BGR: dark grey/blueish)
    cv2.line(img, (50, 250), (450, 250), (85, 80, 75), 14)
    path = os.path.join(os.path.dirname(__file__), "test_thin_road.png")
    cv2.imwrite(path, img)

    res = rgb_landcover_estimation(path)
    dbg = res["debug_water"]
    _validate_debug_payload(dbg)
    print("\n--- TEST 1: Thin Straight Road ---")
    print("Water %:", res["water_pct"], "Debug:", dbg)
    assert dbg["suppressed_road_candidates"] >= 1 or res["water_pct"] < 1.0, "TEST 1 Failed: Thin road not suppressed"
    print("PASSED TEST 1")


def test_2_wide_road():
    """TEST 2: Wide road -> suppressed."""
    img = create_base_canvas()
    # Wide multi-lane highway (45px wide, uniform asphalt)
    cv2.line(img, (250, 50), (250, 450), (90, 85, 80), 45)
    path = os.path.join(os.path.dirname(__file__), "test_wide_road.png")
    cv2.imwrite(path, img)

    res = rgb_landcover_estimation(path)
    dbg = res["debug_water"]
    _validate_debug_payload(dbg)
    print("\n--- TEST 2: Wide Road ---")
    print("Water %:", res["water_pct"], "Debug:", dbg)
    assert dbg["suppressed_road_candidates"] >= 1 or res["water_pct"] < 1.0, "TEST 2 Failed: Wide road not suppressed"
    print("PASSED TEST 2")


def test_3_curved_road():
    """TEST 3: Curved road -> suppressed."""
    img = create_base_canvas()
    # Curved road polyline
    pts = np.array([[50, 100], [180, 250], [300, 200], [450, 400]], dtype=np.int32)
    cv2.polylines(img, [pts], isClosed=False, color=(80, 75, 70), thickness=20)
    path = os.path.join(os.path.dirname(__file__), "test_curved_road.png")
    cv2.imwrite(path, img)

    res = rgb_landcover_estimation(path)
    dbg = res["debug_water"]
    _validate_debug_payload(dbg)
    print("\n--- TEST 3: Curved Road ---")
    print("Water %:", res["water_pct"], "Debug:", dbg)
    assert dbg["suppressed_road_candidates"] >= 1 or res["water_pct"] < 1.0, "TEST 3 Failed: Curved road not suppressed"
    print("PASSED TEST 3")


def test_4_road_intersection():
    """TEST 4: Road intersection -> suppressed."""
    img = create_base_canvas()
    # Cross intersection
    cv2.line(img, (50, 250), (450, 250), (85, 80, 75), 22)
    cv2.line(img, (250, 50), (250, 450), (85, 80, 75), 22)
    path = os.path.join(os.path.dirname(__file__), "test_intersection.png")
    cv2.imwrite(path, img)

    res = rgb_landcover_estimation(path)
    dbg = res["debug_water"]
    _validate_debug_payload(dbg)
    print("\n--- TEST 4: Road Intersection ---")
    print("Water %:", res["water_pct"], "Debug:", dbg)
    assert dbg["suppressed_road_candidates"] >= 1 or res["water_pct"] < 1.0, "TEST 4 Failed: Intersection not suppressed"
    print("PASSED TEST 4")


def test_5_genuine_large_water_body():
    """TEST 5: Genuine large water body -> retained."""
    img = create_base_canvas()
    # Deep blue lake (BGR: B=180, G=100, R=20)
    cv2.circle(img, (250, 250), 120, (180, 100, 20), -1)
    path = os.path.join(os.path.dirname(__file__), "test_large_water.png")
    cv2.imwrite(path, img)

    res = rgb_landcover_estimation(path)
    dbg = res["debug_water"]
    _validate_debug_payload(dbg)
    print("\n--- TEST 5: Large Water Body ---")
    print("Water %:", res["water_pct"], "Debug:", dbg)
    assert dbg["accepted_water_regions"] >= 1 and res["water_pct"] > 10.0, "TEST 5 Failed: Large water body suppressed"
    print("PASSED TEST 5")


def test_6_narrow_river_canal():
    """TEST 6: Narrow river/canal -> retained."""
    img = create_base_canvas()
    # Narrow winding canal with vibrant blue water (BGR: B=200, G=110, R=15)
    pts = np.array([[50, 80], [150, 180], [250, 300], [450, 420]], dtype=np.int32)
    cv2.polylines(img, [pts], isClosed=False, color=(200, 110, 15), thickness=18)
    path = os.path.join(os.path.dirname(__file__), "test_narrow_canal.png")
    cv2.imwrite(path, img)

    res = rgb_landcover_estimation(path)
    dbg = res["debug_water"]
    _validate_debug_payload(dbg)
    print("\n--- TEST 6: Narrow River/Canal ---")
    print("Water %:", res["water_pct"], "Debug:", dbg)
    assert dbg["accepted_water_regions"] >= 1 and res["water_pct"] > 1.0, "TEST 6 Failed: Narrow canal suppressed"
    print("PASSED TEST 6")


def test_7_blue_grey_roof():
    """TEST 7: Blue/grey building roof -> not classified as water."""
    img = create_base_canvas()
    # Rectangular metal blue-grey building roof (BGR: B=130, G=120, R=110, low saturation neutral)
    cv2.rectangle(img, (180, 180), (320, 320), (130, 120, 110), -1)
    path = os.path.join(os.path.dirname(__file__), "test_grey_roof.png")
    cv2.imwrite(path, img)

    res = rgb_landcover_estimation(path)
    dbg = res["debug_water"]
    _validate_debug_payload(dbg)
    print("\n--- TEST 7: Blue/Grey Roof ---")
    print("Water %:", res["water_pct"], "Debug:", dbg)
    assert res["water_pct"] < 1.0, f"TEST 7 Failed: Roof classified as water ({res['water_pct']}%)"
    print("PASSED TEST 7")


def test_8_shadowed_road():
    """TEST 8: Shadowed road -> not classified as water."""
    img = create_base_canvas()
    # Dark shadowed road strip (BGR: B=25, G=25, R=25, textured)
    cv2.line(img, (80, 50), (420, 450), (25, 25, 25), 20)
    path = os.path.join(os.path.dirname(__file__), "test_shadowed_road.png")
    cv2.imwrite(path, img)

    res = rgb_landcover_estimation(path)
    dbg = res["debug_water"]
    _validate_debug_payload(dbg)
    print("\n--- TEST 8: Shadowed Road ---")
    print("Water %:", res["water_pct"], "Debug:", dbg)
    assert res["water_pct"] < 1.5, f"TEST 8 Failed: Shadowed road classified as water ({res['water_pct']}%)"
    print("PASSED TEST 8")


def test_9_mixed_urban_scene():
    """TEST 9: Mixed urban scene containing roads + water -> water retained while road false positives minimized."""
    img = create_base_canvas()
    # Road network (straight + crossing)
    cv2.line(img, (50, 150), (450, 150), (85, 80, 75), 18)
    cv2.line(img, (150, 50), (150, 450), (85, 80, 75), 18)
    
    # Real lake on right side (BGR: B=190, G=105, R=10)
    cv2.circle(img, (350, 350), 75, (190, 105, 10), -1)
    
    path = os.path.join(os.path.dirname(__file__), "test_mixed_urban.png")
    cv2.imwrite(path, img)

    res = rgb_landcover_estimation(path)
    dbg = res["debug_water"]
    _validate_debug_payload(dbg)
    print("\n--- TEST 9: Mixed Urban Scene ---")
    print("Water %:", res["water_pct"], "Debug:", dbg)
    assert dbg["accepted_water_regions"] >= 1, "TEST 9 Failed: Mixed scene lake suppressed"
    assert dbg["suppressed_road_candidates"] >= 1 or res["water_pct"] < 10.0, "TEST 9 Failed: Mixed scene roads not suppressed"
    print("PASSED TEST 9")


if __name__ == "__main__":
    test_1_thin_straight_road()
    test_2_wide_road()
    test_3_curved_road()
    test_4_road_intersection()
    test_5_genuine_large_water_body()
    test_6_narrow_river_canal()
    test_7_blue_grey_roof()
    test_8_shadowed_road()
    test_9_mixed_urban_scene()
    print("\n=============================================")
    print("ALL 9 WATER DETECTION TESTS PASSED CLEANLY!")
    print("=============================================")
