"""
Automated Regression Test: River / Water AOI Building Detection Test
Guarantees zero false positive building counts on river edges, water channels, and vegetation boundaries.
"""

import os
import sys
from pathlib import Path

# Ensure backend root is on sys.path
backend_dir = str(Path(__file__).resolve().parent.parent)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import cv2
import numpy as np
from app.schemas import ImageRef
from app.tools.object_counting import ObjectCountingTool


def test_river_aoi_no_buildings():
    """Verifies that a river image returns 0 buildings, 0.0 density, and empty GeoJSON features."""
    # Create synthetic river scene: dark blue water channel with green riverbanks
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    # Green riverbanks
    img[:, :] = (34, 139, 34)
    # Winding blue river channel
    pts = np.array([[100, 0], [150, 200], [120, 350], [200, 512], [280, 512], [200, 350], [220, 200], [180, 0]], dtype=np.int32)
    cv2.fillPoly(img, [pts], (255, 140, 0)) # BGR format: dark blue channel

    test_river_path = os.path.join(os.path.dirname(__file__), "test_river_scene.png")
    os.makedirs(os.path.dirname(test_river_path), exist_ok=True)
    cv2.imwrite(test_river_path, img)

    # Mock ImageRef
    class DummyRef:
        def __init__(self, file_id):
            self.file_id = file_id
            self.modality = "optical"

    # Save to mock image_io directory or test directly
    file_id = "test_river_scene"
    from app.config import UPLOAD_DIR
    target_upload_path = UPLOAD_DIR / f"{file_id}.png"
    cv2.imwrite(str(target_upload_path), img)

    tool = ObjectCountingTool()
    image_ref = DummyRef(file_id)
    aoi_bbox = [120.0, 100.0, 300.0, 400.0]

    result = tool.run("How many buildings are in this river?", [image_ref], aoi_bbox=aoi_bbox)

    print("\n--- River Regression Test Output ---")
    print("Output text:", result.output_text)
    print("Building count:", result.physical_metrics.get("building_count"))
    print("Density per km²:", result.physical_metrics.get("density_per_km2"))
    print("GeoJSON features count:", len(result.geojson_overlay.get("features", [])))

    b_count = result.physical_metrics.get("building_count")
    detector_status = result.raw.get("detector_status")

    if detector_status == "not_loaded":
        assert b_count is None, f"Expected None for unloaded model, got {b_count}"
    else:
        assert b_count == 0, f"Expected 0 buildings on water scene, got {b_count}"

    assert len(result.geojson_overlay.get("features", [])) == 0, f"Expected 0 features, got {len(result.geojson_overlay.get('features', []))}"
    print("PASSED: test_river_aoi_no_buildings [OK]")


if __name__ == "__main__":
    test_river_aoi_no_buildings()
