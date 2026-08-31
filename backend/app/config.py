"""Central config — swap paths/model names here without touching business logic."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
REPORT_DIR = DATA_DIR / "reports"
AUDIT_DB_PATH = DATA_DIR / "audit.sqlite3"

for _dir in (DATA_DIR, UPLOAD_DIR, REPORT_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# Model identifiers — point these at your fine-tuned checkpoints / LoRA adapters.
VQA_CAPTION_MODEL = os.getenv("VQA_CAPTION_MODEL", "geochat-7b-4bit")
GROUNDING_MODEL = os.getenv("GROUNDING_MODEL", "groundingdino-swin-t")
CHANGE_MODEL = os.getenv("CHANGE_MODEL", "bit-cd")
LORA_ADAPTER_PATH = os.getenv("LORA_ADAPTER_PATH", str(DATA_DIR / "adapters" / "bigearthnet-lora"))
OBJECT_DETECTOR_WEIGHTS = os.getenv("OBJECT_DETECTOR_WEIGHTS", str(DATA_DIR / "models" / "spacenet_building_detector.pt"))
GROUNDING_DINO_CONFIG = os.getenv("GROUNDING_DINO_CONFIG", str(DATA_DIR / "configs" / "GroundingDINO_SwinT_OGC.py"))
GROUNDING_DINO_WEIGHTS = os.getenv("GROUNDING_DINO_WEIGHTS", str(DATA_DIR / "models" / "groundingdino_swint_ogc.pth"))

# Building Detector Hyperparameters
DETECTOR_CONF_THRESH = float(os.getenv("DETECTOR_CONF_THRESH", "0.35"))
DETECTOR_TILE_OVERLAP = float(os.getenv("DETECTOR_TILE_OVERLAP", "0.20"))
DETECTOR_NMS_IOU = float(os.getenv("DETECTOR_NMS_IOU", "0.45"))

# Remote Sensing Spectral & Signal Thresholds (Configurable)
NDVI_THRESHOLD = float(os.getenv("NDVI_THRESHOLD", "0.30"))
NDWI_THRESHOLD = float(os.getenv("NDWI_THRESHOLD", "0.00"))
NDBI_THRESHOLD = float(os.getenv("NDBI_THRESHOLD", "0.00"))
SAR_WATER_THRESHOLD = float(os.getenv("SAR_WATER_THRESHOLD", "-17.0"))
SAR_BUILTUP_PERCENTILE = float(os.getenv("SAR_BUILTUP_PERCENTILE", "70.0"))
SAR_TEXTURE_THRESHOLD = float(os.getenv("SAR_TEXTURE_THRESHOLD", "3.0"))

# Sensor-Aware Spectral Band Mappings (1-based band numbers for rasterio)
SENSOR_BANDS = {
    "SENTINEL_2": {"blue": 2, "green": 3, "red": 4, "nir": 8, "swir": 11},
    "LANDSAT_8": {"blue": 2, "green": 3, "red": 4, "nir": 5, "swir": 6},
    "LANDSAT_9": {"blue": 2, "green": 3, "red": 4, "nir": 5, "swir": 6},
    "CARTOSAT": {"blue": 1, "green": 2, "red": 3, "nir": 4},
    "PLANETSCOPE": {"blue": 1, "green": 2, "red": 3, "nir": 4},
    "DEFAULT": {"red": 1, "green": 2, "nir": 3, "swir": 4},
}

# Confidence below this triggers a "low confidence" flag in the UI.
LOW_CONFIDENCE_THRESHOLD = float(os.getenv("LOW_CONFIDENCE_THRESHOLD", "0.55"))

ALLOWED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

