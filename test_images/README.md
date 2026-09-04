# SatQuery AI Test Images

These satellite scenes are curated for validating SatQuery AI's satellite-image reasoning and analysis capabilities.

## Directory Structure

```
test_images/
├── 01_city_river_vegetation/
│   └── scene_optical.png          (512x512 RGB)
├── 02_dense_industrial_area/
│   └── scene_optical.png          (512x512 RGB)
└── 03_urban_farmland_transition/
    ├── optical_ref.jpg            (800x440 RGB Optical)
    └── sar_ref.jpg                (1087x860 Grayscale SAR)
```

---

## 01 — City + River + Vegetation

**Files:** `scene_optical.png` ($512 \times 512$ RGB)

**Purpose:**
- Scene understanding
- Water/vegetation context
- Built-up vs surrounding landscape
- Spatial reasoning

---

## 02 — Dense Industrial Area

**Files:** `scene_optical.png` ($512 \times 512$ RGB)

**Purpose:**
- Building detection
- Building density
- Development concentration
- Built-environment reasoning

---

## 03 — Urban + Farmland Transition

**Files:** `optical_ref.jpg` ($800 \times 440$ RGB) & `sar_ref.jpg` ($1087 \times 860$ Grayscale SAR)

**Purpose:**
- Land-use pattern reasoning
- Urban/rural transition
- Spatial comparison
- Vegetation and built-up context
- Cross-modal Optical + SAR evidence fusion and dynamic spatial grid alignment testing

---

## Supported Analysis

The dataset is intended to test capabilities including:
- Natural-language satellite queries
- Building detection using the YOLO-based building detector
- Building density estimation ($\text{buildings/km}^2$)
- Spatial comparison and AOI bounding-box containment
- Scene-level land-cover reasoning
- Spectral indices (NDWI, NDBI, NDVI) where compatible RGB/multispectral bands are supplied
- Optical + SAR fusion where co-registered optical and SAR inputs are provided
- Evidence-based confidence estimation
- Limitations/uncertainty reporting
