# SatQuery AI Test Images

These satellite scenes are curated for validating SatQuery AI's satellite-image reasoning and analysis capabilities.

## Directory Structure

```
test_images/
├── 01_city_river_vegetation/
│   ├── scene_01.jpg               (800x440 RGB Optical - River & Urban Landscape)
│   ├── scene_02.jpg               (1024x810 RGB Optical - Meandering River & Greenery)
│   ├── scene_03.jpg               (1024x657 RGB Optical - River Delta & Urban Settlement)
│   ├── scene_04.jpg               (500x500 RGB Optical - River Channel & Vegetation)
│   ├── scene_05.jpg               (1024x767 RGB Optical - River Valley & Agricultural Fields)
│   └── scene_optical.png          (512x512 RGB Optical - Benchmark River Scene)
├── 02_dense_industrial_area/
│   └── scene_optical.png          (512x512 RGB Optical)
└── 03_urban_farmland_transition/
    ├── optical_ref.jpg            (800x440 RGB Optical)
    └── sar_ref.jpg                (1087x860 Grayscale SAR)
```

---

## 01 — City + River + Vegetation

**Files:** `scene_01.jpg`, `scene_02.jpg`, `scene_03.jpg`, `scene_04.jpg`, `scene_05.jpg`, `scene_optical.png`

**Purpose:**
- Scene understanding & river network isolation
- Water/vegetation contextual analysis
- Built-up vs. surrounding natural landscape interpretation
- Spatial directional & proximity reasoning

---

## 02 — Dense Industrial Area

**Files:** `scene_optical.png` ($512 \times 512$ RGB)

**Purpose:**
- Building detection & boundary segmentation
- Building density estimation ($\text{buildings/km}^2$)
- Development concentration analysis
- Built-environment reasoning

---

## 03 — Urban + Farmland Transition

**Files:** `optical_ref.jpg` ($800 \times 440$ RGB) & `sar_ref.jpg` ($1087 \times 860$ Grayscale SAR)

**Purpose:**
- Land-use pattern reasoning
- Urban/rural transition boundaries
- Spatial comparison across regions
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
