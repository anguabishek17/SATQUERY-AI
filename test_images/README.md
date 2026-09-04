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
│   ├── industrial_scene_01.jpg    (1024x682 RGB Optical - Highway Interchange & Warehouses)
│   ├── industrial_scene_02.jpg    (768x1024 RGB Optical - High-Density Factory Grid)
│   ├── industrial_scene_03.jpg    (1024x768 RGB Optical - Industrial Roof Structures)
│   ├── industrial_scene_04.jpg    (800x388 RGB Optical - Logistics Hub & Water Retention)
│   └── industrial_scene_05.jpg    (800x792 RGB Optical - Solar Industrial Complex & Transport)
└── 03_urban_farmland_transition/
    ├── transition_scene_01.jpg    (1024x682 RGB Optical - Split Urban/Farmland Boundary)
    ├── transition_scene_02.jpg    (768x1024 RGB Optical - Circular Crop Fields & Urban Expansion)
    ├── transition_scene_03.jpg    (1024x768 RGB Optical - Mixed Urban Grid & Agricultural Parcels)
    ├── transition_scene_04.jpg    (800x388 RGB Optical - High-Density Urban Margin & Farmland)
    └── transition_scene_05.jpg    (800x792 RGB Optical - Central Urban Core & Peripheral Fields)
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

**Files:** `industrial_scene_01.jpg`, `industrial_scene_02.jpg`, `industrial_scene_03.jpg`, `industrial_scene_04.jpg`, `industrial_scene_05.jpg`

**Purpose:**
- Neural building detection & boundary segmentation
- Industrial building density estimation ($\text{buildings/km}^2$)
- High-density development concentration analysis
- Logistics infrastructure & industrial land-use reasoning

---

## 03 — Urban + Farmland Transition

**Files:** `transition_scene_01.jpg`, `transition_scene_02.jpg`, `transition_scene_03.jpg`, `transition_scene_04.jpg`, `transition_scene_05.jpg`

**Purpose:**
- Land-use pattern reasoning & fringe classification
- Urban vs. agricultural land boundary delineation
- Mixed development transition zone interpretation
- Spatial comparison across agricultural and built-up areas

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
