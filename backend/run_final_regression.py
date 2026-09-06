import requests
import json

BASE_URL = "http://127.0.0.1:8000"

# 1. Upload optical and SAR images
print("Uploading images...")
with open("../test_images/01_city_river_vegetation/scene_01.jpg", "rb") as f:
    r_opt = requests.post(f"{BASE_URL}/api/upload", params={"modality": "optical"}, files={"file": ("scene_01.jpg", f, "image/jpeg")})
    opt_data = r_opt.json()
    opt_file_id = opt_data["file_id"]

with open("../test_images/01_city_river_vegetation/scene_02.jpg", "rb") as f:
    r_sar = requests.post(f"{BASE_URL}/api/upload", params={"modality": "sar"}, files={"file": ("scene_02.jpg", f, "image/jpeg")})
    sar_data = r_sar.json()
    sar_file_id = sar_data["file_id"]

print(f"Optical File ID: {opt_file_id}")
print(f"SAR File ID: {sar_file_id}")

queries = [
    # BUILDINGS (Single Optical)
    ("BUILDINGS", "How many buildings?", [{"file_id": opt_file_id, "filename": "scene_01.jpg", "modality": "optical"}]),
    ("BUILDINGS", "Where are buildings concentrated?", [{"file_id": opt_file_id, "filename": "scene_01.jpg", "modality": "optical"}]),
    
    # WATER (Single Optical)
    ("WATER", "Is there water?", [{"file_id": opt_file_id, "filename": "scene_01.jpg", "modality": "optical"}]),
    ("WATER", "Where is water located?", [{"file_id": opt_file_id, "filename": "scene_01.jpg", "modality": "optical"}]),
    
    # VEGETATION (Single Optical)
    ("VEGETATION", "Where is vegetation concentrated?", [{"file_id": opt_file_id, "filename": "scene_01.jpg", "modality": "optical"}]),
    ("VEGETATION", "Is vegetation more dominant than built-up land?", [{"file_id": opt_file_id, "filename": "scene_01.jpg", "modality": "optical"}]),
    
    # BUILT-UP (Single Optical)
    ("BUILT-UP", "Which areas are built-up?", [{"file_id": opt_file_id, "filename": "scene_01.jpg", "modality": "optical"}]),
    ("BUILT-UP", "What evidence suggests human activity?", [{"file_id": opt_file_id, "filename": "scene_01.jpg", "modality": "optical"}]),
    
    # SAR / FUSION (Optical + SAR)
    ("SAR / FUSION", "How does SAR complement the optical imagery?", [
        {"file_id": opt_file_id, "filename": "scene_01.jpg", "modality": "optical"},
        {"file_id": sar_file_id, "filename": "scene_02.jpg", "modality": "sar"}
    ]),
    ("SAR / FUSION", "What additional information does SAR provide compared with optical imagery?", [
        {"file_id": opt_file_id, "filename": "scene_01.jpg", "modality": "optical"},
        {"file_id": sar_file_id, "filename": "scene_02.jpg", "modality": "sar"}
    ]),
    ("SAR / FUSION", "Where do optical and SAR disagree?", [
        {"file_id": opt_file_id, "filename": "scene_01.jpg", "modality": "optical"},
        {"file_id": sar_file_id, "filename": "scene_02.jpg", "modality": "sar"}
    ]),
    
    # GENERAL (Single Optical)
    ("GENERAL", "What land-cover types are visible?", [{"file_id": opt_file_id, "filename": "scene_01.jpg", "modality": "optical"}]),
    ("GENERAL", "Describe the overall scene.", [{"file_id": opt_file_id, "filename": "scene_01.jpg", "modality": "optical"}])
]

results = []
for category, q_text, imgs in queries:
    payload = {
        "query": q_text,
        "images": imgs
    }
    r = requests.post(f"{BASE_URL}/api/query", json=payload)
    status_code = r.status_code
    if status_code == 200:
        data = r.json()
    else:
        data = {"error": r.text}
    
    results.append({
        "category": category,
        "query": q_text,
        "status_code": status_code,
        "data": data
    })

with open("final_regression_results.json", "w", encoding="utf-8") as out_f:
    json.dump(results, out_f, indent=2)

print("Finished running all 13 queries. Output saved to final_regression_results.json")
