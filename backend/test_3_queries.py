import requests
import json

BASE_URL = "http://127.0.0.1:8000"

# 1. Upload optical image
print("Uploading test optical image...")
with open("../test_images/01_city_river_vegetation/scene_01.jpg", "rb") as f:
    r_opt = requests.post(f"{BASE_URL}/api/upload", params={"modality": "optical"}, files={"file": ("scene_01.jpg", f, "image/jpeg")})
    opt_data = r_opt.json()
    opt_file_id = opt_data["file_id"]

queries = [
    "Where is water located?",
    "What evidence suggests human activity?",
    "Describe the overall scene."
]

for q in queries:
    payload = {
        "query": q,
        "images": [{"file_id": opt_file_id, "filename": "scene_01.jpg", "modality": "optical"}]
    }
    r = requests.post(f"{BASE_URL}/api/query", json=payload)
    data = r.json()
    print(f"\n--- QUERY: {q} ---")
    print(f"HTTP Status: {r.status_code}")
    print(f"TaskType: {data.get('execution_trace', [{}])[2].get('detail', '').replace('query classified as task=', '') if len(data.get('execution_trace', [])) > 2 else data.get('task')}")
    print(f"Selected Tool: {data.get('tools_used', [None])[0]}")
    print(f"Confidence: {data.get('confidence')}")
    print(f"Final Response: {data.get('answer')}")
