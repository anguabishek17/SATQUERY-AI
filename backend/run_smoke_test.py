import requests
import json

BASE_URL = "http://127.0.0.1:8000"

# 1. Upload optical and SAR images
with open("../test_images/01_city_river_vegetation/scene_01.jpg", "rb") as f:
    r_opt = requests.post(f"{BASE_URL}/api/upload", params={"modality": "optical"}, files={"file": ("scene_01.jpg", f, "image/jpeg")})
    opt_data = r_opt.json()
    opt_file_id = opt_data["file_id"]

with open("../test_images/01_city_river_vegetation/scene_02.jpg", "rb") as f:
    r_sar = requests.post(f"{BASE_URL}/api/upload", params={"modality": "sar"}, files={"file": ("scene_02.jpg", f, "image/jpeg")})
    sar_data = r_sar.json()
    sar_file_id = sar_data["file_id"]

smoke_queries = [
    ("How many buildings?", [{"file_id": opt_file_id, "filename": "scene_01.jpg", "modality": "optical"}]),
    ("Is there water?", [{"file_id": opt_file_id, "filename": "scene_01.jpg", "modality": "optical"}]),
    ("Where is vegetation concentrated?", [{"file_id": opt_file_id, "filename": "scene_01.jpg", "modality": "optical"}]),
    ("Which areas are built-up?", [{"file_id": opt_file_id, "filename": "scene_01.jpg", "modality": "optical"}]),
    ("What evidence suggests human activity?", [{"file_id": opt_file_id, "filename": "scene_01.jpg", "modality": "optical"}]),
    ("What additional information does SAR provide compared with optical imagery?", [
        {"file_id": opt_file_id, "filename": "scene_01.jpg", "modality": "optical"},
        {"file_id": sar_file_id, "filename": "scene_02.jpg", "modality": "sar"}
    ]),
    ("Where do optical and SAR disagree?", [
        {"file_id": opt_file_id, "filename": "scene_01.jpg", "modality": "optical"},
        {"file_id": sar_file_id, "filename": "scene_02.jpg", "modality": "sar"}
    ]),
    ("What land-cover types are visible?", [{"file_id": opt_file_id, "filename": "scene_01.jpg", "modality": "optical"}]),
    ("Describe the overall scene.", [{"file_id": opt_file_id, "filename": "scene_01.jpg", "modality": "optical"}])
]

results = []
for q_text, imgs in smoke_queries:
    payload = {"query": q_text, "images": imgs}
    r = requests.post(f"{BASE_URL}/api/query", json=payload)
    status_code = r.status_code
    data = r.json() if status_code == 200 else {"error": r.text}
    
    # TaskType extraction
    exec_trace = data.get("execution_trace", [])
    task_type = "UNKNOWN"
    for step in exec_trace:
        if step.get("step") == "task_classification":
            task_type = step.get("detail", "").replace("query classified as task=", "")
            break
            
    tool_used = data.get("tools_used", [None])[0] if data.get("tools_used") else data.get("tool_name")
    answer = data.get("answer", "")
    conf = data.get("confidence")
    
    results.append({
        "query": q_text,
        "status_code": status_code,
        "task_type": task_type,
        "tool_used": tool_used,
        "confidence": conf,
        "answer": answer,
        "full_data": data
    })

with open("smoke_test_output.json", "w", encoding="utf-8") as out_f:
    json.dump(results, out_f, indent=2)

print("Smoke test finished.")
