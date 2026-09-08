import requests
import json

BASE_URL = "http://127.0.0.1:8000"

print("Uploading test image pair for Bi-Temporal testing...")
with open("../test_images/01_city_river_vegetation/scene_01.jpg", "rb") as f:
    r0 = requests.post(f"{BASE_URL}/api/upload", params={"modality": "optical"}, files={"file": ("scene_01.jpg", f, "image/jpeg")})
    t0_data = r0.json()
    t0_id = t0_data["file_id"]

with open("../test_images/01_city_river_vegetation/scene_02.jpg", "rb") as f:
    r1 = requests.post(f"{BASE_URL}/api/upload", params={"modality": "optical"}, files={"file": ("scene_02.jpg", f, "image/jpeg")})
    t1_data = r1.json()
    t1_id = t1_data["file_id"]

print(f"T0 File ID: {t0_id}")
print(f"T1 File ID: {t1_id}")

bitemporal_images = [
    {"file_id": t0_id, "filename": "scene_01.jpg", "modality": "optical"},
    {"file_id": t1_id, "filename": "scene_02.jpg", "modality": "optical"}
]

test_queries = [
    "What major changes occurred between the two images?",
    "Where are the changes concentrated?",
    "How much area changed?",
    "Has vegetation increased or decreased?",
    "Has built-up area increased?",
    "Is there evidence of new construction?",
    "Based on these changes, what is the likely future trend?",
    "What should I monitor next?",
    "Predict the future trend."
]

answers = {}

for q in test_queries:
    payload = {
        "query": q,
        "images": bitemporal_images
    }
    r = requests.post(f"{BASE_URL}/api/query", json=payload)
    assert r.status_code == 200, f"Query '{q}' failed with {r.status_code}: {r.text}"
    data = r.json()
    ans = data.get("answer", "")
    answers[q] = ans
    print("=" * 60)
    print(f"QUERY: {q}")
    print(f"TASK: {data.get('task')}")
    print(f"CONFIDENCE: {data.get('confidence')}")
    print("RESPONSE:")
    print(ans)
    print("=" * 60)

    # Verification checks
    assert "Future Prediction:" in ans, f"Response missing 'Future Prediction:' for query: {q}"
    assert "across 0 distinct regions" not in ans, f"Logically inconsistent '0 regions' in: {ans}"
    assert "Confidence:" not in ans and "Confidence is" not in ans, f"Leaked confidence percentage in: {ans}"

print("\n\nAll queries returned structured responses with 'Future Prediction:' and zero leaked confidence.")
unique_count = len(set(answers.values()))
print(f"Total queries: {len(test_queries)}, Unique answers: {unique_count}")
assert unique_count >= 6, "Expected at least 6 distinct answers across different sub-intents"
print("SUCCESS: Bi-temporal query-specific structured answers verified!")
