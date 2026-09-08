import time
import requests
import json
import os

BASE_URL = "http://127.0.0.1:8000"

def run_tests():
    print("=" * 70)
    print("SATQUERY AI PERFORMANCE & ROUTING REGRESSION TEST SUITE")
    print("=" * 70)

    # 1. Upload Test Images
    print("Uploading test imagery...")
    opt_path = "../test_images/01_city_river_vegetation/scene_01.jpg"
    sar_path = "../test_images/01_city_river_vegetation/scene_02.jpg"

    with open(opt_path, "rb") as f:
        r_opt = requests.post(f"{BASE_URL}/api/upload", params={"modality": "optical"}, files={"file": ("scene_01.jpg", f, "image/jpeg")})
        assert r_opt.status_code == 200, f"Optical upload failed: {r_opt.text}"
        opt_id = r_opt.json()["file_id"]

    with open(sar_path, "rb") as f:
        r_sar = requests.post(f"{BASE_URL}/api/upload", params={"modality": "sar"}, files={"file": ("scene_02.jpg", f, "image/jpeg")})
        assert r_sar.status_code == 200, f"SAR upload failed: {r_sar.text}"
        sar_id = r_sar.json()["file_id"]

    print(f"Optical ID: {opt_id}")
    print(f"SAR ID:     {sar_id}")
    print("-" * 70)

    opt_img = [{"file_id": opt_id, "filename": "scene_01.jpg", "modality": "optical"}]
    fusion_imgs = [
        {"file_id": opt_id, "filename": "scene_01.jpg", "modality": "optical"},
        {"file_id": sar_id, "filename": "scene_02.jpg", "modality": "sar"}
    ]
    bitemp_imgs = [
        {"file_id": opt_id, "filename": "scene_01.jpg", "modality": "optical"},
        {"file_id": opt_id, "filename": "scene_01_t1.jpg", "modality": "optical"}
    ]

    test_cases = [
        {
            "name": "1. Water Detection (Fresh)",
            "query": "Is there water present?",
            "images": opt_img,
            "expect_yolo": False,
            "max_seconds": 15.0,
            "expected_task": "dynamic_analysis"
        },
        {
            "name": "2. Water Location (Cached / Follow-up)",
            "query": "Where is the water located?",
            "images": opt_img,
            "expect_yolo": False,
            "max_seconds": 15.0,
            "expected_task": "dynamic_analysis"
        },
        {
            "name": "3. Vegetation Analysis",
            "query": "What percentage is vegetation?",
            "images": opt_img,
            "expect_yolo": False,
            "max_seconds": 15.0,
            "expected_task": "dynamic_analysis"
        },
        {
            "name": "4. Building Detection (YOLO execution)",
            "query": "How many buildings are present?",
            "images": opt_img,
            "expect_yolo": True,
            "max_seconds": 25.0,
            "expected_task": "object_counting"
        },
        {
            "name": "5. Building Distribution (Cached boxes partition)",
            "query": "Where are buildings concentrated?",
            "images": opt_img,
            "expect_yolo": True,
            "max_seconds": 15.0,
            "expected_task": "object_counting"
        },
        {
            "name": "6. Scene Description (Lightweight - NO YOLO)",
            "query": "Describe the scene.",
            "images": opt_img,
            "expect_yolo": False,
            "max_seconds": 15.0,
            "expected_task": "dynamic_analysis"
        },
        {
            "name": "7. Optical + SAR Fusion",
            "query": "Compare optical and SAR imagery.",
            "images": fusion_imgs,
            "expect_yolo": False,
            "max_seconds": 20.0,
            "expected_task": "optical_sar_fusion"
        },
        {
            "name": "8. Bi-temporal Change Detection",
            "query": "Detect changes between dates.",
            "images": bitemp_imgs,
            "expect_yolo": False,
            "max_seconds": 20.0,
            "expected_task": "change_vqa"
        }
    ]

    all_passed = True
    results_summary = []

    for tc in test_cases:
        print(f"\nRunning: {tc['name']}")
        print(f"Query: \"{tc['query']}\"")
        payload = {
            "query": tc["query"],
            "images": tc["images"]
        }
        t0 = time.time()
        res = requests.post(f"{BASE_URL}/api/query", json=payload)
        elapsed = time.time() - t0

        if res.status_code != 200:
            print(f"FAILED: Status {res.status_code} - {res.text}")
            all_passed = False
            continue

        data = res.json()
        task = data.get("task")
        answer = data.get("answer", "")
        trace = data.get("execution_trace", [])

        # Check YOLO execution from trace or detector status
        yolo_in_trace = any("YOLO: True" in s.get("detail", "") or "Building Detector" in s.get("detail", "") for s in trace)
        tools_used = data.get("tools_used", [])
        yolo_used = "building-detector" in tools_used or yolo_in_trace

        # Assertions
        yolo_assertion = (yolo_used == tc["expect_yolo"])
        time_assertion = elapsed <= tc["max_seconds"]
        fallback_assertion = "AI explanation layer unavailable" not in answer

        print(f"Status:      {res.status_code} OK")
        print(f"Task:        {task}")
        print(f"Elapsed:     {elapsed:.2f}s (Budget: {tc['max_seconds']}s)")
        print(f"YOLO Ran:    {yolo_used} (Expected: {tc['expect_yolo']}) -> {'PASS' if yolo_assertion else 'FAIL'}")
        print(f"No Fallback: {'PASS' if fallback_assertion else 'FAIL'}")
        print(f"Answer:      {answer[:120]}...")

        passed = yolo_assertion and time_assertion and fallback_assertion
        if not passed:
            all_passed = False
            print(">>> TEST CASE FAILED ASSERTIONS <<<")
        else:
            print(">>> TEST CASE PASSED <<<")

        results_summary.append({
            "name": tc["name"],
            "query": tc["query"],
            "elapsed_seconds": round(elapsed, 2),
            "yolo_used": yolo_used,
            "expect_yolo": tc["expect_yolo"],
            "passed": passed,
            "answer": answer
        })

    with open("perf_test_summary.json", "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL 8 PERFORMANCE & ROUTING REGRESSION TESTS PASSED [OK]")
    else:
        print("SOME TESTS FAILED - REVIEW LOGS")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
