import os
import sys
import time
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath("."))
load_dotenv()

from app.schemas import ImageRef, InputConfig, TaskType
from app.controller.agent_controller import handle_query
from app.services.analysis_cache import analysis_cache
import app.services.image_io as image_io

def run_benchmark():
    print("=" * 70)
    print("SATQUERY AI: FAST WATER QUERY BENCHMARK (1333x1333 SATELLITE IMAGE)")
    print("=" * 70)

    # Locate user uploaded 1333x1333 image
    user_img = "data/uploads/94bb6006-d12e-41ab-a8b7-6394975cd0a5.jpg"
    if not os.path.exists(user_img):
        # Fallback to any recent upload in data/uploads
        import glob
        uploads = glob.glob("data/uploads/*.jpg")
        assert uploads, "No test image found in data/uploads"
        user_img = uploads[0]

    abs_img_path = os.path.abspath(user_img)
    print(f"Target Image: {abs_img_path}")

    # Patch image_io and dynamic_analysis_tool to point to our test image
    import app.services.image_io
    import app.tools.dynamic_analysis_tool
    app.services.image_io.saved_path = lambda fid: abs_img_path
    app.tools.dynamic_analysis_tool.saved_path = lambda fid: abs_img_path

    query = "Is there water present in the given image?"
    images = [ImageRef(file_id="bench_water_01", filename=os.path.basename(user_img), modality="optical")]

    runs = []

    for run_idx in [1, 2, 3]:
        is_cold = (run_idx == 1)
        if is_cold:
            analysis_cache.clear()
            print(f"\n--- RUN {run_idx}: COLD CACHE ---")
        else:
            print(f"\n--- RUN {run_idx}: WARM CACHE ---")

        t_start = time.perf_counter()
        response = handle_query(query, images)
        t_total = (time.perf_counter() - t_start) * 1000

        # Extract metrics
        yolo_executed = any("YOLO" in step.detail and "True" in step.detail for step in response.execution_trace)
        
        # Check assertions
        assert response.task == TaskType.dynamic_analysis or response.task == TaskType.WATER_DETECTION, f"Unexpected task: {response.task}"
        assert not yolo_executed, "Assertion Failed: YOLO was executed!"
        assert t_total < 5000.0, f"Assertion Failed: Total latency {t_total:.2f} ms exceeds 5000 ms budget!"
        assert response.answer, "Assertion Failed: Empty answer!"
        assert "evidence_json" in [s.step for s in response.execution_trace], "Assertion Failed: Evidence JSON missing in trace"
        assert "evidence_validation" in [s.step for s in response.execution_trace], "Assertion Failed: Evidence Validation missing in trace"

        runs.append({
            "run": run_idx,
            "type": "Cold Cache" if is_cold else "Warm Cache",
            "total_ms": t_total,
            "answer": response.answer,
            "trace": response.execution_trace
        })
        print(f"Run {run_idx} Completed in: {t_total:.2f} ms")
        print(f"Answer: {response.answer[:120]}...")

    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY RESULTS")
    print("=" * 70)
    for r in runs:
        print(f"Run {r['run']} ({r['type']}): Total Latency = {r['total_ms']:.2f} ms ({(r['total_ms']/1000):.2f}s)")
    print("ALL RUNS MET <5.0 SECOND TARGET [OK]")

if __name__ == "__main__":
    run_benchmark()
