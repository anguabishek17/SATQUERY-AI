import requests
import json
import time
import os

URL_UPLOAD = 'http://127.0.0.1:8000/api/upload'
URL_QUERY = 'http://127.0.0.1:8000/api/query'

def run_tests():
    opt_path = '../test_images/01_city_river_vegetation/scene_01.jpg'
    sar_path = '../test_images/01_city_river_vegetation/scene_02.jpg'
    
    print("Uploading test images...")
    with open(opt_path, 'rb') as f:
        r_opt = requests.post(URL_UPLOAD, files={'file': ('scene_01.jpg', f, 'image/jpeg')}, params={'modality': 'optical'})
        opt_id = r_opt.json()['file_id']

    with open(sar_path, 'rb') as f:
        r_sar = requests.post(URL_UPLOAD, files={'file': ('scene_02.jpg', f, 'image/jpeg')}, params={'modality': 'sar'})
        sar_id = r_sar.json()['file_id']

    print(f"Uploaded: opt={opt_id}, sar={sar_id}")

    test_queries = [
        "Compare the optical and SAR images",
        "Compare optical and SAR images",
        "How does SAR complement optical imagery?",
        "What additional information does SAR provide?",
        "Where do optical and SAR disagree?",
        "Which is better, optical or SAR?"
    ]

    for q in test_queries:
        print("\n" + "="*70)
        print(f"QUERY: {q}")
        print("="*70)
        payload = {
            'query': q,
            'images': [
                {'file_id': opt_id, 'filename': 'scene_01.jpg', 'modality': 'optical'},
                {'file_id': sar_id, 'filename': 'scene_02.jpg', 'modality': 'sar'}
            ],
            'session_id': 'fusion_quality_test'
        }
        res = requests.post(URL_QUERY, json=payload)
        if res.status_code == 200:
            data = res.json()
            answer = data.get('answer', '')
            print("TASK:", data.get('task'))
            print("RESPONSE:\n" + answer)
            print("-" * 50)
            print("CONFIDENCE:", data.get('confidence'))
            print("FUSION AGREEMENT SCORE:", data.get('fusion_agreement_score'))
            print("METRICS:", data.get('physical_metrics'))
        else:
            print(f"FAILED {res.status_code}: {res.text}")

if __name__ == '__main__':
    run_tests()
