import requests
import json
import time
import os

URL_UPLOAD = 'http://127.0.0.1:8000/api/upload'
URL_QUERY = 'http://127.0.0.1:8000/api/query'

def test_pipeline():
    opt_path = '../test_images/01_city_river_vegetation/scene_01.jpg'
    sar_path = '../test_images/01_city_river_vegetation/scene_02.jpg'
    
    # 1. Upload optical
    print(f"Uploading optical {opt_path}...")
    with open(opt_path, 'rb') as f:
        files = {'file': (os.path.basename(opt_path), f, 'image/jpeg')}
        params = {'modality': 'optical'}
        res = requests.post(URL_UPLOAD, files=files, params=params)
        if res.status_code != 200:
            print(f"Optical Upload failed: {res.text}")
            return
        opt_id = res.json()['file_id']
        print(f"Optical Upload successful. File ID: {opt_id}")

    # 2. Upload SAR
    print(f"Uploading SAR {sar_path}...")
    with open(sar_path, 'rb') as f:
        files = {'file': (os.path.basename(sar_path), f, 'image/jpeg')}
        params = {'modality': 'sar'}
        res = requests.post(URL_UPLOAD, files=files, params=params)
        if res.status_code != 200:
            print(f"SAR Upload failed: {res.text}")
            return
        sar_id = res.json()['file_id']
        print(f"SAR Upload successful. File ID: {sar_id}")

    queries = [
        "How does SAR complement the optical imagery in this scene?",
        "What additional information does the SAR image provide compared with the optical image?",
        "What evidence is supported by both optical and SAR?",
        "Where do optical and SAR disagree?"
    ]

    for q in queries:
        print(f"\n--- Testing Query: '{q}' ---")
        payload = {
            'query': q,
            'images': [
                {'file_id': opt_id, 'filename': os.path.basename(opt_path), 'url': '', 'modality': 'optical'},
                {'file_id': sar_id, 'filename': os.path.basename(sar_path), 'url': '', 'modality': 'sar'}
            ],
            'session_id': 'test_fusion_session'
        }
        res = requests.post(URL_QUERY, json=payload)
        if res.status_code == 200:
            try:
                data = res.json()
                print(f"TaskType: {data.get('task')}")
                print(f"Response:\n{data.get('answer')}")
                print("Full JSON:")
                print(json.dumps(data, indent=2))
            except Exception as e:
                print(f"Error parsing json: {e}, text: {res.text}")
        else:
            print(f"Query failed with status {res.status_code}: {res.text}")

if __name__ == '__main__':
    time.sleep(2)
    test_pipeline()
