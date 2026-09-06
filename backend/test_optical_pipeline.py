import requests
import json
import time
import os

URL_UPLOAD = 'http://127.0.0.1:8000/api/upload'
URL_QUERY = 'http://127.0.0.1:8000/api/query'

def test_pipeline():
    image_path = '../test_images/01_city_river_vegetation/scene_01.jpg'
    
    # 1. Upload image
    print(f"Uploading {image_path}...")
    with open(image_path, 'rb') as f:
        files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
        params = {'modality': 'optical'}
        res = requests.post(URL_UPLOAD, files=files, params=params)
        if res.status_code != 200:
            print(f"Upload failed: {res.text}")
            return
        upload_data = res.json()
        file_id = upload_data['file_id']
        print(f"Upload successful. File ID: {file_id}")

    queries = [
        "Is there water?",
        "Where is vegetation concentrated?",
        "Which areas are built-up?",
        "What percentage of the scene is vegetation?",
        "What percentage of the scene is built-up?"
    ]

    for q in queries:
        print(f"\n--- Testing Query: '{q}' ---")
        payload = {
            'query': q,
            'images': [
                {'file_id': file_id, 'filename': os.path.basename(image_path), 'url': '', 'modality': 'optical'}
            ],
            'session_id': 'test_optical_session'
        }
        res = requests.post(URL_QUERY, json=payload)
        if res.status_code == 200:
            try:
                data = res.json()
                
                # We want to extract specific info as requested by user:
                # - input modality/bands (we know it's RGB from context)
                # - TaskType
                # - selected tool
                # - whether quantitative index was actually calculated
                # - metric returned
                # - confidence
                # - final response
                
                steps = data.get('plan', {}).get('steps', [])
                # The response schema usually has task, tool_name etc. Let's dump relevant fields
                print(f"TaskType: {data.get('task')}")
                # We need to find tool output. `data` might be the Final Answer or similar, let's print the whole thing nicely
                print(f"Response: {data.get('answer')}")
                print("Full JSON:")
                print(json.dumps(data, indent=2))
                
            except Exception as e:
                print(f"Error parsing json: {e}, text: {res.text}")
        else:
            print(f"Query failed with status {res.status_code}: {res.text}")

if __name__ == '__main__':
    # Wait a bit for server to start if just launched
    time.sleep(2)
    test_pipeline()
