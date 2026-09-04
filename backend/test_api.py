
import requests
import json

url = 'http://127.0.0.1:8000/api/query'
payload = {
    'query': 'fuse optical and sar',
    'images': [
        {'file_id': 'test_opt', 'filename': 'test_opt.jpg', 'url': '', 'modality': 'optical'},
        {'file_id': 'test_sar', 'filename': 'test_sar.jpg', 'url': '', 'modality': 'sar'}
    ],
    'session_id': 'test_session'
}
try:
    res = requests.post(url, json=payload)
    print(res.status_code)
    print(res.text)
except Exception as e:
    print('Exception:', e)
