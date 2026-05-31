import requests
r = requests.post('http://127.0.0.1:8000/install?tool=tensor')
print('install status:', r.status_code)
print(r.json())
r2 = requests.get('http://127.0.0.1:8000/status')
print('status:', r2.json())
