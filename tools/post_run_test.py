# Small test script to POST to the local qubo /run endpoint and print full response
import http.client
import json
import sys

HOST = '127.0.0.1'
PORT = 8000

# Replace this code with something simple; wrapper expects a variable named `qc` in the namespace.
# We'll send simple code that intentionally does not define qc so we can inspect error handling.
user_code = """
# demo user code
print('hello from user code')
# no 'qc' defined here, runner should report that
"""

payload = json.dumps({"code": user_code, "shots": 1024})
headers = {"Content-Type": "application/json"}

conn = http.client.HTTPConnection(HOST, PORT, timeout=10)
try:
    conn.request('POST', '/run', payload, headers)
    resp = conn.getresponse()
    body = resp.read().decode('utf-8', errors='replace')
    print('HTTP', resp.status, resp.reason)
    print('HEADERS:')
    for k, v in resp.getheaders():
        print(f'  {k}: {v}')
    print('\nBODY:')
    # try to pretty-print JSON
    try:
        parsed = json.loads(body)
        print(json.dumps(parsed, indent=2))
    except Exception:
        print(body)
except Exception as e:
    print('Request failed:', e)
    sys.exit(2)
finally:
    try:
        conn.close()
    except Exception:
        pass
