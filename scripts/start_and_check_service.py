import subprocess, time, requests, os
# start server
p = subprocess.Popen([r'C:\Users\mayan\OneDrive\Desktop\qubo\.venv\Scripts\python.exe','-m','qubo.service'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
print('Started PID', p.pid)
# wait a bit
time.sleep(2)
try:
    r = requests.get('http://127.0.0.1:8000/')
    print('root len', len(r.text))
    print(r.text[:200])
    s = requests.get('http://127.0.0.1:8000/status')
    print('status', s.json())
except Exception as e:
    print('request error', e)
# do not terminate server here
