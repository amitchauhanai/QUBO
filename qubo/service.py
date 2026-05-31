# security & env
from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
import uvicorn
import importlib
import subprocess
import sys
from typing import List, Optional
import os
import json as _json
import time as _time
import re as _re
from fastapi import Request as _Request
import uuid
from fastapi.responses import JSONResponse

from qubo.exec_helper import run_code_subprocess

app = FastAPI(title='qubo API')

# CORS (allow the UI to call the API when hosted)
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.environ.get('QUBO_ALLOW_ORIGIN', '*')],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# security: optionally require an API key for sensitive endpoints
API_KEY = os.environ.get('QUBO_API_KEY')
# control whether remote pip installs are allowed
ALLOW_INSTALL = os.environ.get('QUBO_ALLOW_INSTALL', 'false').lower() in ('1', 'true', 'yes')

def require_api_key(x_api_key: Optional[str] = Header(None)):
    if API_KEY:
        if not x_api_key or x_api_key != API_KEY:
            raise HTTPException(status_code=401, detail='Missing or invalid API key')
    return True
# serve static web UI from html/ directory if present
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
import os
from fastapi import WebSocket, WebSocketDisconnect
import asyncio

static_dir = os.path.join(os.path.dirname(__file__), '..', 'html')
static_dir = os.path.abspath(static_dir)
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# simple user file storage (safe sandbox inside project)
USER_FILES = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'user_files'))
os.makedirs(USER_FILES, exist_ok=True)

def safe_path(path: str) -> str:
    # prevent path traversal
    full = os.path.abspath(os.path.join(USER_FILES, path))
    if not full.startswith(USER_FILES):
        raise HTTPException(status_code=400, detail='invalid path')
    return full


@app.get('/list-files')
def list_files():
    files = []
    for fn in os.listdir(USER_FILES):
        files.append(fn)
    return files


@app.get('/load')
def load(path: str):
    p = safe_path(path)
    if not os.path.exists(p):
        raise HTTPException(status_code=404, detail='not found')
    with open(p, 'r', encoding='utf-8') as f:
        return {'path': path, 'content': f.read()}


class SaveRequest(BaseModel):
    path: str
    content: str


@app.post('/save')
def save(req: SaveRequest):
    p = safe_path(req.path)
    with open(p, 'w', encoding='utf-8') as f:
        f.write(req.content)
    return {'path': req.path, 'saved': True}

@app.get('/')
def root():
    index_path = os.path.join(static_dir, 'index.html')
    if os.path.exists(index_path):
        return HTMLResponse(open(index_path, 'r', encoding='utf-8').read())
    return {'status': 'qubo API', 'docs': '/docs'}


class RunRequest(BaseModel):
    code: str
    shots: int = 1024


@app.post('/run')
def run_code(req: RunRequest):
    # use subprocess runner to avoid executing user code in process
    try:
        ok, res = run_code_subprocess(req.code, shots=req.shots)
        return {'ok': ok, 'result': res}
    except Exception as e:
        # return exception details to help debug server-side failures during development
        import traceback
        return {'ok': False, 'error': str(e), 'traceback': traceback.format_exc()}


@app.get('/status')
def status():
    """Return availability of optional backends and components."""
    def check_module(name: str):
        try:
            importlib.import_module(name)
            return True
        except Exception:
            return False

    # jax availability by checking the adapter
    jax_ok = False
    try:
        jb = importlib.import_module('qubo.backends.jax_backend')
        jax_ok = getattr(jb, 'is_available', lambda: False)()
    except Exception:
        jax_ok = False

    return {
        'jax_backend': jax_ok,
        'density_backend': check_module('qubo.density'),
        'fastapi_installed': check_module('fastapi'),
        'opt_einsum': check_module('opt_einsum'),
    }


@app.get('/install-help')
def install_help(tool: str):
    """Return suggested pip commands for a named optional tool."""
    hints = {
        'jax': ['pip install jax jaxlib  # follow jax docs for CUDA wheels'],
        'fastapi': ['pip install fastapi uvicorn'],
        'tensor': ['pip install opt_einsum cotengra'],
    }
    return {'tool': tool, 'suggestions': hints.get(tool, ['# no suggestions available'])}


@app.post('/install')
def install(tool: str):
    """Attempt to install a tool via pip using the running Python executable.

    WARNING: this runs pip in the server environment. Intended for local developer use only.
    """
    mapping = {
        'jax': ['jax', 'jaxlib'],
        'fastapi': ['fastapi', 'uvicorn'],
        'tensor': ['opt_einsum', 'cotengra'],
    }
    pkgs = mapping.get(tool)
    if not pkgs:
        raise HTTPException(status_code=400, detail='unknown tool')
    cmd = [sys.executable, '-m', 'pip', 'install'] + pkgs
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=300)
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail='install timed out')
    return {'returncode': proc.returncode, 'stdout': proc.stdout, 'stderr': proc.stderr}


async def stream_subprocess(cmd):
    """Run a subprocess asynchronously and yield stdout lines as they appear."""
    # use asyncio.create_subprocess_exec if available
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    try:
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            yield line.decode('utf-8', errors='replace')
    finally:
        try:
            proc.kill()
        except Exception:
            pass


@app.websocket('/ws')
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    td = None
    try:
        # simple protocol: client sends a JSON line with {"code": "..."}
        data = await websocket.receive_text()
        import json as _json
        try:
            msg = _json.loads(data)
        except Exception:
            await websocket.send_text('invalid protocol')
            await websocket.close()
            return
        code = msg.get('code')
        if not code:
            await websocket.send_text('no code provided')
            await websocket.close()
            return

        # write user code to temp file and run wrapper similar to run_code_subprocess
        import tempfile as _tempfile
        from pathlib import Path as _Path
        td = _tempfile.TemporaryDirectory()
        td_path = _Path(td.name)
        user_file = td_path / 'user_code_ws.py'
        wrapper_file = td_path / 'runner_ws.py'
        user_file.write_text(code, encoding='utf-8')
        # simple wrapper that prints stdout as it comes
        wrapper = f"""import sys
import traceback
try:
    exec(open({str(user_file)!r}).read(), {{}})
except Exception:
    traceback.print_exc()
"""
        wrapper_file.write_text(wrapper, encoding='utf-8')
        # run and stream
        async for line in stream_subprocess([sys.executable, str(wrapper_file)]):
            try:
                await websocket.send_text(line)
            except WebSocketDisconnect:
                break
    except WebSocketDisconnect:
        # client disconnected
        pass
    except Exception as e:
        try:
            await websocket.send_text('server error: ' + str(e))
        except Exception:
            pass
    finally:
        if td is not None:
            try:
                td.cleanup()
            except Exception:
                pass
        try:
            await websocket.close()
        except Exception:
            pass

ASSISTANT_RATE_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'assistant_rate.json'))
ASSISTANT_USAGE_LOG = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'assistant_usage.log'))
_ASSISTANT_RATE = {}
_ASSISTANT_RATE_LIMIT = int(os.environ.get('ASSISTANT_RATE_PER_MIN', '30'))  # default per-client per-minute


def _load_rate_state():
    try:
        if os.path.exists(ASSISTANT_RATE_FILE):
            with open(ASSISTANT_RATE_FILE, 'r', encoding='utf-8') as f:
                data = _json.load(f)
                # convert numeric keys if needed
                for k,v in data.items():
                    _ASSISTANT_RATE[k] = v
    except Exception:
        pass


def _save_rate_state():
    try:
        with open(ASSISTANT_RATE_FILE, 'w', encoding='utf-8') as f:
            _json.dump(_ASSISTANT_RATE, f)
    except Exception:
        pass

# initialize from disk
_load_rate_state()


def _rate_allowed_for(client_id: str) -> bool:
    now = _time.time()
    entry = _ASSISTANT_RATE.get(client_id, {'count': 0, 'window_start': now})
    # reset window if older than 60s
    if now - entry.get('window_start', now) > 60:
        entry['window_start'] = now
        entry['count'] = 0
    if entry.get('count', 0) >= _ASSISTANT_RATE_LIMIT:
        _ASSISTANT_RATE[client_id] = entry
        return False
    entry['count'] = entry.get('count', 0) + 1
    _ASSISTANT_RATE[client_id] = entry
    # persist asynchronously (best-effort)
    try:
        _save_rate_state()
    except Exception:
        pass
    return True


def _extract_code_from_text(text: str) -> str:
    """Try to extract a reasonable code block from free-form assistant output.
    - Prefer fenced code blocks (```python ... ``` or ``` ... ```
    - Fall back to contiguous indented block (4 spaces or tab)
    - Otherwise return full text (caller may trim)
    """
    if not text:
        return ''
    # look for triple-backtick fenced blocks
    m = _re.search(r'```(?:python\n)?([\s\S]*?)```', text, _re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # look for lines starting with 4+ spaces (indented block)
    lines = text.splitlines()
    block = []
    in_block = False
    for ln in lines:
        if _re.match(r'^(\s{4,}|\t)', ln):
            block.append(ln.lstrip())
            in_block = True
        else:
            if in_block:
                break
    if block:
        return '\n'.join(block).strip()
    # no clear code block; try to heuristically pull anything that looks like code (contains def/class/import/print)
    heur = []
    for ln in lines:
        if _re.search(r'\b(def |class |import |from |print\(|qc\.|QuantumCircuit)', ln):
            heur.append(ln)
    if heur:
        return '\n'.join(heur)
    # fallback: return full text (caller can decide)
    return text.strip()


# Enhance the existing /assistant POST to log usage and enforce rate limit
class AssistantRequest(BaseModel):
    prompt: str
    code: Optional[str] = None


@app.post('/assistant')
def assistant(req: AssistantRequest, request: _Request):
    """Synchronous assistant call. Defensive: requires GEMINI_API_KEY to be set.
    Adds rate-limiting and logs usage to a file. Returns model response plus extracted code.
    """
    # identify client id (cookie/header/ip fallback)
    client_id, created = _get_client_id_from_request(request)
    # enforce per-client rate limit
    if not _rate_allowed_for(client_id):
        return JSONResponse(status_code=429, content={'ok': False, 'error': 'rate_limit_exceeded', 'detail': f'Allowed {_ASSISTANT_RATE_LIMIT} requests per minute.'})

    key = os.environ.get('GEMINI_API_KEY')
    if not key:
        entry = {'ts': _time.time(), 'client': client_id, 'prompt_len': len(req.prompt or ''), 'status': 'no_key'}
        _log_assistant_usage(entry)
        resp = JSONResponse(status_code=400, content={'ok': False, 'error': 'GEMINI_API_KEY not configured. Set GEMINI_API_KEY env var on the server.'})
        if created:
            resp.set_cookie('qubo_client_id', client_id, httponly=True, samesite='Lax')
        return resp

    try:
        import requests
    except Exception:
        return {'ok': False, 'error': 'Missing python "requests" package. Install: pip install requests'}

    url = os.environ.get('GEMINI_API_URL') or 'https://generativelanguage.googleapis.com/v1beta2/models/text-bison-001:generate'
    prompt_text = req.prompt or ''
    if req.code:
        prompt_text = prompt_text + "\n\nContext code:\n" + req.code

    payload = {
        'prompt': {'text': prompt_text},
        'temperature': float(os.environ.get('GEMINI_TEMP', '0.2')),
        'maxOutputTokens': int(os.environ.get('GEMINI_MAX_TOKENS', '1024')),
    }
    headers = {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        try:
            body = resp.json()
        except Exception:
            body = {'raw_text': resp.text}

        # build friendly text
        out_text = ''
        if isinstance(body, dict):
            if 'candidates' in body and isinstance(body['candidates'], list) and body['candidates']:
                out_text = '\n'.join([c.get('content', '') if isinstance(c, dict) else str(c) for c in body['candidates']])
            elif 'output' in body:
                out_text = body.get('output')
            else:
                out_text = str(body)
        else:
            out_text = str(body)

        extracted_code = _extract_code_from_text(out_text)

        entry = {'ts': _time.time(), 'client': client_id, 'prompt_len': len(req.prompt or ''), 'resp_len': len(out_text), 'status_code': resp.status_code}
        _log_assistant_usage(entry)

        return {'ok': True, 'status_code': resp.status_code, 'response': body, 'text': out_text, 'extracted_code': extracted_code}
    except Exception as e:
        _log_assistant_usage({'ts': _time.time(), 'client': client_id, 'error': str(e)})
        import traceback as _tb
        return {'ok': False, 'error': str(e), 'traceback': _tb.format_exc()}


# New websocket endpoint for streaming assistant responses incrementally (improved)
@app.websocket('/ws-assistant')
async def ws_assistant(ws: WebSocket):
    await ws.accept()
    client_ip = 'unknown'
    try:
        if getattr(ws, 'client', None):
            client_ip = ws.client[0]
    except Exception:
        pass

    try:
        data = await ws.receive_text()
        try:
            msg = _json.loads(data)
        except Exception:
            await ws.send_text(_json.dumps({'type': 'error', 'error': 'invalid json'}))
            await ws.close()
            return
        prompt = msg.get('prompt', '')
        code_ctx = msg.get('code', '')

        if not _rate_allowed_for(client_ip):
            await ws.send_text(_json.dumps({'type': 'error', 'error': 'rate_limit_exceeded', 'detail': f'Allowed {_ASSISTANT_RATE_LIMIT} requests per minute.'}))
            await ws.close()
            return

        key = os.environ.get('GEMINI_API_KEY')
        if not key:
            await ws.send_text(_json.dumps({'type': 'error', 'error': 'GEMINI_API_KEY not configured on server.'}))
            await ws.close()
            return

        try:
            import requests as _requests
        except Exception:
            await ws.send_text(_json.dumps({'type': 'error', 'error': 'Missing python requests package on server.'}))
            await ws.close()
            return

        url = os.environ.get('GEMINI_API_URL') or 'https://generativelanguage.googleapis.com/v1beta2/models/text-bison-001:generate'
        payload = {'prompt': {'text': (prompt + '\n\nContext code:\n' + code_ctx) if code_ctx else prompt},
                   'temperature': float(os.environ.get('GEMINI_TEMP', '0.2')),
                   'maxOutputTokens': int(os.environ.get('GEMINI_MAX_TOKENS', '1024'))}
        headers = {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}

        final_text = ''
        # Attempt streaming; many providers may not stream, so fallback to chunked sync call
        try:
            with _requests.post(url, json=payload, headers=headers, stream=True, timeout=60) as resp:
                if resp.status_code != 200:
                    body = resp.text
                    await ws.send_text(_json.dumps({'type': 'error', 'error': 'upstream_error', 'detail': body}))
                else:
                    # iterate over bytes and send small deltas; await between sends to allow backpressure
                    buffer = ''
                    chunk_size = 128
                    for chunk in resp.iter_content(chunk_size=256):
                        if not chunk:
                            continue
                        piece = chunk.decode('utf-8', errors='replace')
                        buffer += piece
                        # flush by newline boundary or if buffer grows
                        while '\n' in buffer or len(buffer) >= chunk_size:
                            # flush up to the first newline or chunk_size
                            if '\n' in buffer:
                                idx = buffer.index('\n') + 1
                                out_piece = buffer[:idx]
                                buffer = buffer[idx:]
                            else:
                                out_piece = buffer[:chunk_size]
                                buffer = buffer[chunk_size:]
                            final_text += out_piece
                            await ws.send_text(_json.dumps({'type': 'delta', 'data': out_piece}))
                            # small sleep to avoid tight loop and allow client to keep up
                            await asyncio.sleep(0.005)
                    # flush remainder
                    if buffer:
                        final_text += buffer
                        await ws.send_text(_json.dumps({'type': 'delta', 'data': buffer}))
        except Exception:
            # fallback synchronous call
            resp = _requests.post(url, json=payload, headers=headers, timeout=60)
            try:
                body = resp.json()
                if isinstance(body, dict) and 'candidates' in body and isinstance(body['candidates'], list) and body['candidates']:
                    text = '\n'.join([c.get('content','') if isinstance(c, dict) else str(c) for c in body['candidates']])
                elif isinstance(body, dict) and 'output' in body:
                    text = body.get('output','')
                else:
                    text = _json.dumps(body)
            except Exception:
                text = resp.text
            # stream back in small chunks with pauses
            chunk_size = 120
            for i in range(0, len(text), chunk_size):
                piece = text[i:i+chunk_size]
                final_text += piece
                await ws.send_text(_json.dumps({'type': 'delta', 'data': piece}))
                await asyncio.sleep(0.01)

        extracted = _extract_code_from_text(final_text or '')
        await ws.send_text(_json.dumps({'type': 'done', 'text': final_text or '', 'extracted_code': extracted}))

        _log_assistant_usage({'ts': _time.time(), 'client': client_ip, 'prompt_len': len(prompt or ''), 'resp_len': len(final_text or ''), 'mode': 'ws'})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_text(_json.dumps({'type': 'error', 'error': str(e)}))
        except Exception:
            pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass


def _log_assistant_usage(entry: dict):
    try:
        entry = dict(entry)
        entry.setdefault('ts', _time.time())
        with open(ASSISTANT_USAGE_LOG, 'a', encoding='utf-8') as f:
            f.write(_json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _get_client_id_from_request(req: _Request):
    """Return (client_id, created_bool) — prefer persistent cookie or X-Client-Id header, fallback to IP."""
    client_id = None
    created = False
    try:
        # cookies available on Request
        client_id = req.cookies.get('qubo_client_id') if getattr(req, 'cookies', None) is not None else None
    except Exception:
        client_id = None
    if not client_id:
        try:
            hdr = req.headers.get('x-client-id') if getattr(req, 'headers', None) is not None else None
            if hdr:
                client_id = hdr
        except Exception:
            pass
    if not client_id:
        # fallback to remote host or generate ephemeral id
        try:
            client_id = req.client.host if getattr(req, 'client', None) else None
        except Exception:
            client_id = None
    if not client_id:
        client_id = 'anon-' + uuid.uuid4().hex
        created = True
    return client_id, created


async def _ws_send_with_backoff(ws: WebSocket, msg_text: str, max_retries: int = 4):
    """Send with exponential backoff on transient errors to avoid busy-looping if client is slow."""
    backoff = 0.01
    for attempt in range(max_retries):
        try:
            await ws.send_text(msg_text)
            return True
        except WebSocketDisconnect:
            return False
        except Exception:
            # brief sleep and retry with increased backoff
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 0.5)
    try:
        # final attempt, best-effort
        await ws.send_text(msg_text)
        return True
    except Exception:
        return False