"""Small test script that calls `qubo.copilot.call_gemini` using saved key.

This script reads the key via `get_gemini_key` and makes a single generate call.
"""
from qubo.copilot import call_gemini, get_local_gemini_key

if __name__ == '__main__':
    key = get_local_gemini_key()
    if not key:
        print('No Gemini key found. Set GEMINI_API_KEY or save key via GUI.')
    else:
        print('Found Gemini key (hidden). Calling API...')
        res = call_gemini('Hello from QUBO test. Provide a short greeting.', model='models/gemini-1.0')
        print('Response:')
        print(res)
