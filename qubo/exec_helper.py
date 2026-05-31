import tempfile
import subprocess
import sys
import json
from pathlib import Path
from typing import Tuple


WRAPPER = r"""
import json
import sys
import traceback
import io
import contextlib
ns = {{}}
user_stdout = ''
user_stderr = ''
try:
    # capture stdout/stderr while executing user code
    buf_out = io.StringIO()
    buf_err = io.StringIO()
    with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
        exec(open({userfile!r}).read(), ns)
    user_stdout = buf_out.getvalue()
    user_stderr = buf_err.getvalue()
    result = None
    if 'qc' in ns:
        try:
            from qubo.simulator import StatevectorSimulator
            sim = StatevectorSimulator(ns['qc'])
            res = sim.run(shots={shots})
            # serialize result
            if isinstance(res, dict):
                result = {{'counts': res}}
            else:
                import numpy as _np
                probs = (_np.abs(res) ** 2).real
                labels = [format(i, f'0{{ns["qc"].num_qubits}}b') for i in range(len(probs))]
                result = {{'probabilities': dict(zip(labels, probs.tolist()))}}
        except Exception as e:
            result = {{'error': str(e), 'traceback': traceback.format_exc()}}
    else:
        result = {{'error': "No 'qc' found in executed code"}}
except Exception:
    result = {{'error': 'execution error', 'traceback': traceback.format_exc()}}
# output a JSON object with both result and captured stdout/stderr
sys.stdout.write(json.dumps({{'result': result, 'stdout': user_stdout, 'stderr': user_stderr}}))
"""


def run_code_subprocess(code: str, shots: int = 1024, timeout: int = 5) -> Tuple[bool, dict]:
    """Run user code in a separate Python process and return (success, result_dict).

    This writes the user code to a temp file and runs a small wrapper that executes it
    and prints a JSON-serializable result. Uses subprocess with a timeout to limit execution.
    """
    # create temp directory
    td = tempfile.TemporaryDirectory()
    td_path = Path(td.name)
    user_file = td_path / 'user_code.py'
    wrapper_file = td_path / 'runner.py'

    user_file.write_text(code, encoding='utf-8')
    wrapper_code = WRAPPER.format(userfile=str(user_file), shots=int(shots))
    wrapper_file.write_text(wrapper_code, encoding='utf-8')

    # run interpreter
    try:
        completed = subprocess.run([
            sys.executable, str(wrapper_file)
        ], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        td.cleanup()
        return False, {'error': 'execution timeout'}

    td.cleanup()
    out = completed.stdout.strip()
    if not out:
        # return stderr if no stdout
        return False, {'error': 'no output', 'stderr': completed.stderr}
    try:
        data = json.loads(out)
    except Exception as e:
        return False, {'error': 'invalid json from runner', 'raw': out, 'stderr': completed.stderr}
    return True, data
