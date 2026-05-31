"""QUBO Copilot helpers: ML/DL integration and simple variational training helpers.

This module provides:
- `detect_backends()` to report available ML frameworks (PyTorch/JAX/TensorFlow)
- `train_variational(...)` a small gradient-descent trainer that uses the
  `parameter_shift_gradient` from `qubo.autodiff` to optimize parameterized gates
  in a `QuantumCircuit`.

Note: This is intentionally lightweight — intended as a glue layer for users to
hook into PyTorch/JAX training loops or call the helper for quick experiments.
"""

from typing import List, Callable, Dict, Any, Optional
import os
from pathlib import Path
import json
import urllib.request
import urllib.error

try:
    from .config import get_gemini_key
except Exception:
    # local import fallback for module execution
    try:
        from qubo.config import get_gemini_key
    except Exception:
        def get_gemini_key(path=None):
            return os.environ.get('GEMINI_API_KEY')


def detect_backends() -> Dict[str, Optional[str]]:
    """Return available ML/DL frameworks and their versions (or None if missing)."""
    out = {"torch": None, "jax": None, "tensorflow": None}
    try:
        import torch
        out["torch"] = torch.__version__
    except Exception:
        out["torch"] = None
    try:
        import jax
        out["jax"] = jax.__version__
    except Exception:
        out["jax"] = None
    try:
        import tensorflow as tf
        out["tensorflow"] = tf.__version__
    except Exception:
        out["tensorflow"] = None
    return out


def train_variational(circuit, gate_indices: List[int], observable: Callable[[Any], float],
                      epochs: int = 20, lr: float = 0.1, shots: int = 0,
                      batch_updates: bool = True, clip_grad: Optional[float] = None,
                      momentum: float = 0.0) -> List[Dict[str, Any]]:
    """Perform a simple gradient-descent optimization on parameterized gates.

    - `circuit`: `QuantumCircuit` instance (will be mutated in-place)
    - `gate_indices`: list of indices in `circuit.gates` that contain a scalar parameter
    - `observable`: function(statevector_or_density) -> scalar loss (to minimize)
    - `epochs`: number of gradient steps
    - `lr`: learning rate
    - `shots`: if >0, simulators may use sampling (not implemented for gradients)

    Returns a history list with dicts: {'epoch': i, 'params': [...], 'loss': float}
    """
    from qubo.autodiff import parameter_shift_gradient

    history = []
    # momentum buffer
    vel = [0.0 for _ in gate_indices]

    for ep in range(epochs):
        # current loss
        from qubo.simulator import StatevectorSimulator
        sim = StatevectorSimulator(circuit)
        res = sim.run(shots=shots) if shots > 0 else sim.run()
        loss = float(observable(res))

        # record current params
        cur_params = []
        for gi in gate_indices:
            g = circuit.gates[gi]
            p = float(g.params[0]) if g.params else 0.0
            cur_params.append(p)

        history.append({"epoch": ep, "params": cur_params.copy(), "loss": loss})

        # compute gradients for each parameter
        grads = []
        for gi in gate_indices:
            g = circuit.gates[gi]
            if not g.params:
                grads.append(0.0)
                continue
            grad = parameter_shift_gradient(circuit, gi, observable, shots=shots)
            grads.append(grad)

        # optional gradient clipping
        if clip_grad is not None:
            grads = [max(min(g, clip_grad), -clip_grad) for g in grads]

        if batch_updates:
            # update after all gradients are computed
            for i, gi in enumerate(gate_indices):
                if not circuit.gates[gi].params:
                    continue
                vel[i] = momentum * vel[i] - lr * grads[i]
                update = vel[i]
                circuit.gates[gi].params[0] = float(circuit.gates[gi].params[0]) + update
        else:
            # sequential updates: apply each gradient immediately
            for i, gi in enumerate(gate_indices):
                if not circuit.gates[gi].params:
                    continue
                vel[i] = momentum * vel[i] - lr * grads[i]
                update = vel[i]
                circuit.gates[gi].params[0] = float(circuit.gates[gi].params[0]) + update

    return history


def example_training_run():
    """Small runnable example that minimizes expectation Z on qubit 0 for a single RX gate."""
    try:
        from qubo.circuit import QuantumCircuit
        from qubo.gates import RX
    except Exception:
        return {"error": "imports failed"}
    qc = QuantumCircuit(1)
    # parametric RX gate with initial angle
    qc.add_gate(RX(0, 0.2))

    def loss_fn(state):
        # minimize absolute expectation <Z> on qubit 0 using statevector
        import numpy as _np
        exp = 0.0
        for idx, amp in enumerate(state):
            bit = (idx >> 0) & 1
            val = 1.0 if bit == 0 else -1.0
            exp += val * (abs(amp) ** 2)
        return float(abs(exp))

    hist = train_variational(qc, [0], loss_fn, epochs=6, lr=0.2)
    return hist


def call_gemini(prompt: str, model: str = 'models/gemini-1.0', max_tokens: int = 512) -> Dict[str, Any]:
    """Call Gemini-like Generative API using a raw HTTP POST.

    This helper looks for the API key via `GEMINI_API_KEY` env var or the
    local config file (see `qubo.config.get_gemini_key`). It performs a simple
    POST and returns parsed JSON response or an error dict.
    """
    key = get_gemini_key()
    if not key:
        return {"error": "no_gemini_key"}

    # Build endpoint (Google Generative API style)
    endpoint = f'https://generative.googleapis.com/v1/{model}:generate'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {key}',
        'User-Agent': 'qubo-copilot/1.0'
    }
    body = {
        'prompt': {
            'text': prompt
        },
        'maxOutputTokens': max_tokens,
    }

    req = urllib.request.Request(endpoint, data=json.dumps(body).encode('utf-8'), headers=headers, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = resp.read().decode('utf-8')
            parsed = json.loads(data)
            # Normalize response to a simple dict with 'text' when possible
            # Handle Google Generative API 'candidates'
            if isinstance(parsed, dict):
                if 'candidates' in parsed and isinstance(parsed['candidates'], list):
                    texts = []
                    for c in parsed['candidates']:
                        if isinstance(c, dict):
                            t = c.get('content') or c.get('output') or c.get('text')
                            if t:
                                texts.append(t)
                    return {'text': '\n\n'.join(texts) if texts else parsed, 'raw': parsed}
                # OpenAI-like responses
                if 'choices' in parsed and isinstance(parsed['choices'], list):
                    texts = []
                    for ch in parsed['choices']:
                        txt = ch.get('text') or (ch.get('message') or {}).get('content')
                        if txt:
                            texts.append(txt)
                    return {'text': '\n\n'.join(texts) if texts else parsed, 'raw': parsed}
            return {'text': str(parsed), 'raw': parsed}
    except urllib.error.HTTPError as e:
        # Read error body if present
        try:
            err = e.read().decode('utf-8')
        except Exception:
            err = None
        # Common case: model endpoint not found (404) — offer guidance and optional fallback
        if e.code == 404:
            suggestion = (
                'Model endpoint not found (404).\n'
                'If you are using Google Generative API, verify the `model` string.\n'
                'Common Google model names: "models/text-bison-001", "models/chat-bison-001".\n'
                'For Gemini preview names may differ — check your provider docs.\n'
                'Alternatively, set `OPENAI_API_KEY` to use OpenAI endpoints as a fallback.'
            )
            result = {"error": "http_error", "status": e.code, "body": err, "suggestion": suggestion}

            # Try OpenAI fallback if user has provided an OPENAI_API_KEY
            openai_key = os.environ.get('OPENAI_API_KEY')
            if openai_key:
                try:
                    oa_headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {openai_key}', 'User-Agent': 'qubo-copilot/1.0'}
                    # use chat completions endpoint
                    oa_body = {'model': model or 'gpt-3.5-turbo', 'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': max_tokens}
                    oa_req = urllib.request.Request('https://api.openai.com/v1/chat/completions', data=json.dumps(oa_body).encode('utf-8'), headers=oa_headers, method='POST')
                    with urllib.request.urlopen(oa_req, timeout=20) as oa_resp:
                        oa_data = oa_resp.read().decode('utf-8')
                        oa_parsed = json.loads(oa_data)
                        # normalize OpenAI response
                        texts = []
                        for ch in oa_parsed.get('choices', []):
                            txt = (ch.get('message') or {}).get('content') or ch.get('text')
                            if txt:
                                texts.append(txt)
                        return {'text': '\n\n'.join(texts) if texts else oa_parsed, 'raw': {'google_error': result, 'openai': oa_parsed}}
                except Exception as oa_e:
                    result['openai_fallback_error'] = str(oa_e)
            return result
        try:
            return {"error": "http_error", "status": e.code, "body": err}
        except Exception:
            return {"error": "http_error", "status": e.code}
    except Exception as e:
        return {"error": "exception", "message": str(e)}


def get_local_gemini_key(path: Path | None = None) -> str | None:
    """Convenience wrapper that returns the detected Gemini key.

    Prefers the environment variable, falls back to the local config file.
    """
    return get_gemini_key(path)
