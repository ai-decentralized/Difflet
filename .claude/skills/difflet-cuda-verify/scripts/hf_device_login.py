"""Hugging Face device-code login on a pod, for gated repos such as FLUX.1-dev (skill
difflet-cuda-verify, references/pod-runbook.md). Saves the token under HF_HOME.

Run it detached, then read the log; never `pkill -f` it from inside an ssh command:

    HF_HOME=/data/hf setsid nohup ~/Difflet/.venv/bin/python hf_device_login.py
        > ~/logs/hf-login.log 2>&1 < /dev/null &      (one line)

The log prints `OPEN: <url>` and `CODE: <code>`; the owner opens the URL and enters the code
before it expires (300 s on 2026-10-18), then `LOGIN_OK user: ...` follows.
"""

from huggingface_hub._login import _save_oauth_token
from huggingface_hub.utils._oauth_device import poll_device_token, request_device_code

info = request_device_code()
print("OPEN:", info["verification_uri_complete"], flush=True)
print("CODE:", info["user_code"], "expires_in_s:", info["expires_in"], flush=True)
resp = poll_device_token(info)
name, user = _save_oauth_token(resp)
print("LOGIN_OK user:", user, "token_name:", name, flush=True)
