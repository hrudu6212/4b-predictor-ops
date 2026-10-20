"""Ensure runtime deps exist on the VM before serving."""
import importlib.util
import subprocess
import sys

MAP = {
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "pyngrok": "pyngrok",
    "multipart": "python-multipart",
    "PIL": "Pillow",
    "torch": "torch",
    "torchvision": "torchvision",
}

missing = [pkg for mod, pkg in MAP.items() if importlib.util.find_spec(mod) is None]
print("missing:", missing, flush=True)
if missing:
    p = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing],
                       capture_output=True, text=True)
    print("pip rc:", p.returncode, flush=True)
    print((p.stdout or "")[-800:], flush=True)
    print((p.stderr or "")[-800:], flush=True)
for mod in MAP:
    print(mod, "->", "OK" if importlib.util.find_spec(mod) else "MISSING", flush=True)

# Pre-fetch the ngrok binary here (slow/flaky in-VM download must not happen
# during serve). Retry a few times and verify the path exists.
import os
import time

from pyngrok import installer
from pyngrok.conf import get_default

ngrok_path = get_default().ngrok_path
print("ngrok target:", ngrok_path, flush=True)
for attempt in range(5):
    try:
        if os.path.exists(ngrok_path) and os.path.getsize(ngrok_path) > 1_000_000:
            print("ngrok bin ok:", os.path.getsize(ngrok_path), flush=True)
            break
        installer.install_ngrok(ngrok_path)
    except Exception as exc:
        print(f"ngrok install attempt {attempt} failed: {type(exc).__name__}: {exc}", flush=True)
    time.sleep(5)
else:
    print("ngrok bin: FAILED", flush=True)

print("DEPS_DONE", flush=True)
