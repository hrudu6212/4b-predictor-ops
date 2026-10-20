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
print("DEPS_DONE", flush=True)
