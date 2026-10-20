"""Colab setup B: launch the REAL 4B API + ngrok tunnel (detached), verify local."""
import importlib.util
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

CKPT = "/tmp/4b_data/out/best_4b_chart_model.pt"
assert os.path.exists(CKPT), f"checkpoint missing: {CKPT}"
env = dict(os.environ, MODEL_CHECKPOINT=CKPT)
if os.path.exists("/tmp/4b_work/ngrok"):
    os.chmod("/tmp/4b_work/ngrok", 0o755)
    env["NGROK_BIN"] = "/tmp/4b_work/ngrok"
    print("using pre-uploaded ngrok binary", flush=True)

if importlib.util.find_spec("pyngrok") is None:
    print("installing pyngrok ...", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyngrok"],
                   capture_output=True, text=True)

# Single instance: kill any previous launcher/tunnel/API before starting.
subprocess.run(
    "pkill -f start_4b_tunnel.py; pkill -f vm_keepalive.py; "
    "pkill -f 'uvicorn' ; pkill -f ngrok",
    shell=True, capture_output=True, text=True)
time.sleep(3)

subprocess.Popen(
    "nohup python /tmp/4b_work/start_4b_tunnel.py >> /tmp/4b_serve.log 2>&1 &",
    shell=True, cwd="/tmp/4b_work", env=env, start_new_session=True)
print("launcher detached", flush=True)

# Keep this VM alive past Colab's idle disconnect + supervise the launcher.
try:
    subprocess.Popen(
        "nohup python /tmp/4b_work/vm_keepalive.py >> /tmp/4b_data/keepalive_stdout.log 2>&1 &",
        shell=True, cwd="/tmp/4b_work", env=env, start_new_session=True)
    print("keepalive detached", flush=True)
except Exception as exc:
    print("keepalive launch failed:", exc, flush=True)


def get(url, timeout=10):
    try:
        return urllib.request.urlopen(url, timeout=timeout).read().decode()[:400]
    except urllib.error.HTTPError as e:
        return f"HTTP {e.code} {e.read().decode()[:300]}"
    except Exception as exc:
        return f"ERR {type(exc).__name__} {exc}"


ok = False
for _ in range(120):
    body = get("http://127.0.0.1:8000/healthz", timeout=5)
    if body.startswith("{"):
        print("local /healthz:", body, flush=True)
        ok = True
        break
    time.sleep(2)
print("LOCAL_UP:", ok, flush=True)
print("local /readyz:", get("http://127.0.0.1:8000/readyz"), flush=True)

try:
    with open("/tmp/4b_serve.log") as fh:
        txt = fh.read()
    print("SERVE_LOG_HEAD:", txt[:1200], flush=True)
    print("SERVE_LOG_TAIL:", txt[-2500:], flush=True)
except OSError as exc:
    print("serve log unreadable:", exc, flush=True)
print("SETUP_B_DONE", flush=True)
