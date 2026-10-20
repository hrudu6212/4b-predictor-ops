"""Colab setup B: launch the REAL 4B API + ngrok tunnel (detached), verify local."""
import os
import subprocess
import time
import urllib.error
import urllib.request

CKPT = "/tmp/4b_data/out/best_4b_chart_model.pt"
assert os.path.exists(CKPT), f"checkpoint missing: {CKPT}"
env = dict(os.environ, MODEL_CHECKPOINT=CKPT)

log = open("/tmp/4b_serve.log", "w")
subprocess.Popen(["nohup", "python", "/tmp/4b_work/start_4b_tunnel.py"],
                 cwd="/tmp/4b_work", env=env,
                 stdout=log, stderr=subprocess.STDOUT,
                 stdin=subprocess.DEVNULL, start_new_session=True)
print("launcher detached", flush=True)

# Keep this VM alive past Colab's idle disconnect (runs inside the VM).
try:
    kl_log = open("/tmp/4b_data/keepalive_stdout.log", "a")
    subprocess.Popen(["nohup", "python", "/tmp/4b_work/vm_keepalive.py"],
                     cwd="/tmp/4b_work", env=env,
                     stdout=kl_log, stderr=subprocess.STDOUT,
                     stdin=subprocess.DEVNULL, start_new_session=True)
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

# Diagnostics: always show what the launcher did (ngrok/tunnel status).
try:
    with open("/tmp/4b_serve.log") as fh:
        txt = fh.read()
    print("SERVE_LOG_HEAD:", txt[:1500], flush=True)
    print("SERVE_LOG_TAIL:", txt[-2500:], flush=True)
except OSError as exc:
    print("serve log unreadable:", exc, flush=True)
print("SETUP_B_DONE", flush=True)
