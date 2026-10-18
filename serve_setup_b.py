"""Colab setup B: launch the REAL 4B API + ngrok tunnel (detached), verify local."""
import subprocess, time, urllib.request, urllib.error, os

CKPT = "/tmp/4b_data/out/best_4b_chart_model.pt"
assert os.path.exists(CKPT), f"checkpoint missing: {CKPT}"
env = dict(os.environ, MODEL_CHECKPOINT=CKPT)
log = open("/tmp/4b_serve.log", "w")
subprocess.Popen(["nohup", "python", "/tmp/4b_work/start_4b_tunnel.py"],
                 cwd="/tmp/4b_work", env=env,
                 stdout=log, stderr=subprocess.STDOUT,
                 stdin=subprocess.DEVNULL, start_new_session=True)
print("launcher detached", flush=True)

ok = False
for _ in range(90):
    try:
        body = urllib.request.urlopen("http://127.0.0.1:8000/healthz", timeout=5).read().decode()
        print("local /healthz:", body, flush=True)
        ok = True
        break
    except Exception:
        time.sleep(2)
print("LOCAL_UP:", ok, flush=True)

try:
    print("local /readyz:", urllib.request.urlopen("http://127.0.0.1:8000/readyz", timeout=10).read().decode()[:250], flush=True)
except urllib.error.HTTPError as e:
    print("local /readyz:", e.code, e.read().decode()[:250], flush=True)
print("SETUP_B_DONE", flush=True)
