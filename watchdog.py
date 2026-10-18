import base64
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://portfolio-debate-olive.ngrok-free.dev"
HERE = Path.cwd()
CONFIG = Path.home() / ".config" / "colab-cli"
SESSIONS = HERE / "sessions-4b.json"
COLAB = "colab"
H = {"ngrok-skip-browser-warning": "1"}


def sh(*args, timeout=None):
    p = subprocess.run(list(args), capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def colab_out(*args, timeout=2400):
    env = dict(os.environ)
    p = subprocess.run(
        [COLAB, "--config", str(SESSIONS), *args],
        capture_output=True, text=True, timeout=timeout, env=env,
    )
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def public_ok():
    for _ in range(2):
        try:
            r = urllib.request.urlopen(
                urllib.request.Request(BASE + "/healthz", headers=H), timeout=20)
            if r.status == 200:
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


def prepare_files():
    CONFIG.mkdir(parents=True, exist_ok=True)
    (CONFIG / "token-4b.json").write_bytes(
        base64.b64decode(os.environ["COLAB_TOKEN_4B_B64"]))
    os.environ["COLAB_CLI_TOKEN_FILE"] = str(CONFIG / "token-4b.json")
    SESSIONS.write_bytes(base64.b64decode(os.environ["COLAB_SESSIONS_4B_B64"]))
    (HERE / ".env.4b").write_bytes(base64.b64decode(os.environ["PREDICTION_ENV_4B_B64"]))
    (HERE / "ngrok_4b_token.txt").write_text(os.environ["NGROK_TOKEN_4B"].strip())
    HERE.joinpath("ngrok_4b_domain.txt").write_text("portfolio-debate-olive.ngrok-free.dev")
    root = subprocess.run([sys.executable, "-c",
                           "import site; print(site.getsitepackages()[0])"],
                          capture_output=True, text=True).stdout.strip()
    pkg = Path(root) / "colab_cli"
    pkg.mkdir(exist_ok=True)
    (pkg / "auth.py").write_text((HERE / "vendor" / "colab_cli" / "auth.py").read_text())
    (pkg / "commands").mkdir(exist_ok=True)
    (pkg / "commands" / "automation.py").write_text(
        (HERE / "vendor" / "colab_cli" / "commands" / "automation.py").read_text())


def reconnect(force=False, healthy_first=True):
    if healthy_first and not force and public_ok():
        print("WATCHDOG: tunnel healthy, nothing to do", flush=True)
        return 0
    print("WATCHDOG: tunnel DOWN -> rebuilding", flush=True)
    prepare_files()
    made = None
    for attempt in range(8):
        rc, out = colab_out("new", "-s", "4bgpu", "--gpu", "T4", timeout=300)
        tail = [l for l in out.splitlines() if l.strip()][-1:]
        print(f"try {attempt}: {tail}", flush=True)
        if "READY" in out:
            made = True
            break
        time.sleep(90)
    if not made:
        print("WATCHDOG: GPU unavailable, giving up this run", flush=True)
        return 2
    for f in ["mkdir_cpu.py"]:
        rc, out = colab_out("exec", "-s", "4bgpu", "-f", str(HERE / f),
                            "--timeout", "300", timeout=600)
        print("exec", f, "->", out.strip().splitlines()[-1:], flush=True)
    for fn, remote in [
        ("serve_4b_predictor.py", "/tmp/4b_work/serve_4b_predictor.py"),
        ("start_4b_tunnel.py", "/tmp/4b_work/start_4b_tunnel.py"),
        ("ngrok_4b_token.txt", "/tmp/4b_work/ngrok_4b_token.txt"),
        (".env.4b", "/tmp/4b_work/.env.4b"),
        ("best_4b_chart_model.pt", "/tmp/4b_data/out/best_4b_chart_model.pt"),
    ]:
        ok = False
        for tryup in range(3):
            rc, out = colab_out("upload", str(HERE / fn), remote, timeout=900)
            if "Uploaded" in out:
                ok = True
                break
            print("upload", fn, f"attempt {tryup} failed:", out.strip()[-200:], flush=True)
            time.sleep(5)
        print("upload", fn, "->", "OK" if ok else "FAILED", flush=True)
    for f in ["write_domain.py", "serve_setup_b.py"]:
        colab_out("exec", "-s", "4bgpu", "-f", str(HERE / f), "--timeout", "300", timeout=600)
    rc, out = colab_out("exec", "-s", "4bgpu", "-f", str(HERE / "serve_setup_b.py"),
                        "--timeout", "900", timeout=1500)
    print("serve exec:", out[-600:], flush=True)
    for _ in range(20):
        try:
            r = urllib.request.urlopen(
                urllib.request.Request(BASE + "/healthz", headers=H), timeout=20)
            if r.status == 200:
                print("WATCHDOG: tunnel RESTORED", flush=True)
                return 0
        except urllib.error.HTTPError as e:
            print("healthz:", e.code, flush=True)
        except Exception:
            pass
        time.sleep(10)
    print("WATCHDOG: restore attempt failed", flush=True)
    return 3


if __name__ == "__main__":
    force = "--force" in sys.argv
    sys.exit(reconnect(force=force))
