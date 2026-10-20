"""Start the REAL 4B chart predictor + ngrok tunnel (reserved domain).

Usage:
    python start_4b_tunnel.py
    python start_4b_tunnel.py --port 8000
    python start_4b_tunnel.py --no-tunnel     # local API only

What it does:
  1. Reads ngrok credentials: $NGROK_AUTHTOKEN / ngrok_token.txt,
     $NGROK_DOMAIN / ngrok_domain.txt (your reserved domain).
  2. Reads/creates PREDICTION_API_KEY: $PREDICTION_API_KEY env or .env.4b
     (a long random bearer token is generated on first run).
  3. Boots serve_4b_predictor.api_app on 127.0.0.1:<port> in a thread.
     The predictor is REAL-ONLY: without torch + the trained
     best_4b_chart_model.pt it answers 503 (never fakes a prediction).
  4. Opens the ngrok tunnel on the reserved domain and prints the
     plug-in block for your website backend.

Plug-in (website backend -> Colab-style base URL):
    COLAB_API_URL=https://<your-reserved-domain>/v1
    COLAB_MODEL=4b-stock-movement-resnet18
    COLAB_API_KEY=<printed key>

Secrets stay in ngrok_token.txt / .env.4b (both gitignored) — never hardcoded.
"""

from __future__ import annotations

import argparse
import os
import secrets
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL_ID = "4b-stock-movement-resnet18"


def read_secret(env_name: str, *paths: str) -> str:
    val = os.environ.get(env_name, "").strip()
    if val:
        return val
    for p in paths:
        try:
            fp = Path(p)
            if not fp.is_absolute():
                fp = HERE / p
            if fp.is_file():
                for line in fp.read_text().splitlines():
                    line = line.strip()
                    if line and not line.startswith("#"):
                        return line
        except OSError:
            pass
    return ""


def clean_domain(raw: str) -> str:
    return raw.strip().replace("https://", "").replace("http://", "").strip().strip("/")


def ensure_api_key() -> tuple[str, bool]:
    key = os.environ.get("PREDICTION_API_KEY", "").strip()
    if key:
        return key, False
    dotenv = HERE / ".env.4b"
    if dotenv.is_file():
        for line in dotenv.read_text().splitlines():
            if line.strip().startswith("PREDICTION_API_KEY="):
                key = line.split("=", 1)[1].strip().strip("'\"")
                if key:
                    os.environ["PREDICTION_API_KEY"] = key
                    return key, False
    key = secrets.token_urlsafe(32)
    with dotenv.open("a") as fh:
        fh.write(f"PREDICTION_API_KEY={key}\n")
    os.environ["PREDICTION_API_KEY"] = key
    return key, True


def wait_local(port: int, timeout: int = 60) -> bool:
    for _ in range(timeout):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT_4B", "8000")))
    ap.add_argument("--no-tunnel", action="store_true")
    args = ap.parse_args()
    port = args.port
    os.environ["PORT_4B"] = str(port)

    # Dedicated 4B credentials first (may belong to a different ngrok account
    # than the LLM tunnels), falling back to the shared ones.
    token = read_secret(
        "NGROK_AUTHTOKEN_4B", "ngrok_4b_token.txt", "/tmp/ngrok_4b_token.txt"
    ) or read_secret("NGROK_AUTHTOKEN", "ngrok_token.txt", "/tmp/ngrok_token.txt")
    # Dedicated 4B endpoint: NGROK_DOMAIN_4B / ngrok_4b_domain.txt wins so the two
    # LLM tunnels keep their domains. Falls back to NGROK_DOMAIN / ngrok_domain.txt.
    domain = clean_domain(
        read_secret(
            "NGROK_DOMAIN_4B", "ngrok_4b_domain.txt", "/tmp/ngrok_4b_domain.txt"
        )
        or read_secret("NGROK_DOMAIN", "ngrok_domain.txt", "/tmp/ngrok_domain.txt")
    )
    api_key, generated = ensure_api_key()
    if generated:
        print("[4b] generated PREDICTION_API_KEY and saved it to .env.4b (gitignored).", flush=True)

    if not args.no_tunnel and not token:
        sys.exit("NGROK authtoken missing. Put it in ngrok_4b_token.txt (or ngrok_token.txt) or $NGROK_AUTHTOKEN_4B.")
    if not args.no_tunnel and not domain:
        sys.exit("NGROK domain missing. Reserve one at dashboard.ngrok.com -> Domains, "
                 "put it in ngrok_domain.txt or $NGROK_DOMAIN.")

    # --- boot the real predictor -------------------------------------------
    import uvicorn
    import serve_4b_predictor as predictor

    # Re-read key in case the launcher just generated it (module import order).
    if not predictor.PREDICTION_API_KEY and api_key:
        print("[4b] WARNING: predictor module loaded without a key; restart required.", flush=True)

    server = uvicorn.Server(uvicorn.Config(predictor.app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    if not wait_local(port):
        sys.exit(f"[4b] API did not come up on 127.0.0.1:{port}.")
    print(f"[4b] local API: http://127.0.0.1:{port}", flush=True)
    try:
        body = urllib.request.urlopen(f"http://127.0.0.1:{port}/readyz", timeout=10).read().decode()
        print(f"[4b] local /readyz: {body[:300]}", flush=True)
    except Exception as exc:
        print(f"[4b] local /readyz not ready yet: {exc}", flush=True)
        print("[4b] predictions will return 503 until the trained checkpoint is installed.", flush=True)

    if args.no_tunnel:
        print("[4b] tunnel skipped (--no-tunnel). Press Ctrl+C to stop.", flush=True)
        try:
            while thread.is_alive():
                time.sleep(1)
        except KeyboardInterrupt:
            server.should_exit = True
        return

    # --- ngrok tunnel on the reserved domain --------------------------------
    # Drive the agent binary directly: the current stable agent uses the new
    # "endpoints/--url" schema, which pyngrok's local-API call (domain=/hostname=)
    # rejects with "invalid endpoint configuration".
    import json

    ngrok_bin = os.environ.get("NGROK_BIN", "/tmp/4b_work/ngrok")
    if not os.path.exists(ngrok_bin):
        try:
            print("[4b] ngrok binary missing; downloading ...", flush=True)
            from pyngrok import installer
            installer.install_ngrok(ngrok_bin)
        except Exception as exc:
            sys.exit(f"[4b] ngrok binary missing and download failed: {exc}")
    try:
        os.chmod(ngrok_bin, 0o755)
    except OSError:
        pass
    print(f"[4b] ngrok binary: {ngrok_bin}", flush=True)

    api = "http://127.0.0.1:4040/api/tunnels"
    agent: dict = {"proc": None}

    def _kill():
        p = agent.get("proc")
        if p is not None and p.poll() is None:
            try:
                p.terminate()
                p.wait(timeout=10)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
        agent["proc"] = None

    def open_tunnel():
        _kill()
        subprocess.run(f"pkill -f '{ngrok_bin}'", shell=True, capture_output=True)
        time.sleep(2)
        cmd = [ngrok_bin, "http", str(port), "--authtoken", token, "--log=stdout"]
        if domain:
            cmd += ["--url", f"https://{domain}"]
        print(f"[4b] launching ngrok agent (url={domain or 'auto'})", flush=True)
        env2 = dict(os.environ, NGROK_AUTHTOKEN=token)
        agent["proc"] = subprocess.Popen(cmd, stdout=sys.stdout, stderr=subprocess.STDOUT,
                                         stdin=subprocess.DEVNULL, env=env2)
        for _ in range(60):
            if agent["proc"].poll() is not None:
                raise RuntimeError(f"ngrok agent exited (code {agent['proc'].returncode})")
            try:
                data = json.loads(urllib.request.urlopen(api, timeout=3).read().decode())
                for t in data.get("tunnels", []):
                    url = t.get("public_url", "")
                    if url.startswith("https"):
                        return url
            except Exception:
                pass
            time.sleep(1)
        raise RuntimeError("ngrok agent did not publish a tunnel within 60s")

    try:
        tunnel = None
        last_exc = None
        for attempt in range(4):
            try:
                tunnel = open_tunnel()
                break
            except Exception as exc:
                last_exc = exc
                print(f"[4b] tunnel attempt {attempt} failed: {str(exc)[:200]}", flush=True)
                time.sleep(10)
        if tunnel is None:
            raise last_exc
    except Exception as exc:
        sys.exit(f"[4b] could not open ngrok tunnel: {exc}")
    public = tunnel.replace("http://", "https://", 1)
    base_v1 = public.rstrip("/") + "/v1"

    print("", flush=True)
    print("==============================================================", flush=True)
    print(f"  4B predictor PUBLIC base : {base_v1}", flush=True)
    print(f"  Predict endpoint         : {base_v1}/predict", flush=True)
    print(f"  Model                    : {MODEL_ID}", flush=True)
    print("  Plug these into your website backend:", flush=True)
    print(f"    COLAB_API_URL={base_v1}", flush=True)
    print(f"    COLAB_MODEL={MODEL_ID}", flush=True)
    print(f"    COLAB_API_KEY={api_key}", flush=True)
    print("==============================================================", flush=True)
    print("", flush=True)
    print("curl test (from your backend, key stays server-side):", flush=True)
    print(f'  curl -X POST "{base_v1}/predict" \\', flush=True)
    print(f'    -H "Authorization: Bearer $COLAB_API_KEY" \\', flush=True)
    print('    -F "file=@chart.png;type=image/png" \\', flush=True)
    print('    -F "text=Predict the next movement and show confidence."', flush=True)
    print("[4b] tunnel watchdog running (30s checks). Press Ctrl+C to stop.", flush=True)

    stop = threading.Event()

    def watchdog() -> None:
        while not stop.wait(30):
            try:
                req = urllib.request.Request(
                    public.rstrip("/") + "/healthz",
                    headers={"ngrok-skip-browser-warning": "1"},
                )
                urllib.request.urlopen(req, timeout=15).read()
            except Exception as exc:
                print(f"[ngrok] public check failed ({exc}); reconnecting ...", flush=True)
                try:
                    t2 = open_tunnel()
                    print(f"[ngrok] reconnected: {t2}", flush=True)
                except Exception as exc2:
                    print(f"[ngrok] reconnect failed: {exc2}", flush=True)

    threading.Thread(target=watchdog, daemon=True).start()
    try:
        while thread.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        print("[4b] stopping ...", flush=True)
    finally:
        stop.set()
        server.should_exit = True
        _kill()


if __name__ == "__main__":
    main()
