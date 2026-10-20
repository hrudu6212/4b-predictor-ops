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
    try:
        from pyngrok import conf, ngrok
    except ImportError:
        print("[4b] installing pyngrok ...", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyngrok"], check=True)
        from pyngrok import conf, ngrok  # type: ignore

    # Prefer a pre-uploaded binary (avoids slow/flaky in-VM downloads).
    ngrok_bin = os.environ.get("NGROK_BIN", "/tmp/4b_work/ngrok")
    cfg = conf.PyngrokConfig(ngrok_path=ngrok_bin) if os.path.exists(ngrok_bin) else None
    print(f"[4b] ngrok binary: {ngrok_bin} (exists={cfg is not None})", flush=True)

    def _set_token():
        if cfg is not None:
            ngrok.set_auth_token(token, pyngrok_config=cfg)
        else:
            ngrok.set_auth_token(token)

    def _kill():
        try:
            if cfg is not None:
                ngrok.kill(pyngrok_config=cfg)
            else:
                ngrok.kill()
        except Exception:
            pass

    _set_token()
    _kill()

    def open_tunnel():
        kw = {"pyngrok_config": cfg} if cfg is not None else {}
        if domain:
            return ngrok.connect(port, proto="http", domain=domain, **kw)
        return ngrok.connect(port, proto="http", **kw)

    try:
        tunnel = None
        last_exc = None
        for attempt in range(6):
            try:
                tunnel = open_tunnel()
                break
            except Exception as exc:
                last_exc = exc
                msg = str(exc)
                print(f"[4b] tunnel open attempt {attempt} failed: {msg[:200]}", flush=True)
                if "already online" in msg or "ERR_NGROK_334" in msg:
                    _kill()
                    time.sleep(15)
                    continue
                raise
        if tunnel is None:
            raise last_exc
    except Exception as exc:
        msg = str(exc)
        if "already online" in msg or "ERR_NGROK_334" in msg:
            sys.exit(
                f"[4b] Reserved domain https://{domain} is already online on another "
                "tunnel (right now it serves your Ollama LLMs on :11434).\n"
                "Pick one:\n"
                "  1. Stop the other tunnel using that domain, then re-run.\n"
                "  2. Reserve a 3rd domain for the 4B predictor "
                "(dashboard.ngrok.com -> Domains) and put it in ngrok_4b_domain.txt "
                "or $NGROK_DOMAIN_4B (leaves both LLM tunnels untouched).\n"
                "Sharing one reserved domain between Ollama and the 4B API is not "
                "supported (ngrok would load-balance chart uploads to the LLM)."
            )
        if "reserved for another account" in msg or "ERR_NGROK_320" in msg:
            sys.exit(
                f"[4b] Domain https://{domain} is reserved under a DIFFERENT ngrok "
                "account than this tunnel's authtoken.\n"
                "Fix: log in to dashboard.ngrok.com as the account that reserved it, "
                "copy its authtoken (Get Started -> Your Authtoken), and save it as "
                "ngrok_4b_token.txt (or $NGROK_AUTHTOKEN_4B), then re-run.\n"
                "The 4B token is read first, so your LLM tunnels keep using theirs."
            )
        raise
    public = tunnel.public_url.replace("http://", "https://", 1)
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
                    _kill()
                    t2 = open_tunnel()
                    print(f"[ngrok] reconnected: {t2.public_url}", flush=True)
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
