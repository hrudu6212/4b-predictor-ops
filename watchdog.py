"""4B predictor keepalive — resident GitHub Actions healer.

The chart predictor (4B chart-image model) lives on a Colab VM behind the
reserved ngrok domain portfolio-debate-olive.ngrok-free.dev. This runner keeps
that public endpoint alive:

  * every 45s it checks /healthz (tunnel) and /readyz (model loaded)
  * when the endpoint stays down it rebuilds the whole VM: new Colab session
    (T4 first, CPU fallback), re-push of the server files + checkpoint, boot of
    the API and the ngrok tunnel, then verification
  * the Colab VM keeps itself alive from the inside (vm_keepalive.py pings the
    assignment keep-alive endpoint and supervises the launcher), so this runner
    only has to notice a dead VM and rebuild it

Why the runner stays resident instead of rebuilding once per cron tick:
scheduled workflows in this private repository fire every 4-5 hours in
practice, so after a Colab reclamation the endpoint was dark for hours. The
runner therefore holds the job for ~5h50m and hands over to a successor
workflow run shortly before GitHub's 6-hour job limit. The cron entry stays as
a backstop that restarts a broken chain.

Endpoints:
    https://portfolio-debate-olive.ngrok-free.dev/healthz  tunnel alive
    https://portfolio-debate-olive.ngrok-free.dev/readyz   model loaded
"""

import base64
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE = "https://portfolio-debate-olive.ngrok-free.dev"
HERE = Path.cwd()
CONFIG = Path.home() / ".config" / "colab-cli"
SESSIONS = HERE / "sessions-4b.json"
COLAB = "colab"
H = {"ngrok-skip-browser-warning": "1"}
SESSION_NAME = "4bgpu"

MAX_RUN_SECONDS = 5 * 3600 - 300        # leave room for the handover dispatch
CHECK_INTERVAL_SECONDS = 45
FAILURES_BEFORE_REBUILD = 2              # consecutive failed checks
HANDOVER_LEAD_SECONDS = 120
WORKFLOW_FILE = "watchdog.yml"
REPO = os.environ.get("GITHUB_REPOSITORY", "hrudu6212/4b-predictor-ops")
TOKEN_PATH = CONFIG / "token-4b.json"

# The colab CLI reads this override when its auth module is first imported, so
# it has to be set before anything imports colab_cli. Without it the CLI (and
# the adoption helper below) would talk to the default Google account.
os.environ.setdefault("COLAB_CLI_TOKEN_FILE", str(TOKEN_PATH))

REBUILD_IN_PROGRESS = False


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def sh(*args, timeout=None):
    p = subprocess.run(list(args), capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def colab_out(*args, timeout=2400, input_text=None):
    env = dict(os.environ)
    env["COLAB_CLI_TOKEN_FILE"] = str(TOKEN_PATH)
    p = subprocess.run(
        [COLAB, "--config", str(SESSIONS), *args],
        capture_output=True, text=True, timeout=timeout, env=env, input=input_text,
    )
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def prepare_files():
    CONFIG.mkdir(parents=True, exist_ok=True)
    TOKEN_PATH.write_bytes(base64.b64decode(os.environ["COLAB_TOKEN_4B_B64"]))
    os.environ["COLAB_CLI_TOKEN_FILE"] = str(TOKEN_PATH)
    data = json.loads(base64.b64decode(os.environ["COLAB_SESSIONS_4B_B64"]).decode())
    for info in data.values():
        if isinstance(info, dict) and info.get("token_file"):
            info["token_file"] = str(TOKEN_PATH)
    SESSIONS.write_text(json.dumps(data))
    (HERE / ".env.4b").write_bytes(base64.b64decode(os.environ["PREDICTION_ENV_4B_B64"]))
    (HERE / "ngrok_4b_token.txt").write_text(os.environ["NGROK_TOKEN_4B"].strip())
    HERE.joinpath("ngrok_4b_domain.txt").write_text("portfolio-debate-olive.ngrok-free.dev")
    root = subprocess.run([sys.executable, "-c",
                           "import site; print(site.getsitepackages()[0])"],
                          capture_output=True, text=True).stdout.strip()
    pkg = Path(root) / "colab_cli"
    if pkg.exists():
        shutil.rmtree(pkg)
    shutil.copytree(HERE / "vendor" / "colab_cli", pkg)


def ensure_fresh_token():
    """Refresh the Colab OAuth token before it expires.

    A stale token makes `colab new` fail, which is how an earlier revision of
    this watcher ended up unable to rebuild the VM for hours.
    """
    if not TOKEN_PATH.is_file():
        return
    try:
        conf = json.loads(TOKEN_PATH.read_text())
        expiry = conf.get("expiry")
        needs_refresh = True
        if expiry:
            try:
                exp = datetime.fromisoformat(expiry.rstrip("Z")).replace(tzinfo=timezone.utc)
                needs_refresh = (exp - datetime.now(timezone.utc)).total_seconds() < 600
            except Exception:
                needs_refresh = True
        if needs_refresh and conf.get("refresh_token"):
            log("refreshing Colab OAuth token ...")
            data = urllib.parse.urlencode({
                "client_id": conf.get("client_id"),
                "client_secret": conf.get("client_secret"),
                "refresh_token": conf.get("refresh_token"),
                "grant_type": "refresh_token",
            }).encode()
            req = urllib.request.Request("https://oauth2.googleapis.com/token", data=data)
            with urllib.request.urlopen(req, timeout=20) as res:
                payload = json.loads(res.read().decode())
            conf["token"] = payload["access_token"]
            expires_in = payload.get("expires_in", 3600)
            conf["expiry"] = (datetime.now(timezone.utc)
                              + timedelta(seconds=expires_in)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            TOKEN_PATH.write_text(json.dumps(conf))
            log(f"token refreshed, valid for {expires_in}s")
        elif not needs_refresh:
            log(f"token still valid until {expiry}")
    except Exception as exc:
        log(f"token refresh check failed: {type(exc).__name__}: {exc}")


LIVE_PROBE = (
    "import urllib.request\n"
    "try:\n"
    "    print('LIVE_PROBE', urllib.request.urlopen("
    "'http://127.0.0.1:8000/healthz', timeout=8).status)\n"
    "except Exception as exc:\n"
    "    print('LIVE_PROBE none', type(exc).__name__)\n"
)


def adopt_session(name=SESSION_NAME):
    """Register the server-side assignment locally.

    GitHub runners are ephemeral, so the local session registry starts empty
    (or holds a stale endpoint). Without adoption every `colab exec` would
    report the session as lost even though the VM is alive and serving.

    The account can hold more than one runtime, so each candidate is probed for
    a live local API and the serving one is adopted.
    """
    try:
        from colab_cli.common import state
        from colab_cli.state import SessionState
    except Exception as exc:
        log(f"adopt: import failed: {exc}")
        return ""
    try:
        state.config_path = str(SESSIONS)
        assignments = state.client.list_assignments()
    except Exception as exc:
        log(f"adopt: list_assignments failed: {type(exc).__name__}: {exc}")
        return ""
    if not assignments:
        log("adopt: no server assignments")
        return ""

    def accelerator_of(assignment):
        return str(getattr(assignment.accelerator, "value", assignment.accelerator))

    ordered = sorted(assignments, key=lambda a: 0 if "GPU" in accelerator_of(a) else 1)
    registered = ""
    for assignment in ordered:
        accelerator = accelerator_of(assignment)
        try:
            state.store.add(SessionState(
                name=name,
                endpoint=assignment.endpoint,
                url=assignment.runtime_proxy_info.url,
                token=assignment.runtime_proxy_info.token,
                variant=getattr(assignment.variant, "name", "DEFAULT"),
                accelerator=accelerator,
            ))
        except Exception as exc:
            log(f"adopt: register failed: {type(exc).__name__}: {exc}")
            continue
        registered = name
        rc, out = colab_out("exec", "-s", name, "--timeout", "40",
                            timeout=120, input_text=LIVE_PROBE)
        if "LIVE_PROBE 200" in out:
            log(f"adopted live assignment {assignment.endpoint} as '{name}' ({accelerator})")
            return name
        log(f"adopt: {assignment.endpoint} ({accelerator}) is not serving")
    if registered:
        log(f"adopted assignment {ordered[0].endpoint} as '{name}' (no live API found)")
    return registered


def get(url, timeout=20):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=H), timeout=timeout) as r:
            return r.status, r.read().decode(errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode(errors="replace")
    except Exception as exc:
        return None, str(exc)


def public_ok():
    status, _ = get(BASE + "/healthz", timeout=20)
    return status == 200


def ready_ok():
    status, body = get(BASE + "/readyz", timeout=25)
    return status == 200, body


def session_endpoint(name=SESSION_NAME):
    try:
        data = json.loads(SESSIONS.read_text())
    except Exception:
        return ""
    for key, info in data.items():
        if isinstance(info, dict) and (key == name or info.get("name") == name):
            return info.get("endpoint", "")
    return ""


def create_session():
    """Allocate a Colab runtime: T4 GPU first, CPU as the fallback."""
    for extra, tries, pause in ((["--gpu", "T4"], 2, 30), ([], 4, 20)):
        label = "T4" if extra else "CPU"
        for attempt in range(tries):
            rc, out = colab_out("new", "-s", SESSION_NAME, *extra, timeout=300)
            tail = [l for l in out.splitlines() if l.strip()][-1:]
            log(f"create {label} attempt {attempt + 1}/{tries}: {tail}")
            if "READY" in out:
                log(f"{label} session ready")
                return True
            time.sleep(pause)
    log("no runtime available")
    return False


def upload_file(local_name, remote):
    src = Path(local_name)
    if not src.is_absolute():
        src = HERE / local_name
    for attempt in range(3):
        rc, out = colab_out("upload", str(src), remote, timeout=900)
        if "Uploaded" in out:
            log(f"upload {src.name} -> OK")
            return True
        log(f"upload {src.name} attempt {attempt + 1} failed: {out.strip()[-200:]}")
        time.sleep(5)
    log(f"upload {src.name} -> FAILED")
    return False


def rebuild():
    """Recreate the VM, re-push the model and boot the API + tunnel."""
    global REBUILD_IN_PROGRESS
    REBUILD_IN_PROGRESS = True
    try:
        log("=== REBUILD START ===")
        adopt_session()
        colab_out("stop", "-s", SESSION_NAME, timeout=180)
        if not create_session():
            log("rebuild failed: no runtime")
            return False
        for f in ("mkdir_cpu.py", "deps.py"):
            rc, out = colab_out("exec", "-s", SESSION_NAME, "-f", str(HERE / f),
                                "--timeout", "600", timeout=900)
            log(f"exec {f} -> {out.strip()[-200:]}")
        endpoint = session_endpoint()
        log(f"endpoint: {endpoint or 'MISSING'}")
        (HERE / "keepalive_ep.txt").write_text(endpoint)
        uploads = [
            ("serve_4b_predictor.py", "/tmp/4b_work/serve_4b_predictor.py"),
            ("start_4b_tunnel.py", "/tmp/4b_work/start_4b_tunnel.py"),
            ("vm_keepalive.py", "/tmp/4b_work/vm_keepalive.py"),
            ("ngrok_4b_token.txt", "/tmp/4b_work/ngrok_4b_token.txt"),
            (".env.4b", "/tmp/4b_work/.env.4b"),
            ("keepalive_ep.txt", "/tmp/4b_data/keepalive_ep.txt"),
            (str(TOKEN_PATH), "/tmp/4b_data/token-4b.json"),
            ("best_4b_chart_model.pt", "/tmp/4b_data/out/best_4b_chart_model.pt"),
        ]
        if (HERE / "ngrok").exists():
            uploads.append(("ngrok", "/tmp/4b_work/ngrok"))
        for local_name, remote in uploads:
            upload_file(local_name, remote)
        colab_out("exec", "-s", SESSION_NAME, "-f", str(HERE / "write_domain.py"),
                  "--timeout", "300", timeout=600)
        rc, out = colab_out("exec", "-s", SESSION_NAME, "-f", str(HERE / "serve_setup_b.py"),
                            "--timeout", "900", timeout=1500)
        log(f"serve exec: {out[-1500:]}")
        rc, out = colab_out("exec", "-s", SESSION_NAME, "-f", str(HERE / "diag.py"),
                            "--timeout", "300", timeout=600)
        log(f"diag: {out[-1500:]}")
        for attempt in range(20):
            if public_ok():
                ok, body = ready_ok()
                log(f"tunnel RESTORED, readyz: {body[:200]}")
                return ok
            time.sleep(10)
        log("rebuild failed: endpoint did not come up")
        return False
    finally:
        REBUILD_IN_PROGRESS = False


def dispatch_successor():
    token = os.environ.get("GH_DISPATCH_TOKEN")
    if not token:
        log("no GH_DISPATCH_TOKEN set; successor not dispatched")
        return False
    url = f"https://api.github.com/repos/{REPO}/actions/workflows/{WORKFLOW_FILE}/dispatches"
    payload = json.dumps({"ref": "main"}).encode()
    req = urllib.request.Request(
        url, data=payload, method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "User-Agent": "4b-watchdog/2.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            log(f"successor workflow dispatched (HTTP {resp.status})")
            return True
    except Exception as exc:
        log(f"successor dispatch failed: {type(exc).__name__}: {exc}")
        return False


def main():
    force = "--force" in sys.argv or os.environ.get("INPUT_FORCE", "").lower() == "true"
    log("=== 4B predictor keepalive (GitHub Actions) ===")
    prepare_files()
    ensure_fresh_token()
    adopt_session()

    if force:
        log("--force: rebuilding regardless of health")
        rebuild()

    started = time.time()
    failures = 0
    tick = 0
    while True:
        elapsed = time.time() - started
        remaining = MAX_RUN_SECONDS - elapsed
        tick += 1

        if remaining <= HANDOVER_LEAD_SECONDS:
            log("handover threshold reached; dispatching successor workflow")
            if not REBUILD_IN_PROGRESS:
                dispatch_successor()
            time.sleep(15)
            log("handover complete; exiting")
            return 0

        if tick % 10 == 1:
            ensure_fresh_token()

        if public_ok():
            failures = 0
            ok, body = ready_ok()
            log(f"tick {tick}: tunnel alive, ready={ok} {body[:120]}")
        else:
            failures += 1
            log(f"tick {tick}: endpoint DOWN ({failures}/{FAILURES_BEFORE_REBUILD})")
            if failures >= FAILURES_BEFORE_REBUILD:
                failures = 0
                if rebuild():
                    log("rebuild succeeded")
                else:
                    log("rebuild failed; will retry on the next tick")

        time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    sys.exit(main())
