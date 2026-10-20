"""VM diagnostics: what is running and what the launcher logged."""
import subprocess
import urllib.request


def sh(cmd):
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        return (p.stdout or "") + (p.stderr or "")
    except Exception as exc:
        return f"ERR {exc}"


print("== ps ==")
print(sh("ps -ef | grep -E 'start_4b|uvicorn|ngrok|vm_keepalive' | grep -v grep"))
print("== port 8000 ==")
print(sh("ss -ltnp 2>/dev/null | grep 8000 || true"))
print("== files ==")
print(sh("ls -l /tmp/4b_serve.log /tmp/4b_work/nohup.out /tmp/4b_data/keepalive.log "
         "/tmp/4b_data/keepalive_stdout.log 2>&1"))
print("== serve log ==")
print(sh("tail -c 2000 /tmp/4b_serve.log 2>&1"))
print("== nohup.out ==")
print(sh("tail -c 2000 /tmp/4b_work/nohup.out 2>&1"))
print("== keepalive stdout ==")
print(sh("tail -c 1000 /tmp/4b_data/keepalive_stdout.log 2>&1"))
print("== local readyz ==")
try:
    print(urllib.request.urlopen("http://127.0.0.1:8000/readyz", timeout=10).read().decode()[:300])
except Exception as exc:
    print("ERR", exc)
print("DIAG_DONE")
