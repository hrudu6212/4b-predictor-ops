"""Keep this Colab VM from idle-disconnecting.

Runs INSIDE the VM: every 60s it pings Colab's keep-alive endpoint for this
assignment, which refreshes the idle timer exactly like an open notebook tab.
Requires (uploaded by the watchdog):
  /tmp/4b_data/token-4b.json      OAuth token for the Colab account
  /tmp/4b_data/keepalive_ep.txt   assignment endpoint id (from sessions json)
"""
import json
import os
import sys
import time

import requests
from google.auth.transport.requests import Request as GoogleRequest
from google.oauth2.credentials import Credentials

TOKEN_PATH = os.environ.get("KEEPALIVE_TOKEN", "/tmp/4b_data/token-4b.json")
EP_PATH = os.environ.get("KEEPALIVE_EP", "/tmp/4b_data/keepalive_ep.txt")
LOG_PATH = os.environ.get("KEEPALIVE_LOG", "/tmp/4b_data/keepalive.log")
SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/colaboratory",
    "https://www.googleapis.com/auth/drive.file",
]


def log(msg):
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}"
    print(line, flush=True)
    try:
        with open(LOG_PATH, "a") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


def main():
    endpoint = open(EP_PATH).read().strip()
    creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
    sess = requests.Session()
    url = f"https://colab.research.google.com/tun/m/{endpoint}/keep-alive/"
    log(f"keepalive start endpoint={endpoint}")

    consecutive_4xx = 0
    while True:
        try:
            if not creds.valid:
                creds.refresh(GoogleRequest())
                with open(TOKEN_PATH, "w") as fh:
                    fh.write(creds.to_json())
            headers = {
                "Authorization": f"Bearer {creds.token}",
                "Accept": "application/json",
                "X-Colab-Client-Agent": "colab-cli",
                "X-Colab-Tunnel": "Google",
            }
            try:
                r = sess.get(url, headers=headers, params={"authuser": "0"}, timeout=10)
            except requests.exceptions.ReadTimeout:
                r = None
            if r is not None and 400 <= r.status_code < 500:
                consecutive_4xx += 1
                log(f"keepalive http={r.status_code} body={r.text[:200]}")
                if consecutive_4xx >= 3:
                    log("keepalive stop: assignment gone (consecutive 4xx)")
                    sys.exit(0)
            else:
                consecutive_4xx = 0
        except Exception as exc:
            log(f"keepalive error: {type(exc).__name__}: {exc}")
        time.sleep(60)


if __name__ == "__main__":
    main()
