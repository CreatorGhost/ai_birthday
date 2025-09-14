#!/usr/bin/env python3
"""
Send a message to a Bitrix24 dialog (including Open Lines) using an incoming webhook.

Usage
  export B24_WEBHOOK_BASE="https://your.b24/rest/{user}/{code}"
  python3 b24_send_message.py chat97669 "Hello from terminal"

You can also pass a numeric chat id and it will be normalized to dialog id:
  python3 b24_send_message.py 97669 "Hi"

Notes
- Requires `im` scope. The webhook user must have permission in the dialog
  (e.g., be an operator and joined to the Open Lines session).
"""

import json
import os
import sys
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional


def _load_dotenv(path: str = ".env") -> None:
    if not os.path.isfile(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                if s.startswith("export "):
                    s = s[len("export "):]
                if "=" not in s:
                    continue
                k, v = s.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                os.environ.setdefault(k, v)
    except Exception:
        pass


_load_dotenv()
WEBHOOK_BASE = os.environ.get("B24_WEBHOOK_BASE", "").rstrip("/")


def call_rest(method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not WEBHOOK_BASE:
        raise SystemExit("Please set B24_WEBHOOK_BASE env var to your incoming webhook base URL.")
    url = f"{WEBHOOK_BASE}/{method}.json"
    data = None
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    if params:
        data = urllib.parse.urlencode(params, doseq=True).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=20) as resp:
        payload = resp.read()
        return json.loads(payload.decode("utf-8"))


def normalize_dialog_id(d: str) -> str:
    d = str(d).strip()
    if d.startswith("chat") or d.startswith("user"):
        return d
    if d.isdigit():
        return f"chat{d}"
    return d


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python3 b24_send_message.py <dialog_id|chat_id> <message>")
        sys.exit(2)
    dialog = normalize_dialog_id(sys.argv[1])
    message = " ".join(sys.argv[2:]).strip()
    if not message:
        print("Message is empty; nothing to send.")
        sys.exit(2)

    # Optional: verify profile
    profile = call_rest("profile").get("result", {})
    print(f"Sending as user ID {profile.get('ID')} to {dialog}…")

    res = call_rest("im.message.add", {"DIALOG_ID": dialog, "MESSAGE": message})
    if "result" in res:
        print(f"Sent. Message ID: {res['result']}")
    else:
        print(f"Unexpected response: {res}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
