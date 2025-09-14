#!/usr/bin/env python3
"""
Interactive terminal chat for a single Bitrix24 dialog (incl. Open Lines).

Features
- Reads B24_WEBHOOK_BASE from .env (no external deps) or env.
- Follows one dialog in near real-time and prints new messages.
- Lets you type and send messages to that dialog from the terminal.

Usage
  # .env must contain B24_WEBHOOK_BASE=... (or export it)
  python3 b24_chat_tty.py chat97669
  # or with numeric chat id:
  python3 b24_chat_tty.py 97669

Commands (type and press Enter)
  /quit               exit
  /who                print current dialog id
  /help               show help

Notes
- You must have permission in the dialog: join the Open Line session.
- Requires `im` scope.
"""

import json
import os
import queue
import sys
import threading
import time
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional


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
POLL_INTERVAL_SEC = float(os.environ.get("B24_POLL_SEC", "2"))


def call_rest(method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not WEBHOOK_BASE:
        raise SystemExit("B24_WEBHOOK_BASE is not set. Put it in .env or export the variable.")
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


def get_profile() -> Dict[str, Any]:
    return call_rest("profile").get("result", {})


def get_user_name(uid: int, cache: Dict[int, str]) -> str:
    if uid in cache:
        return cache[uid]
    try:
        r = call_rest("im.user.get", {"ID": uid})
        u = r.get("result") or {}
        name = u.get("name") or u.get("NAME") or str(uid)
    except Exception:
        name = str(uid)
    cache[uid] = name
    return name


def fetch_new_messages(dialog_id: str, since_id: Optional[int]) -> List[Dict[str, Any]]:
    params: Dict[str, Any] = {"DIALOG_ID": dialog_id, "LIMIT": 50}
    if since_id:
        # Fetch messages NEWER than since_id. Use FIRST_ID (not LAST_ID).
        params.update({"FIRST_ID": since_id})
    r = call_rest("im.dialog.messages.get", params)
    res = r.get("result")
    msgs: List[Dict[str, Any]] = []
    if isinstance(res, dict):
        msgs = res.get("messages") or res.get("MESSAGES") or res.get("items") or res.get("ITEMS") or []
    elif isinstance(res, list):
        msgs = res
    # Normalize order oldest->newest
    msgs = sorted(msgs, key=lambda m: int(m.get("id") or m.get("ID") or 0))
    # Filter strictly greater than since_id
    if since_id is not None:
        msgs = [m for m in msgs if (int(m.get("id") or m.get("ID") or 0) > since_id)]
    return msgs


def add_message(dialog_id: str, text: str) -> Optional[int]:
    r = call_rest("im.message.add", {"DIALOG_ID": dialog_id, "MESSAGE": text})
    return r.get("result") if isinstance(r, dict) else None


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python3 b24_chat_tty.py <dialog_id|chat_id>")
        sys.exit(2)
    dialog = normalize_dialog_id(sys.argv[1])

    me = get_profile()
    print(f"Chat TTY as user ID {me.get('ID')} on {dialog}. Type /quit to exit.")

    # Seed last id by fetching the latest page then taking the max id
    seed = fetch_new_messages(dialog, None)
    last_id = 0
    for m in seed:
        mid = int(m.get("id") or m.get("ID") or 0)
        last_id = max(last_id, mid)

    name_cache: Dict[int, str] = {}
    outgoing: "queue.Queue[str]" = queue.Queue()
    stop = threading.Event()

    def input_worker():
        try:
            while not stop.is_set():
                line = input()
                if not line:
                    continue
                if line.strip() in ("/q", "/quit", ":q"):
                    stop.set()
                    break
                if line.strip() in ("/help", "?"):
                    print("Commands: /quit, /who, /help")
                    continue
                if line.strip() == "/who":
                    print(f"Dialog: {dialog}")
                    continue
                outgoing.put(line)
        except EOFError:
            stop.set()

    def poll_worker():
        nonlocal last_id
        while not stop.is_set():
            try:
                # Send queued messages first
                while True:
                    try:
                        text = outgoing.get_nowait()
                    except queue.Empty:
                        break
                    mid = add_message(dialog, text)
                    if mid:
                        last_id = max(last_id, int(mid))
                        print(f"[me] {text}")
                # Fetch new incoming messages
                msgs = fetch_new_messages(dialog, last_id)
                for m in msgs:
                    mid = int(m.get("id") or m.get("ID") or 0)
                    text = m.get("text") or m.get("TEXT") or m.get("message") or m.get("MESSAGE") or ""
                    author = m.get("author_id") or m.get("AUTHOR_ID")
                    if isinstance(author, str) and author.isdigit():
                        author = int(author)
                    who = get_user_name(author, name_cache) if isinstance(author, int) else str(author)
                    print(f"[{who}] {text}")
                    last_id = max(last_id, mid)
            except Exception as e:
                print(f"WARN: {e}")
            time.sleep(POLL_INTERVAL_SEC)

    t_in = threading.Thread(target=input_worker, daemon=True)
    t_poll = threading.Thread(target=poll_worker, daemon=True)
    t_in.start()
    t_poll.start()
    try:
        while not stop.is_set():
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    stop.set()
    print("Bye")


if __name__ == "__main__":
    main()
