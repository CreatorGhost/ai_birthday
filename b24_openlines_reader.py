#!/usr/bin/env python3
"""
Bitrix24 Open Lines (WhatsApp) message reader

Reads incoming messages from Bitrix24 Open Lines via IM REST API by polling.

Requirements
- Incoming webhook with scopes: at minimum `im`; ideally also `imopenlines`.
- Webhook base URL like: https://your.b24.com/rest/{user_id}/{code}

Usage
  export B24_WEBHOOK_BASE="https://leoandloona.bitrix24.com/rest/23/v0x4zd0icufgymjp"
  python3 b24_openlines_reader.py

Notes
- With only `crm` scope, IM/OL methods are unavailable. The script will detect
  this and print instructions to create a new webhook.
- Poll interval is 2 seconds by default. Adjust POLL_INTERVAL_SEC if needed.
"""

import json
import os
import sys
import time
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader (no external deps)."""
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
        raise SystemExit("Please set B24_WEBHOOK_BASE env var to your incoming webhook base URL.")
    url = f"{WEBHOOK_BASE}/{method}.json"
    data = None
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    if params:
        data = urllib.parse.urlencode(params, doseq=True).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            payload = resp.read()
            return json.loads(payload.decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {e.code} calling {method}: {body}") from e


def get_scopes() -> List[str]:
    res = call_rest("scope")
    return res.get("result", []) or []


def get_profile() -> Dict[str, Any]:
    res = call_rest("profile")
    return res.get("result", {})


def try_recent_list() -> Optional[List[Dict[str, Any]]]:
    """Try im.recent.list, fallback to im.recent.get.

    Returns a list of items, or None if neither method is available.
    """
    for method in ("im.recent.list", "im.recent.get"):
        try:
            res = call_rest(method)
        except RuntimeError as e:
            # Method may not exist due to missing scope
            if "ERROR_METHOD_NOT_FOUND" in str(e) or "denied" in str(e):
                continue
            raise
        if "result" in res:
            items = res["result"].get("items") if isinstance(res["result"], dict) else res["result"]
            if isinstance(items, list):
                return items
    return None


def get_openlines_configs() -> List[Dict[str, Any]]:
    """Return list of Open Lines configs with queues.

    Uses imopenlines.config.list.get which is available with 'imopenlines' scope.
    """
    try:
        res = call_rest("imopenlines.config.list.get")
    except Exception:
        return []
    items = res.get("result")
    if isinstance(items, list):
        return items
    return []


def extract_last_message_info(item: Dict[str, Any]) -> Tuple[Optional[int], Optional[str]]:
    """Extract last message id and text from an im.recent.* item.

    Tries several known shapes to be resilient across portal versions.
    """
    msg = None
    for key in ("message", "MESSAGE", "last_message", "LAST_MESSAGE"):
        if isinstance(item.get(key), dict):
            msg = item[key]
            break
    if not isinstance(msg, dict):
        return None, None
    mid = None
    for k in ("id", "ID"):
        if isinstance(msg.get(k), int):
            mid = msg[k]
            break
        # some portals return string ids
        if isinstance(msg.get(k), str) and msg[k].isdigit():
            mid = int(msg[k])
            break
    text = None
    for k in ("text", "TEXT", "message", "MESSAGE"):
        if isinstance(msg.get(k), str):
            text = msg[k]
            break
    return mid, text


def item_dialog_id(item: Dict[str, Any]) -> Optional[str]:
    for k in ("dialogId", "DIALOG_ID", "dialog_id"):
        v = item.get(k)
        if isinstance(v, str):
            return v
    # some shapes store dialog under item["id"] or chat["id"]
    v = item.get("id")
    if isinstance(v, str):
        return v
    chat = item.get("chat") or item.get("CHAT")
    if isinstance(chat, dict):
        for k in ("dialogId", "DIALOG_ID", "dialog_id"):
            v = chat.get(k)
            if isinstance(v, str):
                return v
        # sometimes dialog id is chat{ID} where ID is numeric
        for k in ("id", "ID"):
            cv = chat.get(k)
            if isinstance(cv, int):
                return f"chat{cv}"
            if isinstance(cv, str) and cv.isdigit():
                return f"chat{cv}"
    return None


def item_is_openlines(item: Dict[str, Any]) -> bool:
    """Best-effort check if the recent item is an Open Lines dialog.

    We check typical flags: chat.type == 'LINES' or entityType == 'LINES'.
    If uncertain, return True to avoid missing messages — you can refine later.
    """
    chat = item.get("chat") or item.get("CHAT")
    if isinstance(chat, dict):
        ctype = chat.get("type") or chat.get("TYPE") or chat.get("entityType") or chat.get("ENTITY_TYPE")
        if isinstance(ctype, str) and ctype.upper() == "LINES":
            return True
    et = item.get("entityType") or item.get("ENTITY_TYPE")
    if isinstance(et, str) and et.upper() == "LINES":
        return True
    # Fallback: not sure, allow
    return True


def main() -> None:
    print("Bitrix24 Open Lines reader starting…", flush=True)
    if not WEBHOOK_BASE:
        print("ERROR: B24_WEBHOOK_BASE not set.")
        sys.exit(2)

    scopes = get_scopes()
    if "im" not in scopes:
        print("ERROR: Your webhook lacks 'im' scope. Current scopes:", scopes)
        print("- Create a new Incoming Webhook in Bitrix24 with at least 'im' (and ideally 'imopenlines').")
        print("- Then set B24_WEBHOOK_BASE to that new webhook and rerun.")
        sys.exit(3)

    profile = get_profile()
    my_uid = profile.get("ID")
    print(f"Authorized as user ID {my_uid}. Polling every {POLL_INTERVAL_SEC:.0f}s…")

    # Check Open Lines queue membership if imopenlines scope is present
    if "imopenlines" in scopes:
        configs = get_openlines_configs()
        if configs:
            in_lines = []
            for cfg in configs:
                line_id = cfg.get("ID")
                line_name = cfg.get("LINE_NAME") or cfg.get("NAME")
                queue = set(cfg.get("QUEUE", []))
                if isinstance(my_uid, int):
                    in_q = str(my_uid) in queue
                else:
                    in_q = my_uid in queue
                in_lines.append((line_id, line_name, in_q))
            not_in_any = all(not x[2] for x in in_lines)
            if not_in_any:
                print("WARNING: This user is not in any Open Lines queue.")
                for lid, lname, _ in in_lines:
                    print(f"- Line {lid}: {lname} — not in queue")
                print("Add this user to the WhatsApp Open Line queue (or grant Supervisor) and rerun.")
            else:
                print("Open Lines access:")
                for lid, lname, in_q in in_lines:
                    status = "in queue" if in_q else "not in queue"
                    print(f"- Line {lid}: {lname} — {status}")

    items = try_recent_list()
    if items is None:
        print("ERROR: Neither im.recent.list nor im.recent.get is available. Check webhook scopes/plan.")
        sys.exit(4)

    last_seen: Dict[str, int] = {}
    # Seed last seen from current state to avoid replay
    for it in items:
        if not item_is_openlines(it):
            continue
        did = item_dialog_id(it)
        mid, _ = extract_last_message_info(it)
        if did and isinstance(mid, int):
            last_seen[did] = mid

    print(f"Seeded {len(last_seen)} dialogs from recents.")
    if not last_seen:
        print("NOTE: No IM recents visible to this user yet. If this is an operator account, send a test WhatsApp message and ensure the user is in the Open Line queue.")

    # Poll loop
    while True:
        time.sleep(POLL_INTERVAL_SEC)
        try:
            items = try_recent_list()
        except Exception as e:
            print(f"WARN: failed to poll recent: {e}")
            continue
        if not items:
            continue
        for it in items:
            if not item_is_openlines(it):
                continue
            did = item_dialog_id(it)
            mid, text = extract_last_message_info(it)
            if not did or not isinstance(mid, int):
                continue
            prev = last_seen.get(did)
            if prev is None:
                # new dialog appeared — treat as new
                last_seen[did] = mid
                if text:
                    print(f"[NEW DIALOG {did}] {text}")
                else:
                    print(f"[NEW DIALOG {did}] message id {mid}")
                continue
            if mid > prev:
                last_seen[did] = mid
                # Print last message text if available
                if text:
                    print(f"[{did}] {text}")
                else:
                    print(f"[{did}] new message id {mid}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
