#!/usr/bin/env python3
"""
Bitrix24 Open Lines → AI Chatbot bridge (HTTP)

Listens for new WhatsApp messages via IM recents polling and forwards them
to an HTTP webhook. Sends the chatbot replies back to the Bitrix chat.

Environment (.env or shell)
- B24_WEBHOOK_BASE: Incoming webhook base URL (required)
- B24_POLL_SEC: poll interval seconds (default 2)
- B24_BUSINESS_PHONE: optional, e.g. +971501754133
- BOT_HTTP_URL: chatbot webhook endpoint (default http://localhost:8000/webhook)
- BOT_HTTP_AUTH: optional bearer or token string to send as Authorization

Limitations
- Incoming webhooks cannot receive push events — we poll.
- The webhook user must be allowed to see dialogs (queue member or joined).
"""

import json
import os
import sys
import time
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple


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
BOT_HTTP_URL = os.environ.get("BOT_HTTP_URL", "http://35.232.52.16:8001/webhook/whatsapp")
BOT_HTTP_AUTH = os.environ.get("BOT_HTTP_AUTH", "")
BOT_HTTP_TIMEOUT = float(os.environ.get("BOT_HTTP_TIMEOUT_SEC", "120"))
BUSINESS_PHONE = os.environ.get("B24_BUSINESS_PHONE", "+971501754133")


def call_rest(method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not WEBHOOK_BASE:
        raise SystemExit("B24_WEBHOOK_BASE is not set")
    url = f"{WEBHOOK_BASE}/{method}.json"
    data = None
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    if params:
        data = urllib.parse.urlencode(params, doseq=True).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=25) as resp:
        payload = resp.read()
        return json.loads(payload.decode("utf-8"))


def http_post_json(url: str, payload: Dict[str, Any], auth: str = "") -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if auth:
        headers["Authorization"] = auth
    req = urllib.request.Request(url, data=data, headers=headers)
    start = time.time()
    with urllib.request.urlopen(req, timeout=BOT_HTTP_TIMEOUT) as resp:
        raw = resp.read().decode("utf-8")
        dur = time.time() - start
        print(f"BOT HTTP {resp.status} in {dur:.1f}s")
        try:
            return json.loads(raw) if raw else {}
        except Exception:
            return {"raw": raw}


def recent_items() -> List[Dict[str, Any]]:
    for method in ("im.recent.list", "im.recent.get"):
        try:
            res = call_rest(method)
        except Exception:
            continue
        r = res.get("result")
        if isinstance(r, dict) and isinstance(r.get("items"), list):
            return r["items"]
        if isinstance(r, list):
            return r
    return []


def item_dialog_id(item: Dict[str, Any]) -> Optional[str]:
    for k in ("dialogId", "DIALOG_ID", "dialog_id"):
        v = item.get(k)
        if isinstance(v, str):
            return v
    v = item.get("id")
    if isinstance(v, str):
        return v
    chat = item.get("chat") or item.get("CHAT")
    if isinstance(chat, dict):
        for k in ("id", "ID"):
            cv = chat.get(k)
            if isinstance(cv, int):
                return f"chat{cv}"
            if isinstance(cv, str) and cv.isdigit():
                return f"chat{cv}"
    return None


def is_lines_chat(item: Dict[str, Any]) -> bool:
    """Return True only for Open Lines chats.

    We check known flags and avoid permissive fallback to prevent 400s
    from imopenlines.* endpoints for non-lines chats.
    """
    chat = item.get("chat") or item.get("CHAT")
    if isinstance(chat, dict):
        et = chat.get("entity_type") or chat.get("ENTITY_TYPE")
        tp = chat.get("type") or chat.get("TYPE")
        if isinstance(et, str) and et.upper() == "LINES":
            return True
        if isinstance(tp, str) and tp.lower() in ("lines", "l"):
            return True
    et_top = item.get("entityType") or item.get("ENTITY_TYPE")
    if isinstance(et_top, str) and et_top.upper() == "LINES":
        return True
    # Not an Open Lines chat
    return False


def get_lines_dialog_meta(dialog_id: str) -> Dict[str, Any]:
    # dialog_id is like chat97669 → extract numeric id
    chat_id = int(dialog_id.replace("chat", "")) if dialog_id.startswith("chat") else int(dialog_id)
    res = call_rest("imopenlines.dialog.get", {"CHAT_ID": chat_id})
    return res.get("result", {})


def get_dialog_users(dialog_id: str) -> List[Dict[str, Any]]:
    res = call_rest("im.dialog.users.list", {"DIALOG_ID": dialog_id})
    r = res.get("result")
    return r if isinstance(r, list) else []


def parse_phone_from_meta(meta: Dict[str, Any]) -> Optional[str]:
    # Prefer entity_id: e.g., "olchat_wa_connector_2|1|917704090366|39771"
    eid = meta.get("entity_id") or meta.get("ENTITY_ID")
    if isinstance(eid, str) and "|" in eid:
        parts = eid.split("|")
        # heuristic: the phone is the last numeric with >= 8 digits or the 3rd field
        for p in reversed(parts):
            if p.isdigit() and len(p) >= 8:
                return "+" + p if not p.startswith("+") else p
        if len(parts) >= 3 and parts[2]:
            p = parts[2]
            if p.replace("+", "").isdigit():
                return "+" + p if not p.startswith("+") else p
    # fallback: try chat name like "+917704... - Open Channel ..."
    name = meta.get("name") or meta.get("NAME")
    if isinstance(name, str) and name.strip().startswith("+"):
        ph = name.split(" ")[0]
        return ph
    return None


def get_connector_user_and_phone(dialog_id: str) -> Tuple[Optional[int], Optional[str]]:
    users = get_dialog_users(dialog_id)
    connector_id = None
    phone = None
    for u in users:
        if u.get("connector") or u.get("external_auth_id") == "imconnector":
            connector_id = int(u.get("id")) if str(u.get("id")).isdigit() else None
            ph = None
            phones = u.get("phones") or {}
            ph = phones.get("personal_mobile") or phones.get("work_phone")
            if isinstance(ph, str) and ph:
                phone = "+" + ph if not ph.startswith("+") else ph
            break
    if not phone:
        meta = get_lines_dialog_meta(dialog_id)
        phone = parse_phone_from_meta(meta)
    return connector_id, phone


def fetch_new_messages(dialog_id: str, since_id: Optional[int]) -> List[Dict[str, Any]]:
    params: Dict[str, Any] = {"DIALOG_ID": dialog_id, "LIMIT": 50}
    if since_id:
        params["FIRST_ID"] = since_id
    r = call_rest("im.dialog.messages.get", params)
    res = r.get("result")
    msgs: List[Dict[str, Any]] = []
    if isinstance(res, dict):
        msgs = res.get("messages") or res.get("MESSAGES") or []
    elif isinstance(res, list):
        msgs = res
    msgs = sorted(msgs, key=lambda m: int(m.get("id") or m.get("ID") or 0))
    if since_id is not None:
        msgs = [m for m in msgs if int(m.get("id") or m.get("ID") or 0) > since_id]
    return msgs


def send_b24_message(dialog_id: str, text: str) -> Optional[int]:
    r = call_rest("im.message.add", {"DIALOG_ID": dialog_id, "MESSAGE": text})
    return r.get("result") if isinstance(r, dict) else None


def main() -> None:
    scopes = call_rest("scope").get("result", [])
    if "im" not in scopes:
        print("ERROR: Webhook lacks 'im' scope. Current:", scopes)
        sys.exit(2)
    me = call_rest("profile").get("result", {})
    my_id = int(me.get("ID")) if str(me.get("ID")).isdigit() else None
    print(f"Bridge running as user {me.get('ID')} poll={POLL_INTERVAL_SEC}s")
    dialogs: Dict[str, Dict[str, Any]] = {}

    while True:
        time.sleep(POLL_INTERVAL_SEC)
        try:
            items = recent_items()
        except Exception as e:
            print("WARN recent:", e)
            continue
        for it in items:
            # Skip non-Open Lines chats early
            if not is_lines_chat(it):
                continue
            did = item_dialog_id(it)
            if not did:
                continue
            # Initialize state
            st = dialogs.setdefault(did, {"last_id": 0, "connector_id": None, "phone": None})
            # Read connector mapping lazily
            if st["connector_id"] is None or st["phone"] is None:
                try:
                    cid, phone = get_connector_user_and_phone(did)
                    st["connector_id"], st["phone"] = cid, phone
                except Exception as e:
                    print(f"WARN users/meta for {did}: {e}")
            # Seed last_id from recents' last_id to avoid replay on first sight
            last_id_field = None
            chat = it.get("chat") or {}
            last_id_field = chat.get("last_id") or it.get("last_id")
            if st["last_id"] == 0 and isinstance(last_id_field, int):
                st["last_id"] = last_id_field
                continue
            # Fetch newer messages
            try:
                msgs = fetch_new_messages(did, st["last_id"])
            except Exception as e:
                print(f"WARN fetch messages {did}: {e}")
                continue
            if not msgs:
                continue
            for m in msgs:
                mid = int(m.get("id") or m.get("ID") or 0)
                author_id = m.get("author_id") or m.get("AUTHOR_ID")
                if isinstance(author_id, str) and author_id.isdigit():
                    author_id = int(author_id)
                text = m.get("text") or m.get("TEXT") or ""
                st["last_id"] = max(st["last_id"], mid)
                # Inbound = connector message
                if author_id is not None and author_id == st.get("connector_id"):
                    # Build payload for your bot. It expects at least
                    # {"phone": "+<customer>", "message": "..."}
                    payload = {
                        "phone": st.get("phone"),
                        "message": text,
                        # Extras for context/debugging
                        "platform": "bitrix24",
                        "dialog_id": did,
                        "message_id": mid,
                        "business_phone": BUSINESS_PHONE,
                        "author_id": author_id,
                    }
                    try:
                        resp = http_post_json(BOT_HTTP_URL, payload, BOT_HTTP_AUTH)
                    except Exception as e:
                        print(f"WARN bot http: {e}")
                        continue
                    # Parse a few common response shapes
                    replies: List[str] = []
                    if isinstance(resp, dict):
                        if isinstance(resp.get("reply_text"), str):
                            replies = [resp["reply_text"]]
                        elif isinstance(resp.get("replies"), list):
                            replies = [str(x) for x in resp["replies"]]
                        elif isinstance(resp.get("text"), str):
                            replies = [resp["text"]]
                        elif isinstance(resp.get("response"), str):
                            replies = [resp["response"]]
                        elif isinstance(resp.get("messages"), list):
                            replies = [str(x) for x in resp["messages"]]
                    for rtext in replies:
                        if not rtext:
                            continue
                        try:
                            send_b24_message(did, rtext)
                            print(f"[{did}] -> sent reply")
                        except Exception as e:
                            print(f"WARN send reply {did}: {e}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
