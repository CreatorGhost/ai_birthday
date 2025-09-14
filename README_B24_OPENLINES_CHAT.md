# Bitrix24 Open Lines (WhatsApp) – Real‑Time Reader and Terminal Chat

This repo includes a tiny, dependency‑free toolkit to read and send WhatsApp messages from Bitrix24 Open Lines using the REST API. It supports

- Near real‑time message detection (2s poll) for new and existing chats
- An interactive terminal chat (send + receive) bound to a single dialog
- One‑off send command for scripting/automation

It uses an Incoming Webhook and the Bitrix IM/Open Lines API. No sockets or external packages required.

---

## What You Get

- `b24_openlines_reader.py` – Receive‑only poller that prints new OL messages as they arrive
- `b24_chat_tty.py` – Interactive terminal chat (type to send, prints new replies)
- `b24_send_message.py` – One‑off send to a dialog

All scripts auto‑load `.env` (no external library) and respect:

- `B24_WEBHOOK_BASE` – your incoming webhook base URL (required)
- `B24_POLL_SEC` – optional poll interval (default 2s)

---

## Prerequisites

1. Bitrix24 portal with Open Lines connected to WhatsApp (e.g., OLChat connector).
2. Incoming Webhook with scopes:
   - `im` (required)
   - `imopenlines` (recommended; enables line/session metadata)
3. Permissions for the webhook user:
   - Either add the user to the Open Line queue (Operator or Supervisor)
   - Or click “Join”/“Take” in the chat UI for each session you want to read

Why: Open Lines messages are IM chats of type `LINES`. The API returns only dialogs the user is allowed to see (queue member or joined session).

---

## Configure `.env`

Create/ensure a `.env` in the project root with:

```
B24_WEBHOOK_BASE=https://<your-portal>.bitrix24.<tld>/rest/<user_id>/<code>/
# Optional: poll interval (seconds)
B24_POLL_SEC=2
```

Notes
- Keep this URL secret; it grants API access as the selected user.
- A trailing slash is OK; scripts normalize it.

There’s also an example in `.env.example`.

---

## Quick Start

Verify the webhook and scopes:

```
curl -sS "$B24_WEBHOOK_BASE/scope.json" | jq -r '.'
curl -sS "$B24_WEBHOOK_BASE/profile.json" | jq -r '.'
```

You should see `im` in scopes and your user profile data.

### 1) Receive‑only reader

```
python3 b24_openlines_reader.py
```

Expected behavior:
- Prints the authenticated user and poll interval
- Warns if the user is not in any Open Lines queue
- Seeds from `im.recent.list` and then prints new messages like:

```
[NEW DIALOG chat97669] <text>
[chat97669] <new incoming text>
```

Tip: If you’re not in the queue, open the chat in the UI and click “Join”. The reader then sees that dialog.

### 2) Interactive terminal chat (send + receive)

```
python3 b24_chat_tty.py chat97669
# or with numeric id:
python3 b24_chat_tty.py 97669
```

Behavior:
- Prints new incoming messages every ~2s
- Lets you type outbound messages; they are sent with `im.message.add`
- Commands: `/quit`, `/who`, `/help`

Implementation detail: The script uses `im.dialog.messages.get` with `FIRST_ID` to fetch messages strictly newer than the last seen id (correct for Bitrix paging semantics).

### 3) One‑off send

```
python3 b24_send_message.py chat97669 "Hello from terminal"
```

### 4) Bridge to your AI bot (HTTP)

Use this to automatically forward inbound WhatsApp messages to your bot and send its replies back to Bitrix.

```
# .env should include B24_WEBHOOK_BASE; you can also add:
# BOT_HTTP_URL=http://35.232.52.16:8001/webhook/whatsapp
# BOT_HTTP_AUTH=Bearer <token-if-needed>
# B24_BUSINESS_PHONE=+971501754133
python3 b24_ol_bridge.py
```

Behavior:
- Polls IM recents, detects new messages in LINES chats.
- Identifies the visitor (connector) user via `im.dialog.users.list` and extracts their phone.
- For each inbound (connector) message, POSTs JSON to `BOT_HTTP_URL`:

```json
{
  "platform": "bitrix24",
  "direction": "inbound",
  "dialog_id": "chat97669",
  "message_id": 55018227,
  "text": "hello",
  "business_phone": "+971501754133",
  "customer_phone": "+917704090366",
  "author_id": 39771
}
```

It expects a simple JSON response; any of these shapes are supported:

```json
{ "reply_text": "Hello!" }
{ "replies": ["Hello!", "How can I help?"] }
{ "text": "Hello!" }
{ "messages": ["Hello!"] }
```

Replies are sent back to Bitrix via `im.message.add`.

---

## Current Implementation (What’s Live Now)

- Reading: We use a polling reader against `im.recent.list` + `im.dialog.messages.get` to detect new messages within ~2–10s.
- Sending: Replies are posted with `im.message.add` as the webhook user.
- Operator visibility: The webhook user must be a queue operator or join each session. For full coverage, use a dedicated service user added to all WhatsApp Open Lines.
- Bridge: `b24_ol_bridge.py` forwards inbound messages to your bot and posts the bot’s reply back to Bitrix.
- Bot payload: the bridge calls your bot with `{ "phone": "+<customer>", "message": "..." }` and accepts replies in any of: `response`, `reply_text`, `text`, `replies[]`, `messages[]`.
- Timeout control: Set `BOT_HTTP_TIMEOUT_SEC` (default 120) so the bridge waits long enough for RAG turns.
- Filtering: Only Open Lines (`LINES`) chats are processed; general/user chats are ignored to avoid API 400s.

Environment summary

```
B24_WEBHOOK_BASE=https://<portal>/rest/<user>/<code>/
BOT_HTTP_URL=http://35.232.52.16:8001/webhook/whatsapp
BOT_HTTP_TIMEOUT_SEC=120
B24_POLL_SEC=2
```

Operational notes
- If your bot is offline, use `mock_bot_server.py` locally for end‑to‑end verification.
- If some turns time out, increase `BOT_HTTP_TIMEOUT_SEC` or make your bot return 200 quickly and process async.
- For lowest latency and no polling, switch to Outgoing Webhooks or a private app with `pull`/`imbot`.

---

## How It Works

- Open Lines (WhatsApp) messages live in the Bitrix IM module as dialogs of type `LINES`.
- The reader polls `im.recent.list` (fallback `im.recent.get`) and tracks the last message id per dialog to detect new arrivals quickly.
- The TTY chat fetches messages via `im.dialog.messages.get` using `FIRST_ID` and sends via `im.message.add`.
- Incoming Webhooks cannot subscribe to push events (`event.bind` is denied). Polling is used for near real‑time (<2–10s). For full push, configure an Outgoing Webhook or an app subscription.

Key endpoints used:

- `scope`, `profile`
- `im.recent.list`, `im.recent.get`
- `im.dialog.messages.get`, `im.message.add`
- `imopenlines.config.list.get` (for queue introspection)

---

## Operator Access

You have two ways to make a chat visible to the webhook user:

1) Queue membership
   - Contact Center → Open Channels → select your WhatsApp line → “Queue”
   - Add the webhook’s user as Operator (or Supervisor)

2) Join the session
   - Open the chat window and click “Join/Take” (bottom right)
   - This grants visibility for the current session only

The reader prints your queue status per line if `imopenlines` scope is available.

---

## Test Scenario (<= 10s)

1. Start the reader: `python3 b24_openlines_reader.py`
2. Join the target WhatsApp chat (or ensure you are in the queue and Available)
3. From a phone, send: `test message` to the business number
4. Within 2–10 seconds you should see the message printed

For two‑way terminal chat, use `b24_chat_tty.py` instead and keep the session joined.

---

## Troubleshooting

- `Your webhook lacks 'im' scope` → Create a new Incoming Webhook with `im` (and `imopenlines`).
- Reader shows no LINES dialogs → You’re not in the queue. Add the user as Operator or click “Join” in the chat.
- `ACCESS_DENIED` on send → You don’t have rights in that dialog; join it or have it assigned to you.
- `event.bind` errors → Incoming webhooks cannot bind events; polling is by design.
- Nothing prints after joining → Send a fresh message after the script is running (old messages may have been seeded as “last seen”).

---

## Security

- Treat `B24_WEBHOOK_BASE` as a secret; don’t share publicly or commit real values.
- Rotate the webhook if it is exposed.

---

## Roadmap (Optional)

- Recent dialog picker for `b24_chat_tty.py`
- Multi‑dialog stream with per‑dialog routing
- Push receiver using Outgoing Webhook (server callback)

---

## Maintenance Notes

- Scripts are dependency‑free and use Python’s stdlib only.
- `.env` parsing is minimal: supports `export VAR=value` and `VAR=value`; `#` comments are ignored.
- If Bitrix returns a different envelope, tweak the accessors where marked in code to be shape‑tolerant.

---

## Limitations & Choices

- Incoming Webhooks cannot subscribe to events. `event.bind` returns `WRONG_AUTH_TYPE`. Polling is the only option with an Incoming Webhook.
- Visibility is user‑scoped. The webhook user only sees Open Lines dialogs they participate in (assigned or joined) or are allowed to view by permissions. To guarantee visibility for every new WhatsApp message, either:
  - Add the webhook user as Operator (or Supervisor + Operator) to every WhatsApp Open Line queue; or
  - Use a Push approach (Outgoing Webhook or App) described below.
- Rate limits apply (typical ~2 RPS per webhook). Keep intervals modest and add backoff if you extend the scripts.

---

## Admin Checklist — Make All WhatsApp Messages Readable

This is the shortest path to 100% coverage using the provided scripts.

1) Create a dedicated service user (e.g., “AI Integration”).
   - Paid seat. This user will own the webhook and be used by the terminal tools.

2) Add the service user to every WhatsApp Open Line queue.
   - Contact Center → Open Channels → choose the WhatsApp line → Queue → Add user.
   - For this portal we saw line IDs and names via API:
     - ID 1: “Open Channel LeoLoona”
     - ID 3: “Open Chanel HelloKids (WhatsApp)”
     - ID 5: “Open Channel 3”
   - Keep normal distribution rules; adding this user does not block other operators.

3) Grant Open Lines permissions to the service user.
   - Access/Permissions: assign at least Operator; Supervisor is recommended to view all sessions and history.

4) Ensure the user can receive/see sessions.
   - Either keep the user “Available” or allow the user to Join sessions on demand.
   - Optionally disable “Check that an agent is online when routing enquiries” if you don’t want availability gating.

5) Create a new Incoming Webhook for the service user.
   - Scopes: `im` (required) and `imopenlines` (recommended).
   - Share the link as `B24_WEBHOOK_BASE` in `.env`.

6) Verify
   - Run `python3 b24_openlines_reader.py` and send a WhatsApp message to the business number. You should see it within 2–10s.

Copy‑paste for admin:

> Please create a dedicated Bitrix24 user “AI Integration” and add it as Operator (and Supervisor) to all WhatsApp Open Lines. Then create an Incoming Webhook for this user with scopes `im` and `imopenlines`, and send me the full webhook URL. We’ll use it to read all WhatsApp messages in real time.

---

## Recommended Push Option (Production)

Polling works, but push is more robust and lower latency. Two supported paths:

1) Outgoing Webhook (no app needed)
   - Contact Center → Webhooks → Outgoing webhook.
   - Select IM/Open Lines message‑added events (names vary by portal build; pick the IM/OL chat message events).
   - Set handler URL to your server endpoint (HTTPS). Bitrix will POST on each new message; no polling required.
   - Keep your existing Incoming Webhook for sending replies.

2) Local Application (full OAuth app)
   - Create a private app with scopes: `im`, `imopenlines`, `imbot`, `pull`.
   - Optionally register a Chatbot and attach it to the Open Line. The bot receives events and can reply programmatically.
   - Best path for a production chatbot fully inside Bitrix.

We can provide a minimal receiver (Python/Node) for Outgoing Webhook events on request.

---

## Integration Notes for the AI Chatbot

- Use `b24_chat_tty.py` logic as the skeleton: for each dialog, track the last message ID and call `im.dialog.messages.get` with `FIRST_ID` for new messages.
- To reply, call `im.message.add` with `DIALOG_ID=chatNNNNN` (or userNNNNN for user dialog).
- Persist last seen per dialog if you build a service; on restart, reconcile by fetching with `FIRST_ID`.
- To map dialogs to CRM entities, use `imopenlines.session.history.get` with chat or session id and CRM bindings.
