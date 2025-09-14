# Bitrix24 WhatsApp → AI Chatbot Integration Roadmap

This plan takes us from the current polling bridge to a robust, always‑on integration that answers all WhatsApp queries on the business number.

Current status
- Live bridge: `b24_ol_bridge.py` polls Open Lines, posts inbound messages to the bot HTTP endpoint, and sends bot replies back to Bitrix.
- Terminal tools: `b24_chat_tty.py` (two‑way chat) and `b24_openlines_reader.py` (read‑only) are available for debugging.

Key actors
- Product/Owner: Aditya
- Bitrix Admin: Admin team (manages Open Lines, roles, webhooks)
- Backend Engineer: AI/bot owner (manages bot server and endpoints)
- DevOps: Infra for services, domains, TLS, monitoring

---

## Phase 0 — Preconditions (Admin + Infra)

- Service user (Owner: Bitrix Admin)
  - Create “AI assistant” user (ID 38005 exists) and keep it Available.
  - Add to every WhatsApp Open Line queue as Operator; optionally Supervisor to view all sessions.

- Webhook (Owner: Bitrix Admin)
  - Create Incoming Webhook for user 38005 with scopes `im`, `imopenlines`.
  - Provide `B24_WEBHOOK_BASE` URL to the team.

- Network & TLS (Owner: DevOps)
  - Expose bot over HTTPS (e.g., `https://bot.yourdomain.com/webhook/whatsapp`).
  - Open required ports; add firewall rules; ensure stable public IP/DNS.

Outcome: Bridge can see all WhatsApp messages and post replies portal‑wide.

---

## Phase 1 — Stabilize Polling Bridge (Owner: Backend Engineer)

- Configure `.env` for bridge
  - `B24_WEBHOOK_BASE=<webhook of user 38005>`
  - `BOT_HTTP_URL=https://bot.yourdomain.com/webhook/whatsapp`
  - `BOT_HTTP_TIMEOUT_SEC=120` (tune for RAG latency)
  - `B24_POLL_SEC=2`

- Run as service
  - Create a `systemd` service or PM2/supervisor entry for `b24_ol_bridge.py`.
  - Log to file; rotate logs.

- Resilience
  - Add retry/backoff on 5xx/timeouts from bot.
  - De‑dup protection by message id.

- KPIs
  - Delivery latency p50/p95, successful reply rate, error rate.

Deliverable: Reliable auto‑reply for all chats via polling with metrics.

---

## Phase 2 — Push Delivery (Owner: Bitrix Admin + Backend Engineer)

Option A: Outgoing Webhook (faster)
- Admin: Configure Outgoing Webhook for IM/Open Lines “new message” events and point to `BOT_HTTP_URL`.
- Backend: Add a small verification secret; accept events and reply via `im.message.add`.

Option B: Private App (best)
- Create a private Bitrix app with scopes: `im`, `imopenlines`, `imbot`, `pull`.
- Register a chatbot; connect it to the Open Line.
- Receive events over PULL; reply via bot APIs; optional typing indicators, read receipts.

Deliverable: True real‑time, no polling, lower API load.

---

## Phase 3 — Bot Quality & Safety (Owner: Backend Engineer)

- SLA & timeouts
  - Keep average response < 10s; implement async processing with early 200 ACK if needed.

- Guardrails & fallbacks
  - Graceful fallback message on errors/timeouts.
  - Escalation keywords ("agent", "help") → auto‑transfer to operator.

- Personalization & memory
  - Persist per‑phone profile (name, location, previous context) in a DB.
  - Respect privacy and retention policies.

Deliverable: Consistent replies with safe fallbacks and smooth handoff.

---

## Phase 4 — CRM Binding & Analytics (Owner: Backend Engineer)

- Session → CRM mapping
  - Use `imopenlines.dialog.get` + `imopenlines.session.*` to link chat to Lead/Deal.
  - Write bot interactions to CRM timeline.

- Reporting
  - Log per‑message metadata (latency, success, transfer events).
  - Dashboard for volumes, CSAT tags (thumbs up/down), escalation rate.

Deliverable: Full traceability from WhatsApp to CRM.

---

## Phase 5 — Operations, Security, and Cost (Owner: DevOps)

- Ops
  - Health checks, uptime alerts, error alerts.
  - Daily log rotation and retention.

- Security
  - HTTPS everywhere; restrict webhook by IP allow‑list or signature.
  - Rotate Bitrix webhook quarterly.

- Cost
  - Cache answers for FAQs; rate‑limit abusive sessions.

Deliverable: Production‑grade reliability and governance.

---

## Acceptance Tests

1) Fresh inbound WhatsApp message is responded to within 10s (p95 < 15s).
2) Multiple quick turns (≥5 consecutive) delivered without timeout.
3) Escalation phrase routes to live agent and bot stops replying.
4) Messages logged against correct CRM lead; timeline includes bot replies.
5) Bridge restarts do not duplicate messages (idempotent handling).

---

## Open Items & Owners

- [ ] Admin: Add user 38005 to all WhatsApp Open Lines as Operator/Supervisor
- [ ] Admin: Create Incoming Webhook for user 38005 (scopes: im, imopenlines)
- [ ] DevOps: HTTPS domain for bot + firewall rules
- [ ] Backend: Enable async processing or increase timeout safely
- [ ] Backend: Add retries/backoff to bridge
- [ ] Backend: Outgoing Webhook receiver or Private App for push
- [ ] Backend: Telemetry dashboard (latency, success rate)

