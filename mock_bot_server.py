#!/usr/bin/env python3
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload: dict):
        body = json.dumps(payload).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == '/status':
            self._send(200, {"service": "mock-bot", "status": "ok"})
            return
        self._send(404, {"error": "not found"})

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != '/webhook/whatsapp':
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            raw = self.rfile.read(length)
            data = json.loads(raw.decode('utf-8') or '{}')
            text = data.get('text') or data.get('message') or ''
            phone = data.get('customer_phone') or data.get('phone') or ''
            reply = f"Hi {phone or 'there'}! You said: {text}"
            self._send(200, {"reply_text": reply})
        except Exception as e:
            self._send(500, {"error": str(e)})


def run(host='127.0.0.1', port=8001):
    httpd = HTTPServer((host, port), Handler)
    print(f"Mock bot server running at http://{host}:{port}")
    httpd.serve_forever()


if __name__ == '__main__':
    run()

