"""Servidor do dashboard + API. Usa http.server da stdlib (zero dependência) para garantir
que `radar serve` funcione offline. Endpoints:
  GET /                -> Mapa de Alavancagem (web/index.html)
  GET /api/gold        -> gold set diagnosticado pelo motor real
  GET /api/brief?name= -> briefing markdown
  GET /api/ask?q=      -> RAG
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from ..config import WEB_DIR


class Handler(BaseHTTPRequestHandler):
    def _send(self, body: bytes, ctype="application/json"):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):  # silencioso
        pass

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path in ("/", "/index.html"):
                self._send((WEB_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif u.path == "/api/gold":
                from ..eval.gold_eval import map_data
                self._send(json.dumps(map_data(), ensure_ascii=False).encode())
            elif u.path == "/api/brief":
                from ..graph import run
                st = run(q.get("name", [""])[0])
                self._send(json.dumps({"briefing": st.get("briefing", ""),
                                       "errors": st.get("errors", [])}, ensure_ascii=False).encode())
            elif u.path == "/api/ask":
                from ..rag.pipeline import ask
                self._send(json.dumps(ask(q.get("q", [""])[0]), ensure_ascii=False).encode())
            elif u.path in ("/map_data.js", "/map_data.json"):
                f = WEB_DIR / u.path.lstrip("/")
                ctype = "application/javascript" if u.path.endswith(".js") else "application/json"
                if f.exists():
                    self._send(f.read_bytes(), f"{ctype}; charset=utf-8")
                else:
                    self.send_response(404); self.end_headers()
            else:
                self.send_response(404); self.end_headers()
        except Exception as e:
            self._send(json.dumps({"error": str(e)}).encode())


def serve(port: int = 8000):
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"NVIDIA Startup AI Radar em http://127.0.0.1:{port}  (Ctrl+C para sair)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nencerrando.")
