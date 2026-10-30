"""Loopback, read-only evaluation artifacts; no model imports or controls."""
from functools import partial
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control','no-store')
        super().end_headers()

    def log_message(self,*args): pass


if __name__=='__main__':
    root=Path(__file__).resolve().parent
    server=ThreadingHTTPServer(('127.0.0.1',8769),partial(Handler,directory=str(root)))
    print('Rudra DINOv3 ACT 200K evaluation: http://127.0.0.1:8769/',flush=True)
    server.serve_forever()
