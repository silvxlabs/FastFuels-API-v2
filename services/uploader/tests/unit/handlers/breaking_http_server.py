"""HTTP server for the cut-short GeoJSON read tests.

Serves FILE at any path, one request per connection, and prints its port. In
a failing MODE, the second GET of a path drops the connection half-way through
the body (``drop``) or answers 503 (``unavailable``). It runs in its own
process: GDAL holds the GIL while it reads, so a server thread in the test
process would never answer.

usage: breaking_http_server.py FILE {ok,drop,unavailable}
"""

import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BODY = open(sys.argv[1], "rb").read()
MODE = sys.argv[2]
GETS: dict[str, int] = {}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _headers(self, status: int, length: int) -> None:
        self.send_response(status)
        self.send_header("Content-Length", str(length))
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

    def do_HEAD(self):
        self._headers(200, len(BODY))

    def do_GET(self):
        GETS[self.path] = GETS.get(self.path, 0) + 1
        fail = MODE != "ok" and GETS[self.path] == 2
        if fail and MODE == "unavailable":
            self._headers(503, 0)
            return
        self._headers(200, len(BODY))
        self.wfile.write(BODY[: len(BODY) // 2] if fail else BODY)


server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
print(server.server_port, flush=True)
server.serve_forever()
