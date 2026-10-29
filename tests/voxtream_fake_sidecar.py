"""A stand-in for the VoXtream sidecar process, for model-free tests.

Run with the test interpreter as if it were the VoXtream environment's python:

    python tests/voxtream_fake_sidecar.py --host 127.0.0.1 --port N [--device D]

It accepts the real server's arguments and serves ``GET /health`` with 200. The
environment variable ``VOXTREAM_FAKE_MODE`` changes its behaviour:

- ``ready`` (default): answer 200 at once.
- ``slow``: answer 503 for ``VOXTREAM_FAKE_DELAY_S`` seconds (default 1), then 200,
  the way the real sidecar does while the model loads.
- ``never``: answer 503 forever.
- ``exit``: exit with code 3 at once, like a sidecar whose import failed.

``VOXTREAM_FAKE_ARGS_FILE`` names a file the fake writes its argument list to, so a
test can check what it was started with. The fake keeps running until it is
terminated.
"""
import argparse
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--config-dir", default=None)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args(argv)

    args_file = os.environ.get("VOXTREAM_FAKE_ARGS_FILE")
    if args_file:
        with open(args_file, "w", encoding="utf-8") as f:
            json.dump(sys.argv[1:], f)

    mode = os.environ.get("VOXTREAM_FAKE_MODE", "ready")
    if mode == "exit":
        sys.exit(3)
    ready_at = time.monotonic()
    if mode == "slow":
        ready_at += float(os.environ.get("VOXTREAM_FAKE_DELAY_S", "1"))
    if mode == "never":
        ready_at = float("inf")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 (http.server naming)
            if self.path != "/health":
                self.send_response(404)
                self.end_headers()
                return
            ready = time.monotonic() >= ready_at
            body = json.dumps({"status": "ok" if ready else "loading",
                               "sample_rate": 24000}).encode()
            self.send_response(200 if ready else 503)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):  # quiet
            return

    server = HTTPServer((args.host, args.port), Handler)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
