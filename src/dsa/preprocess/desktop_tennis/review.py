"""Local-only point editor; range requests allow seeking through large video files."""
from __future__ import annotations

import json
import re
import shutil
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .annotations import load, score_timeline, validate_log, write_outputs


def serve(out: Path, port: int = 8765):
    out = out.resolve()
    _, source = load(out)

    class Handler(BaseHTTPRequestHandler):
        def json_response(self, data, status=200):
            payload = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            route = urlsplit(self.path).path
            if route == "/":
                payload = Path(__file__).with_name("review.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            elif route == "/api/log":
                log, _ = load(out)
                self.json_response({"log": log, "source": {k: source[k] for k in ("fps", "frames", "original_video")},
                                    "scores": {s["id"]: score_timeline(s) for s in log["segments"]}})
            elif route == "/video":
                path = Path(source["video"])
                size = path.stat().st_size
                start, end = 0, size - 1
                header = self.headers.get("Range")
                if header:
                    match = re.fullmatch(r"bytes=(\d+)-(\d*)", header)
                    if not match:
                        self.send_error(416)
                        return
                    start = int(match[1])
                    end = min(int(match[2]), end) if match[2] else end
                    if start > end:
                        self.send_error(416)
                        return
                self.send_response(206 if header else 200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(end - start + 1))
                if header:
                    self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.end_headers()
                try:
                    with path.open("rb") as handle:
                        handle.seek(start)
                        remaining = end - start + 1
                        while remaining:
                            chunk = handle.read(min(1024 * 1024, remaining))
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            remaining -= len(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_error(404)

        def do_POST(self):
            origin = self.headers.get("Origin")
            if origin and urlsplit(origin).netloc != self.headers.get("Host"):
                self.send_error(403)
                return
            if self.path != "/api/log":
                self.send_error(404)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 5_000_000:
                    raise ValueError("Invalid point log size")
                log = json.loads(self.rfile.read(size))
                validate_log(log, source)
                path = out / "annotations.json"
                temporary = out / "annotations.pending.json"
                temporary.write_text(json.dumps(log, indent=2))
                shutil.copy2(path, out / "annotations.previous.json")
                temporary.replace(path)
                write_outputs(log, source, out)
                self.json_response({"saved": True,
                                    "scores": {s["id"]: score_timeline(s) for s in log["segments"]}})
            except (ValueError, KeyError, TypeError) as exc:
                self.json_response({"error": str(exc)}, 400)

        def log_message(self, fmt, *args):
            pass

    print(f"Point review: http://127.0.0.1:{port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
