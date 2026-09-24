"""Watch labelled clips with the labels drawn over the video; flag what looks wrong.

Serves one page (review_video.html) on 127.0.0.1. Each clip plays in a
<video> with a canvas overlay redrawn on every presented frame. Labels come
from the raw clip source (dsa.label.clips); what the consistency filters
masked (the `flags` table of <source>_clean) is drawn dim red. Videos are
served as per-clip H.264 proxies cut with ffmpeg into data/scratch/review_proxy/
(clips are windows of long videos, some AV1 or 10-bit HEVC), with HTTP Range
support for seeking.

Keys: Space play / pause, , . one frame, [ ] speed, F then a digit flags the
moment (1 ball, 2 court, 3 pose, 4 players, 5 racket, 6 view / in play, 0 other),
Backspace removes the flag at or before the playhead, N / P next / previous clip, H hide labels,
D mark the clip done. Flags are saved atomically per clip to
data/gold/<version>/video_flags/<clip>.json; `summary` prints flag rates per
tag per collection and camera over the clips marked done.

    python -m dsa.label.review_video serve --source clips_v1 --version v1
    python -m dsa.label.review_video summary --version v1
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import threading
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from dsa.data import schema
from dsa.data.paths import DATA, GOLD, LABELS, SCRATCH

REVIEWER = "human:peter"
TAGS = {"1": "ball", "2": "court", "3": "pose", "4": "players", "5": "racket", "6": "view / in play", "0": "other"}
LABELERS = {"ball": "WASB", "court": "Astra", "pose": "RF-DETR + ViTPose", "players": "Astra + tracker",
            "racket": "RacketVision", "view / in play": "Astra", "other": ""}
PROXY = SCRATCH / "review_proxy"
_locks = defaultdict(threading.Lock)


def clips_of(source: str) -> list[dict]:
    fr = schema.read(source, "frames")
    out = []
    for clip, g in fr.groupby("clip", sort=False):
        r = g.iloc[0]
        out.append({"clip": clip, "media": r["media"], "start": int(g["frame"].min()), "end": int(g["frame"].max()) + 1,
                    "fps": float(r["fps"]), "width": int(r["width"]), "height": int(r["height"]),
                    **{k: (None if pd.isna(r.get(k)) else r.get(k)) for k in ("collection", "video", "camera", "surface", "level")}})
    return out


def proxy(c: dict) -> Path:
    """The clip cut from its video and re-encoded to H.264, frame 0 = clip start."""
    out = PROXY / f"{c['clip']}.mp4"
    with _locks[c["clip"]]:
        if not out.exists():
            if not shutil.which("ffmpeg"):
                raise RuntimeError("ffmpeg not found")
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(".tmp.mp4")
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{c['start'] / c['fps']:.6f}", "-i", str(DATA / c["media"]),
                            "-frames:v", str(c["end"] - c["start"]), "-fps_mode", "passthrough", "-c:v", "libx264",
                            "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart",
                            str(tmp)], check=True)
            tmp.replace(out)
    return out


def r1(v) -> list:
    return [None if not np.isfinite(x) else round(float(x), 1) for x in np.asarray(v, float).ravel()]


def clip_labels(source: str, clean: str | None, c: dict) -> dict:
    """Every frame's labels for one clip, with what the filters masked, as plain JSON."""
    def rd(t: str, root: str = source) -> pd.DataFrame:
        path = LABELS / root / f"{t}.parquet"
        if not path.exists() or pq.read_metadata(path).num_rows == 0:
            return pd.DataFrame(columns=schema.TABLES[t])
        return pd.read_parquet(path, filters=[("sample", "in", samples)])

    fr = pd.read_parquet(LABELS / source / "frames.parquet", filters=[("clip", "==", c["clip"])])
    samples = fr["sample"].tolist()
    off = dict(zip(fr["sample"], fr["frame"] - c["start"]))
    frames = {int(o): {"scope": s, "people": [], "rackets": []} for o, s in zip(off.values(), fr["scope"])}
    masked = defaultdict(set)
    if clean and (LABELS / clean / "flags.parquet").exists():
        for r in rd("flags", clean).itertuples():
            masked[(r.sample, r.head, None if pd.isna(r.person) else int(r.person))].add(
                (None if pd.isna(r.point) else int(r.point), r.reason))
    why = lambda key: sorted({reason for _, reason in masked.get(key, ())})
    for r in rd("scene").itertuples():
        frames[off[r.sample]]["scene"] = {"view": None if pd.isna(r.view) else bool(r.view),
                                          "in_play": None if pd.isna(r.in_play) else bool(r.in_play),
                                          "singles": None if pd.isna(getattr(r, "singles", np.nan)) else bool(r.singles),
                                          "masked": {h: why((r.sample, h, None)) for h in ("view", "in_play")}}
    for r in rd("ball").itertuples():
        frames[off[r.sample]]["ball"] = {"x": r1([r.x])[0], "y": r1([r.y])[0], "visible": bool(r.visible),
                                         "masked": why((r.sample, "ball", None))}
    for r in rd("court").itertuples():
        frames[off[r.sample]]["court"] = {"kp": r1(r.kp), "masked": sorted(
            {p for p, _ in masked.get((r.sample, "court", None), ())}), "why": why((r.sample, "court", None))}
    for r in rd("people").itertuples():
        pm = masked.get((r.sample, "pose", int(r.person)), set())
        frames[off[r.sample]]["people"].append({
            "box": r1(r.box), "kp": r1(r.kp), "track": r.track, "player": bool(getattr(r, "player", False)),
            "masked": sorted({p for p, _ in pm}), "why": sorted({w for _, w in pm}),
            "player_masked": why((r.sample, "players", int(r.person)))})
    for r in rd("rackets").itertuples():
        frames[off[r.sample]]["rackets"].append({"kp": r1(r.kp), "person": None if pd.isna(r.person) else int(r.person)})
    return {**c, "frames": frames}


def flags_path(version: str, clip: str) -> Path:
    return GOLD / version / "video_flags" / f"{clip}.json"


def serve(source: str, clean: str | None, version: str, port: int) -> None:
    clips = clips_of(source)
    (GOLD / version / "video_flags").mkdir(parents=True, exist_ok=True)
    page = Path(__file__).with_name("review_video.html")

    def prefetch(i: int):
        if 0 <= i < len(clips):
            threading.Thread(target=proxy, args=(clips[i],), daemon=True).start()

    class Handler(BaseHTTPRequestHandler):
        def send(self, payload: bytes, kind: str, status: int = 200, extra: dict | None = None):
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(payload)

        def video(self, path: Path):
            size = path.stat().st_size
            m = re.match(r"bytes=(\d*)-(\d*)$", self.headers.get("Range", ""))
            a, b = 0, size - 1
            if m and (m[1] or m[2]):
                a, b = (int(m[1]), int(m[2]) if m[2] else size - 1) if m[1] else (size - int(m[2]), size - 1)
                if a >= size or a > b:
                    self.send(b"", "video/mp4", 416, {"Content-Range": f"bytes */{size}"})
                    return
                b = min(b, size - 1)
            with path.open("rb") as f:
                f.seek(a)
                data = f.read(b - a + 1)
            self.send(data, "video/mp4", 206 if m and (m[1] or m[2]) else 200,
                      {"Accept-Ranges": "bytes", **({"Content-Range": f"bytes {a}-{b}/{size}"} if m and (m[1] or m[2]) else {})})

        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            route = urlsplit(self.path).path
            try:
                if route == "/":
                    self.send(page.read_bytes(), "text/html; charset=utf-8")
                elif route == "/api/index":
                    status = {}
                    for c in clips:
                        p = flags_path(version, c["clip"])
                        if p.exists():
                            body = json.loads(p.read_text())
                            status[c["clip"]] = {"done": body.get("done", False), "flags": len(body.get("flags", []))}
                    self.send(json.dumps({"clips": clips, "status": status, "tags": TAGS}).encode(), "application/json")
                elif m := re.fullmatch(r"/api/clip/(\d+)", route):
                    i = int(m[1])
                    p = flags_path(version, clips[i]["clip"])
                    saved = json.loads(p.read_text()) if p.exists() else {"flags": [], "done": False}
                    body = {**clip_labels(source, clean, clips[i]), "saved": saved}
                    self.send(json.dumps(body).replace("NaN", "null").encode(), "application/json")
                    prefetch(i + 1)
                elif m := re.fullmatch(r"/video/(\d+)\.mp4", route):
                    self.video(proxy(clips[int(m[1])]))
                else:
                    self.send_error(404)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self):
            origin = self.headers.get("Origin")
            if origin and urlsplit(origin).netloc != self.headers.get("Host"):
                self.send_error(403)
                return
            m = re.fullmatch(r"/api/clip/(\d+)/flags", urlsplit(self.path).path)
            if not m:
                self.send_error(404)
                return
            c = clips[int(m[1])]
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            flags = [{"frame": int(f["frame"]), "t": round(float(f["frame"]) / c["fps"], 3), "tag": f["tag"]}
                     for f in body.get("flags", []) if f.get("tag") in TAGS.values()]
            path = flags_path(version, c["clip"])
            tmp = path.with_suffix(".pending")
            tmp.write_text(json.dumps({"clip": c["clip"], "done": bool(body.get("done")), "flags": flags,
                                       "seconds": round((c["end"] - c["start"]) / c["fps"], 2), "reviewer": REVIEWER,
                                       **{k: c[k] for k in ("collection", "video", "camera", "surface", "level")}},
                                      indent=1))
            tmp.replace(path)
            self.send(b'{"saved": true}', "application/json")

        def log_message(self, fmt, *args):
            pass

    prefetch(0)
    print(f"Video review ({len(clips)} clips): http://127.0.0.1:{port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


def summary(version: str) -> pd.DataFrame:
    rows = []
    for p in sorted((GOLD / version / "video_flags").glob("*.json")):
        if p.name.startswith("._"):
            continue
        b = json.loads(p.read_text())
        if b.get("done"):
            rows.append({"collection": b.get("collection"), "camera": b.get("camera"), "clip": b["clip"],
                         "minutes": b["seconds"] / 60, **{t: sum(f["tag"] == t for f in b["flags"]) for t in TAGS.values()}})
    if not rows:
        print("no clips marked done")
        return pd.DataFrame()
    df = pd.DataFrame(rows).fillna({"collection": "?", "camera": "?"})
    g = df.groupby(["collection", "camera"])
    out = g.agg(clips=("clip", "size"), minutes=("minutes", "sum"))
    for t in TAGS.values():
        out[f"{t} /min"] = (g[t].sum() / out["minutes"]).round(2)
    total = pd.DataFrame([{"clips": len(df), "minutes": df["minutes"].sum(),
                           **{f"{t} /min": round(df[t].sum() / df["minutes"].sum(), 2) for t in TAGS.values()}}],
                         index=pd.MultiIndex.from_tuples([("all", "")], names=["collection", "camera"]))
    out = pd.concat([out, total])
    out["minutes"] = out["minutes"].round(1)
    print("flags per minute of video watched; labelers: " + ", ".join(f"{t} = {l}" for t, l in LABELERS.items() if l))
    print(out.to_string())
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["serve", "summary"])
    p.add_argument("--source", help="raw clip labels source (dsa.label.clips)")
    p.add_argument("--clean", help="filtered source with the flags table; default <source>_clean if it exists")
    p.add_argument("--version", default="v1")
    p.add_argument("--port", type=int, default=8771)
    a = p.parse_args()
    if a.command == "summary":
        summary(a.version)
        return
    if not a.source:
        p.error("serve needs --source")
    clean = a.clean or (f"{a.source}_clean" if (LABELS / f"{a.source}_clean").exists() else None)
    serve(a.source, clean, a.version, a.port)


if __name__ == "__main__":
    main()
