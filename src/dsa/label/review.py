"""Review tool for the gold set: correct pre-filled labels frame by frame.

Serves one page (review.html) on localhost. Each frame shows every pre-filled
label on the image; drag points to correct them, toggle visibility, fix the
ball, court, view and in play, then save. Each save writes one JSON file per
frame to data/gold/<version>/reviewed/, so progress survives restarts and
nothing is overwritten in bulk. `export` turns the reviewed files into the
labels source gold_<version> (labeler "human:peter").

    python -m dsa.label.review serve --version v1 --prefill gold_v1_prefill
    python -m dsa.label.review export --version v1 --prefill gold_v1_prefill
"""
from __future__ import annotations

import argparse
import json
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import cv2
import numpy as np
import pandas as pd

from dsa.data import schema
from dsa.data.paths import DATA, GOLD

REVIEWER = "human:peter"


def load_prefill(prefill: str) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Frames and, per sample, the pre-filled labels as plain JSON."""
    frames = schema.read(prefill, "frames")
    tables = {t: schema.read(prefill, t) for t in ("people", "rackets", "ball", "court", "scene")
              if (Path(schema.LABELS) / prefill / f"{t}.parquet").exists()}
    labels = {s: {"people": [], "ball": None, "court": None, "view": None, "in_play": None} for s in frames["sample"]}
    for r in tables.get("people", pd.DataFrame()).itertuples():
        labels[r.sample]["people"].append({"box": list(r.box), "kp": np.asarray(r.kp, float).tolist(),
                                           "player": bool(getattr(r, "player", True)), "labeler": r.labeler})
    for r in tables.get("ball", pd.DataFrame()).itertuples():
        labels[r.sample]["ball"] = {"x": None if pd.isna(r.x) else r.x, "y": None if pd.isna(r.y) else r.y,
                                    "visible": bool(r.visible)}
    for r in tables.get("court", pd.DataFrame()).itertuples():
        labels[r.sample]["court"] = np.asarray(r.kp, float).tolist()
    for r in tables.get("scene", pd.DataFrame()).itertuples():
        labels[r.sample].update(view=bool(r.view), in_play=bool(r.in_play))
    # JSON has no NaN: send null.
    return frames, json.loads(json.dumps(labels).replace("NaN", "null"))


def reviewed_path(version: str, sample: str) -> Path:
    return GOLD / version / "reviewed" / (sample.replace("/", "__") + ".json")


def serve(version: str, prefill: str, port: int) -> None:
    frames, labels = load_prefill(prefill)
    rows = frames.to_dict("records")
    index = [{"sample": r["sample"], "collection": r.get("collection"), "camera": r.get("camera"),
              "width": r["width"], "height": r["height"]} for r in rows]
    (GOLD / version / "reviewed").mkdir(parents=True, exist_ok=True)

    @lru_cache(maxsize=64)
    def jpeg(i: int) -> bytes:
        r = rows[i]
        cap = cv2.VideoCapture(str(DATA / r["media"]))
        cap.set(cv2.CAP_PROP_POS_FRAMES, r["frame"])
        ok, bgr = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"cannot read {r['media']} frame {r['frame']}")
        return cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()

    class Handler(BaseHTTPRequestHandler):
        def send(self, payload: bytes, kind: str, status: int = 200):
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            route = urlsplit(self.path).path
            if route == "/":
                self.send(Path(__file__).with_name("review.html").read_bytes(), "text/html; charset=utf-8")
            elif route == "/api/index":
                done = {p.stem for p in (GOLD / version / "reviewed").glob("*.json") if not p.name.startswith("._")}
                status = {}
                for p in (GOLD / version / "reviewed").glob("*.json"):
                    if not p.name.startswith("._"):
                        status[p.stem] = json.loads(p.read_text()).get("status", "done")
                self.send(json.dumps({"frames": index, "status": {s["sample"]: status.get(
                    s["sample"].replace("/", "__")) for s in index if s["sample"].replace("/", "__") in done}})
                    .encode(), "application/json")
            elif route.startswith("/api/frame/"):
                i = int(route.rsplit("/", 1)[1])
                sample = rows[i]["sample"]
                path = reviewed_path(version, sample)
                body = json.loads(path.read_text()) if path.exists() else {"labels": labels[sample], "status": None}
                self.send(json.dumps({"sample": sample, "prefill": labels[sample], **body}).encode(),
                          "application/json")
            elif route.startswith("/image/"):
                self.send(jpeg(int(route.rsplit("/", 1)[1].split(".")[0])), "image/jpeg")
            else:
                self.send_error(404)

        def do_POST(self):
            origin = self.headers.get("Origin")
            if origin and urlsplit(origin).netloc != self.headers.get("Host"):
                self.send_error(403)
                return
            route = urlsplit(self.path).path
            if not route.startswith("/api/frame/"):
                self.send_error(404)
                return
            i = int(route.rsplit("/", 1)[1])
            size = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(size))
            if body.get("status") not in ("done", "unsure", "skip"):
                self.send(b'{"error": "status must be done, unsure or skip"}', "application/json", 400)
                return
            path = reviewed_path(version, rows[i]["sample"])
            tmp = path.with_suffix(".pending")
            tmp.write_text(json.dumps({"labels": body["labels"], "status": body["status"],
                                       "note": body.get("note", ""), "reviewer": REVIEWER}, indent=1))
            tmp.replace(path)
            self.send(b'{"saved": true}', "application/json")

        def log_message(self, fmt, *args):
            pass

    print(f"Gold review ({len(rows)} frames): http://127.0.0.1:{port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


def export(version: str, prefill: str) -> None:
    """Reviewed frames (status done) to the labels source gold_<version>."""
    frames = schema.read(prefill, "frames")
    people, ball, court, scene, keep = [], [], [], [], []
    for s in frames["sample"]:
        path = reviewed_path(version, s)
        if not path.exists():
            continue
        body = json.loads(path.read_text())
        if body["status"] != "done":
            continue
        keep.append(s)
        lab = body["labels"]
        nan = lambda v: np.nan if v is None else v
        for i, p in enumerate(lab["people"]):
            people.append({"sample": s, "person": i, "track": None, "box": [nan(v) for v in p["box"]],
                           "kp": [nan(v) for v in p["kp"]], "stroke": None, "player": p.get("player", False),
                           "labeler": REVIEWER})
        if lab.get("ball") is not None:
            b = lab["ball"]
            ball.append({"sample": s, "x": nan(b["x"]), "y": nan(b["y"]), "visible": b["visible"],
                         "labeler": REVIEWER})
        if lab.get("court") is not None:
            court.append({"sample": s, "kp": [nan(v) for v in np.ravel(lab["court"])], "labeler": REVIEWER})
        scene.append({"sample": s, "view": lab.get("view"), "in_play": lab.get("in_play"), "labeler": REVIEWER})
    f = frames[frames["sample"].isin(keep)]
    out = schema.write(f"gold_{version}", {
        "frames": f,
        "people": pd.DataFrame(people, columns=schema.TABLES["people"] + ["player"]),
        "ball": pd.DataFrame(ball, columns=schema.TABLES["ball"]),
        "court": pd.DataFrame(court, columns=schema.TABLES["court"]),
        "scene": pd.DataFrame(scene, columns=schema.TABLES["scene"])})
    print(f"{len(f)} reviewed frames -> {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["serve", "export"])
    p.add_argument("--version", default="v1")
    p.add_argument("--prefill", required=True)
    p.add_argument("--port", type=int, default=8770)
    a = p.parse_args()
    serve(a.version, a.prefill, a.port) if a.command == "serve" else export(a.version, a.prefill)


if __name__ == "__main__":
    main()
