"""Astra on hit timing, hitter and fine stroke type, scored on E2E-Spot and F3Set.

Labels (both human, on US Open broadcasts downloaded at 720p):

  E2E-Spot tennis test, US Open 2019 final: every serve, swing and bounce,
      tagged near / far court, swings with a coarse stroke comment.
  F3Set tennis test, 2 matches: every shot, with hitter side, court position,
      forehand / backhand, technique (ground stroke, slice, volley, smash,
      drop, lob), direction and outcome.

Frame numbers are at 29.97 fps; the Jabeur video is 59.94 fps, so its source
frames are doubled.

Two tests, one Astra call per sample:

  timing  a 4 x 4 sheet of 16 consecutive frames (0.53 s). 80 windows hold a
          labelled hit (serve or swing) at a random position, 20 hold none.
          Astra returns the contact frame (or none), the hitter (near / far),
          and the first bounce frame in the window (or none; E2E-Spot only).
          A hit is found within 1 or 2 frames of the label.
  stroke  F3Set shots: 12 frames every 4th frame (0.3 s before contact to
          1.2 s after), with the hitter and contact frame given. Astra returns
          forehand / backhand, technique and direction.

    python -m dsa.astra.events --n 100
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from dsa.astra.codex import ask

VIDEOS = Path("data/event_videos")
E2E = Path("data/e2e_spot_repo/data/tennis/test.json")
F3 = Path("data/f3set_repo/data/f3set-tennis/test.json")
MATCHES = {"usopen_2019_mens_final_medvedev_nadal": "e2e",
           "20210910-M-US_Open-SF-Novak_Djokovic-Alexander_Zverev": "f3",
           "20220906-W-US_Open-QF-Ajla_Tomljanovic-Ons_Jabeur": "f3"}
TECH = {"gs": "ground stroke (topspin or flat)", "slice": "slice", "volley": "volley", "smash": "smash / overhead",
        "drop": "drop shot", "lob": "lob"}
DIRS = {"CC": "cross-court", "DL": "down the line", "DM": "down the middle", "II": "inside-in",
        "IO": "inside-out"}
SERVE_DIRS = {"T": "down the T", "B": "at the body", "W": "out wide"}

TIMING_SCHEMA = {
    "type": "object",
    "properties": {"hit_frame": {"type": "integer"}, "hitter": {"type": "string", "enum": ["near", "far", "none"]},
                   "bounce_frame": {"type": "integer"}},
    "required": ["hit_frame", "hitter", "bounce_frame"], "additionalProperties": False,
}
TIMING_PROMPT = (
    "The attached sheet shows 16 consecutive frames (1/30 s apart) of a tennis broadcast, numbered 1-16 "
    "in reading order. The near player is the one at the bottom of the court, the far player at the top. "
    "Find the frame where a player's racket strikes the ball (a serve or any shot) and who hits it. If no "
    "racket touches the ball in these frames, set hit_frame to 0 and hitter to none. If there are two "
    "hits, give the first. Also give the first frame where the ball bounces on the court, or 0 if it "
    "does not bounce in these frames. Look at the image directly; do not run any commands."
)
STROKE_SCHEMA = {
    "type": "object",
    "properties": {"hand": {"type": "string", "enum": ["forehand", "backhand"]},
                   "technique": {"type": "string", "enum": list(TECH)},
                   "direction": {"type": "string", "enum": list(DIRS) + list(SERVE_DIRS)}},
    "required": ["hand", "technique", "direction"], "additionalProperties": False,
}


def stroke_prompt(hitter: str, contact: int, serve: bool) -> str:
    kind = "serves" if serve else "hits the ball"
    dirs = SERVE_DIRS if serve else DIRS
    return (
        f"The attached sheet shows 12 frames of a tennis broadcast, 4/30 s apart, numbered 1-12 in reading "
        f"order. The {hitter} player ({'bottom' if hitter == 'near' else 'top'} of the court) {kind} at about "
        f"frame {contact}. Classify that shot. hand: forehand or backhand (a serve counts as forehand). "
        f"technique: " + "; ".join(f"{k} = {v}" for k, v in TECH.items()) + " (a serve is gs). direction, "
        "where the ball goes: " + "; ".join(f"{k} = {v}" for k, v in dirs.items())
        + ("" if serve else ". Inside-out and inside-in are forehands hit from the backhand side (or "
           "backhands from the forehand side), inside-out going cross-court, inside-in down the line")
        + ". Look at the image directly; do not run any commands."
    )


def clips() -> list[dict]:
    out = []
    for path, src in ((E2E, "e2e"), (F3, "f3")):
        for c in json.load(open(path)):
            match, start, _ = c["video"].rsplit("_", 2)
            if MATCHES.get(match) == src:
                out.append({**c, "match": match, "start": int(start), "src": src})
    return out


def is_hit(ev: dict) -> bool:
    """E2E-Spot labels serves, swings and bounces; F3Set labels only shots."""
    return not ev["label"].endswith("bounce")


def side(ev: dict) -> str:
    return ev["label"].split("_")[0]


def read(match: str, frames: list[int]) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(VIDEOS / f"{match}.mp4"))
    mult = round(cap.get(cv2.CAP_PROP_FPS) / 29.97)
    want = [f * mult for f in frames]
    cap.set(cv2.CAP_PROP_POS_FRAMES, want[0])
    out, f = [], want[0]
    for w in want:
        while f < w:
            cap.grab()
            f += 1
        ok, bgr = cap.read()
        f += 1
        if not ok:
            raise RuntimeError(f"cannot read {match} frame {w}")
        out.append(bgr)
    cap.release()
    return out


def sheet(frames: list[np.ndarray], cols: int, width: int) -> np.ndarray:
    tiles = []
    for i, f in enumerate(frames):
        t = cv2.resize(f, (width, width * 9 // 16))
        cv2.putText(t, str(i + 1), (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
        tiles.append(t)
    return np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])


def timing_samples(cs: list[dict], n: int, rng) -> list[dict]:
    hits, empty = [], []
    for c in cs:
        hit_frames = [e["frame"] for e in c["events"] if is_hit(e)]
        for e in c["events"]:
            if is_hit(e):
                hits.append((c, e))
        for f in range(20, c["num_frames"] - 36, 8):
            if all(abs(f + 8 - h) > 30 for h in hit_frames):
                empty.append((c, f))
    k = round(0.8 * n)
    out = []
    for c, e in (hits[i] for i in rng.choice(len(hits), k, replace=False)):
        first = e["frame"] - int(rng.integers(0, 16))
        out.append({"clip": c, "first": first})
    for c, f in (empty[i] for i in rng.choice(len(empty), n - k, replace=False)):
        out.append({"clip": c, "first": f})
    for s in out:
        c, lo = s["clip"], s["first"]
        inside = [e for e in c["events"] if lo <= e["frame"] < lo + 16]
        h = [e for e in inside if is_hit(e)]
        b = [e for e in inside if not is_hit(e)]
        s.update(id=f"{c['video']}@{lo}", hit=h[0]["frame"] - lo + 1 if h else 0, hitter=side(h[0]) if h else "none",
                 bounce=b[0]["frame"] - lo + 1 if b else 0)
    return out


def stroke_samples(cs: list[dict], n: int, rng) -> list[dict]:
    shots = [(c, e) for c in cs if c["src"] == "f3" for e in c["events"]
             if e["frame"] - 9 >= 0 and e["frame"] + 36 < c["num_frames"]]
    out = []
    for c, e in (shots[i] for i in rng.choice(len(shots), n, replace=False)):
        parts = e["label"].split("_")
        out.append({"clip": c, "first": e["frame"] - 8, "id": f"{c['video']}@{e['frame']}", "hitter": parts[0],
                    "serve": parts[2] == "serve", "hand": {"fh": "forehand", "bh": "backhand"}.get(parts[3], "forehand"),
                    "technique": parts[4] if parts[4] != "-" else "gs", "direction": parts[5]})
    return out


def label_timing(s: dict, images: Path, cache: Path) -> dict:
    c = s["clip"]
    path = images / f"timing_{s['id']}.jpg"
    if not path.exists():
        cv2.imwrite(str(path), sheet(read(c["match"], [c["start"] + s["first"] + i for i in range(16)]), 4, 480))
    ans, sec = ask(TIMING_PROMPT, [path.resolve()], TIMING_SCHEMA, cache)
    return {k: v for k, v in s.items() if k != "clip"} | {
        "src": c["src"], "pred_hit": ans["hit_frame"], "pred_hitter": ans["hitter"],
        "pred_bounce": ans["bounce_frame"], "sec": sec, "image": str(path)}


def label_stroke(s: dict, images: Path, cache: Path) -> dict:
    c = s["clip"]
    path = images / f"stroke_{s['id']}.jpg"
    if not path.exists():
        cv2.imwrite(str(path), sheet(read(c["match"], [c["start"] + s["first"] + 4 * i for i in range(12)]), 4, 480))
    ans, sec = ask(stroke_prompt(s["hitter"], 3, s["serve"]), [path.resolve()], STROKE_SCHEMA, cache)
    return {k: v for k, v in s.items() if k != "clip"} | {
        "pred_hand": ans["hand"], "pred_technique": ans["technique"], "pred_direction": ans["direction"],
        "sec": sec, "image": str(path)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default="output/astra_eval/events")
    a = p.parse_args()

    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    cs = clips()
    timing, stroke = timing_samples(cs, a.n, rng), stroke_samples(cs, a.n, rng)
    print(f"{len(cs)} clips; {len(timing)} timing windows, {len(stroke)} strokes", flush=True)
    with ThreadPoolExecutor(a.workers) as pool:
        t = pd.DataFrame(pool.map(lambda s: label_timing(s, out / "images", out / "cache"), timing))
        s = pd.DataFrame(pool.map(lambda s: label_stroke(s, out / "images", out / "cache"), stroke))
    t.to_parquet(out / "timing.parquet", index=False)
    s.to_parquet(out / "stroke.parquet", index=False)

    has = t[t.hit > 0]
    d = (has.pred_hit - has.hit).abs().where(has.pred_hit > 0)
    b = t[(t.src == "e2e") & (t.bounce > 0)]
    db = (b.pred_bounce - b.bounce).abs().where(b.pred_bounce > 0)
    summary = {
        "timing": {"windows_with_hit": len(has), "found_within1": float((d <= 1).mean()),
                   "found_within2": float((d <= 2).mean()), "median_frames_off": float(d.median()),
                   "hitter_correct": float((has.pred_hitter == has.hitter).mean()),
                   "empty_windows": int((t.hit == 0).sum()),
                   "hit_claimed_in_empty": int(((t.hit == 0) & (t.pred_hit > 0)).sum()),
                   "bounces": len(b), "bounce_within2": float((db <= 2).mean())},
        "stroke": {"n": len(s), "serves": int(s.serve.sum()),
                   "hand": float((s[~s.serve].pred_hand == s[~s.serve].hand).mean()),
                   "technique": float((s[~s.serve].pred_technique == s[~s.serve].technique).mean()),
                   "direction": float((s.pred_direction == s.direction).mean()),
                   "direction_rallies": float((s[~s.serve].pred_direction == s[~s.serve].direction).mean())},
    }
    new = pd.concat([t.sec, s.sec])
    summary["sec_per_call"] = float(new[new > 0].mean()) if (new > 0).any() else None
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    rally = s[~s.serve]
    for col in ("technique", "direction"):
        print(f"\n{col}\n" + pd.crosstab(rally[col], rally[f"pred_{col}"]).to_string())


if __name__ == "__main__":
    main()
