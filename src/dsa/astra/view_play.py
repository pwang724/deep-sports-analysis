"""Astra on view and in-play, scored against our reviewed cut lists.

Neither label is exact ground truth; both are the best reviewed labels we have:

  view     USO final highlights, `shots.json`: keep = the high end-on court
           camera, discard = everything else (other angles, close-ups, crowd,
           graphics). Method output tuned and inspected on this clip; `review`
           shots are excluded, and so are frames within 0.5 s of a shot edge.
           Our definition also counts the low end-on camera as usable (it is
           the phone view); the USO recipe discarded it, so those disagree.
  in play  Richard / Dylan phone recordings, `cuts.json`: frames at least 2 s
           inside a retained activity window are in play; frames at least 3 s
           from any window are not. Windows keep some short resets, so a few
           "in play" frames can show a pause.

View: Astra sees one frame. In play: a 2 x 2 sheet of four frames 0.5 s apart
centred on the sample time, since one frame cannot show whether play is on.

    python -m dsa.astra.view_play --n 50
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

USO_VIDEO = "data/raw/uso2026_final_highlights.mp4"
USO_SHOTS = "output/preprocess/uso2026_highlights/shots.json"
PHONE = "output/preprocess/desktop_tennis"

SCHEMA = {"type": "object", "properties": {"answer": {"type": "boolean"}},
          "required": ["answer"], "additionalProperties": False}
VIEW_PROMPT = (
    "The attached frame is from a tennis video. Is it a usable analysis view: a camera behind one "
    "baseline, at any height, looking down the length of the court, with the whole court and both "
    "players' positions in view? Answer false for side-on or other angles, close-ups, replays zoomed "
    "on part of the court, crowd, stadium overviews and graphics. Look at the image directly; do "
    "not run any commands."
)
PLAY_PROMPT = (
    "The attached sheet shows four frames of a phone recording of a tennis session, 0.5 s apart, "
    "in reading order (time in the corner). Consider only the main court in view, the one whose "
    "baseline is nearest the camera; ignore anyone on neighbouring courts. Is tennis being played "
    "on that court at this moment: a serve being hit, or a rally under way, including warm-up "
    "rallies? A point runs from the serve toss to the end of the rally. Answer false when its "
    "players are collecting balls, walking, bouncing the ball before a serve, changing ends, "
    "resting or otherwise waiting between points, or when the court is empty. "
    "Look at the image directly; do not run any commands."
)


def read_frame(cap: cv2.VideoCapture, frame: int, width: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, bgr = cap.read()
    if not ok:
        raise RuntimeError(f"cannot read frame {frame}")
    return cv2.resize(bgr, (width, round(bgr.shape[0] * width / bgr.shape[1])))


def view_samples(n: int, rng: np.random.Generator) -> list[dict]:
    d = json.load(open(USO_SHOTS))
    fps, pad = d["fps"], round(0.5 * d["fps"])
    out = []
    for label in ("keep", "discard"):
        shots = [s for s in d["shots"] if s["label"] == label and s["end_frame"] - s["start_frame"] > 3 * pad]
        weights = np.array([s["end_frame"] - s["start_frame"] - 2 * pad for s in shots], float)
        for k in rng.choice(len(shots), n, p=weights / weights.sum()):
            s = shots[k]
            f = int(rng.integers(s["start_frame"] + pad, s["end_frame"] - pad))
            out.append({"task": "view", "source": "uso", "frame": f, "time": f / fps, "truth": label == "keep"})
    return out


def play_samples(n: int, rng: np.random.Generator) -> list[dict]:
    out = []
    for who in ("richard", "dylan"):
        cuts = json.load(open(f"{PHONE}/{who}/cuts.json"))
        fps = cuts["fps"]
        total = int(cv2.VideoCapture(f"{PHONE}/{who}/analysis.mp4").get(cv2.CAP_PROP_FRAME_COUNT))
        clips = [(c["start_frame"], c["end_frame"]) for c in cuts["clips"]]
        inside = np.zeros(total, bool)
        near = np.zeros(total, bool)
        for a, b in clips:
            inside[a + int(2 * fps):max(a + int(2 * fps), b - int(2 * fps))] = True
            near[max(0, a - int(3 * fps)):b + int(3 * fps)] = True
        edge = int(2 * fps)  # sheet frames must exist on both sides
        for truth, pool in ((True, inside), (False, ~near)):
            pool[:edge] = pool[-edge:] = False
            for f in rng.choice(np.flatnonzero(pool), n // 2, replace=False):
                out.append({"task": "play", "source": who, "frame": int(f), "time": f / fps, "truth": truth})
    return out


def play_sheet(cap: cv2.VideoCapture, frame: int, fps: float) -> np.ndarray:
    tiles = []
    for dt in (-0.75, -0.25, 0.25, 0.75):
        f = frame + round(dt * fps)
        tile = read_frame(cap, f, 640)
        cv2.putText(tile, f"{f / fps:.2f} s", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        tiles.append(tile)
    return np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])


def label_one(s: dict, images: Path, cache: Path) -> dict:
    path = images / f"{s['task']}_{s['source']}_{s['frame']}.jpg"
    if not path.exists():
        if s["task"] == "view":
            cap = cv2.VideoCapture(USO_VIDEO)
            cv2.imwrite(str(path), read_frame(cap, s["frame"], 1280))
        else:
            cap = cv2.VideoCapture(f"{PHONE}/{s['source']}/analysis.mp4")
            cv2.imwrite(str(path), play_sheet(cap, s["frame"], cap.get(cv2.CAP_PROP_FPS)))
        cap.release()
    answer, sec = ask(VIEW_PROMPT if s["task"] == "view" else PLAY_PROMPT, [path], SCHEMA, cache)
    return {**s, "pred": bool(answer["answer"]), "sec": sec, "image": str(path)}


def mistakes_sheet(df: pd.DataFrame, out: Path, n: int = 8) -> None:
    tiles = []
    for _, r in df[df.pred != df.truth].head(n).iterrows():
        tile = cv2.resize(cv2.imread(r.image), (480, 270))
        cv2.putText(tile, f"{r.source} {r.time:.0f}s truth={r.truth}", (6, 262), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        tiles.append(tile)
    if not tiles:
        return
    while len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(out), np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=50, help="samples per class per task")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default="output/astra_eval/view_play")
    a = p.parse_args()

    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    samples = view_samples(a.n, rng) + play_samples(a.n, rng)
    print(f"{len(samples)} samples", flush=True)
    with ThreadPoolExecutor(a.workers) as pool:
        df = pd.DataFrame(pool.map(lambda s: label_one(s, out / "images", out / "cache"), samples))
    df.to_parquet(out / "per_sample.parquet", index=False)

    summary = {}
    for task, g in df.groupby("task"):
        mistakes_sheet(g, out / f"mistakes_{task}.jpg")
        new = g.sec[g.sec > 0]
        summary[task] = {"n": len(g), "accuracy": float((g.pred == g.truth).mean()),
                         "recall_true": float(g[g.truth].pred.mean()),
                         "recall_false": float((~g[~g.truth].pred).mean()),
                         "sec_per_call": float(new.mean()) if len(new) else None}
        print(f"\n{task}: accuracy {summary[task]['accuracy']:.1%} on {len(g)}")
        print(pd.crosstab(g.truth, g.pred, rownames=["truth"], colnames=["astra"]).to_string())
    (out / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
