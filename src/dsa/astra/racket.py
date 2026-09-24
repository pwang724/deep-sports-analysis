"""Astra as a racket keypoint labeler, scored on RacketVision tennis.

RacketVision (MIT) labels rackets in 1080p broadcast frames with a box and
five points: top and bottom of the head along its long axis, the handle end,
and the left and right edges of the head. Rackets are small: median 55 px on
the long side. Labels are per-frame JSON (tennis/all/<match>/racket/<clip>/
<frame>.json) over clips tennis/videos/<match>_<clip>.mp4; only matches whose
clips are downloaded are used.

Each sampled racket gets a 360 x 360 px crop that contains it at a random
offset (as a player crop would), upscaled to 768 px. Astra returns the five
points. Scored by error relative to racket length (top to handle), counting
labelled-visible points only, against a baseline that puts every point on the
box centre. Left and right are ambiguous on a racket seen edge-on, so a
side-agnostic score is reported too.

    python -m dsa.astra.racket --n 100
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

ROOT = Path("data/racketvision/tennis")
POINTS = ["top", "bottom", "handle", "left", "right"]
CROP, LONG_SIDE = 360, 768

SCHEMA = {
    "type": "object",
    "properties": {"found": {"type": "boolean"},
                   **{j: {"type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
                          "required": ["x", "y"], "additionalProperties": False} for j in POINTS}},
    "required": ["found"] + POINTS,
    "additionalProperties": False,
}


def prompt(w: int, h: int) -> str:
    return (
        f"The attached image is {w} x {h} pixels, a crop of a tennis broadcast frame containing a tennis "
        "racket held by a player. Return pixel coordinates (x from the left edge, y from the top edge) of "
        "five points on that racket: top = the tip of the head, furthest from the handle; bottom = where "
        "the head meets the throat, on the head's long axis; handle = the butt end of the handle; left and "
        "right = the widest points on the two sides of the head, left being the one further left in the "
        "image. Set found to false if you cannot see a racket, and still give your best guess. "
        "Look at the image directly; do not run any commands."
    )


def sample(n: int, seed: int = 0) -> list[dict]:
    items = []
    for video in sorted((ROOT / "videos").glob("*.mp4")):
        match, clip = video.stem.rsplit("_", 1)
        for f in sorted((ROOT / "all" / match / "racket" / clip).glob("*.json")):
            for k, r in enumerate(json.load(open(f))):
                kp = np.array(r["keypoints"], float)
                if (kp[:, 2] > 0).sum() >= 4 and kp[0, 2] > 0 and kp[2, 2] > 0:
                    items.append({"video": str(video), "frame": int(f.stem), "k": k,
                                  "kp": kp, "box": np.array(r["bbox_xywh"], float)})
    rng = np.random.default_rng(seed)
    picked = [items[i] for i in sorted(rng.choice(len(items), min(n, len(items)), replace=False))]
    for it in picked:
        cx, cy = it["box"][:2] + it["box"][2:] / 2
        dx, dy = rng.uniform(-0.3, 0.3, 2) * CROP
        it["origin"] = np.array([cx + dx - CROP / 2, cy + dy - CROP / 2]).round()
    return picked


def score(name: str, xy: np.ndarray, it: dict) -> dict:
    kp = it["kp"]
    length = np.linalg.norm(kp[0, :2] - kp[2, :2])
    vis = kp[:, 2] > 0
    err = np.linalg.norm(xy - kp[:, :2], axis=1) / length
    swapped = xy[[0, 1, 2, 4, 3]]
    err_sw = np.linalg.norm(swapped - kp[:, :2], axis=1) / length
    if err_sw[vis].mean() < err[vis].mean():
        err_any = err_sw
    else:
        err_any = err
    r = {"id": f"{Path(it['video']).stem}_{it['frame']}_{it['k']}", "backend": name, "length_px": length,
         "mean_err": float(err[vis].mean()), "pck10": float((err[vis] < 0.1).mean()),
         "pck20": float((err[vis] < 0.2).mean()), "pck10_any_side": float((err_any[vis] < 0.1).mean())}
    for k, j in enumerate(POINTS):
        r[f"{j}_err"] = err[k] if vis[k] else np.nan
    return r


def label_one(it: dict, crops: Path, cache: Path) -> list[dict]:
    cap = cv2.VideoCapture(it["video"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, it["frame"])
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read frame {it['frame']} of {it['video']}")
    H, W = bgr.shape[:2]
    x0, y0 = np.clip(it["origin"], 0, [W - CROP, H - CROP]).astype(int)
    it["origin"] = np.array([x0, y0], float)
    crop = cv2.resize(bgr[y0:y0 + CROP, x0:x0 + CROP], (LONG_SIDE, LONG_SIDE), interpolation=cv2.INTER_CUBIC)
    path = crops / f"{Path(it['video']).stem}_{it['frame']}_{it['k']}.png"
    cv2.imwrite(str(path), crop)
    answer, sec = ask(prompt(LONG_SIDE, LONG_SIDE), [path], SCHEMA, cache)
    scale = LONG_SIDE / CROP
    xy = np.array([[answer[j]["x"], answer[j]["y"]] for j in POINTS]) / scale + it["origin"]
    row = score("astra", xy, it)
    row.update(found=bool(answer["found"]), sec=sec, crop=str(path),
               **{f"{j}_{c}": xy[k, i] for k, j in enumerate(POINTS) for i, c in enumerate("xy")})
    centre = it["box"][:2] + it["box"][2:] / 2
    return [row, score("box_centre", np.tile(centre, (5, 1)), it)]


def worst_sheet(df: pd.DataFrame, items: dict, out: Path, n: int = 12) -> None:
    """Worst crops, labelled points green, Astra red, around the racket."""
    tiles = []
    for _, r in df.nlargest(n, "mean_err").iterrows():
        it = items[r.id]
        img = cv2.imread(r.crop)
        s = LONG_SIDE / CROP
        for pts, color in ((it["kp"][:, :2], (0, 200, 0)),
                           (np.array([[r[f"{j}_x"], r[f"{j}_y"]] for j in POINTS]), (0, 0, 255))):
            p = ((pts - it["origin"]) * s).astype(int)
            for a, b in ((0, 1), (1, 2), (3, 4)):
                cv2.line(img, tuple(p[a]), tuple(p[b]), color, 2)
            for q in p:
                cv2.circle(img, tuple(q), 5, color, -1)
        tile = cv2.resize(img, (256, 256))
        cv2.putText(tile, f"{r.mean_err:.2f}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        tiles.append(tile)
    while len(tiles) % 6:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(out), np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default="output/astra_eval/racket")
    a = p.parse_args()

    out = Path(a.out)
    (out / "crops").mkdir(parents=True, exist_ok=True)
    items = sample(a.n)
    print(f"{len(items)} rackets", flush=True)
    rows = []
    with ThreadPoolExecutor(a.workers) as pool:
        for r in pool.map(lambda it: label_one(it, out / "crops", out / "cache"), items):
            rows += r
    df = pd.DataFrame(rows)
    df.to_parquet(out / "per_racket.parquet", index=False)
    worst_sheet(df[df.backend == "astra"], {r["id"]: it for r, it in zip(rows[::2], items)}, out / "worst.jpg")

    summary = {}
    for b, g in df.groupby("backend"):
        summary[b] = {"n": len(g), "mean_err": float(g.mean_err.mean()), "pck10": float(g.pck10.mean()),
                      "pck20": float(g.pck20.mean()), "pck10_any_side": float(g.pck10_any_side.mean()),
                      "point_err_median": {j: float(g[f"{j}_err"].median()) for j in POINTS}}
    astra = df[df.backend == "astra"]
    summary["astra"]["found"] = float(astra.found.mean())
    new = astra.sec[astra.sec > 0]
    summary["astra"]["sec_per_call"] = float(new.mean()) if len(new) else None
    summary["racket_length_px_median"] = float(astra.length_px.median())
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
